int cmlq_afloat_subtract_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_afloat_subtract_afloat")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            FLOAT_subtract(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                FLOAT_subtract(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 0);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        FLOAT_subtract(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", FLOAT_subtract(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            FLOAT_subtract(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", FLOAT_subtract(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_afloat_subtract_afloat")
    return 0;
fail:
    return -1;
}

int cmlq_afloat_inplace_subtract_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_afloat_inplace_subtract_afloat")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

            result = lhs;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            FLOAT_subtract(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
    } else if (CACHE_MATCH_ITERATOR()) {

            result = lhs;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                FLOAT_subtract(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
            fixed_strides[2] = PyArray_STRIDES(lhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
                fixed_strides[2] = PyArray_ITEMSIZE(lhs);
            }
        }
    }

        // inplace operation, the result is the same as the lhs
        Py_INCREF(lhs);
        result = lhs;

    if(fast_path == 1) {

        assert(lhs == result);

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        FLOAT_subtract(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", FLOAT_subtract(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        // this is an inplace operation. We do not cache the result here because no result array is allocated anyway

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        assert(ops[0] == ops[2]);

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            FLOAT_subtract(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", FLOAT_subtract(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        // no locality cache, we always need to deallocate the iterator
        // this is an inplace operation. We do not cache the result here because no result array is allocated anyway

        int should_deallocate = 1;

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_afloat_inplace_subtract_afloat")
    return 0;
fail:
    return -1;
}

int cmlq_afloat_add_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_afloat_add_afloat")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            FLOAT_add(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                FLOAT_add(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 1);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        FLOAT_add(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", FLOAT_add(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            FLOAT_add(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", FLOAT_add(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_afloat_add_afloat")
    return 0;
fail:
    return -1;
}

int cmlq_afloat_inplace_add_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_afloat_inplace_add_afloat")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

            result = lhs;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            FLOAT_add(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
    } else if (CACHE_MATCH_ITERATOR()) {

            result = lhs;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                FLOAT_add(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
            fixed_strides[2] = PyArray_STRIDES(lhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
                fixed_strides[2] = PyArray_ITEMSIZE(lhs);
            }
        }
    }

        // inplace operation, the result is the same as the lhs
        Py_INCREF(lhs);
        result = lhs;

    if(fast_path == 1) {

        assert(lhs == result);

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        FLOAT_add(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", FLOAT_add(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        // this is an inplace operation. We do not cache the result here because no result array is allocated anyway

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        assert(ops[0] == ops[2]);

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            FLOAT_add(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", FLOAT_add(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        // no locality cache, we always need to deallocate the iterator
        // this is an inplace operation. We do not cache the result here because no result array is allocated anyway

        int should_deallocate = 1;

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_afloat_inplace_add_afloat")
    return 0;
fail:
    return -1;
}

int cmlq_afloat_multiply_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_afloat_multiply_afloat")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            FLOAT_multiply(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                FLOAT_multiply(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 1);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        FLOAT_multiply(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", FLOAT_multiply(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            FLOAT_multiply(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", FLOAT_multiply(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_afloat_multiply_afloat")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_subtract_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_subtract_adouble")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_subtract(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_subtract(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 0);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_subtract(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_subtract(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_subtract(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_subtract(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_subtract_adouble")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_inplace_subtract_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_inplace_subtract_adouble")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

            result = lhs;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_subtract(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
    } else if (CACHE_MATCH_ITERATOR()) {

            result = lhs;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_subtract(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
            fixed_strides[2] = PyArray_STRIDES(lhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
                fixed_strides[2] = PyArray_ITEMSIZE(lhs);
            }
        }
    }

        // inplace operation, the result is the same as the lhs
        Py_INCREF(lhs);
        result = lhs;

    if(fast_path == 1) {

        assert(lhs == result);

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_subtract(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_subtract(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        // this is an inplace operation. We do not cache the result here because no result array is allocated anyway

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        assert(ops[0] == ops[2]);

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_subtract(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_subtract(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        // no locality cache, we always need to deallocate the iterator
        // this is an inplace operation. We do not cache the result here because no result array is allocated anyway

        int should_deallocate = 1;

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_inplace_subtract_adouble")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_subtract_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_subtract_sfloat")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyFloat_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;

        PyArrayObject *rhs = (PyArrayObject *)PyArray_FromAny(m2, right_descr, 0, 0, 0, NULL);

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_subtract(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_subtract(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 0);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_subtract(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_subtract(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_subtract(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_subtract(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_subtract_sfloat")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_subtract_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_subtract_sfloat_broadcast_cache")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyFloat_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;

        assert(elem->state == BROADCAST && elem->result);
        PyArrayObject *rhs = elem->result;

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 0);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_subtract(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_subtract(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_subtract(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_subtract(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        // no locality cache, we always need to deallocate the iterator

        int should_deallocate = 1;

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);

    // the rhs is a cached broadcast array, no decref

    Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_subtract_sfloat_broadcast_cache")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_add_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_add_adouble")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_add(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_add(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 1);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_add(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_add(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_add(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_add(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_add_adouble")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_inplace_add_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_inplace_add_adouble")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

            result = lhs;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_add(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
    } else if (CACHE_MATCH_ITERATOR()) {

            result = lhs;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_add(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
            fixed_strides[2] = PyArray_STRIDES(lhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
                fixed_strides[2] = PyArray_ITEMSIZE(lhs);
            }
        }
    }

        // inplace operation, the result is the same as the lhs
        Py_INCREF(lhs);
        result = lhs;

    if(fast_path == 1) {

        assert(lhs == result);

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_add(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_add(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        // this is an inplace operation. We do not cache the result here because no result array is allocated anyway

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        assert(ops[0] == ops[2]);

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_add(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_add(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        // no locality cache, we always need to deallocate the iterator
        // this is an inplace operation. We do not cache the result here because no result array is allocated anyway

        int should_deallocate = 1;

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_inplace_add_adouble")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_add_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_add_sfloat")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

    if (PyFloat_CheckExact(m1)) {
        PyObject *tmp = m1;
        m1 = m2;
        m2 = tmp;
    }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyFloat_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;

        PyArrayObject *rhs = (PyArrayObject *)PyArray_FromAny(m2, right_descr, 0, 0, 0, NULL);

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_add(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_add(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 1);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_add(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_add(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_add(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_add(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_add_sfloat")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_add_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_add_sfloat_broadcast_cache")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

    if (PyFloat_CheckExact(m1)) {
        PyObject *tmp = m1;
        m1 = m2;
        m2 = tmp;
    }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyFloat_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;

        assert(elem->state == BROADCAST && elem->result);
        PyArrayObject *rhs = elem->result;

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 1);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_add(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_add(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_add(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_add(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        // no locality cache, we always need to deallocate the iterator

        int should_deallocate = 1;

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);

    // the rhs is a cached broadcast array, no decref

    Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_add_sfloat_broadcast_cache")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_inplace_add_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_inplace_add_sfloat")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

    if (PyFloat_CheckExact(m1)) {
        PyObject *tmp = m1;
        m1 = m2;
        m2 = tmp;
    }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyFloat_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;

        PyArrayObject *rhs = (PyArrayObject *)PyArray_FromAny(m2, right_descr, 0, 0, 0, NULL);

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

            result = lhs;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_add(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
    } else if (CACHE_MATCH_ITERATOR()) {

            result = lhs;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_add(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

        // inplace operation, the result is the same as the lhs
        Py_INCREF(lhs);
        result = lhs;

    if(fast_path == 1) {

        assert(lhs == result);

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_add(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_add(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        // this is an inplace operation. We do not cache the result here because no result array is allocated anyway

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        assert(ops[0] == ops[2]);

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_add(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_add(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        // no locality cache, we always need to deallocate the iterator
        // this is an inplace operation. We do not cache the result here because no result array is allocated anyway

        int should_deallocate = 1;

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_inplace_add_sfloat")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_inplace_add_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_inplace_add_sfloat_broadcast_cache")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

    if (PyFloat_CheckExact(m1)) {
        PyObject *tmp = m1;
        m1 = m2;
        m2 = tmp;
    }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyFloat_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;

        assert(elem->state == BROADCAST && elem->result);
        PyArrayObject *rhs = elem->result;

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

        // inplace operation, the result is the same as the lhs
        Py_INCREF(lhs);
        result = lhs;

    if(fast_path == 1) {

        assert(lhs == result);

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_add(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_add(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        // this is an inplace operation. We do not cache the result here because no result array is allocated anyway

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        assert(ops[0] == ops[2]);

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_add(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_add(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        // no locality cache, we always need to deallocate the iterator
        // this is an inplace operation. We do not cache the result here because no result array is allocated anyway

        int should_deallocate = 1;

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);

    // the rhs is a cached broadcast array, no decref

    Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_inplace_add_sfloat_broadcast_cache")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_multiply_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_multiply_adouble")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_multiply(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_multiply(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 1);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_multiply(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_multiply(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_multiply(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_multiply(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_multiply_adouble")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_multiply_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_multiply_slong")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

    if (PyLong_CheckExact(m1)) {
        PyObject *tmp = m1;
        m1 = m2;
        m2 = tmp;
    }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyLong_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;
        right_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArrayObject *rhs = (PyArrayObject *)PyArray_FromAny(m2, right_descr, 0, 0, 0, NULL);

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_multiply(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_multiply(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 1);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_multiply(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_multiply(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_multiply(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_multiply(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_multiply_slong")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_multiply_slong_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_multiply_slong_broadcast_cache")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

    if (PyLong_CheckExact(m1)) {
        PyObject *tmp = m1;
        m1 = m2;
        m2 = tmp;
    }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyLong_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;
        right_descr = PyArray_DescrFromType(NPY_DOUBLE);

        assert(elem->state == BROADCAST && elem->result);
        PyArrayObject *rhs = elem->result;

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 1);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_multiply(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_multiply(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_multiply(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_multiply(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        // no locality cache, we always need to deallocate the iterator

        int should_deallocate = 1;

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);

    // the rhs is a cached broadcast array, no decref

    Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_multiply_slong_broadcast_cache")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_multiply_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_multiply_sfloat")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

    if (PyFloat_CheckExact(m1)) {
        PyObject *tmp = m1;
        m1 = m2;
        m2 = tmp;
    }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyFloat_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;

        PyArrayObject *rhs = (PyArrayObject *)PyArray_FromAny(m2, right_descr, 0, 0, 0, NULL);

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_multiply(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_multiply(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 1);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_multiply(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_multiply(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_multiply(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_multiply(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_multiply_sfloat")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_multiply_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_multiply_sfloat_broadcast_cache")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

    if (PyFloat_CheckExact(m1)) {
        PyObject *tmp = m1;
        m1 = m2;
        m2 = tmp;
    }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyFloat_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;

        assert(elem->state == BROADCAST && elem->result);
        PyArrayObject *rhs = elem->result;

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 1);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_multiply(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_multiply(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_multiply(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_multiply(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        // no locality cache, we always need to deallocate the iterator

        int should_deallocate = 1;

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);

    // the rhs is a cached broadcast array, no decref

    Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_multiply_sfloat_broadcast_cache")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_inplace_multiply_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_inplace_multiply_sfloat")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

    if (PyFloat_CheckExact(m1)) {
        PyObject *tmp = m1;
        m1 = m2;
        m2 = tmp;
    }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyFloat_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;

        PyArrayObject *rhs = (PyArrayObject *)PyArray_FromAny(m2, right_descr, 0, 0, 0, NULL);

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

            result = lhs;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_multiply(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
    } else if (CACHE_MATCH_ITERATOR()) {

            result = lhs;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_multiply(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

        // inplace operation, the result is the same as the lhs
        Py_INCREF(lhs);
        result = lhs;

    if(fast_path == 1) {

        assert(lhs == result);

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_multiply(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_multiply(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        // this is an inplace operation. We do not cache the result here because no result array is allocated anyway

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        assert(ops[0] == ops[2]);

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_multiply(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_multiply(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        // no locality cache, we always need to deallocate the iterator
        // this is an inplace operation. We do not cache the result here because no result array is allocated anyway

        int should_deallocate = 1;

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_inplace_multiply_sfloat")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_inplace_multiply_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_inplace_multiply_sfloat_broadcast_cache")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

    if (PyFloat_CheckExact(m1)) {
        PyObject *tmp = m1;
        m1 = m2;
        m2 = tmp;
    }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyFloat_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;

        assert(elem->state == BROADCAST && elem->result);
        PyArrayObject *rhs = elem->result;

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

        // inplace operation, the result is the same as the lhs
        Py_INCREF(lhs);
        result = lhs;

    if(fast_path == 1) {

        assert(lhs == result);

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_multiply(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_multiply(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        // this is an inplace operation. We do not cache the result here because no result array is allocated anyway

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        assert(ops[0] == ops[2]);

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_multiply(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_multiply(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        // no locality cache, we always need to deallocate the iterator
        // this is an inplace operation. We do not cache the result here because no result array is allocated anyway

        int should_deallocate = 1;

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);

    // the rhs is a cached broadcast array, no decref

    Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_inplace_multiply_sfloat_broadcast_cache")
    return 0;
fail:
    return -1;
}

int cmlq_along_multiply_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_along_multiply_slong")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

    if (PyLong_CheckExact(m1)) {
        PyObject *tmp = m1;
        m1 = m2;
        m2 = tmp;
    }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_LONG)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyLong_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;

        PyArrayObject *rhs = (PyArrayObject *)PyArray_FromAny(m2, right_descr, 0, 0, 0, NULL);

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            LONG_multiply(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                LONG_multiply(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 1);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_LONG);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        LONG_multiply(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", LONG_multiply(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_LONG);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            LONG_multiply(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", LONG_multiply(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_along_multiply_slong")
    return 0;
fail:
    return -1;
}

int cmlq_along_multiply_slong_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_along_multiply_slong_broadcast_cache")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

    if (PyLong_CheckExact(m1)) {
        PyObject *tmp = m1;
        m1 = m2;
        m2 = tmp;
    }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_LONG)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyLong_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;

        assert(elem->state == BROADCAST && elem->result);
        PyArrayObject *rhs = elem->result;

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 1);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_LONG);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        LONG_multiply(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", LONG_multiply(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_LONG);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            LONG_multiply(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", LONG_multiply(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        // no locality cache, we always need to deallocate the iterator

        int should_deallocate = 1;

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);

    // the rhs is a cached broadcast array, no decref

    Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_along_multiply_slong_broadcast_cache")
    return 0;
fail:
    return -1;
}

int cmlq_along_true_divide_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_along_true_divide_sfloat")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_LONG)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyFloat_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;

        PyArrayObject *rhs = (PyArrayObject *)PyArray_FromAny(m2, right_descr, 0, 0, 0, NULL);

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_divide(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_divide(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 0);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_divide(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_divide(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_divide(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_divide(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_along_true_divide_sfloat")
    return 0;
fail:
    return -1;
}

int cmlq_along_true_divide_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_along_true_divide_sfloat_broadcast_cache")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_LONG)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyFloat_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;

        assert(elem->state == BROADCAST && elem->result);
        PyArrayObject *rhs = elem->result;

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 0);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_divide(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_divide(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_divide(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_divide(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        // no locality cache, we always need to deallocate the iterator

        int should_deallocate = 1;

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);

    // the rhs is a cached broadcast array, no decref

    Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_along_true_divide_sfloat_broadcast_cache")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_true_divide_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_true_divide_sfloat")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyFloat_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;

        PyArrayObject *rhs = (PyArrayObject *)PyArray_FromAny(m2, right_descr, 0, 0, 0, NULL);

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_divide(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_divide(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 0);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_divide(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_divide(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_divide(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_divide(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_true_divide_sfloat")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_true_divide_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_true_divide_sfloat_broadcast_cache")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyFloat_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;

        assert(elem->state == BROADCAST && elem->result);
        PyArrayObject *rhs = elem->result;

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 0);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_divide(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_divide(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_divide(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_divide(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        // no locality cache, we always need to deallocate the iterator

        int should_deallocate = 1;

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);

    // the rhs is a cached broadcast array, no decref

    Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_true_divide_sfloat_broadcast_cache")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_true_divide_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_true_divide_slong")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyLong_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;
        right_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArrayObject *rhs = (PyArrayObject *)PyArray_FromAny(m2, right_descr, 0, 0, 0, NULL);

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_divide(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_divide(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 0);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_divide(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_divide(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_divide(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_divide(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_true_divide_slong")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_true_divide_slong_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_true_divide_slong_broadcast_cache")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyLong_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;
        right_descr = PyArray_DescrFromType(NPY_DOUBLE);

        assert(elem->state == BROADCAST && elem->result);
        PyArrayObject *rhs = elem->result;

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 0);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_divide(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_divide(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_divide(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_divide(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        // no locality cache, we always need to deallocate the iterator

        int should_deallocate = 1;

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);

    // the rhs is a cached broadcast array, no decref

    Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_true_divide_slong_broadcast_cache")
    return 0;
fail:
    return -1;
}

int cmlq_sfloat_true_divide_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_sfloat_true_divide_adouble")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyFloat_CheckExact(m1))) {

            goto deopt;
        }
        PyArray_Descr *left_descr = NULL;

        PyArrayObject *lhs = (PyArrayObject *)PyArray_FromAny(m1, left_descr, 0, 0, 0, NULL);

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(rhs);
    npy_intp *result_shape = PyArray_SHAPE(rhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_divide(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_divide(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(rhs) == 1) {
        fixed_strides[1] = PyArray_STRIDES(rhs)[0];
    } else {
        if (!(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[1] = PyArray_ITEMSIZE(rhs);
        }
    }

    fixed_strides[0] = 0;

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 0);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_divide(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_divide(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_divide(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_divide(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_sfloat_true_divide_adouble")
    return 0;
fail:
    return -1;
}

int cmlq_sfloat_true_divide_adouble_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_sfloat_true_divide_adouble_broadcast_cache")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyFloat_CheckExact(m1))) {

            goto deopt;
        }
        PyArray_Descr *left_descr = NULL;

        assert(elem->state == BROADCAST && elem->result);
        PyArrayObject *lhs = elem->result;

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(rhs);
    npy_intp *result_shape = PyArray_SHAPE(rhs);

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(rhs) == 1) {
        fixed_strides[1] = PyArray_STRIDES(rhs)[0];
    } else {
        if (!(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[1] = PyArray_ITEMSIZE(rhs);
        }
    }

    fixed_strides[0] = 0;

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 0);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_divide(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_divide(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_divide(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_divide(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        // no locality cache, we always need to deallocate the iterator

        int should_deallocate = 1;

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    // the lhs is a cached broadcast array, no decref

    Py_DECREF(rhs);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_sfloat_true_divide_adouble_broadcast_cache")
    return 0;
fail:
    return -1;
}

int cmlq_adouble_true_divide_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
    //CMLQ_PAPI_BEGIN("cmlq_adouble_true_divide_adouble")
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_divide(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_divide(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

            // try to avoid creating a temporary array
            determine_elide_temp_binary(m1, m2, (PyObject **)&result, 0);

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[2] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[2] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_divide(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_divide(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_divide(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_divide(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {

        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } else if (elem->state == ITERATOR) {
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }

        elem->state = UNUSED;
        elem->result = NULL;

    }
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(rhs);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("cmlq_adouble_true_divide_adouble")
    return 0;
fail:
    return -1;
}



int cmlq_adouble_power_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyFloat_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;

        PyArrayObject *rhs = (PyArrayObject *)PyArray_FromAny(m2, right_descr, 0, 0, 0, NULL);

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    double exponent = PyFloat_AsDouble(m2);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_power(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_power(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

    if(fast_path == 1) {

                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_power(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_power(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_power(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_power(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);

        Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;
fail:
    return -1;
}



int cmlq_adouble_square_power_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyFloat_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;

        PyArrayObject *rhs = (PyArrayObject *)PyArray_FromAny(m2, right_descr, 0, 0, 0, NULL);

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    double exponent = PyFloat_AsDouble(m2);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_square(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[2];
            op_it[0] = lhs; // is always LHS

            op_it[1] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_square(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    if (exponent != 2.0) {

        goto deopt;
    }
    ops[1] = NULL;

    npy_intp fixed_strides[2];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

            // try to avoid creating a temporary array
            if (can_elide_temp_unary(m1)) {
                result = m1;
            }

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[1] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[1] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[1] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[2];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_square(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_square(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[2] = {PyArray_DESCR(lhs), result_descr};
        npy_uint32 op_flags[2] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[1] = result;

        NpyIter *iter = NpyIter_AdvancedNew(2, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[1];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[2];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_square(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_square(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);

        Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;
fail:
    return -1;
}



int cmlq_adouble_square_power_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyLong_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;

        PyArrayObject *rhs = (PyArrayObject *)PyArray_FromAny(m2, right_descr, 0, 0, 0, NULL);

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    long exponent = PyLong_AsLong(m2);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_square(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[2];
            op_it[0] = lhs; // is always LHS

            op_it[1] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_square(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    if (exponent != 2) {

        goto deopt;
    }
    ops[1] = NULL;

    npy_intp fixed_strides[2];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

            // try to avoid creating a temporary array
            if (can_elide_temp_unary(m1)) {
                result = m1;
            }

    if(fast_path == 1) {

            if (result == NULL) {
                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[1] = PyArray_DESCR(result)->elsize;
            } else {
                // there was temp elision
                Py_INCREF(result);

                // copy the strides from the lhs (because it is the result as well)
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[1] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[1] = PyArray_ITEMSIZE(lhs);
                }
            }

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[2];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_square(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_square(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[2] = {PyArray_DESCR(lhs), result_descr};
        npy_uint32 op_flags[2] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[1] = result;

        NpyIter *iter = NpyIter_AdvancedNew(2, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[1];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[2];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_square(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_square(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);

        Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;
fail:
    return -1;
}



int cmlq_adouble_power_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyLong_CheckExact(m2))) {

            goto deopt;
        }
        PyArray_Descr *right_descr = NULL;

        PyArrayObject *rhs = (PyArrayObject *)PyArray_FromAny(m2, right_descr, 0, 0, 0, NULL);

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    long exponent = PyLong_AsLong(m2);

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_power(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_power(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) == 1) {
        fixed_strides[0] = PyArray_STRIDES(lhs)[0];
    } else {
        if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
            fast_path = 2;
        } else {
            fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        }
    }

    fixed_strides[1] = 0;

    if(fast_path == 1) {

                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_power(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_power(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_power(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_power(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);

        Py_DECREF(m2);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;
fail:
    return -1;
}



int cmlq_minimum_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_INT)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_INT)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-3];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            INT_minimum(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                INT_minimum(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

    if(fast_path == 1) {

                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_INT);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        INT_minimum(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", INT_minimum(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_INT);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            INT_minimum(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", INT_minimum(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 3;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}



int cmlq_minimum_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-3];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            FLOAT_minimum(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                FLOAT_minimum(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

    if(fast_path == 1) {

                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        FLOAT_minimum(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", FLOAT_minimum(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            FLOAT_minimum(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", FLOAT_minimum(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 3;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}



int cmlq_minimum_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-3];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_minimum(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_minimum(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

    if(fast_path == 1) {

                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_minimum(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_minimum(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_minimum(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_minimum(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 3;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}



int cmlq_maximum_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_INT)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_INT)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-3];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            INT_maximum(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                INT_maximum(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

    if(fast_path == 1) {

                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_INT);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        INT_maximum(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", INT_maximum(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_INT);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            INT_maximum(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", INT_maximum(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 3;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}



int cmlq_maximum_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-3];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            FLOAT_maximum(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                FLOAT_maximum(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

    if(fast_path == 1) {

                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        FLOAT_maximum(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", FLOAT_maximum(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            FLOAT_maximum(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", FLOAT_maximum(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 3;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}



int cmlq_maximum_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-3];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_maximum(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_maximum(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

    if(fast_path == 1) {

                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_maximum(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_maximum(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_maximum(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_maximum(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 3;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}



int cmlq_add_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_INT)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_INT)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-3];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            INT_add(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                INT_add(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

    if(fast_path == 1) {

                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_INT);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        INT_add(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", INT_add(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_INT);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            INT_add(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", INT_add(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 3;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}



int cmlq_add_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-3];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            FLOAT_add(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                FLOAT_add(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

    if(fast_path == 1) {

                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        FLOAT_add(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", FLOAT_add(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            FLOAT_add(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", FLOAT_add(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 3;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}



int cmlq_add_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-3];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_add(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_add(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

    if(fast_path == 1) {

                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_add(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_add(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_add(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_add(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 3;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}



int cmlq_subtract_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_INT)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_INT)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-3];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            INT_subtract(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                INT_subtract(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

    if(fast_path == 1) {

                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_INT);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        INT_subtract(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", INT_subtract(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_INT);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            INT_subtract(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", INT_subtract(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 3;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}



int cmlq_subtract_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-3];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            FLOAT_subtract(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                FLOAT_subtract(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

    if(fast_path == 1) {

                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        FLOAT_subtract(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", FLOAT_subtract(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            FLOAT_subtract(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", FLOAT_subtract(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 3;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}



int cmlq_subtract_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-3];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_subtract(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_subtract(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

    if(fast_path == 1) {

                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_subtract(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_subtract(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_subtract(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_subtract(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 3;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}



int cmlq_multiply_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_INT)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_INT)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-3];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            INT_multiply(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                INT_multiply(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

    if(fast_path == 1) {

                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_INT);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        INT_multiply(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", INT_multiply(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_INT);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            INT_multiply(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", INT_multiply(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 3;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}



int cmlq_multiply_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-3];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            FLOAT_multiply(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                FLOAT_multiply(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

    if(fast_path == 1) {

                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        FLOAT_multiply(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", FLOAT_multiply(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            FLOAT_multiply(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", FLOAT_multiply(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 3;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}



int cmlq_multiply_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-3];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_multiply(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_multiply(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

    if(fast_path == 1) {

                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_multiply(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_multiply(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_multiply(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_multiply(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 3;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}



int cmlq_matmul_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_DOUBLE)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-3];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_matmul(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_matmul(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

    if(fast_path == 1) {

                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_matmul(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_matmul(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            DOUBLE_matmul(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_matmul(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 3;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}



int cmlq_matmul_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_FLOAT)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-3];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            FLOAT_matmul(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                FLOAT_matmul(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

    if(fast_path == 1) {

                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        FLOAT_matmul(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", FLOAT_matmul(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            FLOAT_matmul(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", FLOAT_matmul(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 3;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}



int cmlq_matmul_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];

        if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

            goto deopt;
        }
        PyArrayObject *lhs = (PyArrayObject *)m1;
        if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_INT)) {

            goto deopt;
        }

        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {

            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != NPY_INT)) {

            goto deopt;
        }

    // No need to check for ufunc overrides as we check for ndarray exact type
    // PyUFuncOverride_GetNonDefaultArrayUfunc even has a fast path for this case because there are no overrides

    PyArrayObject *ops[] = {lhs, rhs, NULL};
    PyArrayObject *result = NULL;

    // the non-scalar side determines the result shape
    npy_intp result_ndims = PyArray_NDIM(lhs);
    npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-3];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            INT_matmul(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // is always LHS

            op_it[1] = rhs;

            op_it[2] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                INT_matmul(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (PyArray_NDIM(lhs) != PyArray_NDIM(rhs) ||
        !PyArray_CompareLists(PyArray_SHAPE(lhs), PyArray_SHAPE(rhs), PyArray_NDIM(lhs))) {
        fast_path = 2;
    } else {
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
            fixed_strides[1] = PyArray_STRIDES(rhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
                !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
                fixed_strides[1] = PyArray_ITEMSIZE(rhs);
            }
        }
    }

    if(fast_path == 1) {

                // there was no temp elision so we need to create a new array
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_INT);

                // at this time we know that the arrays can't have different dimensions otherwise we would have taken the iterator path
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[2] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        INT_matmul(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", INT_matmul(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                elem->trivial.count = count;
                elem->trivial.fixed_strides[0] = fixed_strides[0];
                elem->trivial.fixed_strides[1] = fixed_strides[1];
                elem->trivial.fixed_strides[2] = fixed_strides[2];
                elem->state = TRIVIAL;

            } else {
                // warm up the result cache
                elem->miss_counter++;
            }
        }

        goto success;
    }

    if (fast_path == 2) {

        // use the standard flags here
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_INT);

        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

        // if result is not NULL, it means we need to reuse it, if it is NULL the iterator will create a new array
        ops[2] = result;

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* Set the output array as output (the iterator might have created an array) */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];

        // the result will be pushed to the stack
        Py_INCREF(result);

        /* Only do the loop if the iteration size is non-zero */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[3];

        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);
        baseptrs[2] = PyArray_BYTES(op_it[2]);

        /* Get the variables needed for the loop */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* The reset may copy the first buffer chunk, which could cause FPEs */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* Execute the loop */
        do {
            INT_matmul(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", INT_matmul(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        int should_deallocate = 0;

        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->state = ITERATOR;

                elem->iterator.countptr = countptr;
                elem->iterator.dataptr = dataptr;
                elem->iterator.strides = strides;

                // we do not need to increase the refcnt here because the iterator holds the reference
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                elem->iterator.cached_iter = iter;
                elem->iterator.iter_next = *iternext;

            } else {
                // warm up the result cache
                elem->miss_counter++;
                should_deallocate = 1;
            }
        } else {
            // the iterator is not cached, so we need to deallocate it
            should_deallocate = 1;
        }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // some other error we can't handle
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 3;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}



int cmlq_square_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

PyObject *m1 = (*stack_pointer_ptr)[-1];

    if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

        goto deopt;
    }
    PyArrayObject *lhs = (PyArrayObject *)m1;
    if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_INT)) {

        goto deopt;
    }

PyArrayObject *ops[] = {lhs, NULL};
PyArrayObject *result = NULL;

npy_intp result_ndims = PyArray_NDIM(lhs);
npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-2];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            INT_square(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[2];
            op_it[0] = lhs; // is always LHS

            op_it[1] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                INT_square(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[2];
    int fast_path = 1;
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
            }
        }

    if(fast_path == 1) {

                // 没有临时数组优化，需要创建新数组
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_INT);

                // 此时已知操作数维数匹配，直接用 lhs 的 ndim/shape 创建结果
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[1] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        INT_square(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", INT_square(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    Py_INCREF(elem->result);

                    elem->trivial.count = count;
                    elem->trivial.fixed_strides[0] = fixed_strides[0];
                    elem->trivial.fixed_strides[1] = fixed_strides[1];
                    elem->state = TRIVIAL;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                }
            }

        goto success;
    }

    if (fast_path == 2) {

        // 使用迭代器处理不满足 fast_path 的情况
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_INT);

            PyArray_Descr *descriptors[2] = {PyArray_DESCR(lhs), result_descr};
            npy_uint32 op_flags[2] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

            // 若 result 非空，则复用，否则迭代器将创建新数组
            ops[1] = result;

        NpyIter *iter = NpyIter_AdvancedNew(2, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* 将输出数组设为结果（迭代器可能已经创建了数组） */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[1];

        // 结果将压入堆栈
        Py_INCREF(result);

        /* 仅当迭代大小非零时执行循环 */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[2];
        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);

        /* 获取迭代所需变量 */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* 重设基础指针（可能复制首个缓冲区块，需防止浮点异常） */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* 执行循环 */
        do {
            INT_square(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", INT_square(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

            int should_deallocate = 0;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->state = ITERATOR;

                    elem->iterator.countptr = countptr;
                    elem->iterator.dataptr = dataptr;
                    elem->iterator.strides = strides;

                    // 迭代器持有引用，无需增加 refcnt
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    elem->iterator.cached_iter = iter;
                    elem->iterator.iter_next = *iternext;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                    should_deallocate = 1;
                }
            } else {
                // 若未缓存，则释放迭代器
                should_deallocate = 1;
            }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // 无法处理的其它错误情况
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 2;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}


int cmlq_square_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

PyObject *m1 = (*stack_pointer_ptr)[-1];

    if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

        goto deopt;
    }
    PyArrayObject *lhs = (PyArrayObject *)m1;
    if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

        goto deopt;
    }

PyArrayObject *ops[] = {lhs, NULL};
PyArrayObject *result = NULL;

npy_intp result_ndims = PyArray_NDIM(lhs);
npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-2];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_square(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[2];
            op_it[0] = lhs; // is always LHS

            op_it[1] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_square(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[2];
    int fast_path = 1;
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
            }
        }

    if(fast_path == 1) {

                // 没有临时数组优化，需要创建新数组
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // 此时已知操作数维数匹配，直接用 lhs 的 ndim/shape 创建结果
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[1] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_square(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_square(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    Py_INCREF(elem->result);

                    elem->trivial.count = count;
                    elem->trivial.fixed_strides[0] = fixed_strides[0];
                    elem->trivial.fixed_strides[1] = fixed_strides[1];
                    elem->state = TRIVIAL;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                }
            }

        goto success;
    }

    if (fast_path == 2) {

        // 使用迭代器处理不满足 fast_path 的情况
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

            PyArray_Descr *descriptors[2] = {PyArray_DESCR(lhs), result_descr};
            npy_uint32 op_flags[2] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

            // 若 result 非空，则复用，否则迭代器将创建新数组
            ops[1] = result;

        NpyIter *iter = NpyIter_AdvancedNew(2, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* 将输出数组设为结果（迭代器可能已经创建了数组） */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[1];

        // 结果将压入堆栈
        Py_INCREF(result);

        /* 仅当迭代大小非零时执行循环 */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[2];
        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);

        /* 获取迭代所需变量 */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* 重设基础指针（可能复制首个缓冲区块，需防止浮点异常） */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* 执行循环 */
        do {
            DOUBLE_square(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_square(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

            int should_deallocate = 0;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->state = ITERATOR;

                    elem->iterator.countptr = countptr;
                    elem->iterator.dataptr = dataptr;
                    elem->iterator.strides = strides;

                    // 迭代器持有引用，无需增加 refcnt
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    elem->iterator.cached_iter = iter;
                    elem->iterator.iter_next = *iternext;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                    should_deallocate = 1;
                }
            } else {
                // 若未缓存，则释放迭代器
                should_deallocate = 1;
            }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // 无法处理的其它错误情况
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 2;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}


int cmlq_square_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

PyObject *m1 = (*stack_pointer_ptr)[-1];

    if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

        goto deopt;
    }
    PyArrayObject *lhs = (PyArrayObject *)m1;
    if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_FLOAT)) {

        goto deopt;
    }

PyArrayObject *ops[] = {lhs, NULL};
PyArrayObject *result = NULL;

npy_intp result_ndims = PyArray_NDIM(lhs);
npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-2];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            FLOAT_square(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[2];
            op_it[0] = lhs; // is always LHS

            op_it[1] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                FLOAT_square(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[2];
    int fast_path = 1;
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
            }
        }

    if(fast_path == 1) {

                // 没有临时数组优化，需要创建新数组
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

                // 此时已知操作数维数匹配，直接用 lhs 的 ndim/shape 创建结果
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[1] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        FLOAT_square(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", FLOAT_square(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    Py_INCREF(elem->result);

                    elem->trivial.count = count;
                    elem->trivial.fixed_strides[0] = fixed_strides[0];
                    elem->trivial.fixed_strides[1] = fixed_strides[1];
                    elem->state = TRIVIAL;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                }
            }

        goto success;
    }

    if (fast_path == 2) {

        // 使用迭代器处理不满足 fast_path 的情况
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

            PyArray_Descr *descriptors[2] = {PyArray_DESCR(lhs), result_descr};
            npy_uint32 op_flags[2] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

            // 若 result 非空，则复用，否则迭代器将创建新数组
            ops[1] = result;

        NpyIter *iter = NpyIter_AdvancedNew(2, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* 将输出数组设为结果（迭代器可能已经创建了数组） */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[1];

        // 结果将压入堆栈
        Py_INCREF(result);

        /* 仅当迭代大小非零时执行循环 */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[2];
        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);

        /* 获取迭代所需变量 */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* 重设基础指针（可能复制首个缓冲区块，需防止浮点异常） */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* 执行循环 */
        do {
            FLOAT_square(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", FLOAT_square(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

            int should_deallocate = 0;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->state = ITERATOR;

                    elem->iterator.countptr = countptr;
                    elem->iterator.dataptr = dataptr;
                    elem->iterator.strides = strides;

                    // 迭代器持有引用，无需增加 refcnt
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    elem->iterator.cached_iter = iter;
                    elem->iterator.iter_next = *iternext;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                    should_deallocate = 1;
                }
            } else {
                // 若未缓存，则释放迭代器
                should_deallocate = 1;
            }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // 无法处理的其它错误情况
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 2;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}


int cmlq_sqrt_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

PyObject *m1 = (*stack_pointer_ptr)[-1];

    if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

        goto deopt;
    }
    PyArrayObject *lhs = (PyArrayObject *)m1;
    if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

        goto deopt;
    }

PyArrayObject *ops[] = {lhs, NULL};
PyArrayObject *result = NULL;

npy_intp result_ndims = PyArray_NDIM(lhs);
npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-2];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_sqrt(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[2];
            op_it[0] = lhs; // is always LHS

            op_it[1] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_sqrt(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[2];
    int fast_path = 1;
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
            }
        }

    if(fast_path == 1) {

                // 没有临时数组优化，需要创建新数组
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // 此时已知操作数维数匹配，直接用 lhs 的 ndim/shape 创建结果
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[1] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_sqrt(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_sqrt(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    Py_INCREF(elem->result);

                    elem->trivial.count = count;
                    elem->trivial.fixed_strides[0] = fixed_strides[0];
                    elem->trivial.fixed_strides[1] = fixed_strides[1];
                    elem->state = TRIVIAL;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                }
            }

        goto success;
    }

    if (fast_path == 2) {

        // 使用迭代器处理不满足 fast_path 的情况
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

            PyArray_Descr *descriptors[2] = {PyArray_DESCR(lhs), result_descr};
            npy_uint32 op_flags[2] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

            // 若 result 非空，则复用，否则迭代器将创建新数组
            ops[1] = result;

        NpyIter *iter = NpyIter_AdvancedNew(2, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* 将输出数组设为结果（迭代器可能已经创建了数组） */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[1];

        // 结果将压入堆栈
        Py_INCREF(result);

        /* 仅当迭代大小非零时执行循环 */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[2];
        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);

        /* 获取迭代所需变量 */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* 重设基础指针（可能复制首个缓冲区块，需防止浮点异常） */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* 执行循环 */
        do {
            DOUBLE_sqrt(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_sqrt(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

            int should_deallocate = 0;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->state = ITERATOR;

                    elem->iterator.countptr = countptr;
                    elem->iterator.dataptr = dataptr;
                    elem->iterator.strides = strides;

                    // 迭代器持有引用，无需增加 refcnt
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    elem->iterator.cached_iter = iter;
                    elem->iterator.iter_next = *iternext;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                    should_deallocate = 1;
                }
            } else {
                // 若未缓存，则释放迭代器
                should_deallocate = 1;
            }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // 无法处理的其它错误情况
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 2;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}


int cmlq_sqrt_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

PyObject *m1 = (*stack_pointer_ptr)[-1];

    if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

        goto deopt;
    }
    PyArrayObject *lhs = (PyArrayObject *)m1;
    if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_FLOAT)) {

        goto deopt;
    }

PyArrayObject *ops[] = {lhs, NULL};
PyArrayObject *result = NULL;

npy_intp result_ndims = PyArray_NDIM(lhs);
npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-2];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            FLOAT_sqrt(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[2];
            op_it[0] = lhs; // is always LHS

            op_it[1] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                FLOAT_sqrt(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[2];
    int fast_path = 1;
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
            }
        }

    if(fast_path == 1) {

                // 没有临时数组优化，需要创建新数组
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

                // 此时已知操作数维数匹配，直接用 lhs 的 ndim/shape 创建结果
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[1] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        FLOAT_sqrt(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", FLOAT_sqrt(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    Py_INCREF(elem->result);

                    elem->trivial.count = count;
                    elem->trivial.fixed_strides[0] = fixed_strides[0];
                    elem->trivial.fixed_strides[1] = fixed_strides[1];
                    elem->state = TRIVIAL;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                }
            }

        goto success;
    }

    if (fast_path == 2) {

        // 使用迭代器处理不满足 fast_path 的情况
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

            PyArray_Descr *descriptors[2] = {PyArray_DESCR(lhs), result_descr};
            npy_uint32 op_flags[2] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

            // 若 result 非空，则复用，否则迭代器将创建新数组
            ops[1] = result;

        NpyIter *iter = NpyIter_AdvancedNew(2, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* 将输出数组设为结果（迭代器可能已经创建了数组） */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[1];

        // 结果将压入堆栈
        Py_INCREF(result);

        /* 仅当迭代大小非零时执行循环 */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[2];
        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);

        /* 获取迭代所需变量 */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* 重设基础指针（可能复制首个缓冲区块，需防止浮点异常） */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* 执行循环 */
        do {
            FLOAT_sqrt(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", FLOAT_sqrt(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

            int should_deallocate = 0;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->state = ITERATOR;

                    elem->iterator.countptr = countptr;
                    elem->iterator.dataptr = dataptr;
                    elem->iterator.strides = strides;

                    // 迭代器持有引用，无需增加 refcnt
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    elem->iterator.cached_iter = iter;
                    elem->iterator.iter_next = *iternext;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                    should_deallocate = 1;
                }
            } else {
                // 若未缓存，则释放迭代器
                should_deallocate = 1;
            }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // 无法处理的其它错误情况
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 2;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}


int cmlq_absolute_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

PyObject *m1 = (*stack_pointer_ptr)[-1];

    if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

        goto deopt;
    }
    PyArrayObject *lhs = (PyArrayObject *)m1;
    if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_INT)) {

        goto deopt;
    }

PyArrayObject *ops[] = {lhs, NULL};
PyArrayObject *result = NULL;

npy_intp result_ndims = PyArray_NDIM(lhs);
npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-2];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            INT_absolute(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[2];
            op_it[0] = lhs; // is always LHS

            op_it[1] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                INT_absolute(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[2];
    int fast_path = 1;
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
            }
        }

    if(fast_path == 1) {

                // 没有临时数组优化，需要创建新数组
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_INT);

                // 此时已知操作数维数匹配，直接用 lhs 的 ndim/shape 创建结果
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[1] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        INT_absolute(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", INT_absolute(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    Py_INCREF(elem->result);

                    elem->trivial.count = count;
                    elem->trivial.fixed_strides[0] = fixed_strides[0];
                    elem->trivial.fixed_strides[1] = fixed_strides[1];
                    elem->state = TRIVIAL;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                }
            }

        goto success;
    }

    if (fast_path == 2) {

        // 使用迭代器处理不满足 fast_path 的情况
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_INT);

            PyArray_Descr *descriptors[2] = {PyArray_DESCR(lhs), result_descr};
            npy_uint32 op_flags[2] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

            // 若 result 非空，则复用，否则迭代器将创建新数组
            ops[1] = result;

        NpyIter *iter = NpyIter_AdvancedNew(2, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* 将输出数组设为结果（迭代器可能已经创建了数组） */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[1];

        // 结果将压入堆栈
        Py_INCREF(result);

        /* 仅当迭代大小非零时执行循环 */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[2];
        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);

        /* 获取迭代所需变量 */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* 重设基础指针（可能复制首个缓冲区块，需防止浮点异常） */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* 执行循环 */
        do {
            INT_absolute(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", INT_absolute(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

            int should_deallocate = 0;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->state = ITERATOR;

                    elem->iterator.countptr = countptr;
                    elem->iterator.dataptr = dataptr;
                    elem->iterator.strides = strides;

                    // 迭代器持有引用，无需增加 refcnt
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    elem->iterator.cached_iter = iter;
                    elem->iterator.iter_next = *iternext;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                    should_deallocate = 1;
                }
            } else {
                // 若未缓存，则释放迭代器
                should_deallocate = 1;
            }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // 无法处理的其它错误情况
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 2;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}


int cmlq_absolute_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

PyObject *m1 = (*stack_pointer_ptr)[-1];

    if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

        goto deopt;
    }
    PyArrayObject *lhs = (PyArrayObject *)m1;
    if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

        goto deopt;
    }

PyArrayObject *ops[] = {lhs, NULL};
PyArrayObject *result = NULL;

npy_intp result_ndims = PyArray_NDIM(lhs);
npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-2];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_absolute(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[2];
            op_it[0] = lhs; // is always LHS

            op_it[1] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_absolute(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[2];
    int fast_path = 1;
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
            }
        }

    if(fast_path == 1) {

                // 没有临时数组优化，需要创建新数组
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // 此时已知操作数维数匹配，直接用 lhs 的 ndim/shape 创建结果
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[1] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_absolute(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_absolute(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    Py_INCREF(elem->result);

                    elem->trivial.count = count;
                    elem->trivial.fixed_strides[0] = fixed_strides[0];
                    elem->trivial.fixed_strides[1] = fixed_strides[1];
                    elem->state = TRIVIAL;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                }
            }

        goto success;
    }

    if (fast_path == 2) {

        // 使用迭代器处理不满足 fast_path 的情况
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

            PyArray_Descr *descriptors[2] = {PyArray_DESCR(lhs), result_descr};
            npy_uint32 op_flags[2] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

            // 若 result 非空，则复用，否则迭代器将创建新数组
            ops[1] = result;

        NpyIter *iter = NpyIter_AdvancedNew(2, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* 将输出数组设为结果（迭代器可能已经创建了数组） */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[1];

        // 结果将压入堆栈
        Py_INCREF(result);

        /* 仅当迭代大小非零时执行循环 */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[2];
        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);

        /* 获取迭代所需变量 */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* 重设基础指针（可能复制首个缓冲区块，需防止浮点异常） */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* 执行循环 */
        do {
            DOUBLE_absolute(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_absolute(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

            int should_deallocate = 0;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->state = ITERATOR;

                    elem->iterator.countptr = countptr;
                    elem->iterator.dataptr = dataptr;
                    elem->iterator.strides = strides;

                    // 迭代器持有引用，无需增加 refcnt
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    elem->iterator.cached_iter = iter;
                    elem->iterator.iter_next = *iternext;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                    should_deallocate = 1;
                }
            } else {
                // 若未缓存，则释放迭代器
                should_deallocate = 1;
            }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // 无法处理的其它错误情况
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 2;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}


int cmlq_absolute_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

PyObject *m1 = (*stack_pointer_ptr)[-1];

    if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

        goto deopt;
    }
    PyArrayObject *lhs = (PyArrayObject *)m1;
    if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_FLOAT)) {

        goto deopt;
    }

PyArrayObject *ops[] = {lhs, NULL};
PyArrayObject *result = NULL;

npy_intp result_ndims = PyArray_NDIM(lhs);
npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-2];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            FLOAT_absolute(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[2];
            op_it[0] = lhs; // is always LHS

            op_it[1] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                FLOAT_absolute(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[2];
    int fast_path = 1;
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
            }
        }

    if(fast_path == 1) {

                // 没有临时数组优化，需要创建新数组
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

                // 此时已知操作数维数匹配，直接用 lhs 的 ndim/shape 创建结果
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[1] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        FLOAT_absolute(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", FLOAT_absolute(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    Py_INCREF(elem->result);

                    elem->trivial.count = count;
                    elem->trivial.fixed_strides[0] = fixed_strides[0];
                    elem->trivial.fixed_strides[1] = fixed_strides[1];
                    elem->state = TRIVIAL;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                }
            }

        goto success;
    }

    if (fast_path == 2) {

        // 使用迭代器处理不满足 fast_path 的情况
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

            PyArray_Descr *descriptors[2] = {PyArray_DESCR(lhs), result_descr};
            npy_uint32 op_flags[2] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

            // 若 result 非空，则复用，否则迭代器将创建新数组
            ops[1] = result;

        NpyIter *iter = NpyIter_AdvancedNew(2, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* 将输出数组设为结果（迭代器可能已经创建了数组） */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[1];

        // 结果将压入堆栈
        Py_INCREF(result);

        /* 仅当迭代大小非零时执行循环 */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[2];
        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);

        /* 获取迭代所需变量 */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* 重设基础指针（可能复制首个缓冲区块，需防止浮点异常） */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* 执行循环 */
        do {
            FLOAT_absolute(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", FLOAT_absolute(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

            int should_deallocate = 0;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->state = ITERATOR;

                    elem->iterator.countptr = countptr;
                    elem->iterator.dataptr = dataptr;
                    elem->iterator.strides = strides;

                    // 迭代器持有引用，无需增加 refcnt
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    elem->iterator.cached_iter = iter;
                    elem->iterator.iter_next = *iternext;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                    should_deallocate = 1;
                }
            } else {
                // 若未缓存，则释放迭代器
                should_deallocate = 1;
            }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // 无法处理的其它错误情况
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 2;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}


int cmlq_reciprocal_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

PyObject *m1 = (*stack_pointer_ptr)[-1];

    if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

        goto deopt;
    }
    PyArrayObject *lhs = (PyArrayObject *)m1;
    if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_FLOAT)) {

        goto deopt;
    }

PyArrayObject *ops[] = {lhs, NULL};
PyArrayObject *result = NULL;

npy_intp result_ndims = PyArray_NDIM(lhs);
npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-2];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            FLOAT_reciprocal(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[2];
            op_it[0] = lhs; // is always LHS

            op_it[1] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                FLOAT_reciprocal(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[2];
    int fast_path = 1;
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
            }
        }

    if(fast_path == 1) {

                // 没有临时数组优化，需要创建新数组
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

                // 此时已知操作数维数匹配，直接用 lhs 的 ndim/shape 创建结果
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[1] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        FLOAT_reciprocal(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", FLOAT_reciprocal(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    Py_INCREF(elem->result);

                    elem->trivial.count = count;
                    elem->trivial.fixed_strides[0] = fixed_strides[0];
                    elem->trivial.fixed_strides[1] = fixed_strides[1];
                    elem->state = TRIVIAL;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                }
            }

        goto success;
    }

    if (fast_path == 2) {

        // 使用迭代器处理不满足 fast_path 的情况
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_FLOAT);

            PyArray_Descr *descriptors[2] = {PyArray_DESCR(lhs), result_descr};
            npy_uint32 op_flags[2] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

            // 若 result 非空，则复用，否则迭代器将创建新数组
            ops[1] = result;

        NpyIter *iter = NpyIter_AdvancedNew(2, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* 将输出数组设为结果（迭代器可能已经创建了数组） */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[1];

        // 结果将压入堆栈
        Py_INCREF(result);

        /* 仅当迭代大小非零时执行循环 */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[2];
        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);

        /* 获取迭代所需变量 */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* 重设基础指针（可能复制首个缓冲区块，需防止浮点异常） */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* 执行循环 */
        do {
            FLOAT_reciprocal(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", FLOAT_reciprocal(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

            int should_deallocate = 0;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->state = ITERATOR;

                    elem->iterator.countptr = countptr;
                    elem->iterator.dataptr = dataptr;
                    elem->iterator.strides = strides;

                    // 迭代器持有引用，无需增加 refcnt
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    elem->iterator.cached_iter = iter;
                    elem->iterator.iter_next = *iternext;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                    should_deallocate = 1;
                }
            } else {
                // 若未缓存，则释放迭代器
                should_deallocate = 1;
            }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // 无法处理的其它错误情况
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 2;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}


int cmlq_reciprocal_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

PyObject *m1 = (*stack_pointer_ptr)[-1];

    if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

        goto deopt;
    }
    PyArrayObject *lhs = (PyArrayObject *)m1;
    if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_DOUBLE)) {

        goto deopt;
    }

PyArrayObject *ops[] = {lhs, NULL};
PyArrayObject *result = NULL;

npy_intp result_ndims = PyArray_NDIM(lhs);
npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-2];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            DOUBLE_reciprocal(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[2];
            op_it[0] = lhs; // is always LHS

            op_it[1] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                DOUBLE_reciprocal(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[2];
    int fast_path = 1;
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
            }
        }

    if(fast_path == 1) {

                // 没有临时数组优化，需要创建新数组
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

                // 此时已知操作数维数匹配，直接用 lhs 的 ndim/shape 创建结果
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[1] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        DOUBLE_reciprocal(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", DOUBLE_reciprocal(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    Py_INCREF(elem->result);

                    elem->trivial.count = count;
                    elem->trivial.fixed_strides[0] = fixed_strides[0];
                    elem->trivial.fixed_strides[1] = fixed_strides[1];
                    elem->state = TRIVIAL;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                }
            }

        goto success;
    }

    if (fast_path == 2) {

        // 使用迭代器处理不满足 fast_path 的情况
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_DOUBLE);

            PyArray_Descr *descriptors[2] = {PyArray_DESCR(lhs), result_descr};
            npy_uint32 op_flags[2] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

            // 若 result 非空，则复用，否则迭代器将创建新数组
            ops[1] = result;

        NpyIter *iter = NpyIter_AdvancedNew(2, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* 将输出数组设为结果（迭代器可能已经创建了数组） */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[1];

        // 结果将压入堆栈
        Py_INCREF(result);

        /* 仅当迭代大小非零时执行循环 */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[2];
        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);

        /* 获取迭代所需变量 */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* 重设基础指针（可能复制首个缓冲区块，需防止浮点异常） */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* 执行循环 */
        do {
            DOUBLE_reciprocal(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", DOUBLE_reciprocal(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

            int should_deallocate = 0;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->state = ITERATOR;

                    elem->iterator.countptr = countptr;
                    elem->iterator.dataptr = dataptr;
                    elem->iterator.strides = strides;

                    // 迭代器持有引用，无需增加 refcnt
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    elem->iterator.cached_iter = iter;
                    elem->iterator.iter_next = *iternext;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                    should_deallocate = 1;
                }
            } else {
                // 若未缓存，则释放迭代器
                should_deallocate = 1;
            }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // 无法处理的其它错误情况
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 2;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}


int cmlq_reciprocal_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr)
{
        CMLQLocalityCacheElem *restrict elem = (CMLQLocalityCacheElem *restrict)external_cache_pointer;

PyObject *m1 = (*stack_pointer_ptr)[-1];

    if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {

        goto deopt;
    }
    PyArrayObject *lhs = (PyArrayObject *)m1;
    if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != NPY_INT)) {

        goto deopt;
    }

PyArrayObject *ops[] = {lhs, NULL};
PyArrayObject *result = NULL;

npy_intp result_ndims = PyArray_NDIM(lhs);
npy_intp *result_shape = PyArray_SHAPE(lhs);

    PyObject *callable = (*stack_pointer_ptr)[-2];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {

        // ufunc has user loops or is generalized
        goto deopt;
    }

// we can only use the result cache if we are the sole owner of the result object
// and if the object properties match the required result properties
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

#define CACHE_MATCH_TRIVIAL() \
    elem->state == TRIVIAL

#define CACHE_MATCH_ITERATOR() \
    elem->state == ITERATOR

    if (CACHE_MATCH_TRIVIAL()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // in addition to the cache, the result will be pushed to the stack
            Py_INCREF(result);

            char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(elem->trivial.count);

            NpyAuxData *auxdata = NULL;
            INT_reciprocal(data, &elem->trivial.count, elem->trivial.fixed_strides, auxdata);

            NPY_END_THREADS;

            goto success;
        }
        else {

            trivial_cache_miss(elem);
        }
    } else if (CACHE_MATCH_ITERATOR()) {

        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;

            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[2];
            op_it[0] = lhs; // is always LHS

            op_it[1] = result;

            // can reuse the old one, nothing changes there...
            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            /* Execute the loop */
            do {
                INT_reciprocal(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;

            // standard epilogue here...
            goto success;
       } else {

            iterator_cache_miss(elem);
       }
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);

        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);

        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }

    npy_intp fixed_strides[2];
    int fast_path = 1;
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
            }
        }

    if(fast_path == 1) {

                // 没有临时数组优化，需要创建新数组
                PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_INT);

                // 此时已知操作数维数匹配，直接用 lhs 的 ndim/shape 创建结果
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[1] = PyArray_DESCR(result)->elsize;

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[2];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        INT_reciprocal(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", INT_reciprocal(data, &count, fixed_strides, auxdata));

        NPY_END_THREADS;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    Py_INCREF(elem->result);

                    elem->trivial.count = count;
                    elem->trivial.fixed_strides[0] = fixed_strides[0];
                    elem->trivial.fixed_strides[1] = fixed_strides[1];
                    elem->state = TRIVIAL;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                }
            }

        goto success;
    }

    if (fast_path == 2) {

        // 使用迭代器处理不满足 fast_path 的情况
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(NPY_INT);

            PyArray_Descr *descriptors[2] = {PyArray_DESCR(lhs), result_descr};
            npy_uint32 op_flags[2] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

            // 若 result 非空，则复用，否则迭代器将创建新数组
            ops[1] = result;

        NpyIter *iter = NpyIter_AdvancedNew(2, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* 将输出数组设为结果（迭代器可能已经创建了数组） */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[1];

        // 结果将压入堆栈
        Py_INCREF(result);

        /* 仅当迭代大小非零时执行循环 */
        npy_intp full_size = NpyIter_GetIterSize(iter);
        if (full_size == 0) {
            Py_DECREF(result_descr);
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
            goto success;
        }

        char *baseptrs[2];
        baseptrs[0] = PyArray_BYTES(op_it[0]);
        baseptrs[1] = PyArray_BYTES(op_it[1]);

        /* 获取迭代所需变量 */
        NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);
        if (iternext == NULL) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }
        char **dataptr = NpyIter_GetDataPtrArray(iter);
        npy_intp *strides = NpyIter_GetInnerStrideArray(iter);
        npy_intp *countptr = NpyIter_GetInnerLoopSizePtr(iter);

        NpyAuxData *auxdata = NULL;

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(full_size);

        /* 重设基础指针（可能复制首个缓冲区块，需防止浮点异常） */
        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        /* 执行循环 */
        do {
            INT_reciprocal(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", INT_reciprocal(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

            int should_deallocate = 0;

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->state = ITERATOR;

                    elem->iterator.countptr = countptr;
                    elem->iterator.dataptr = dataptr;
                    elem->iterator.strides = strides;

                    // 迭代器持有引用，无需增加 refcnt
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    elem->iterator.cached_iter = iter;
                    elem->iterator.iter_next = *iternext;

                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                    should_deallocate = 1;
                }
            } else {
                // 若未缓存，则释放迭代器
                should_deallocate = 1;
            }

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    // 无法处理的其它错误情况
    raise(SIGTRAP);

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 2;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}
