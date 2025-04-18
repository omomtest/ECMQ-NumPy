<%namespace file="cache_stats_macro.mako" import="*"/>
${signature}
{
    %if locality_cache or locality_stats:
    <%include file="load_cache_elem.mako"/>
    %endif
    fprintf(stderr,"1\n");
    PyObject *m1 = (*stack_pointer_ptr)[-2];
    PyObject *m2 = (*stack_pointer_ptr)[-1];
    

    if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {
        <%count_stat("left_type_misses")%>
        goto deopt;
    }
    PyArrayObject *lhs = (PyArrayObject *)m1;
    if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != ${left_numpy_name})) {
        <%count_stat("left_type_misses")%>
        goto deopt;
    }


        if (NPY_UNLIKELY(!PyArray_CheckExact(m2))) {
            <%count_stat("right_type_misses")%>
            goto deopt;
        }
        PyArrayObject *rhs = (PyArrayObject *)m2;
        if (NPY_UNLIKELY(PyArray_DESCR(rhs)->type_num != ${right_numpy_name})) {
            <%count_stat("right_type_misses")%>
            goto deopt;
        }

    fprintf(stderr,"2\n");
    // 检查矩阵乘法的基本要求：
    // 两个输入都至少应为二维数组，并且 lhs 的最后一维必须与 rhs 的倒数第二维相等。
    npy_intp nd_lhs = PyArray_NDIM(lhs);
    npy_intp nd_rhs = PyArray_NDIM(rhs);
    if (nd_lhs < 2 || nd_rhs < 2) {
        goto deopt;
    }
    npy_intp *shape_lhs = PyArray_SHAPE(lhs);
    npy_intp *shape_rhs = PyArray_SHAPE(rhs);
    if (shape_lhs[nd_lhs - 1] != shape_rhs[nd_rhs - 2]) {
        goto deopt;
    }
    fprintf(stderr,"3\n");
    // 计算结果形状：对于二维情况，若 lhs: (m, n) 且 rhs: (n, p)，则结果: (m, p)
    npy_intp m = shape_lhs[nd_lhs - 2];
    npy_intp n = shape_lhs[nd_lhs - 1];
    npy_intp p = shape_rhs[nd_rhs - 1];
    npy_intp result_shape[2] = { m, p };
    int result_ndims = 2;
    fprintf(stderr,"4\n");
    %if locality_cache:
    #define RESULT_CACHE_VALID(elem) \
        ((PyObject *)(elem->result))->ob_refcnt == 1 && \
        PyArray_NDIM(elem->result) == result_ndims && \
        PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)

    #define CACHE_MATCH_TRIVIAL() ${"\\"}
        elem->state == TRIVIAL

    #define CACHE_MATCH_ITERATOR() ${"\\"}
        elem->state == ITERATOR

    PyArrayObject *result = NULL;
    if (CACHE_MATCH_TRIVIAL()) {
          fprintf(stderr,"trival\n");
        %if not inplace:
        if (RESULT_CACHE_VALID(elem)) {
        %endif
            %if inplace:
            result = lhs;
            %else:
            result = elem->result;
            %endif
            Py_INCREF(result);

            /* 构造传递给 DOUBLE_matmul 的参数：
             * dimensions: {dOuter, dm, dn, dp}，由于二维情况下 dOuter 为1。
             * steps 数组:前三个为外部步长(这里均为0,因为仅一块数据),
             * 接着为 lhs、rhs、result 的核心二维步长。
             */
            npy_intp dims[4] = {1, m, n, p};
            npy_intp steps[9];
            /* 外层步长:二维无外层广播,设为0 */
            steps[0] = 0;
            steps[1] = 0;
            steps[2] = 0;
            /* 对于 lhs,取倒数第二维（行）和最后一维（列）的步长 */
            npy_intp *lhs_strides = PyArray_STRIDES(lhs);
            steps[3] = lhs_strides[nd_lhs - 2];
            steps[4] = lhs_strides[nd_lhs - 1];
            /* 对于 rhs,取倒数第二维和最后一维;注意:rhs 的倒数第二维必须等于
             * n */
            npy_intp *rhs_strides = PyArray_STRIDES(rhs);
            steps[5] = rhs_strides[nd_rhs - 2];
            steps[6] = rhs_strides[nd_rhs - 1];
            /* 对于输出 result,二维数组的行列步长 */
            npy_intp *res_strides = PyArray_STRIDES(result);
            steps[7] = res_strides[0];
            steps[8] = res_strides[1];

            char *data[3];
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(rhs);
            data[2] = PyArray_BYTES(result);

            NPY_BEGIN_THREADS_DEF;
             /* 使用 dimensions 为 dims 数组,元素个数为4,
             * 使用 steps 数组传递完整的步长信息
             */
            NPY_BEGIN_THREADS_THRESHOLDED(dims[1] * dims[3]);

            NpyAuxData *auxdata = NULL;
            ${loop_function}(data, dims, steps, auxdata);

            NPY_END_THREADS;
            goto success;
        %if not inplace:
        } else {
            trivial_cache_miss(elem);
        }
        %endif
    } else if (CACHE_MATCH_ITERATOR()) {
          fprintf(stderr,"ITERATOR\n");
        %if not inplace:
        if (RESULT_CACHE_VALID(elem)) {
        %endif
            %if inplace:
            result = lhs;
            %else:
            result = elem->result;
            %endif

            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[3];
            op_it[0] = lhs; // lhs 固定为第一个操作数
            op_it[1] = rhs;
            op_it[2] = result;

            NpyIter_IterNextFunc *iter_next = elem->iterator.iter_next;
            char **dataptr = elem->iterator.dataptr;
            npy_intp *strides = elem->iterator.strides;

            char *baseptrs[3];
            baseptrs[0] = PyArray_BYTES(op_it[0]);
            baseptrs[1] = PyArray_BYTES(op_it[1]);
            baseptrs[2] = PyArray_BYTES(op_it[2]);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(elem->iterator.cached_iter));

            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }

            NpyAuxData *auxdata = NULL;

            do {
                ${loop_function}(dataptr, elem->iterator.countptr, strides, auxdata);
            } while (iter_next(iter));

            NPY_END_THREADS;
            goto success;
        %if not inplace:
        } else {
            iterator_cache_miss(elem);
        }
        %endif
    } else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            trivial_cache_miss(elem);
        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);
        }

        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }
    %endif



    npy_intp fixed_strides[3];
    int fast_path = 1;

    if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS)) ||
        !(PyArray_FLAGS(rhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
        fast_path = 2;
    } else {
        fixed_strides[0] = PyArray_ITEMSIZE(lhs);
        fixed_strides[1] = PyArray_ITEMSIZE(rhs);
        %if inplace:
        fixed_strides[2] = PyArray_ITEMSIZE(lhs);
        %endif
    }
        




    if(fast_path == 1) {
        <%count_stat("trivial_case")%>
        %if not inplace:

          // 如果 elide 失败，则需要创建一个新的结果数组
          PyArray_Descr *result_descr = PyArray_DescrFromType(${result_type});
          result = (PyArrayObject *)PyArray_NewFromDescr(
                  &PyArray_Type, result_descr, result_ndims,
                  result_shape, NULL, NULL,
                  0, NULL);
          fixed_strides[2] = PyArray_ITEMSIZE(result);

        %else:
            assert(lhs == result);
        %endif

       npy_intp count = m * p; 
        /* 构造 dimensions 和 steps 数组 */
        npy_intp dims[4] = {1, m, n, p};
        npy_intp steps[9];
        steps[0] = 0;
        steps[1] = 0;
        steps[2] = 0;
        {
            npy_intp *lhs_strides = PyArray_STRIDES(lhs);
            npy_intp *rhs_strides = PyArray_STRIDES(rhs);
            npy_intp *res_strides = PyArray_STRIDES(result);
            steps[3] = lhs_strides[nd_lhs - 2];
            steps[4] = lhs_strides[nd_lhs - 1];
            steps[5] = rhs_strides[nd_rhs - 2];
            steps[6] = rhs_strides[nd_rhs - 1];
            steps[7] = res_strides[0];
            steps[8] = res_strides[1];
        }
        /* 同样加入步长检查 */
        if (PyArray_STRIDES(lhs)[nd_lhs - 1] !=
                    (npy_intp)PyArray_ITEMSIZE(lhs) ||
            PyArray_STRIDES(rhs)[nd_rhs - 1] !=
                    (npy_intp)PyArray_ITEMSIZE(rhs) ||
            PyArray_STRIDES(result)[1] != (npy_intp)PyArray_ITEMSIZE(result)) {
            goto deopt;
        }

        char *data[3];
        data[0] = PyArray_BYTES(lhs);
        data[1] = PyArray_BYTES(rhs);
        data[2] = PyArray_BYTES(result);

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;

        ${loop_function}(data,dims, steps, auxdata);
        NPY_END_THREADS;

        %if locality_cache and not inplace:
            %if locality_cache_size_limit is not UNDEFINED:
            if (PyArray_NBYTES(result) >= ${locality_cache_size_limit}) {
                <%count_stat("result_too_big")%>
                cache_miss(elem);
                elem->state = DISABLED;
            }
            %endif

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |=
                NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);
                elem->trivial.count = dims[1] * dims[3];
                memcpy(elem->trivial.fixed_strides, fixed_strides,
                       3 * sizeof(npy_intp));
                elem->state = TRIVIAL;
                }    else {
                    elem->miss_counter++;
                }
            }
        %endif

        %if inplace:
            /* inplace 运算时不做缓存 */
        %endif

        goto success;
    }

    if (fast_path == 2) {
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(${result_type});
        PyArrayObject *ops[3] = {lhs, rhs, result};
        PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS,
                                  NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};


        %if inplace:
        assert(ops[0] == ops[2]);
        %endif

        NpyIter *iter = NpyIter_AdvancedNew(3, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[2];
        Py_INCREF(result);

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

        if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
            NpyIter_Deallocate(iter);
            Py_DECREF(result_descr);
            goto fail;
        }

        do {
            ${loop_function}(dataptr, countptr, strides, auxdata);
        } while (iternext(iter));
        NPY_END_THREADS;

        %if locality_cache and not inplace:
            int should_deallocate = 0;
            %if locality_cache_size_limit is not UNDEFINED:
            if (PyArray_NBYTES(result) >= ${locality_cache_size_limit}) {
                cache_miss(elem);
                elem->state = DISABLED;
            }
            %endif

            if (elem->state != DISABLED && result != lhs) {
                if (elem->miss_counter >= 0) {
                    elem->state = ITERATOR;
                    elem->iterator.countptr = countptr;
                    elem->iterator.dataptr = dataptr;
                    elem->iterator.strides = strides;
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    elem->iterator.cached_iter = iter;
                    elem->iterator.iter_next = *iternext;
                } else {
                    elem->miss_counter++;
                    should_deallocate = 1;
                }
            } else {
                should_deallocate = 1;
            }
        %else:
            %if inplace:
            /* inplace 操作不缓存 */
            %endif
            int should_deallocate = 1;
        %endif

        if (should_deallocate) {
            if (!NpyIter_Deallocate(iter)) {
                goto fail;
            }
        }

        Py_DECREF(result_descr);
        goto success;
    }

    raise(SIGTRAP);

deopt:
    %if locality_cache:
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
    %endif
    return 2;

success:
fprintf(stderr,"success\n");
%if cache_broadcast_array and left_scalar_name is not UNDEFINED:
    // the lhs is a cached broadcast array, no decref
%else:
    Py_DECREF(lhs);
%endif

%if cache_broadcast_array and left_scalar_name is UNDEFINED and right_scalar_name is not UNDEFINED:
    // the rhs is a cached broadcast array, no decref
%else:
    Py_DECREF(rhs);
%endif

%if right_scalar_name:
    Py_DECREF(m2);
%endif

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    //CMLQ_PAPI_END("${opname}")
    return 0;
fail:
    return -1;
}
