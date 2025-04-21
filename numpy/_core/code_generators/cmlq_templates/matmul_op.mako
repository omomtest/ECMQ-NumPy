<%namespace file="cache_stats_macro.mako" import="*"/>
${signature}
{   
    %if locality_cache or locality_stats:
    <%include file="load_cache_elem.mako"/>
    %endif
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

    // 检查矩阵乘法的基本要求：
    // 两个输入都至少应为二维数组，并且 lhs 的最后一维必须与 rhs 的倒数第二维相等。
    npy_intp nd_lhs = PyArray_NDIM(lhs);
    npy_intp nd_rhs = PyArray_NDIM(rhs);
    if (nd_lhs > 2|| nd_rhs >2) {
        goto deopt;
    }
    npy_intp *shape_lhs = PyArray_SHAPE(lhs);
    npy_intp *shape_rhs = PyArray_SHAPE(rhs);
    PyArrayObject *result = NULL;
    int path=1;
    if (nd_lhs == 1&&nd_rhs == 1) {
       path=2;
    }else if (nd_lhs==1){
        path=3;
    }else if (nd_rhs==1){
        path=4;
    }
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims && \
    PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)
#define CACHE_MATCH_TRIVIAL() ${"\\"}
    elem->state == TRIVIAL
#define CACHE_MATCH_ITERATOR() ${"\\"}
    elem->state == ITERATOR

switch(path)
{
case 1:{
        if (shape_lhs[nd_lhs - 1] != shape_rhs[nd_rhs - 2]) {
        goto deopt;
        }
        // 计算结果形状：对于二维情况，若 lhs: (m, n) 且 rhs: (n, p)，则结果: (m, p)
        int result_ndims = 2;
        npy_intp m = shape_lhs[nd_lhs - 2];
        npy_intp n = shape_lhs[nd_lhs - 1];
        npy_intp p = shape_rhs[nd_rhs - 1];
        npy_intp result_shape[2] = { m, p };
        if (m < 0 || n < 0 || p < 0) {
        goto fail;
        }
        npy_intp dims[4] = {1, m, n, p};
        %if locality_cache:

        if (CACHE_MATCH_TRIVIAL()) {

            %if not inplace:
            if (RESULT_CACHE_VALID(elem)) {
            %endif
                %if inplace:
                result = lhs;
                %else:
                result = elem->result;
                %endif
                Py_INCREF(result);
                char *data[3];
                data[0] = PyArray_BYTES(lhs);
                data[1] = PyArray_BYTES(rhs);
                data[2] = PyArray_BYTES(result);
                NpyAuxData *auxdata = NULL;
                NPY_BEGIN_THREADS_DEF;
                NPY_BEGIN_THREADS_THRESHOLDED(dims[1] *dims[2]*dims[3]);
                ${loop_function}(data, dims, elem->trivial.steps, auxdata);
                NPY_END_THREADS;
                goto success;
            %if not inplace:
            } else {
                trivial_cache_miss(elem);
            }
            %endif
        }  else {
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

        determine_elide_temp_binary(m1, m2, (PyObject **)&result, 1);
        // 缓存未命中：创建新结果数组并计算
    
        if (result == NULL) {
            PyArray_Descr *result_descr = PyArray_DescrFromType(${result_type});
            result = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, result_descr,
                                                       result_ndims, result_shape,
                                                       NULL, NULL, 0, NULL);
        }else{
            Py_INCREF(result);
        }
        npy_intp steps[9] = {0,
                             0,
                             0,
                             PyArray_STRIDES(lhs)[nd_lhs - 2],
                             PyArray_STRIDES(lhs)[nd_lhs - 1],
                             PyArray_STRIDES(rhs)[nd_rhs - 2],
                             PyArray_STRIDES(rhs)[nd_rhs - 1],
                             PyArray_STRIDES(result)[0],
                             PyArray_STRIDES(result)[1]};
        char *data[3] = {PyArray_BYTES(lhs), PyArray_BYTES(rhs),
                         PyArray_BYTES(result)};
        NpyAuxData *auxdata = NULL;
        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(dims[1] *dims[2]* dims[3]);
        ${loop_function}(data, dims, steps, auxdata);
        NPY_END_THREADS;

        %if locality_cache_size_limit is not UNDEFINED:
        if (PyArray_NBYTES(result) >= ${locality_cache_size_limit}) {
            <%count_stat("result_too_big")%>
            cache_miss(elem);
            elem->state = DISABLED;
        }
        %endif


        // 更新缓存
        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {

                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                for(int i =0;i<9;i++){
                    elem->trivial.steps[i]=steps[i];
                }
                
                elem->state=TRIVIAL;
            }
            else {
                elem->miss_counter++;
            }
        }

        goto success;
}   
case 2:{
    int result_ndims = 0;
    // 确保 lhs 和 rhs 都是一维数组
    if (shape_lhs[0] != shape_rhs[0]) {
        goto deopt; // 如果维度不匹配，则跳转到 deopt
    }


    npy_intp result_shape[1] = {1}; // 结果是一个标量
    
    %if locality_cache:
    if (CACHE_MATCH_TRIVIAL()) {
        
        %if not inplace:
        if (RESULT_CACHE_VALID(elem)) {
        %endif
            %if inplace:
            result = lhs;
            %else:
            result = elem->result;
            %endif
            Py_INCREF(result);

            NPY_BEGIN_THREADS_DEF;
            NPY_BEGIN_THREADS_THRESHOLDED(shape_lhs[0]);
            ${res_c_type}_dot(PyArray_BYTES(lhs),elem->trivial.fixed_strides[0], PyArray_BYTES(rhs),
            elem->trivial.fixed_strides[1], PyArray_BYTES(result),shape_lhs[0],NULL);
            NPY_END_THREADS;
            //DOUBLE_dot(ip1, is1_n, ip2, is2_n, op, dn, NULL);
            goto success;
        %if not inplace:
        } else {
            trivial_cache_miss(elem);
        }
        %endif
    }  else {
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
    
    determine_elide_temp_binary(m1, m2, (PyObject **)&result, 1);
    // 缓存未命中：创建新结果数组并计算
    if (result == NULL) {
        PyArray_Descr *result_descr = PyArray_DescrFromType(${result_type});
        result = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, result_descr,
                                                   0, result_shape, NULL, NULL, 0, NULL);
    }else{
        Py_INCREF(result);
    }
    NPY_BEGIN_THREADS_DEF;
    NPY_BEGIN_THREADS_THRESHOLDED(shape_lhs[0]);
    ${res_c_type}_dot(PyArray_BYTES(lhs),PyArray_STRIDES(lhs)[0], PyArray_BYTES(rhs),
             PyArray_STRIDES(rhs)[0], PyArray_BYTES(result),shape_lhs[0],NULL);
    NPY_END_THREADS;
    //DOUBLE_dot(ip1, is1_n, ip2, is2_n, op, dn, NULL);

    %if locality_cache_size_limit is not UNDEFINED:
        if (PyArray_NBYTES(result) >= ${locality_cache_size_limit}) {
            <%count_stat("result_too_big")%>
            cache_miss(elem);
            elem->state = DISABLED;
        }
    %endif

    //更新缓存
    if (elem->state != DISABLED && result != lhs) {
        if (elem->miss_counter >= 0) {
            elem->result = result;
            ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
            Py_INCREF(elem->result);
            elem->trivial.fixed_strides[0]=PyArray_STRIDES(lhs)[0];
            elem->trivial.fixed_strides[1]=PyArray_STRIDES(rhs)[0];
            elem->state=TRIVIAL;
        }
        else {
            elem->miss_counter++;
        }
    }
    goto success;
}
case 3:{
    if (shape_lhs[0] != shape_rhs[nd_rhs - 2]) {
        goto deopt;
        }
        // 计算结果形状：对于二维情况，若 lhs: (n,) 且 rhs: (n, p)，则结果: (p,)
        int result_ndims = 1;
        npy_intp n = shape_lhs[0];
        npy_intp p = shape_rhs[nd_rhs - 1];
        npy_intp result_shape[1] = {p};
        npy_intp dims[4] = {1, 1, n, p};
        %if locality_cache:
    
        if (CACHE_MATCH_TRIVIAL()) {

            %if not inplace:
            if (RESULT_CACHE_VALID(elem)) {
            %endif
                %if inplace:
                result = lhs;
                %else:
                result = elem->result;
                %endif
                Py_INCREF(result);
                char *data[3];
                data[0] = PyArray_BYTES(lhs);
                data[1] = PyArray_BYTES(rhs);
                data[2] = PyArray_BYTES(result);
                NpyAuxData *auxdata = NULL;
                NPY_BEGIN_THREADS_DEF;
                NPY_BEGIN_THREADS_THRESHOLDED(dims[1] *dims[2]* dims[3]);
                ${loop_function}(data, dims, elem->trivial.steps, auxdata);
                NPY_END_THREADS;
                goto success;
            %if not inplace:
            } else {
                trivial_cache_miss(elem);
            }
            %endif
        }  else {
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

        determine_elide_temp_binary(m1, m2, (PyObject **)&result, 1);
        // 缓存未命中：创建新结果数组并计算
    
        if (result == NULL) {
            PyArray_Descr *result_descr = PyArray_DescrFromType(${result_type});
            result = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, result_descr,
                                                       result_ndims, result_shape,
                                                       NULL, NULL, 0, NULL);
        }else{
            Py_INCREF(result);
        }
        npy_intp steps[9] = {0,
                             0,
                             0,
                             PyArray_STRIDES(lhs)[0]*n,
                             PyArray_STRIDES(lhs)[0],
                             PyArray_STRIDES(rhs)[nd_rhs - 2],
                             PyArray_STRIDES(rhs)[nd_rhs - 1],
                             PyArray_STRIDES(result)[0]*p,
                             PyArray_STRIDES(result)[0]};
        char *data[3] = {PyArray_BYTES(lhs), PyArray_BYTES(rhs),
                         PyArray_BYTES(result)};
        NpyAuxData *auxdata = NULL;
        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(dims[1] *dims[2]* dims[3]);
        ${loop_function}(data, dims, steps, auxdata);
        NPY_END_THREADS;

        // 更新缓存
        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                for(int i =0;i<9;i++){
                    elem->trivial.steps[i]=steps[i];
                }
                elem->state=TRIVIAL;
            }
            else {
                elem->miss_counter++;
            }
        }

        goto success;
}
case 4:{
    if (shape_lhs[nd_lhs - 1] != shape_rhs[0]) {
        goto deopt;
        }
        // 计算结果形状：对于二维情况，若 lhs: (m, n) 且 rhs: (n,)，则结果: (m,)
        int result_ndims = 1;
        npy_intp m = shape_lhs[nd_lhs - 2];
        npy_intp n = shape_rhs[0];
        npy_intp result_shape[1] = { m };

        npy_intp dims[4] = {1, m, n, 1};
        %if locality_cache:
        
        if (CACHE_MATCH_TRIVIAL()) {

            %if not inplace:
            if (RESULT_CACHE_VALID(elem)) {
            %endif
                %if inplace:
                result = lhs;
                %else:
                result = elem->result;
                %endif
                Py_INCREF(result);
                char *data[3];
                data[0] = PyArray_BYTES(lhs);
                data[1] = PyArray_BYTES(rhs);
                data[2] = PyArray_BYTES(result);
                NpyAuxData *auxdata = NULL;
                NPY_BEGIN_THREADS_DEF;
                NPY_BEGIN_THREADS_THRESHOLDED(dims[1]*dims[2]*dims[3]);
                ${loop_function}(data, dims, elem->trivial.steps, auxdata);
                NPY_END_THREADS;
                goto success;
            %if not inplace:
            } else {
                trivial_cache_miss(elem);
            }
            %endif
        }  else {
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

        determine_elide_temp_binary(m1, m2, (PyObject **)&result, 1);
        // 缓存未命中：创建新结果数组并计算
    
        if (result == NULL) {
            PyArray_Descr *result_descr = PyArray_DescrFromType(${result_type});
            result = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, result_descr,
                                                       result_ndims, result_shape,
                                                       NULL, NULL, 0, NULL);
        }else{
            Py_INCREF(result);
        }
        npy_intp steps[9] = {0,
                             0,
                             0,
                             PyArray_STRIDES(lhs)[nd_lhs - 2],
                             PyArray_STRIDES(lhs)[nd_lhs - 1],
                             PyArray_STRIDES(rhs)[0],
                             PyArray_STRIDES(rhs)[0],
                             PyArray_STRIDES(result)[0],
                             PyArray_STRIDES(result)[0]};
        char *data[3] = {PyArray_BYTES(lhs), PyArray_BYTES(rhs),
                         PyArray_BYTES(result)};
        NpyAuxData *auxdata = NULL;
        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(dims[1]*dims[2] * dims[3]);
        ${loop_function}(data, dims, steps, auxdata);
        NPY_END_THREADS;

        %if locality_cache_size_limit is not UNDEFINED:
        if (PyArray_NBYTES(result) >= ${locality_cache_size_limit}) {
            <%count_stat("result_too_big")%>
            cache_miss(elem);
            elem->state = DISABLED;
        }
        %endif

        // 更新缓存
        if (elem->state != DISABLED && result != lhs) {
            if (elem->miss_counter >= 0) {
                elem->result = result;
                ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                Py_INCREF(elem->result);

                for(int i =0;i<9;i++){
                    elem->trivial.steps[i]=steps[i];
                }
                elem->state=TRIVIAL;
            }
            else {
                elem->miss_counter++;
            }
        }

        goto success;
}
default:{
        goto deopt;
    }
}


deopt:
    %if locality_cache:
    elem = (CMLQLocalityCacheElem *)external_cache_pointer;
    if (elem->state != UNUSED) {
        elem->result = NULL;
        if (elem->state == TRIVIAL) {
            Py_XDECREF(elem->result);
        } 
        elem->state = UNUSED;
        
    }
    %endif
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);
%if right_scalar_name:
    Py_DECREF(m2);
%endif
    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;
fail:
    return -1;
}
