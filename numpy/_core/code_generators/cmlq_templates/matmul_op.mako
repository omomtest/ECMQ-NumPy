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
    }else if (nd_lhs>2||nd_rhs>2){
        path=5;
    }
#define RESULT_CACHE_VALID(elem) \
    ((PyObject *)(elem->result))->ob_refcnt == 1 && \
    PyArray_NDIM(elem->result) == result_ndims  && \
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
case 5:{
    ##多维矩阵点乘
    if (shape_lhs[nd_lhs - 1] != shape_rhs[nd_rhs - 2]) {
        goto deopt;
        }
        
    int broadcast_ndim, iter_ndim;
    broadcast_ndim = 0;
    int n=nd_lhs>nd_rhs?nd_lhs:nd_rhs;
    int result_ndims=n;
    broadcast_ndim = n-2;
    npy_intp iter_shape[n];
        int idim;
        for (idim = 0; idim < broadcast_ndim; ++idim) {
        iter_shape[idim] = -1;
    }

    iter_shape[broadcast_ndim] = shape_lhs[nd_lhs - 2];
    iter_shape[broadcast_ndim + 1] = shape_rhs[nd_rhs - 1];

    npy_intp inner_dimensions[4] = {
    0,  
    iter_shape[broadcast_ndim],  
    shape_lhs[nd_lhs - 1],         
    iter_shape[broadcast_ndim+1]
    };
    npy_intp result_shape[n];
    if (CACHE_MATCH_ITERATOR()) {
            if(nd_lhs>=nd_rhs){
        int diff=nd_lhs-nd_rhs;
        for(int i=0;i<diff;i++){
            result_shape[i]=shape_lhs[i];
        }
        for(int i=diff;i<broadcast_ndim ;i++){
            result_shape[i]=shape_lhs[i]>shape_rhs[i-diff]?shape_lhs[i]:shape_rhs[i-diff];
        }
    }else{
        int diff=nd_rhs-nd_lhs;
        for(int i=0;i<diff;i++){
            result_shape[i]=shape_rhs[i];
        }
        for(int i=diff;i<broadcast_ndim ;i++){
            result_shape[i]=shape_rhs[i]>shape_lhs[i-diff]?shape_rhs[i]:shape_lhs[i-diff];
        }
    }
        result_shape[n-2] = shape_lhs[nd_lhs - 2];
        result_shape[n-1] = shape_rhs[nd_rhs - 1];
        if (RESULT_CACHE_VALID(elem)) {

            result = elem->result;
            // the result will be pushed to the stack
            Py_INCREF(result);

            NpyIter *iter = elem->iterator.cached_iter;

            PyArrayObject *op_it[2];
            op_it[0] = lhs; // is always LHS
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

            /* The reset may copy the first buffer chunk, which could cause FPEs */
            if (NpyIter_ResetBasePointers(iter, baseptrs, NULL) != NPY_SUCCEED) {
                NpyIter_Deallocate(iter);
                goto fail;
            }
            /* Execute the loop */
            do {
                inner_dimensions[0] =*(elem->iterator.countptr);
                ${loop_function}(dataptr, inner_dimensions, strides, NULL);
            } while (iter_next(iter));

            NPY_END_THREADS;
            goto success;
       } else {
            <%count_stat("result_cache_misses")%>
            <%count_stat("refcnt_misses", "((PyObject *)(elem->result))->ob_refcnt != 1")%>
            <%count_stat("ndims_misses", "PyArray_NDIM(elem->result) != result_ndims")%>
            <%count_stat("shape_misses", "!PyArray_CompareLists(result_shape, PyArray_SHAPE(elem->result), result_ndims)")%>

            iterator_cache_miss(elem);
       }
    }
    else {
        assert(elem->state == UNUSED || elem->state == DISABLED);

        if (elem->state == TRIVIAL) {
            // cache collision, free the old result element
            trivial_cache_miss(elem);
            <%count_stat("trivial_cache_collisions")%>
        }

        if (elem->state == ITERATOR) {
            iterator_cache_miss(elem);
            <%count_stat("iterator_cache_collisions")%>
        }

        // initialize cache element
        if (elem->state == UNUSED) {
            elem->result = NULL;
        }
    }



    npy_uint32 op_flags[3] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

    npy_uint32 iter_flags = NPY_ITER_MULTI_INDEX | NPY_ITER_REFS_OK |
             NPY_ITER_ZEROSIZE_OK | NPY_ITER_COPY_IF_OVERLAP |
             NPY_ITER_DELAY_BUFALLOC;    


    int op_axes_arrays[3][n];
    iter_ndim = broadcast_ndim + 2;


    if(nd_lhs>=nd_rhs){
        int diff=nd_lhs-nd_rhs;
        for(int i=0;i<diff;i++){
            op_axes_arrays[0][i]=i;
            op_axes_arrays[1][i]=-1;
            op_axes_arrays[2][i]=i;
            result_shape[i]=shape_lhs[i];
        }
        for(int i=diff;i<broadcast_ndim ;i++){
            op_axes_arrays[0][i]=i;
            op_axes_arrays[1][i]=i-diff;
            op_axes_arrays[2][i]=i;
            result_shape[i]=shape_lhs[i]>shape_rhs[i-diff]?shape_lhs[i]:shape_rhs[i-diff];
        }
    }else{
        int diff=nd_rhs-nd_lhs;
        for(int i=0;i<diff;i++){
            op_axes_arrays[0][i]=-1;
            op_axes_arrays[1][i]=i;
            op_axes_arrays[2][i]=i;
            result_shape[i]=shape_rhs[i];
        }
        for(int i=diff;i<broadcast_ndim ;i++){
            op_axes_arrays[0][i]=i-diff;
            op_axes_arrays[1][i]=i;
            op_axes_arrays[2][i]=i;
            result_shape[i]=shape_rhs[i]>shape_lhs[i-diff]?shape_rhs[i]:shape_lhs[i-diff];
        }
    }
        // lhs 的点乘轴
    op_axes_arrays[0][n - 2] =NPY_ITER_REDUCTION_AXIS(-1);           
    op_axes_arrays[0][n - 1] = NPY_ITER_REDUCTION_AXIS(-1); 
    // rhs 的点乘轴
    op_axes_arrays[1][n - 2] = NPY_ITER_REDUCTION_AXIS(-1); 
    op_axes_arrays[1][n - 1] = NPY_ITER_REDUCTION_AXIS(-1);       
    // result 的输出轴
    op_axes_arrays[2][n - 2] = broadcast_ndim;
    op_axes_arrays[2][n - 1] = broadcast_ndim + 1;
    result_shape[n-2] = shape_lhs[nd_lhs - 2];
    result_shape[n-1] = shape_rhs[nd_rhs - 1];
    

    int *op_axes[3] = { op_axes_arrays[0],
                        op_axes_arrays[1],
                        op_axes_arrays[2]};

    ## op_flags[0] =  NPY_UFUNC_DEFAULT_INPUT_FLAGS;
    ## op_flags[1] =op_flags[0] ;
    ## op_flags[2] =(NPY_ITER_WRITEONLY| NPY_ITER_UPDATEIFCOPY| NPY_UFUNC_DEFAULT_OUTPUT_FLAGS)
    ## & ~NPY_ITER_OVERLAP_ASSUME_ELEMENTWISE;



    
    PyArray_Descr *result_descr = PyArray_DescrFromType(${result_type});


    result = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, result_descr,
                                                       n, result_shape,
                                                       NULL, NULL, 0, NULL);
    PyArrayObject *ops[3]={lhs, rhs, result};
    PyArray_Descr *descriptors[3] = {PyArray_DESCR(lhs), PyArray_DESCR(rhs), result_descr};
        
    NpyIter *iter = NpyIter_AdvancedNew(3,ops ,iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,op_flags, 
                                            descriptors, iter_ndim, op_axes,iter_shape, 0);
    if (iter == NULL) {
        PyErr_Print();  // 哪一步参数有问题
        goto fail;         
    }
    ops[2] = NpyIter_GetOperandArray(iter);
    Py_INCREF(ops[2]);
        for (int i = broadcast_ndim; i < iter_ndim; ++i) {
        if (NpyIter_RemoveAxis(iter, broadcast_ndim) != NPY_SUCCEED) {
   
            goto fail;
        }
    }
    if (NpyIter_RemoveMultiIndex(iter) != NPY_SUCCEED) {
        goto fail;
    }
    if (NpyIter_EnableExternalLoop(iter) != NPY_SUCCEED) {
        goto fail;
    }
    npy_intp full_size = NpyIter_GetIterSize(iter);
    if (full_size == 0) {
         Py_DECREF(result_descr);
        if (!NpyIter_Deallocate(iter)) {
            goto fail;
        }
        goto success;
    }
    // 从 iterator 获取 innerloop 所需的指针和 shape
    char **dataptrs       = NpyIter_GetDataPtrArray(iter);
    npy_intp *count_ptr   = NpyIter_GetInnerLoopSizePtr(iter);
    npy_intp inner_strides[9] = {
        NpyIter_GetInnerStrideArray(iter)[0],
        NpyIter_GetInnerStrideArray(iter)[1],
        NpyIter_GetInnerStrideArray(iter)[2],
        PyArray_STRIDES(lhs)[nd_lhs - 2],
        PyArray_STRIDES(lhs)[nd_lhs - 1],
        PyArray_STRIDES(rhs)[nd_rhs - 2],
        PyArray_STRIDES(rhs)[nd_rhs - 1],
        PyArray_STRIDES(result)[n - 2],
        PyArray_STRIDES(result)[n - 1]
    };
    
    ## for(int i=0;i<9;i++){
    ##     fprintf(stderr,"inner_strides[%d]:%ld\n",i,inner_strides[i]);
    ## }

    NpyIter_IterNextFunc *iternext = NpyIter_GetIterNext(iter, NULL);

    ## for(int i=0;i<4;i++){
    ##     fprintf(stderr,"inner_dimensions[%d]:%ld\n",i,inner_dimensions[i]);
    ## }
    NPY_BEGIN_THREADS_DEF; 
    NPY_BEGIN_THREADS_THRESHOLDED(full_size);

    do{
        inner_dimensions[0] = *count_ptr;
       ${loop_function}(dataptrs, inner_dimensions, inner_strides, NULL);
    }while (iternext(iter));

    NPY_END_THREADS;
    int should_deallocate = 0;
    %if locality_cache_size_limit is not UNDEFINED:
        if (PyArray_NBYTES(result) >= ${locality_cache_size_limit}) {
            <%count_stat("result_too_big")%>
            cache_miss(elem);
            elem->state = DISABLED;
        }
    %endif
    if (elem->state != DISABLED && result != lhs) {
    if (elem->miss_counter >= 0) {
        elem->state = ITERATOR;

        elem->iterator.countptr = count_ptr;
        elem->iterator.dataptr = dataptrs;
        elem->iterator.strides = PyMem_Calloc(9 ,sizeof(npy_intp));
        memcpy(elem->iterator.strides, inner_strides, 9 * sizeof(npy_intp));


        // we do not need to increase the refcnt here because the iterator holds the reference
        elem->result = result;
        ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
        elem->iterator.cached_iter = iter;
        elem->iterator.iter_next = *iternext;
        <%count_stat("iterator_cache_init")%>
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
        ## if (elem->iterator.strides) {
        ##     PyDataMem_FREE(elem->iterator.strides);
        ##     elem->iterator.strides = NULL;
        ## }
            if (!NpyIter_Deallocate(iter)) {
                 goto fail;
            }
        }
    Py_DECREF(result_descr);

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
        } else if (elem->state == ITERATOR) {
            if (elem->iterator.strides) {
            PyDataMem_FREE(elem->iterator.strides);
            elem->iterator.strides = NULL;
        }
            NpyIter_Deallocate(elem->iterator.cached_iter);
            elem->iterator.cached_iter = NULL;
        }
        elem->state = UNUSED;
        
    }
    %endif
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(rhs);

    (*stack_pointer_ptr)--;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;
fail:
    return -1;
}
