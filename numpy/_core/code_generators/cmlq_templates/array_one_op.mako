<%page args="result_type, try_elide_temp=False, auxdata_init=None, arity=1, inplace=False"/>
<% assert arity == 2 or arity == 1 %>
<%namespace file="cache_stats_macro.mako" import="*"/>

    npy_intp fixed_strides[${arity + 1}];
    int fast_path = 1;
    %if left_numpy_name is not UNDEFINED:
        if (PyArray_NDIM(lhs) == 1) {
            fixed_strides[0] = PyArray_STRIDES(lhs)[0];
        } else {
            if (!(PyArray_FLAGS(lhs) & (NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS))) {
                fast_path = 2;
            } else {
                fixed_strides[0] = PyArray_ITEMSIZE(lhs);
            }
        }
    %endif
    %if not inplace:
        %if try_elide_temp:
            // 尝试避免创建临时数组（单目运算）
            if (can_elide_temp_unary(m1)) {
                result = m1;
            }
        %endif
    %else:
        // inplace 操作，结果与 lhs 相同
        Py_INCREF(lhs);
        result = lhs;
    %endif

    <%count_stat("temp_elision_hits", "result != NULL")%>

    if(fast_path == 1) {
        <%count_stat("trivial_case")%>
        %if not inplace:
            %if try_elide_temp:
            if (result == NULL) {
            %endif
                // 没有临时数组优化，需要创建新数组
                PyArray_Descr *result_descr = PyArray_DescrFromType(${result_type});

                // 此时已知操作数维数匹配，直接用 lhs 的 ndim/shape 创建结果
                result = (PyArrayObject *)PyArray_NewFromDescr(
                        &PyArray_Type, result_descr, result_ndims,
                        result_shape, NULL, NULL,
                        0, NULL);
                fixed_strides[${arity}] = PyArray_ITEMSIZE(result);;
            %if try_elide_temp:
            } else {
                // 存在临时数组优化，复用结果并复制 lhs 的 strides
                Py_INCREF(result);
                if (PyArray_NDIM(lhs) == 1) {
                    fixed_strides[${arity}] = PyArray_STRIDES(lhs)[0];
                } else {
                    fixed_strides[${arity}] = PyArray_ITEMSIZE(lhs);
                }
            }
            %endif
        %else:
            assert(lhs == result);
        %endif

        npy_intp count = PyArray_MultiplyList(result_shape, result_ndims);

        char *data[${arity + 1}];
        %if arity == 1:
            data[0] = PyArray_BYTES(lhs);
            data[1] = PyArray_BYTES(result);
        %endif

        NPY_BEGIN_THREADS_DEF;
        NPY_BEGIN_THREADS_THRESHOLDED(count);

        NpyAuxData *auxdata = NULL;
        %if auxdata_init is not None:
            ${auxdata_init()}
        %endif

        ${loop_function}(data, &count, fixed_strides, auxdata);
        //CMLQ_PAPI_REGION("core_loop", ${loop_function}(data, &count, fixed_strides, auxdata));

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
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    Py_INCREF(elem->result);

                    elem->trivial.count = count;
                    elem->trivial.fixed_strides[0] = fixed_strides[0];
                    elem->trivial.fixed_strides[1] = fixed_strides[1];
                    elem->state = TRIVIAL;
                    <%count_stat("trivial_cache_init")%>
                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                }
            }
        %endif
        %if inplace:
            // inplace 操作不缓存结果
        %endif

        goto success;
    }

    if (fast_path == 2) {
        <%count_stat("iterator_case")%>
        // 使用迭代器处理不满足 fast_path 的情况
        npy_uint32 iter_flags = NPY_ITER_EXTERNAL_LOOP | NPY_ITER_REFS_OK |
                                NPY_ITER_ZEROSIZE_OK | NPY_ITER_BUFFERED |
                                NPY_ITER_GROWINNER | NPY_ITER_DELAY_BUFALLOC |
                                NPY_ITER_COPY_IF_OVERLAP;

        PyArray_Descr *result_descr = PyArray_DescrFromType(${result_type});

        %if arity == 1:
            PyArray_Descr *descriptors[2] = {PyArray_DESCR(lhs), result_descr};
            npy_uint32 op_flags[2] = {NPY_UFUNC_DEFAULT_INPUT_FLAGS, NPY_ITER_WRITEONLY | NPY_UFUNC_DEFAULT_OUTPUT_FLAGS};

            // 若 result 非空，则复用，否则迭代器将创建新数组
            ops[1] = result;
        %endif

        %if inplace:
            assert(ops[0] == ops[${arity}]);
        %endif

        NpyIter *iter = NpyIter_AdvancedNew(${arity + 1}, ops,
                                            iter_flags,
                                            NPY_KEEPORDER, NPY_UNSAFE_CASTING,
                                            op_flags, descriptors,
                                            -1, NULL, NULL, NPY_BUFSIZE);
        if (iter == NULL) {
            goto fail;
        }

        /* 将输出数组设为结果（迭代器可能已经创建了数组） */
        PyArrayObject **op_it = NpyIter_GetOperandArray(iter);
        result = op_it[${arity}];

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

        char *baseptrs[${arity + 1}];
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
        %if auxdata_init is not None:
            ${auxdata_init()}
        %endif

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
            ${loop_function}(dataptr, countptr, strides, auxdata);
            //CMLQ_PAPI_REGION("core_loop", ${loop_function}(dataptr, countptr, strides, auxdata));
        } while (iternext(iter));

        NPY_END_THREADS;

        %if locality_cache and not inplace:
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

                    elem->iterator.countptr = countptr;
                    elem->iterator.dataptr = dataptr;
                    elem->iterator.strides = strides;

                    // 迭代器持有引用，无需增加 refcnt
                    elem->result = result;
                    ((PyArrayObject_fields *)elem->result)->flags |= NPY_ARRAY_IN_LOCALITY_CACHE;
                    elem->iterator.cached_iter = iter;
                    elem->iterator.iter_next = *iternext;
                    <%count_stat("iterator_cache_init")%>
                } else {
                    // 预热结果缓存
                    elem->miss_counter++;
                    should_deallocate = 1;
                }
            } else {
                // 若未缓存，则释放迭代器
                should_deallocate = 1;
            }
        %else:
            // 无 locality cache，总是释放迭代器
            %if inplace:
                // inplace 操作不缓存结果
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

    // 无法处理的其它错误情况
    raise(SIGTRAP);
