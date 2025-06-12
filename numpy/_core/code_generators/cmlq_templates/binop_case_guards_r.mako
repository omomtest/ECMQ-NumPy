    %if left_scalar_name is not UNDEFINED or right_scalar_name is not UNDEFINED and commutative:
    if(
        %if left_scalar_name is not UNDEFINED:
        Py${left_scalar_name}_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == ${right_numpy_name}
        %elif right_scalar_name is not UNDEFINED and commutative:
        Py${right_scalar_name}_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == ${left_numpy_name}
        %else:
        0
        %endif
    )
    {
    <%block name="install_handler">
        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

        %if with_broadcast_cache_variant:

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

            %if left_scalar_name is not UNDEFINED:

                %if left_promotion is not None:
                descr = PyArray_DescrFromType(${left_promotion});
                %endif

                cache_possible = specializer_info.IsOperandConstant(instr, *stack_pointer, 1);
                op = lhs;


            %endif


            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, ${slot_name}_BROADCAST_CACHE, ${opname}_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, ${slot_name}, ${opname}, &locality_cache[next_result_cache_index]);
            }

        %else:
            specializer_info.SpecializeInstruction(instr, ${slot_name}, ${opname}, &locality_cache[next_result_cache_index]);
        %endif
        return 1;
    </%block>
    }
    %endif