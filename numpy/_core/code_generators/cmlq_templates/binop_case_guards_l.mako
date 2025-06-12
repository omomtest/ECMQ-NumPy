    %if left_scalar_name is not UNDEFINED and commutative or right_scalar_name is not UNDEFINED :
    if(
        %if left_scalar_name is not UNDEFINED :
        %if commutative:
        Py${left_scalar_name}_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == ${right_numpy_name}
        %endif
        %elif  right_scalar_name is not UNDEFINED:
        Py${right_scalar_name}_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == ${left_numpy_name}
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


            %if right_scalar_name is not UNDEFINED and left_scalar_name is UNDEFINED:

                %if right_promotion is not None:
                descr = PyArray_DescrFromType(${right_promotion});
                %endif

                cache_possible = specializer_info.IsOperandConstant(instr, *stack_pointer, 0);
                op = rhs;

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