
    if(
        PyFloat_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_MULTIPLY_SDOUBLE_ADOUBLE_KW, cmlq_multiply_sdouble_adouble_kw, &locality_cache[next_result_cache_index]);
        return 1;

    }


	report_missing_binop_case(instr, lhs, rhs);

