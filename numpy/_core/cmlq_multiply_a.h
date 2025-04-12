    if((
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_INT &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_INT
       )
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_MULTIPLY_AINT_AINT, cmlq_multiply_aint_aint, &locality_cache[next_result_cache_index]);
        return 1;

    }
    if((
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_FLOAT &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_FLOAT
       )
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_MULTIPLY_AFLOAT_AFLOAT, cmlq_multiply_afloat_afloat, &locality_cache[next_result_cache_index]);
        return 1;

    }
    if((
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_DOUBLE
       )
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_MULTIPLY_ADOUBLE_ADOUBLE, cmlq_multiply_adouble_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
	report_missing_binop_case(instr, lhs, rhs);

