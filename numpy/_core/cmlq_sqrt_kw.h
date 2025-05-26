    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_SQRT_ADOUBLE_KW, cmlq_sqrt_adouble_kw, &locality_cache[next_result_cache_index]);
        return 1;

    }
	//missing func_one_Op

