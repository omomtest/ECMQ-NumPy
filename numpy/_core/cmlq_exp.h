	PyObject *lhs = STACK_ELEMENT(-1);
    if(
        PyArray_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_FLOAT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_EXP_AFLOAT, cmlq_exp_afloat, &locality_cache[next_result_cache_index]);
        return 1;

    }
    if(
        PyFloat_CheckExact(lhs) 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_EXP_SFLOAT, cmlq_exp_sfloat, &locality_cache[next_result_cache_index]);
        return 1;

    }
    if(
        PyArray_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_EXP_ADOUBLE, cmlq_exp_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
    if(
        PyFloat_CheckExact(lhs) 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_EXP_SDOUBLE, cmlq_exp_sdouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
	//missing func_one_Op

