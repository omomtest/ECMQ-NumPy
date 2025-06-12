case UFUNC_SQUARE:
{
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_INT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_SQUARE_AINT, cmlq_square_aint, &locality_cache[next_result_cache_index]);
        return 1;

    }
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_SQUARE_ADOUBLE, cmlq_square_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_FLOAT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_SQUARE_AFLOAT, cmlq_square_afloat, &locality_cache[next_result_cache_index]);
        return 1;

    }
	report_missing_binop_case(instr, lhs, rhs);

}
case UFUNC_SQRT:
{
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_SQRT_ADOUBLE, cmlq_sqrt_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_FLOAT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_SQRT_AFLOAT, cmlq_sqrt_afloat, &locality_cache[next_result_cache_index]);
        return 1;

    }
	report_missing_binop_case(instr, lhs, rhs);

}
case UFUNC_ABSOLUTE:
{
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_INT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_ABSOLUTE_AINT, cmlq_absolute_aint, &locality_cache[next_result_cache_index]);
        return 1;

    }
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_ABSOLUTE_ADOUBLE, cmlq_absolute_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_FLOAT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_ABSOLUTE_AFLOAT, cmlq_absolute_afloat, &locality_cache[next_result_cache_index]);
        return 1;

    }
	report_missing_binop_case(instr, lhs, rhs);

}
case UFUNC_RECIPROCAL:
{
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_FLOAT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_RECIPROCAL_AFLOAT, cmlq_reciprocal_afloat, &locality_cache[next_result_cache_index]);
        return 1;

    }
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_RECIPROCAL_ADOUBLE, cmlq_reciprocal_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_INT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_RECIPROCAL_AINT, cmlq_reciprocal_aint, &locality_cache[next_result_cache_index]);
        return 1;

    }
	report_missing_binop_case(instr, lhs, rhs);

}
case UFUNC_TANH:
{
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_FLOAT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_TANH_AFLOAT, cmlq_tanh_afloat, &locality_cache[next_result_cache_index]);
        return 1;

    }
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_TANH_ADOUBLE, cmlq_tanh_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
	report_missing_binop_case(instr, lhs, rhs);

}
case UFUNC_EXP:
{
    if(
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
	report_missing_binop_case(instr, lhs, rhs);

}
