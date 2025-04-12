case NB_SUBTRACT:
{
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

            specializer_info.SpecializeInstruction(instr, SLOT_AFLOAT_SUBTRACT_AFLOAT, cmlq_afloat_subtract_afloat, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_SUBTRACT_ADOUBLE, cmlq_adouble_subtract_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }


	report_missing_binop_case(instr, lhs, rhs);
	break;
}
case NB_INPLACE_SUBTRACT:
{
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

            specializer_info.SpecializeInstruction(instr, SLOT_AFLOAT_INPLACE_SUBTRACT_AFLOAT, cmlq_afloat_inplace_subtract_afloat, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_INPLACE_SUBTRACT_ADOUBLE, cmlq_adouble_inplace_subtract_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }

	report_missing_binop_case(instr, lhs, rhs);
	break;
}
case NB_ADD:
{
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

            specializer_info.SpecializeInstruction(instr, SLOT_AFLOAT_ADD_AFLOAT, cmlq_afloat_add_afloat, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_ADD_ADOUBLE, cmlq_adouble_add_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }


	report_missing_binop_case(instr, lhs, rhs);
	break;
}
case NB_INPLACE_ADD:
{
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

            specializer_info.SpecializeInstruction(instr, SLOT_AFLOAT_INPLACE_ADD_AFLOAT, cmlq_afloat_inplace_add_afloat, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_INPLACE_ADD_ADOUBLE, cmlq_adouble_inplace_add_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }


	report_missing_binop_case(instr, lhs, rhs);
	break;
}
case NB_MULTIPLY:
{
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

            specializer_info.SpecializeInstruction(instr, SLOT_AFLOAT_MULTIPLY_AFLOAT, cmlq_afloat_multiply_afloat, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_MULTIPLY_ADOUBLE, cmlq_adouble_multiply_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }




	report_missing_binop_case(instr, lhs, rhs);
	break;
}
case NB_INPLACE_MULTIPLY:
{

	report_missing_binop_case(instr, lhs, rhs);
	break;
}
case NB_TRUE_DIVIDE:
{




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

            specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_TRUE_DIVIDE_ADOUBLE, cmlq_adouble_true_divide_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }

	report_missing_binop_case(instr, lhs, rhs);
	break;
}
case NB_POWER:
{




	report_missing_binop_case(instr, lhs, rhs);
	break;
}
