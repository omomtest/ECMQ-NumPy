case UFUNC_MINIMUM:
{


    if(
        PyLong_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_FLOAT
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_MINIMUM_AFLOAT_SLONG, cmlq_minimum_afloat_slong, &locality_cache[next_result_cache_index]);
        return 1;

    }


	report_missing_binop_case(instr, lhs, rhs);
return 0;

}
case UFUNC_MAXIMUM:
{


    if(
        PyLong_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_MAXIMUM_ADOUBLE_SLONG, cmlq_maximum_adouble_slong, &locality_cache[next_result_cache_index]);
        return 1;

    }

    if(
        PyLong_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_FLOAT
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_MAXIMUM_AFLOAT_SLONG, cmlq_maximum_afloat_slong, &locality_cache[next_result_cache_index]);
        return 1;

    }


	report_missing_binop_case(instr, lhs, rhs);
return 0;

}
case UFUNC_LOGICAL_NOT:
{

	report_missing_binop_case(instr, lhs, rhs);
return 0;

}
case UFUNC_LESS_EQUAL:
{

	report_missing_binop_case(instr, lhs, rhs);
return 0;

}
case UFUNC_LOGICAL_AND:
{

	report_missing_binop_case(instr, lhs, rhs);
return 0;

}
case UFUNC_ARCTAN2:
{

	report_missing_binop_case(instr, lhs, rhs);
return 0;

}
case UFUNC_ADD:
{



	report_missing_binop_case(instr, lhs, rhs);
return 0;

}
case UFUNC_SUBTRACT:
{



	report_missing_binop_case(instr, lhs, rhs);
return 0;

}
case UFUNC_MULTIPLY:
{



	report_missing_binop_case(instr, lhs, rhs);
return 0;

}
