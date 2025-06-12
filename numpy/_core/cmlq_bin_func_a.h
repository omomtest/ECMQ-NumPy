case UFUNC_MINIMUM:
{
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
            specializer_info.SpecializeInstruction(instr, SLOT_MINIMUM_AINT_AINT, cmlq_minimum_aint_aint, &locality_cache[next_result_cache_index]);
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
            specializer_info.SpecializeInstruction(instr, SLOT_MINIMUM_AFLOAT_AFLOAT, cmlq_minimum_afloat_afloat, &locality_cache[next_result_cache_index]);
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
            specializer_info.SpecializeInstruction(instr, SLOT_MINIMUM_ADOUBLE_ADOUBLE, cmlq_minimum_adouble_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }

	report_missing_binop_case(instr, lhs, rhs);

}
case UFUNC_MAXIMUM:
{
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
            specializer_info.SpecializeInstruction(instr, SLOT_MAXIMUM_AINT_AINT, cmlq_maximum_aint_aint, &locality_cache[next_result_cache_index]);
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
            specializer_info.SpecializeInstruction(instr, SLOT_MAXIMUM_AFLOAT_AFLOAT, cmlq_maximum_afloat_afloat, &locality_cache[next_result_cache_index]);
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
            specializer_info.SpecializeInstruction(instr, SLOT_MAXIMUM_ADOUBLE_ADOUBLE, cmlq_maximum_adouble_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }

	report_missing_binop_case(instr, lhs, rhs);

}
case UFUNC_LOGICAL_NOT:
{
    if((
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_BOOL &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_BOOL
       )
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif
            specializer_info.SpecializeInstruction(instr, SLOT_LOGICAL_NOT_ABOOL_ABOOL, cmlq_logical_not_abool_abool, &locality_cache[next_result_cache_index]);
        return 1;

    }

	report_missing_binop_case(instr, lhs, rhs);

}
case UFUNC_LESS_EQUAL:
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
            specializer_info.SpecializeInstruction(instr, SLOT_LESS_EQUAL_ADOUBLE_ADOUBLE, cmlq_less_equal_adouble_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }

	report_missing_binop_case(instr, lhs, rhs);

}
case UFUNC_LOGICAL_AND:
{
    if((
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_BOOL &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_BOOL
       )
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif
            specializer_info.SpecializeInstruction(instr, SLOT_LOGICAL_AND_ABOOL_ABOOL, cmlq_logical_and_abool_abool, &locality_cache[next_result_cache_index]);
        return 1;

    }

	report_missing_binop_case(instr, lhs, rhs);

}
case UFUNC_ARCTAN2:
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
            specializer_info.SpecializeInstruction(instr, SLOT_ARCTAN2_ADOUBLE_ADOUBLE, cmlq_arctan2_adouble_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }

	report_missing_binop_case(instr, lhs, rhs);

}
case UFUNC_ADD:
{
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
            specializer_info.SpecializeInstruction(instr, SLOT_ADD_AINT_AINT, cmlq_add_aint_aint, &locality_cache[next_result_cache_index]);
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
            specializer_info.SpecializeInstruction(instr, SLOT_ADD_AFLOAT_AFLOAT, cmlq_add_afloat_afloat, &locality_cache[next_result_cache_index]);
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
            specializer_info.SpecializeInstruction(instr, SLOT_ADD_ADOUBLE_ADOUBLE, cmlq_add_adouble_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }

	report_missing_binop_case(instr, lhs, rhs);

}
case UFUNC_SUBTRACT:
{
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
            specializer_info.SpecializeInstruction(instr, SLOT_SUBTRACT_AINT_AINT, cmlq_subtract_aint_aint, &locality_cache[next_result_cache_index]);
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
            specializer_info.SpecializeInstruction(instr, SLOT_SUBTRACT_AFLOAT_AFLOAT, cmlq_subtract_afloat_afloat, &locality_cache[next_result_cache_index]);
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
            specializer_info.SpecializeInstruction(instr, SLOT_SUBTRACT_ADOUBLE_ADOUBLE, cmlq_subtract_adouble_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }

	report_missing_binop_case(instr, lhs, rhs);

}
case UFUNC_MULTIPLY:
{
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

}
