case NB_SUBTRACT:
{




	report_missing_binop_case(instr, lhs, rhs);
	break;
}
case NB_INPLACE_SUBTRACT:
{


	report_missing_binop_case(instr, lhs, rhs);
	break;
}
case NB_ADD:
{


    if(
        PyFloat_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_ADD_SFLOAT_BROADCAST_CACHE, cmlq_adouble_add_sfloat_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_ADD_SFLOAT, cmlq_adouble_add_sfloat, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }


    if(
        PyComplex_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_CDOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_ACOMPLEX_ADD_SCOMPLEX_BROADCAST_CACHE, cmlq_acomplex_add_scomplex_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_ACOMPLEX_ADD_SCOMPLEX, cmlq_acomplex_add_scomplex, &locality_cache[next_result_cache_index]);
            }

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

            specializer_info.SpecializeInstruction(instr, SLOT_ADD_AFLOAT_SLONG_KW, cmlq_add_afloat_slong_kw, &locality_cache[next_result_cache_index]);
        return 1;

    }



	report_missing_binop_case(instr, lhs, rhs);
	break;
}
case NB_INPLACE_ADD:
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

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_AFLOAT_INPLACE_ADD_SLONG_BROADCAST_CACHE, cmlq_afloat_inplace_add_slong_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_AFLOAT_INPLACE_ADD_SLONG, cmlq_afloat_inplace_add_slong, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }

    if(
        PyLong_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_INPLACE_ADD_SLONG_BROADCAST_CACHE, cmlq_adouble_inplace_add_slong_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_INPLACE_ADD_SLONG, cmlq_adouble_inplace_add_slong, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }


    if(
        PyFloat_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_INPLACE_ADD_SFLOAT_BROADCAST_CACHE, cmlq_adouble_inplace_add_sfloat_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_INPLACE_ADD_SFLOAT, cmlq_adouble_inplace_add_sfloat, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }

	report_missing_binop_case(instr, lhs, rhs);
	break;
}
case NB_MULTIPLY:
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

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_MULTIPLY_SLONG_BROADCAST_CACHE, cmlq_adouble_multiply_slong_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_MULTIPLY_SLONG, cmlq_adouble_multiply_slong, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }

    if(
        PyFloat_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_MULTIPLY_SFLOAT_BROADCAST_CACHE, cmlq_adouble_multiply_sfloat_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_MULTIPLY_SFLOAT, cmlq_adouble_multiply_sfloat, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }

    if(
        PyLong_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_LONG
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_ALONG_MULTIPLY_SLONG_BROADCAST_CACHE, cmlq_along_multiply_slong_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_ALONG_MULTIPLY_SLONG, cmlq_along_multiply_slong, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }


    if(
        PyComplex_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_CDOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_ACOMPLEX_MULTIPLY_SCOMPLEX_BROADCAST_CACHE, cmlq_acomplex_multiply_scomplex_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_ACOMPLEX_MULTIPLY_SCOMPLEX, cmlq_acomplex_multiply_scomplex, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }

    if(
        PyLong_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_CDOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_ACOMPLEX_MULTIPLY_SLONG_BROADCAST_CACHE, cmlq_acomplex_multiply_slong_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_ACOMPLEX_MULTIPLY_SLONG, cmlq_acomplex_multiply_slong, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }


    if(
        PyFloat_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_CDOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_ACOMPLEX_MULTIPLY_SDOUBLE_BROADCAST_CACHE, cmlq_acomplex_multiply_sdouble_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_ACOMPLEX_MULTIPLY_SDOUBLE, cmlq_acomplex_multiply_sdouble, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }

    if(
        PyFloat_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_CDOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_ACOMPLEX_MULTIPLY_SFLOAT_BROADCAST_CACHE, cmlq_acomplex_multiply_sfloat_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_ACOMPLEX_MULTIPLY_SFLOAT, cmlq_acomplex_multiply_sfloat, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }

    if(
        PyComplex_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_FLOAT
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_AFLOAT_MULTIPLY_SCOMPLEX_BROADCAST_CACHE, cmlq_afloat_multiply_scomplex_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_AFLOAT_MULTIPLY_SCOMPLEX, cmlq_afloat_multiply_scomplex, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }

    if(
        PyComplex_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_MULTIPLY_SCOMPLEX_BROADCAST_CACHE, cmlq_adouble_multiply_scomplex_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_MULTIPLY_SCOMPLEX, cmlq_adouble_multiply_scomplex, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }

    if(
        PyComplex_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_LONG
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_ALONG_MULTIPLY_SCOMPLEX_BROADCAST_CACHE, cmlq_along_multiply_scomplex_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_ALONG_MULTIPLY_SCOMPLEX, cmlq_along_multiply_scomplex, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }


    if(
        PyFloat_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_MULTIPLY_ADOUBLE_SDOUBLE_KW, cmlq_multiply_adouble_sdouble_kw, &locality_cache[next_result_cache_index]);
        return 1;

    }


	report_missing_binop_case(instr, lhs, rhs);
	break;
}
case NB_INPLACE_MULTIPLY:
{
    if(
        PyFloat_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_INPLACE_MULTIPLY_SFLOAT_BROADCAST_CACHE, cmlq_adouble_inplace_multiply_sfloat_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_INPLACE_MULTIPLY_SFLOAT, cmlq_adouble_inplace_multiply_sfloat, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }

	report_missing_binop_case(instr, lhs, rhs);
	break;
}
case NB_TRUE_DIVIDE:
{

    if(
        PyComplex_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_CDOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_ACOMPLEX_TRUE_DIVIDE_SCOMPLEX_BROADCAST_CACHE, cmlq_acomplex_true_divide_scomplex_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_ACOMPLEX_TRUE_DIVIDE_SCOMPLEX, cmlq_acomplex_true_divide_scomplex, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }




    if(
        PyFloat_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

                cache_possible = specializer_info.IsOperandConstant(instr, *stack_pointer, 1);
                op = lhs;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_SFLOAT_TRUE_DIVIDE_ADOUBLE_BROADCAST_CACHE, cmlq_sfloat_true_divide_adouble_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_SFLOAT_TRUE_DIVIDE_ADOUBLE, cmlq_sfloat_true_divide_adouble, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }


    if(
        PyLong_CheckExact(lhs) &&
        PyArray_DESCR((PyArrayObject *)rhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_TRUE_DIVIDE_ADOUBLE_SLONG_KW, cmlq_true_divide_adouble_slong_kw, &locality_cache[next_result_cache_index]);
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
case NB_MATRIX_MULTIPLY:
{

	report_missing_binop_case(instr, lhs, rhs);
	break;
}
