case NB_SUBTRACT:
{


    if(
        PyFloat_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

                cache_possible = PyFloat_CheckExact(rhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 0);
                op = rhs;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_SUBTRACT_SFLOAT_BROADCAST_CACHE, cmlq_adouble_subtract_sfloat_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_SUBTRACT_SFLOAT, cmlq_adouble_subtract_sfloat, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }

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
        PyFloat_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

                cache_possible = PyFloat_CheckExact(rhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 0);
                op = rhs;

                // try the commutative case
                if (!cache_possible) {

                    cache_possible = PyFloat_CheckExact(lhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 1);
                    op = lhs;
                }

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

	report_missing_binop_case(instr, lhs, rhs);
	break;
}
case NB_INPLACE_ADD:
{

    if(
        PyLong_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_FLOAT
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

                cache_possible = PyLong_CheckExact(rhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 0);
                op = rhs;

                // try the commutative case
                if (!cache_possible) {

                    cache_possible = PyLong_CheckExact(lhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 1);
                    op = lhs;
                }

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
        PyLong_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

                cache_possible = PyLong_CheckExact(rhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 0);
                op = rhs;

                // try the commutative case
                if (!cache_possible) {

                    cache_possible = PyLong_CheckExact(lhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 1);
                    op = lhs;
                }

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
        PyFloat_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

                cache_possible = PyFloat_CheckExact(rhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 0);
                op = rhs;

                // try the commutative case
                if (!cache_possible) {

                    cache_possible = PyFloat_CheckExact(lhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 1);
                    op = lhs;
                }

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
        PyLong_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

                descr = PyArray_DescrFromType(NPY_DOUBLE);

                cache_possible = PyLong_CheckExact(rhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 0);
                op = rhs;

                // try the commutative case
                if (!cache_possible) {

                    cache_possible = PyLong_CheckExact(lhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 1);
                    op = lhs;
                }

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
        PyFloat_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

                cache_possible = PyFloat_CheckExact(rhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 0);
                op = rhs;

                // try the commutative case
                if (!cache_possible) {

                    cache_possible = PyFloat_CheckExact(lhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 1);
                    op = lhs;
                }

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
        PyLong_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_LONG
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

                cache_possible = PyLong_CheckExact(rhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 0);
                op = rhs;

                // try the commutative case
                if (!cache_possible) {

                    cache_possible = PyLong_CheckExact(lhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 1);
                    op = lhs;
                }

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

	report_missing_binop_case(instr, lhs, rhs);
	break;
}
case NB_INPLACE_MULTIPLY:
{
    if(
        PyFloat_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

                cache_possible = PyFloat_CheckExact(rhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 0);
                op = rhs;

                // try the commutative case
                if (!cache_possible) {

                    cache_possible = PyFloat_CheckExact(lhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 1);
                    op = lhs;
                }

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
        PyFloat_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_LONG
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

                cache_possible = PyFloat_CheckExact(rhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 0);
                op = rhs;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_ALONG_TRUE_DIVIDE_SFLOAT_BROADCAST_CACHE, cmlq_along_true_divide_sfloat_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_ALONG_TRUE_DIVIDE_SFLOAT, cmlq_along_true_divide_sfloat, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }

    if(
        PyFloat_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

                cache_possible = PyFloat_CheckExact(rhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 0);
                op = rhs;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_TRUE_DIVIDE_SFLOAT_BROADCAST_CACHE, cmlq_adouble_true_divide_sfloat_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_TRUE_DIVIDE_SFLOAT, cmlq_adouble_true_divide_sfloat, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }

    if(
        PyLong_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            PyArrayObject *op = NULL;
            PyArray_Descr *descr = NULL;
            int cache_possible = 0;

                descr = PyArray_DescrFromType(NPY_DOUBLE);

                cache_possible = PyLong_CheckExact(rhs) && specializer_info.IsOperandConstant(instr, *stack_pointer, 0);
                op = rhs;

            if (cache_possible) {
                // precompute the broadcast array
                locality_cache[next_result_cache_index].state = BROADCAST;
                locality_cache[next_result_cache_index].result = (PyArrayObject *)PyArray_FromAny(op, descr, 0, 0, 0, NULL);
                specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_TRUE_DIVIDE_SLONG_BROADCAST_CACHE, cmlq_adouble_true_divide_slong_broadcast_cache, &locality_cache[next_result_cache_index]);
            } else {
                specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_TRUE_DIVIDE_SLONG, cmlq_adouble_true_divide_slong, &locality_cache[next_result_cache_index]);
            }

        return 1;

    }



	report_missing_binop_case(instr, lhs, rhs);
	break;
}
case NB_POWER:
{
    if(
        PyFloat_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_POWER_SFLOAT, cmlq_adouble_power_sfloat, &locality_cache[next_result_cache_index]);
        return 1;

    }

    if(
        PyFloat_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_POWER_SFLOAT, cmlq_adouble_square_power_sfloat, &locality_cache[next_result_cache_index]);
        return 1;

    }

    if(
        PyLong_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_POWER_SLONG, cmlq_adouble_square_power_slong, &locality_cache[next_result_cache_index]);
        return 1;

    }

    if(
        PyLong_CheckExact(rhs) &&
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE
    )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_ADOUBLE_POWER_SLONG, cmlq_adouble_power_slong, &locality_cache[next_result_cache_index]);
        return 1;

    }

	report_missing_binop_case(instr, lhs, rhs);
	break;
}
