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
        PyFloat_CheckExact(lhs) 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_TANH_SFLOAT, cmlq_tanh_sfloat, &locality_cache[next_result_cache_index]);
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
    if(
        PyFloat_CheckExact(lhs) 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_TANH_SDOUBLE, cmlq_tanh_sdouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
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
}
case UFUNC_EXP2:
{
    if(
        PyFloat_CheckExact(lhs) 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_EXP2_SDOUBLE, cmlq_exp2_sdouble, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_EXP2_ADOUBLE, cmlq_exp2_adouble, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_EXP2_AFLOAT, cmlq_exp2_afloat, &locality_cache[next_result_cache_index]);
        return 1;

    }
}
case UFUNC_ARCCOS:
{
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_FLOAT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_ARCCOS_AFLOAT, cmlq_arccos_afloat, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_ARCCOS_ADOUBLE, cmlq_arccos_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
}
case UFUNC_ARCCOSH:
{
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_ARCCOSH_ADOUBLE, cmlq_arccosh_adouble, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_ARCCOSH_AFLOAT, cmlq_arccosh_afloat, &locality_cache[next_result_cache_index]);
        return 1;

    }
}
case UFUNC_ARCSIN:
{
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_FLOAT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_ARCSIN_AFLOAT, cmlq_arcsin_afloat, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_ARCSIN_ADOUBLE, cmlq_arcsin_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
}
case UFUNC_ARCSINH:
{
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_DOUBLE 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_ARCSINH_ADOUBLE, cmlq_arcsinh_adouble, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_ARCSINH_AFLOAT, cmlq_arcsinh_afloat, &locality_cache[next_result_cache_index]);
        return 1;

    }
}
case UFUNC_ARCTANH:
{
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_FLOAT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_ARCTANH_AFLOAT, cmlq_arctanh_afloat, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_ARCTANH_ADOUBLE, cmlq_arctanh_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
}
case UFUNC_ARCTAN:
{
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_FLOAT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_ARCTAN_AFLOAT, cmlq_arctan_afloat, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_ARCTAN_ADOUBLE, cmlq_arctan_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
}
case UFUNC_CBRT:
{
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_FLOAT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_CBRT_AFLOAT, cmlq_cbrt_afloat, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_CBRT_ADOUBLE, cmlq_cbrt_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
}
case UFUNC_CEIL:
{
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_FLOAT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_CEIL_AFLOAT, cmlq_ceil_afloat, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_CEIL_ADOUBLE, cmlq_ceil_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_BOOL 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_CEIL_ABOOL, cmlq_ceil_abool, &locality_cache[next_result_cache_index]);
        return 1;

    }
}
case UFUNC_CONJUGATE:
{
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_FLOAT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_CONJUGATE_AFLOAT, cmlq_conjugate_afloat, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_CONJUGATE_ADOUBLE, cmlq_conjugate_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_LONG 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_CONJUGATE_ALONG, cmlq_conjugate_along, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_CONJUGATE_AINT, cmlq_conjugate_aint, &locality_cache[next_result_cache_index]);
        return 1;

    }
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_CDOUBLE 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_CONJUGATE_ACOMPLEX, cmlq_conjugate_acomplex, &locality_cache[next_result_cache_index]);
        return 1;

    }
}
case UFUNC_EXPM1:
{
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_FLOAT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_EXPM1_AFLOAT, cmlq_expm1_afloat, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_EXPM1_ADOUBLE, cmlq_expm1_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
}
case UFUNC_FLOOR:
{
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_FLOAT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_FLOOR_AFLOAT, cmlq_floor_afloat, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_FLOOR_ADOUBLE, cmlq_floor_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
}
case UFUNC_FREXP:
{
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_FLOAT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_FREXP_AFLOAT, cmlq_frexp_afloat, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_FREXP_ADOUBLE, cmlq_frexp_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
}
case UFUNC_INVERT:
{
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_LONG 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_INVERT_ALONG, cmlq_invert_along, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_INVERT_AINT, cmlq_invert_aint, &locality_cache[next_result_cache_index]);
        return 1;

    }
}
case UFUNC_ISFINITE:
{
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_FLOAT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_ISFINITE_AFLOAT, cmlq_isfinite_afloat, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_ISFINITE_ADOUBLE, cmlq_isfinite_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_LONG 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_ISFINITE_ALONG, cmlq_isfinite_along, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_ISFINITE_AINT, cmlq_isfinite_aint, &locality_cache[next_result_cache_index]);
        return 1;

    }
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_BOOL 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_ISFINITE_ABOOL, cmlq_isfinite_abool, &locality_cache[next_result_cache_index]);
        return 1;

    }
}
case UFUNC_ISINF:
{
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_FLOAT 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_ISINF_AFLOAT, cmlq_isinf_afloat, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_ISINF_ADOUBLE, cmlq_isinf_adouble, &locality_cache[next_result_cache_index]);
        return 1;

    }
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_LONG 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_ISINF_ALONG, cmlq_isinf_along, &locality_cache[next_result_cache_index]);
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

            specializer_info.SpecializeInstruction(instr, SLOT_ISINF_AINT, cmlq_isinf_aint, &locality_cache[next_result_cache_index]);
        return 1;

    }
    if(
        PyArray_DESCR((PyArrayObject *)lhs)->type_num == NPY_BOOL 
       )
    {

        next_result_cache_index++;
        #ifdef CMLQ_STATS
        locality_cache[next_result_cache_index].stats.instr_ptr = instr;
        #endif

            specializer_info.SpecializeInstruction(instr, SLOT_ISINF_ABOOL, cmlq_isinf_abool, &locality_cache[next_result_cache_index]);
        return 1;

    }
}
