<%namespace file="cache_stats_macro.mako" import="*"/>

${signature}
{
    ## fprintf(stderr,"%s\n","${signature}");
    %if locality_cache or locality_stats:
    <%include file="load_cache_elem.mako"/>
    %endif

    <%include file="prepare_binary_args.mako" args="kw=1"/>
    //kwnames key1 arg1 arg2 self_or_null callable

    PyObject *callable = (*stack_pointer_ptr)[-6];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    PyObject *out = (*stack_pointer_ptr)[-2];

    if (NPY_UNLIKELY(!ufunc->specializable)) {
        <%count_stat("ufunc_type_misses")%>
        // ufunc has user loops or is generalized
        fprintf(stderr,"ufunc_type_misses\n");
        goto deopt;
    }

    %if locality_cache:
    <%include file="locality_cache_kw.mako"/>
    %endif

    <%include file="array_op.mako" args="try_elide_temp=False"/>

deopt:
## fprintf(stderr, "deopt\n");
    return 2;

success:
## fprintf(stderr, "success\n");
    Py_DECREF(lhs);
    Py_DECREF(rhs);

    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 5;
    assert(PyArray_CheckExact(out));
    (*stack_pointer_ptr)[-1] = (PyObject *)out;
    return 0;

fail:
## fprintf(stderr, "fail\n");
    return -1;
}
