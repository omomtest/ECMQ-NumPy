<%namespace file="cache_stats_macro.mako" import="*"/>

${signature}
{
    %if locality_cache or locality_stats:
    <%include file="load_cache_elem.mako"/>
    %endif

    <%include file="prepare_oneop_args.mako"/>

    PyObject *callable = (*stack_pointer_ptr)[-3];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    if (NPY_UNLIKELY(!ufunc->specializable)) {
        <%count_stat("ufunc_type_misses")%>
        // ufunc has user loops or is generalized
        goto deopt;
    }

    %if locality_cache:
    <%include file="locality_cache_one.mako"/>
    %endif

    <%include file="array_one_op.mako" args="try_elide_temp=False"/>

deopt:
    return 2;

success:
    Py_DECREF(lhs);
    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 2;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}