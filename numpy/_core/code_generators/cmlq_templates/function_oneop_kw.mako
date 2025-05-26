<%namespace file="cache_stats_macro.mako" import="*"/>

${signature}
{
    %if locality_cache or locality_stats:
    <%include file="load_cache_elem.mako"/>
    %endif

    <%include file="prepare_oneop_args.mako" args="kw=1"/>

    PyObject *callable = (*stack_pointer_ptr)[-5];
    PyUFuncObject *ufunc = (PyUFuncObject *)callable;

    PyObject *out = (*stack_pointer_ptr)[-2];

    if (NPY_UNLIKELY(!ufunc->specializable)) {
        <%count_stat("ufunc_type_misses")%>
        // ufunc has user loops or is generalized
        goto deopt;
    }

    %if locality_cache:
    <%include file="locality_cache_kw.mako" args="arity = 1"/>
    %endif

    <%include file="array_op_kw.mako" args="try_elide_temp=False ,arity = 1"/>

deopt:
    return 2;

success:
    Py_DECREF(lhs);

    Py_DECREF(callable);

    // skip the arguments and the NULL on the stack
    *stack_pointer_ptr -= 4;
    assert(PyArray_CheckExact(result));
    (*stack_pointer_ptr)[-1] = (PyObject *)result;
    return 0;

fail:
    return -1;
}