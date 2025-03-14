<%namespace file="cache_stats_macro.mako" import="*"/>

PyObject *m1 = (*stack_pointer_ptr)[-1];

%if left_scalar_name is not UNDEFINED:
    if (NPY_UNLIKELY(!Py${left_scalar_name}_CheckExact(m1))) {
        <%count_stat("type_misses")%>
        goto deopt;
    }
    PyArray_Descr *descr = NULL;
    %if left_promotion is not None:
        descr = PyArray_DescrFromType(${left_promotion});
    %endif

    %if not cache_broadcast_array:
        PyArrayObject *lhs = (PyArrayObject *)PyArray_FromAny(m1, descr, 0, 0, 0, NULL);
    %else:
        assert(elem->state == BROADCAST && elem->result);
        PyArrayObject *lhs = elem->result;
    %endif
%else:
    if (NPY_UNLIKELY(!PyArray_CheckExact(m1))) {
        <%count_stat("type_misses")%>
        goto deopt;
    }
    PyArrayObject *lhs = (PyArrayObject *)m1;
    if (NPY_UNLIKELY(PyArray_DESCR(lhs)->type_num != ${left_numpy_name})) {
        <%count_stat("type_misses")%>
        goto deopt;
    }
%endif



PyArrayObject *ops[] = {lhs, NULL};
PyArrayObject *result = NULL;


npy_intp result_ndims = PyArray_NDIM(lhs);
npy_intp *result_shape = PyArray_SHAPE(lhs);
