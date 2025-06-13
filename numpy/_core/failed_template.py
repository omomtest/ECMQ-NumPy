from mako import runtime, filters, cache
UNDEFINED = runtime.UNDEFINED
STOP_RENDERING = runtime.STOP_RENDERING
__M_dict_builtin = dict
__M_locals_builtin = locals
_magic_number = 10
_modified_time = 1749784001.734094
_enable_loop = True
_template_filename = '/home/dyb/np224cmq/numpy/_core/code_generators/cmlq_templates/arith_binop.mako'
_template_uri = 'arith_binop.mako'
_source_encoding = 'utf-8'
_exports = []


def render_body(context,**pageargs):
    __M_caller = context.caller_stack._push_frame()
    try:
        __M_locals = __M_dict_builtin(pageargs=pageargs)
        right_scalar_name = context.get('right_scalar_name', UNDEFINED)
        opname = context.get('opname', UNDEFINED)
        signature = context.get('signature', UNDEFINED)
        left_scalar_name = context.get('left_scalar_name', UNDEFINED)
        locality_stats = context.get('locality_stats', UNDEFINED)
        cache_broadcast_array = context.get('cache_broadcast_array', UNDEFINED)
        locality_cache = context.get('locality_cache', UNDEFINED)
        __M_writer = context.writer()
        __M_writer(str(signature))
        __M_writer('\n{\n    //CMLQ_PAPI_BEGIN("')
        __M_writer(str(opname))
        __M_writer('")\n')
        if locality_cache or locality_stats or cache_broadcast_array:
            __M_writer('    ')
            runtime._include_file(context, 'load_cache_elem.mako', _template_uri)
            __M_writer('\n')
        __M_writer('\n    ')
        runtime._include_file(context, 'prepare_binary_args.mako', _template_uri)
        __M_writer('\n\n')
        if locality_cache:
            __M_writer('    ')
            runtime._include_file(context, 'locality_cache.mako', _template_uri)
            __M_writer('\n')
        __M_writer('\n    ')
        runtime._include_file(context, 'array_op.mako', _template_uri, try_elide_temp=True)
        __M_writer('\n\ndeopt:\n')
        if locality_cache:
            __M_writer('    elem = (CMLQLocalityCacheElem *)external_cache_pointer;\n    if (elem->state != UNUSED) {\n\n        if (elem->state == TRIVIAL) {\n            Py_XDECREF(elem->result);\n        } else if (elem->state == ITERATOR) {\n            NpyIter_Deallocate(elem->iterator.cached_iter);\n            elem->iterator.cached_iter = NULL;\n        }\n\n        elem->state = UNUSED;\n        elem->result = NULL;\n        backoff_CMLQCounter(&(elem->counter));\n\n    }\n')
        __M_writer('    return 2;\n\nsuccess:\n')
        if cache_broadcast_array and left_scalar_name is not UNDEFINED:
            __M_writer('    // the lhs is a cached broadcast array, no decref\n')
        else:
            __M_writer('    Py_DECREF(lhs);\n')
        __M_writer('\n')
        if cache_broadcast_array and left_scalar_name is UNDEFINED and right_scalar_name is not UNDEFINED:
            __M_writer('    // the rhs is a cached broadcast array, no decref\n')
        else:
            __M_writer('    Py_DECREF(rhs);\n')
        __M_writer('\n')
        if right_scalar_name:
            __M_writer('    Py_DECREF(m2);\n')
        __M_writer('\n    (*stack_pointer_ptr)--;\n    assert(PyArray_CheckExact(result));\n    (*stack_pointer_ptr)[-1] = (PyObject *)result;\n    //CMLQ_PAPI_END("')
        __M_writer(str(opname))
        __M_writer('")\n    return 0;\nfail:\n    return -1;\n}\n')
        return ''
    finally:
        context.caller_stack._pop_frame()


"""
__M_BEGIN_METADATA
{"filename": "/home/dyb/np224cmq/numpy/_core/code_generators/cmlq_templates/arith_binop.mako", "uri": "arith_binop.mako", "source_encoding": "utf-8", "line_map": {"15": 0, "27": 1, "28": 1, "29": 3, "30": 3, "31": 4, "32": 5, "33": 5, "34": 5, "35": 7, "36": 8, "37": 8, "38": 10, "39": 11, "40": 11, "41": 11, "42": 13, "43": 14, "44": 14, "45": 17, "46": 18, "47": 34, "48": 37, "49": 38, "50": 39, "51": 40, "52": 42, "53": 43, "54": 44, "55": 45, "56": 46, "57": 48, "58": 49, "59": 50, "60": 52, "61": 56, "62": 56, "68": 62}}
__M_END_METADATA
"""
