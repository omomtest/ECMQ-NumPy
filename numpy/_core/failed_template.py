from mako import runtime, filters, cache
UNDEFINED = runtime.UNDEFINED
STOP_RENDERING = runtime.STOP_RENDERING
__M_dict_builtin = dict
__M_locals_builtin = locals
_magic_number = 10
_modified_time = 1748246495.1947908
_enable_loop = True
_template_filename = '/home/dyb/np224cmq/numpy/_core/code_generators/cmlq_templates/function_binop_kw.mako'
_template_uri = 'function_binop_kw.mako'
_source_encoding = 'utf-8'
_exports = []


def _mako_get_namespace(context, name):
    try:
        return context.namespaces[(__name__, name)]
    except KeyError:
        _mako_generate_namespaces(context)
        return context.namespaces[(__name__, name)]
def _mako_generate_namespaces(context):
    ns = runtime.TemplateNamespace('__anon_0x7f434d19fa70', context._clean_inheritance_tokens(), templateuri='cache_stats_macro.mako', callables=None,  calling_uri=_template_uri)
    context.namespaces[(__name__, '__anon_0x7f434d19fa70')] = ns

def render_body(context,**pageargs):
    __M_caller = context.caller_stack._push_frame()
    try:
        __M_locals = __M_dict_builtin(pageargs=pageargs)
        _import_ns = {}
        _mako_get_namespace(context, '__anon_0x7f434d19fa70')._populate(_import_ns, ['*'])
        count_stat = _import_ns.get('count_stat', context.get('count_stat', UNDEFINED))
        signature = _import_ns.get('signature', context.get('signature', UNDEFINED))
        locality_cache = _import_ns.get('locality_cache', context.get('locality_cache', UNDEFINED))
        locality_stats = _import_ns.get('locality_stats', context.get('locality_stats', UNDEFINED))
        __M_writer = context.writer()
        __M_writer('\n\n')
        __M_writer(str(signature))
        __M_writer('\n{\n')
        if locality_cache or locality_stats:
            __M_writer('    ')
            runtime._include_file(context, 'load_cache_elem.mako', _template_uri)
            __M_writer('\n')
        __M_writer('\n    ')
        runtime._include_file(context, 'prepare_binary_args.mako', _template_uri, kw=1)
        __M_writer('\n    //kwnames key1 arg1 arg2 self_or_null callable\n\n    PyObject *callable = (*stack_pointer_ptr)[-6];\n    PyUFuncObject *ufunc = (PyUFuncObject *)callable;\n\n    PyObject *out = (*stack_pointer_ptr)[-2];\n\n    if (NPY_UNLIKELY(!ufunc->specializable)) {\n        ')
        count_stat("ufunc_type_misses")
        
        __M_writer('\n        // ufunc has user loops or is generalized\n        fprintf(stderr,"ufunc_type_misses\\n");\n        goto deopt;\n    }\n\n')
        if locality_cache:
            __M_writer('    ')
            runtime._include_file(context, 'locality_cache_kw.mako', _template_uri)
            __M_writer('\n')
        __M_writer('\n    ')
        runtime._include_file(context, 'array_op_kw.mako', _template_uri, try_elide_temp=False)
        __M_writer('\n\ndeopt:\n')
        __M_writer('    return 2;\n\nsuccess:\n')
        __M_writer('    Py_DECREF(lhs);\n    Py_DECREF(rhs);\n\n    Py_DECREF(callable);\n\n    // skip the arguments and the NULL on the stack\n    *stack_pointer_ptr -= 5;\n    assert(PyArray_CheckExact(out));\n    (*stack_pointer_ptr)[-1] = (PyObject *)out;\n    return 0;\n\nfail:\n')
        __M_writer('    return -1;\n}\n')
        return ''
    finally:
        context.caller_stack._pop_frame()


"""
__M_BEGIN_METADATA
{"filename": "/home/dyb/np224cmq/numpy/_core/code_generators/cmlq_templates/function_binop_kw.mako", "uri": "function_binop_kw.mako", "source_encoding": "utf-8", "line_map": {"22": 1, "25": 0, "36": 1, "37": 3, "38": 3, "39": 6, "40": 7, "41": 7, "42": 7, "43": 9, "44": 10, "45": 10, "46": 19, "47": 20, "48": 19, "49": 25, "50": 26, "51": 26, "52": 26, "53": 28, "54": 29, "55": 29, "56": 33, "57": 37, "58": 50, "64": 58}}
__M_END_METADATA
"""
