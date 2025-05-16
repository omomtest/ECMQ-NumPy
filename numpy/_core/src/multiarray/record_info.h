#include <Python.h>  // for _Py_CODEUNIT, PyObject, etc.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef NPY_COLLECT_BYTECODE

static FILE *bc_log_file = NULL;
static int bc_log_initialized = 0;

static void
init_bytecode_log(void)
{
    if (bc_log_initialized) {
        return;
    }
    const char *path = "bytecode_log.txt";
    bc_log_file = fopen(path, "a");
    if (bc_log_file == NULL) {
        fprintf(stderr, "Error: cannot open %s for writing bytecode info\n",
                path);
        return;
    }

    fprintf(bc_log_file, "\n==== Bytecode collection started====\n");
    fflush(bc_log_file);
    bc_log_initialized = 1;
}

static void
collect_base_info(_Py_CODEUNIT *instr)
{
    if (!bc_log_initialized) {
        init_bytecode_log();
        if (!bc_log_file) {
            return;
        }
    }

    time_t t = time(NULL);
    struct tm tm;
    localtime_r(&t, &tm);
    char timestr[32];
    strftime(timestr, sizeof(timestr), "%H:%M:%S", &tm);
    fprintf(bc_log_file, "[%s] opcode=%d\t",
            timestr, instr->op.code);
}
static void
collect_binop_subscr_info(_Py_CODEUNIT *instr, PyObject **stack)
{

    fprintf(bc_log_file, "arg=%d\t",instr->op.arg);

    PyObject *top1 = stack[-1];
    PyObject *top2 = stack[-2];
        fprintf(bc_log_file, "types: %s, %s\n",
                Py_TYPE(top1)->tp_name,
                Py_TYPE(top2)->tp_name);



    fflush(bc_log_file);
}
static void
collect_call_info(_Py_CODEUNIT *instr, PyObject **stack,PyObject *callable)
{
    fprintf(bc_log_file, "arg=%d\t", instr->op.arg);
    if (callable != NULL && PyCallable_Check(callable)) {
        PyObject *name_attr = PyObject_GetAttrString(callable, "__name__");
        const char *func_name = NULL;
        if (name_attr!=NULL) {
            func_name = PyUnicode_AsUTF8(name_attr);
        }else {
            PyErr_Clear();
        }
            if (func_name != NULL) {
                fprintf(bc_log_file, "func_name:%s\t", func_name);
            }
            Py_XDECREF(name_attr);
    }
    else {
        PyErr_Clear();
    }
        PyObject *top = NULL;

        for (int i = 0; i < instr->op.arg; i++) {
            top = stack[-i - 1];
            fprintf(bc_log_file, "types: %s \t",
                    Py_TYPE(top)->tp_name);
    }
  
        fprintf(bc_log_file, "\n");



    fflush(bc_log_file);
    }
#endif /* NPY_COLLECT_BYTECODE */
