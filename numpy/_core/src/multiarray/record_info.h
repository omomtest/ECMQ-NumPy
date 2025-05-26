#include <Python.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef NPY_COLLECT_BYTECODE

#define BUFFER_SIZE 8192  // 缓冲区大小

static FILE *bc_log_file = NULL;
static int bc_log_initialized = 0;
static char log_buffer[BUFFER_SIZE];
static size_t buffer_offset = 0;

// 写缓冲区到文件
static void
flush_bytecode_log(void)
{
    if (bc_log_file && buffer_offset > 0) {
        fwrite(log_buffer, 1, buffer_offset, bc_log_file);
        fflush(bc_log_file);
        buffer_offset = 0;
    }
}

static void
append_to_buffer(const char *format, ...)
{
    if (!bc_log_initialized)
        return;

    va_list args;
    va_start(args, format);
    int remaining = BUFFER_SIZE - buffer_offset;

    // 第一次尝试写入
    int written =
            vsnprintf(log_buffer + buffer_offset, remaining, format, args);
    va_end(args);

    if (written < 0) {
        return;  // 写入失败，退出
    }

    if (written >= remaining) {
        // 缓冲区不足，先刷新
        flush_bytecode_log();

        // 重新初始化 va_list 再次写入
        va_start(args, format);
        written = vsnprintf(log_buffer, BUFFER_SIZE, format, args);
        va_end(args);

        if (written < 0 || written >= BUFFER_SIZE) {
            return;  // 无法写入，退出
        }
        buffer_offset = written;
    }
    else {
        buffer_offset += written;
    }
}

static void
init_bytecode_log(void)
{
    if (bc_log_initialized)
        return;

    const char *path = "bytecode_log.txt";
    bc_log_file = fopen(path, "a");
    if (!bc_log_file) {
        fprintf(stderr, "Error: cannot open %s\n", path);
        return;
    }

    bc_log_initialized = 1;
    append_to_buffer("\n==== Bytecode collection started ====\n");
}

static void
collect_base_info(_Py_CODEUNIT *instr)
{
    if (!bc_log_initialized) {
        init_bytecode_log();
        if (!bc_log_initialized)
            return;
    }
    time_t t = time(NULL);
    struct tm tm;
    localtime_r(&t, &tm);
    char timestr[32];
    strftime(timestr, sizeof(timestr), "%H:%M:%S", &tm);
    append_to_buffer("[%s] opcode=%d\t", timestr, instr->op.code);
}

static void
collect_binop_subscr_info(_Py_CODEUNIT *instr, PyObject **stack)
{
    append_to_buffer("arg=%d\t", instr->op.arg);

    PyObject *top1 = stack[-1];
    PyObject *top2 = stack[-2];
    append_to_buffer("types: %s, %s\n", Py_TYPE(top1)->tp_name,
                     Py_TYPE(top2)->tp_name);
}

static void
collect_call_info(_Py_CODEUNIT *instr, PyObject **stack, PyObject *callable)
{
    append_to_buffer("arg=%d\t", instr->op.arg);
    if (callable && PyCallable_Check(callable)) {
        PyObject *name_attr = PyObject_GetAttrString(callable, "__name__");
        const char *func_name = NULL;
        if (name_attr) {
            func_name = PyUnicode_AsUTF8(name_attr);
        }
        else {
            PyErr_Clear();
        }
        if (func_name) {
            append_to_buffer("func_name:%s\t", func_name);
        }
        Py_XDECREF(name_attr);
    }
    else {
        PyErr_Clear();
    }
    for (int i = 0; i < instr->op.arg; i++) {
        PyObject *top = stack[-i - 1];
        append_to_buffer("types: %s \t", Py_TYPE(top)->tp_name);
    }
    append_to_buffer("\n");
}
static void
collect_callkw_info(_Py_CODEUNIT *instr, PyObject **stack, PyObject *callable,
                    PyObject *kwnames)
{
    append_to_buffer("arg=%d\t", instr->op.arg);
    if (callable && PyCallable_Check(callable)) {
        PyObject *name_attr = PyObject_GetAttrString(callable, "__name__");
        const char *func_name = NULL;
        if (name_attr) {
            func_name = PyUnicode_AsUTF8(name_attr);
        }
        else {
            PyErr_Clear();
        }
        if (func_name) {
            append_to_buffer("func_name:%s\t", func_name);
        }
        Py_XDECREF(name_attr);
    }
    else {
        PyErr_Clear();
    }
    Py_ssize_t nkwargs = PyTuple_GET_SIZE(kwnames);
    append_to_buffer("keyword:");
    for(int i=0;i<nkwargs;i++) {
        PyObject *key = PyTuple_GET_ITEM(kwnames, i);
        if (PyUnicode_Check(key)) {
            append_to_buffer("%s\t", PyUnicode_AsUTF8(key));
        }
        else {
            append_to_buffer("<non-string>\t");
        }
    }
    append_to_buffer("types:");
    for (int i = 0; i < instr->op.arg; i++) {
        PyObject *top = stack[-i - 2];
        append_to_buffer("%s\t", Py_TYPE(top)->tp_name);
    }
    append_to_buffer("\n");
}

// 在程序退出时刷新缓冲区
__attribute__((destructor)) static void
cleanup_bytecode_log(void)
{
    flush_bytecode_log();
    if (bc_log_file) {
        fclose(bc_log_file);
        bc_log_file = NULL;
    }
}

#endif /* NPY_COLLECT_BYTECODE */
