typedef struct {
            const char *key;
            int value;
        } LookupEntry;

#define UFUNC_ABSOLUTE 0
#define UFUNC_ADD 1
#define UFUNC_ARCCOS 2
#define UFUNC_ARCCOSH 3
#define UFUNC_ARCSIN 4
#define UFUNC_ARCSINH 5
#define UFUNC_ARCTAN 6
#define UFUNC_ARCTAN2 7
#define UFUNC_ARCTANH 8
#define UFUNC_BITWISE_AND 9
#define UFUNC_BITWISE_COUNT 10
#define UFUNC_BITWISE_OR 11
#define UFUNC_BITWISE_XOR 12
#define UFUNC_CBRT 13
#define UFUNC_CEIL 14
#define UFUNC_CONJUGATE 15
#define UFUNC_COPYSIGN 16
#define UFUNC_COS 17
#define UFUNC_COSH 18
#define UFUNC_DIVMOD 19
#define UFUNC_EXP 20
#define UFUNC_EXP2 21
#define UFUNC_EXPM1 22
#define UFUNC_FLOOR 23
#define UFUNC_FLOOR_DIVIDE 24
#define UFUNC_FMAX 25
#define UFUNC_FMIN 26
#define UFUNC_FREXP 27
#define UFUNC_GCD 28
#define UFUNC_INVERT 29
#define UFUNC_ISFINITE 30
#define UFUNC_ISINF 31
#define UFUNC_LESS_EQUAL 32
#define UFUNC_LOGICAL_AND 33
#define UFUNC_LOGICAL_NOT 34
#define UFUNC_MAXIMUM 35
#define UFUNC_MINIMUM 36
#define UFUNC_MULTIPLY 37
#define UFUNC_RECIPROCAL 38
#define UFUNC_SQRT 39
#define UFUNC_SQUARE 40
#define UFUNC_SUBTRACT 41
#define UFUNC_TANH 42
#define UFUNC_TRUE_DIVIDE 43

static const LookupEntry STATIC_TABLE[] = {
    {"absolute", UFUNC_ABSOLUTE},
    {"add", UFUNC_ADD},
    {"arccos", UFUNC_ARCCOS},
    {"arccosh", UFUNC_ARCCOSH},
    {"arcsin", UFUNC_ARCSIN},
    {"arcsinh", UFUNC_ARCSINH},
    {"arctan", UFUNC_ARCTAN},
    {"arctan2", UFUNC_ARCTAN2},
    {"arctanh", UFUNC_ARCTANH},
    {"bitwise_and", UFUNC_BITWISE_AND},
    {"bitwise_count", UFUNC_BITWISE_COUNT},
    {"bitwise_or", UFUNC_BITWISE_OR},
    {"bitwise_xor", UFUNC_BITWISE_XOR},
    {"cbrt", UFUNC_CBRT},
    {"ceil", UFUNC_CEIL},
    {"conjugate", UFUNC_CONJUGATE},
    {"copysign", UFUNC_COPYSIGN},
    {"cos", UFUNC_COS},
    {"cosh", UFUNC_COSH},
    {"divmod", UFUNC_DIVMOD},
    {"exp", UFUNC_EXP},
    {"exp2", UFUNC_EXP2},
    {"expm1", UFUNC_EXPM1},
    {"floor", UFUNC_FLOOR},
    {"floor_divide", UFUNC_FLOOR_DIVIDE},
    {"fmax", UFUNC_FMAX},
    {"fmin", UFUNC_FMIN},
    {"frexp", UFUNC_FREXP},
    {"gcd", UFUNC_GCD},
    {"invert", UFUNC_INVERT},
    {"isfinite", UFUNC_ISFINITE},
    {"isinf", UFUNC_ISINF},
    {"less_equal", UFUNC_LESS_EQUAL},
    {"logical_and", UFUNC_LOGICAL_AND},
    {"logical_not", UFUNC_LOGICAL_NOT},
    {"maximum", UFUNC_MAXIMUM},
    {"minimum", UFUNC_MINIMUM},
    {"multiply", UFUNC_MULTIPLY},
    {"reciprocal", UFUNC_RECIPROCAL},
    {"sqrt", UFUNC_SQRT},
    {"square", UFUNC_SQUARE},
    {"subtract", UFUNC_SUBTRACT},
    {"tanh", UFUNC_TANH},
    {"true_divide", UFUNC_TRUE_DIVIDE},
};


static const size_t TABLE_SIZE =sizeof(STATIC_TABLE) / sizeof(STATIC_TABLE[0]);

int get_value(const char *key)
    {
        int low = 0;
        int high = TABLE_SIZE - 1;

        while (low <= high) {
            int mid = low + (high - low) / 2;
            int cmp = strcmp(key, STATIC_TABLE[mid].key);

            if (cmp == 0) {
                return STATIC_TABLE[mid].value;  // 命中返回对应值
            }
            else if (cmp < 0) {
                high = mid - 1;
            }
            else {
                low = mid + 1;
            }
        }

        return -1;  // 未找到
        }
