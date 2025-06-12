typedef struct {
            const char *key;
            int value;
        } LookupEntry;

#define UFUNC_ABSOLUTE 0
#define UFUNC_ADD 1
#define UFUNC_ARCTAN2 2
#define UFUNC_EXP 3
#define UFUNC_LESS_EQUAL 4
#define UFUNC_LOGICAL_AND 5
#define UFUNC_LOGICAL_NOT 6
#define UFUNC_MAXIMUM 7
#define UFUNC_MINIMUM 8
#define UFUNC_MULTIPLY 9
#define UFUNC_RECIPROCAL 10
#define UFUNC_SQRT 11
#define UFUNC_SQUARE 12
#define UFUNC_SUBTRACT 13
#define UFUNC_TANH 14
#define UFUNC_TRUE_DIVIDE 15

static const LookupEntry STATIC_TABLE[] = {
    {"absolute", UFUNC_ABSOLUTE},
    {"add", UFUNC_ADD},
    {"arctan2", UFUNC_ARCTAN2},
    {"exp", UFUNC_EXP},
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
