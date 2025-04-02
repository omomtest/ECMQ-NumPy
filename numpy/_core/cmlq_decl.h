int cmlq_afloat_subtract_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_AFLOAT_SUBTRACT_AFLOAT 1

int cmlq_afloat_inplace_subtract_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_AFLOAT_INPLACE_SUBTRACT_AFLOAT 2

int cmlq_afloat_add_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_AFLOAT_ADD_AFLOAT 3

int cmlq_afloat_inplace_add_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_AFLOAT_INPLACE_ADD_AFLOAT 4

int cmlq_afloat_multiply_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_AFLOAT_MULTIPLY_AFLOAT 5

int cmlq_adouble_subtract_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_SUBTRACT_ADOUBLE 6

int cmlq_adouble_inplace_subtract_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_INPLACE_SUBTRACT_ADOUBLE 7

int cmlq_adouble_subtract_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_SUBTRACT_SFLOAT 8

int cmlq_adouble_subtract_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_SUBTRACT_SFLOAT_BROADCAST_CACHE 9

int cmlq_adouble_add_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_ADD_ADOUBLE 10

int cmlq_adouble_inplace_add_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_INPLACE_ADD_ADOUBLE 11

int cmlq_adouble_add_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_ADD_SFLOAT 12

int cmlq_adouble_add_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_ADD_SFLOAT_BROADCAST_CACHE 13

int cmlq_adouble_inplace_add_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_INPLACE_ADD_SFLOAT 14

int cmlq_adouble_inplace_add_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_INPLACE_ADD_SFLOAT_BROADCAST_CACHE 15

int cmlq_adouble_multiply_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_MULTIPLY_ADOUBLE 16

int cmlq_adouble_multiply_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_MULTIPLY_SLONG 17

int cmlq_adouble_multiply_slong_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_MULTIPLY_SLONG_BROADCAST_CACHE 18

int cmlq_adouble_multiply_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_MULTIPLY_SFLOAT 19

int cmlq_adouble_multiply_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_MULTIPLY_SFLOAT_BROADCAST_CACHE 20

int cmlq_adouble_inplace_multiply_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_INPLACE_MULTIPLY_SFLOAT 21

int cmlq_adouble_inplace_multiply_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_INPLACE_MULTIPLY_SFLOAT_BROADCAST_CACHE 22

int cmlq_along_multiply_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ALONG_MULTIPLY_SLONG 23

int cmlq_along_multiply_slong_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ALONG_MULTIPLY_SLONG_BROADCAST_CACHE 24

int cmlq_along_true_divide_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ALONG_TRUE_DIVIDE_SFLOAT 25

int cmlq_along_true_divide_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ALONG_TRUE_DIVIDE_SFLOAT_BROADCAST_CACHE 26

int cmlq_adouble_true_divide_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_TRUE_DIVIDE_SFLOAT 27

int cmlq_adouble_true_divide_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_TRUE_DIVIDE_SFLOAT_BROADCAST_CACHE 28

int cmlq_adouble_true_divide_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_TRUE_DIVIDE_SLONG 29

int cmlq_adouble_true_divide_slong_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_TRUE_DIVIDE_SLONG_BROADCAST_CACHE 30

int cmlq_sfloat_true_divide_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SFLOAT_TRUE_DIVIDE_ADOUBLE 31

int cmlq_sfloat_true_divide_adouble_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SFLOAT_TRUE_DIVIDE_ADOUBLE_BROADCAST_CACHE 32

int cmlq_adouble_true_divide_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_TRUE_DIVIDE_ADOUBLE 33

int cmlq_adouble_power_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_POWER_SFLOAT 34

int cmlq_adouble_square_power_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_POWER_SFLOAT 35

int cmlq_adouble_square_power_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_POWER_SLONG 36

int cmlq_adouble_power_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_POWER_SLONG 37

int cmlq_minimum_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MINIMUM_AINT_AINT 38

int cmlq_minimum_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MINIMUM_AFLOAT_AFLOAT 39

int cmlq_minimum_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MINIMUM_ADOUBLE_ADOUBLE 40

int cmlq_maximum_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MAXIMUM_AINT_AINT 41

int cmlq_maximum_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MAXIMUM_AFLOAT_AFLOAT 42

int cmlq_maximum_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MAXIMUM_ADOUBLE_ADOUBLE 43

int cmlq_add_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADD_AINT_AINT 44

int cmlq_add_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADD_AFLOAT_AFLOAT 45

int cmlq_add_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADD_ADOUBLE_ADOUBLE 46

int cmlq_subtract_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SUBTRACT_AINT_AINT 47

int cmlq_subtract_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SUBTRACT_AFLOAT_AFLOAT 48

int cmlq_subtract_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SUBTRACT_ADOUBLE_ADOUBLE 49

int cmlq_multiply_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MULTIPLY_AINT_AINT 50

int cmlq_multiply_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MULTIPLY_AFLOAT_AFLOAT 51

int cmlq_multiply_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MULTIPLY_ADOUBLE_ADOUBLE 52

int cmlq_square_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SQUARE_AINT 53

int cmlq_square_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SQUARE_ADOUBLE 54

int cmlq_square_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SQUARE_AFLOAT 55

int cmlq_sqrt_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SQRT_ADOUBLE 56

int cmlq_sqrt_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SQRT_AFLOAT 57

int cmlq_absolute_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ABSOLUTE_AINT 58

int cmlq_absolute_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ABSOLUTE_ADOUBLE 59

int cmlq_absolute_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ABSOLUTE_AFLOAT 60

int cmlq_reciprocal_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_RECIPROCAL_AFLOAT 61

int cmlq_reciprocal_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_RECIPROCAL_ADOUBLE 62

int cmlq_reciprocal_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_RECIPROCAL_AINT 63

