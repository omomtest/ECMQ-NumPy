int cmlq_afloat_subtract_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_AFLOAT_SUBTRACT_AFLOAT 1

int cmlq_afloat_inplace_subtract_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_AFLOAT_INPLACE_SUBTRACT_AFLOAT 2

int cmlq_afloat_add_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_AFLOAT_ADD_AFLOAT 3

int cmlq_afloat_inplace_add_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_AFLOAT_INPLACE_ADD_AFLOAT 4

int cmlq_afloat_inplace_add_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_AFLOAT_INPLACE_ADD_SLONG 5

int cmlq_afloat_inplace_add_slong_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_AFLOAT_INPLACE_ADD_SLONG_BROADCAST_CACHE 6

int cmlq_adouble_inplace_add_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_INPLACE_ADD_SLONG 7

int cmlq_adouble_inplace_add_slong_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_INPLACE_ADD_SLONG_BROADCAST_CACHE 8

int cmlq_afloat_multiply_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_AFLOAT_MULTIPLY_AFLOAT 9

int cmlq_adouble_subtract_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_SUBTRACT_ADOUBLE 10

int cmlq_adouble_inplace_subtract_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_INPLACE_SUBTRACT_ADOUBLE 11

int cmlq_adouble_subtract_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_SUBTRACT_SFLOAT 12

int cmlq_adouble_subtract_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_SUBTRACT_SFLOAT_BROADCAST_CACHE 13

int cmlq_adouble_add_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_ADD_ADOUBLE 14

int cmlq_adouble_inplace_add_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_INPLACE_ADD_ADOUBLE 15

int cmlq_adouble_add_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_ADD_SFLOAT 16

int cmlq_adouble_add_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_ADD_SFLOAT_BROADCAST_CACHE 17

int cmlq_adouble_inplace_add_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_INPLACE_ADD_SFLOAT 18

int cmlq_adouble_inplace_add_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_INPLACE_ADD_SFLOAT_BROADCAST_CACHE 19

int cmlq_adouble_multiply_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_MULTIPLY_ADOUBLE 20

int cmlq_adouble_multiply_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_MULTIPLY_SLONG 21

int cmlq_adouble_multiply_slong_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_MULTIPLY_SLONG_BROADCAST_CACHE 22

int cmlq_adouble_multiply_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_MULTIPLY_SFLOAT 23

int cmlq_adouble_multiply_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_MULTIPLY_SFLOAT_BROADCAST_CACHE 24

int cmlq_adouble_inplace_multiply_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_INPLACE_MULTIPLY_SFLOAT 25

int cmlq_adouble_inplace_multiply_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_INPLACE_MULTIPLY_SFLOAT_BROADCAST_CACHE 26

int cmlq_along_multiply_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ALONG_MULTIPLY_SLONG 27

int cmlq_along_multiply_slong_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ALONG_MULTIPLY_SLONG_BROADCAST_CACHE 28

int cmlq_acomplex_add_acomplex(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ACOMPLEX_ADD_ACOMPLEX 29

int cmlq_acomplex_add_scomplex(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ACOMPLEX_ADD_SCOMPLEX 30

int cmlq_acomplex_add_scomplex_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ACOMPLEX_ADD_SCOMPLEX_BROADCAST_CACHE 31

int cmlq_acomplex_true_divide_acomplex(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ACOMPLEX_TRUE_DIVIDE_ACOMPLEX 32

int cmlq_acomplex_true_divide_scomplex(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ACOMPLEX_TRUE_DIVIDE_SCOMPLEX 33

int cmlq_acomplex_true_divide_scomplex_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ACOMPLEX_TRUE_DIVIDE_SCOMPLEX_BROADCAST_CACHE 34

int cmlq_acomplex_subtract_acomplex(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ACOMPLEX_SUBTRACT_ACOMPLEX 35

int cmlq_acomplex_multiply_acomplex(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ACOMPLEX_MULTIPLY_ACOMPLEX 36

int cmlq_acomplex_multiply_scomplex(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ACOMPLEX_MULTIPLY_SCOMPLEX 37

int cmlq_acomplex_multiply_scomplex_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ACOMPLEX_MULTIPLY_SCOMPLEX_BROADCAST_CACHE 38

int cmlq_acomplex_multiply_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ACOMPLEX_MULTIPLY_SLONG 39

int cmlq_acomplex_multiply_slong_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ACOMPLEX_MULTIPLY_SLONG_BROADCAST_CACHE 40

int cmlq_acomplex_multiply_sdouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ACOMPLEX_MULTIPLY_SDOUBLE 41

int cmlq_acomplex_multiply_sdouble_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ACOMPLEX_MULTIPLY_SDOUBLE_BROADCAST_CACHE 42

int cmlq_acomplex_multiply_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ACOMPLEX_MULTIPLY_SFLOAT 43

int cmlq_acomplex_multiply_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ACOMPLEX_MULTIPLY_SFLOAT_BROADCAST_CACHE 44

int cmlq_afloat_multiply_scomplex(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_AFLOAT_MULTIPLY_SCOMPLEX 45

int cmlq_afloat_multiply_scomplex_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_AFLOAT_MULTIPLY_SCOMPLEX_BROADCAST_CACHE 46

int cmlq_adouble_multiply_scomplex(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_MULTIPLY_SCOMPLEX 47

int cmlq_adouble_multiply_scomplex_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_MULTIPLY_SCOMPLEX_BROADCAST_CACHE 48

int cmlq_along_multiply_scomplex(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ALONG_MULTIPLY_SCOMPLEX 49

int cmlq_along_multiply_scomplex_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ALONG_MULTIPLY_SCOMPLEX_BROADCAST_CACHE 50

int cmlq_along_true_divide_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ALONG_TRUE_DIVIDE_SFLOAT 51

int cmlq_along_true_divide_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ALONG_TRUE_DIVIDE_SFLOAT_BROADCAST_CACHE 52

int cmlq_adouble_true_divide_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_TRUE_DIVIDE_SFLOAT 53

int cmlq_adouble_true_divide_sfloat_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_TRUE_DIVIDE_SFLOAT_BROADCAST_CACHE 54

int cmlq_adouble_true_divide_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_TRUE_DIVIDE_SLONG 55

int cmlq_adouble_true_divide_slong_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_TRUE_DIVIDE_SLONG_BROADCAST_CACHE 56

int cmlq_sfloat_true_divide_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SFLOAT_TRUE_DIVIDE_ADOUBLE 57

int cmlq_sfloat_true_divide_adouble_broadcast_cache(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SFLOAT_TRUE_DIVIDE_ADOUBLE_BROADCAST_CACHE 58

int cmlq_adouble_true_divide_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_TRUE_DIVIDE_ADOUBLE 59

int cmlq_adouble_power_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_POWER_SFLOAT 60

int cmlq_adouble_matmul_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_MATMUL_ADOUBLE 61

int cmlq_adouble_square_power_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_SQUARE_POWER_SFLOAT 62

int cmlq_adouble_square_power_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_SQUARE_POWER_SLONG 63

int cmlq_adouble_power_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADOUBLE_POWER_SLONG 64

int cmlq_minimum_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MINIMUM_AINT_AINT 65

int cmlq_minimum_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MINIMUM_AFLOAT_AFLOAT 66

int cmlq_minimum_afloat_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MINIMUM_AFLOAT_SFLOAT 67

int cmlq_minimum_afloat_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MINIMUM_AFLOAT_SLONG 68

int cmlq_minimum_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MINIMUM_ADOUBLE_ADOUBLE 69

int cmlq_minimum_adouble_sdouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MINIMUM_ADOUBLE_SDOUBLE 70

int cmlq_maximum_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MAXIMUM_AINT_AINT 71

int cmlq_maximum_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MAXIMUM_AFLOAT_AFLOAT 72

int cmlq_maximum_afloat_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MAXIMUM_AFLOAT_SFLOAT 73

int cmlq_maximum_adouble_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MAXIMUM_ADOUBLE_SLONG 74

int cmlq_maximum_afloat_slong(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MAXIMUM_AFLOAT_SLONG 75

int cmlq_maximum_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MAXIMUM_ADOUBLE_ADOUBLE 76

int cmlq_maximum_adouble_sdouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MAXIMUM_ADOUBLE_SDOUBLE 77

int cmlq_logical_not_abool_abool(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_LOGICAL_NOT_ABOOL_ABOOL 78

int cmlq_less_equal_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_LESS_EQUAL_ADOUBLE_ADOUBLE 79

int cmlq_logical_and_abool_abool(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_LOGICAL_AND_ABOOL_ABOOL 80

int cmlq_arctan2_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ARCTAN2_ADOUBLE_ADOUBLE 81

int cmlq_arctan2_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ARCTAN2_AFLOAT_AFLOAT 82

int cmlq_add_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADD_AINT_AINT 83

int cmlq_add_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADD_AFLOAT_AFLOAT 84

int cmlq_add_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADD_ADOUBLE_ADOUBLE 85

int cmlq_subtract_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SUBTRACT_AINT_AINT 86

int cmlq_subtract_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SUBTRACT_AFLOAT_AFLOAT 87

int cmlq_subtract_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SUBTRACT_ADOUBLE_ADOUBLE 88

int cmlq_multiply_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MULTIPLY_AINT_AINT 89

int cmlq_multiply_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MULTIPLY_AFLOAT_AFLOAT 90

int cmlq_multiply_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MULTIPLY_ADOUBLE_ADOUBLE 91

int cmlq_multiply_adouble_sdouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MULTIPLY_ADOUBLE_SDOUBLE 92

int cmlq_multiply_adouble_along(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MULTIPLY_ADOUBLE_ALONG 93

int cmlq_true_divide_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_TRUE_DIVIDE_ADOUBLE_ADOUBLE 94

int cmlq_true_divide_adouble_sdouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_TRUE_DIVIDE_ADOUBLE_SDOUBLE 95

int cmlq_bitwise_and_along_along(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_BITWISE_AND_ALONG_ALONG 96

int cmlq_bitwise_and_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_BITWISE_AND_AINT_AINT 97

int cmlq_bitwise_count_along_along(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_BITWISE_COUNT_ALONG_ALONG 98

int cmlq_bitwise_count_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_BITWISE_COUNT_AINT_AINT 99

int cmlq_bitwise_or_along_along(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_BITWISE_OR_ALONG_ALONG 100

int cmlq_bitwise_or_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_BITWISE_OR_AINT_AINT 101

int cmlq_bitwise_xor_along_along(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_BITWISE_XOR_ALONG_ALONG 102

int cmlq_bitwise_xor_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_BITWISE_XOR_AINT_AINT 103

int cmlq_copysign_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_COPYSIGN_AFLOAT_AFLOAT 104

int cmlq_copysign_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_COPYSIGN_ADOUBLE_ADOUBLE 105

int cmlq_cos_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_COS_AFLOAT_AFLOAT 106

int cmlq_cos_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_COS_ADOUBLE_ADOUBLE 107

int cmlq_cosh_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_COSH_AFLOAT_AFLOAT 108

int cmlq_cosh_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_COSH_ADOUBLE_ADOUBLE 109

int cmlq_divmod_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_DIVMOD_AFLOAT_AFLOAT 110

int cmlq_divmod_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_DIVMOD_ADOUBLE_ADOUBLE 111

int cmlq_floor_divide_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FLOOR_DIVIDE_AFLOAT_AFLOAT 112

int cmlq_floor_divide_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FLOOR_DIVIDE_ADOUBLE_ADOUBLE 113

int cmlq_fmax_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FMAX_AFLOAT_AFLOAT 114

int cmlq_fmax_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FMAX_ADOUBLE_ADOUBLE 115

int cmlq_fmax_acomplex_acomplex(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FMAX_ACOMPLEX_ACOMPLEX 116

int cmlq_fmin_afloat_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FMIN_AFLOAT_AFLOAT 117

int cmlq_fmin_adouble_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FMIN_ADOUBLE_ADOUBLE 118

int cmlq_fmin_acomplex_acomplex(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FMIN_ACOMPLEX_ACOMPLEX 119

int cmlq_gcd_aint_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_GCD_AINT_AINT 120

int cmlq_gcd_along_along(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_GCD_ALONG_ALONG 121

int cmlq_square_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SQUARE_AINT 122

int cmlq_square_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SQUARE_ADOUBLE 123

int cmlq_square_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SQUARE_AFLOAT 124

int cmlq_sqrt_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SQRT_ADOUBLE 125

int cmlq_sqrt_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SQRT_AFLOAT 126

int cmlq_absolute_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ABSOLUTE_AINT 127

int cmlq_absolute_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ABSOLUTE_ADOUBLE 128

int cmlq_absolute_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ABSOLUTE_AFLOAT 129

int cmlq_reciprocal_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_RECIPROCAL_AFLOAT 130

int cmlq_reciprocal_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_RECIPROCAL_ADOUBLE 131

int cmlq_reciprocal_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_RECIPROCAL_AINT 132

int cmlq_tanh_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_TANH_AFLOAT 133

int cmlq_tanh_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_TANH_SFLOAT 134

int cmlq_tanh_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_TANH_ADOUBLE 135

int cmlq_tanh_sdouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_TANH_SDOUBLE 136

int cmlq_exp_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_EXP_AFLOAT 137

int cmlq_exp_sfloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_EXP_SFLOAT 138

int cmlq_exp_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_EXP_ADOUBLE 139

int cmlq_exp_sdouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_EXP_SDOUBLE 140

int cmlq_exp2_sdouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_EXP2_SDOUBLE 141

int cmlq_exp2_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_EXP2_ADOUBLE 142

int cmlq_exp2_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_EXP2_AFLOAT 143

int cmlq_arccos_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ARCCOS_AFLOAT 144

int cmlq_arccos_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ARCCOS_ADOUBLE 145

int cmlq_arccosh_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ARCCOSH_ADOUBLE 146

int cmlq_arccosh_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ARCCOSH_AFLOAT 147

int cmlq_arcsin_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ARCSIN_AFLOAT 148

int cmlq_arcsin_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ARCSIN_ADOUBLE 149

int cmlq_arcsinh_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ARCSINH_ADOUBLE 150

int cmlq_arcsinh_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ARCSINH_AFLOAT 151

int cmlq_arctanh_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ARCTANH_AFLOAT 152

int cmlq_arctanh_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ARCTANH_ADOUBLE 153

int cmlq_arctan_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ARCTAN_AFLOAT 154

int cmlq_arctan_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ARCTAN_ADOUBLE 155

int cmlq_cbrt_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_CBRT_AFLOAT 156

int cmlq_cbrt_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_CBRT_ADOUBLE 157

int cmlq_ceil_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_CEIL_AFLOAT 158

int cmlq_ceil_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_CEIL_ADOUBLE 159

int cmlq_ceil_abool(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_CEIL_ABOOL 160

int cmlq_conjugate_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_CONJUGATE_AFLOAT 161

int cmlq_conjugate_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_CONJUGATE_ADOUBLE 162

int cmlq_conjugate_along(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_CONJUGATE_ALONG 163

int cmlq_conjugate_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_CONJUGATE_AINT 164

int cmlq_conjugate_acomplex(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_CONJUGATE_ACOMPLEX 165

int cmlq_expm1_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_EXPM1_AFLOAT 166

int cmlq_expm1_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_EXPM1_ADOUBLE 167

int cmlq_floor_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FLOOR_AFLOAT 168

int cmlq_floor_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FLOOR_ADOUBLE 169

int cmlq_frexp_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FREXP_AFLOAT 170

int cmlq_frexp_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FREXP_ADOUBLE 171

int cmlq_invert_along(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_INVERT_ALONG 172

int cmlq_invert_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_INVERT_AINT 173

int cmlq_isfinite_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ISFINITE_AFLOAT 174

int cmlq_isfinite_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ISFINITE_ADOUBLE 175

int cmlq_isfinite_along(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ISFINITE_ALONG 176

int cmlq_isfinite_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ISFINITE_AINT 177

int cmlq_isfinite_abool(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ISFINITE_ABOOL 178

int cmlq_isinf_afloat(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ISINF_AFLOAT 179

int cmlq_isinf_adouble(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ISINF_ADOUBLE 180

int cmlq_isinf_along(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ISINF_ALONG 181

int cmlq_isinf_aint(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ISINF_AINT 182

int cmlq_isinf_abool(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ISINF_ABOOL 183

int cmlq_add_aint_aint_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADD_AINT_AINT_KW 184

int cmlq_add_afloat_slong_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADD_AFLOAT_SLONG_KW 185

int cmlq_true_divide_adouble_slong_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_TRUE_DIVIDE_ADOUBLE_SLONG_KW 186

int cmlq_add_afloat_afloat_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADD_AFLOAT_AFLOAT_KW 187

int cmlq_add_adouble_adouble_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ADD_ADOUBLE_ADOUBLE_KW 188

int cmlq_multiply_aint_aint_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MULTIPLY_AINT_AINT_KW 189

int cmlq_multiply_adouble_sdouble_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MULTIPLY_ADOUBLE_SDOUBLE_KW 190

int cmlq_multiply_adouble_adouble_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_MULTIPLY_ADOUBLE_ADOUBLE_KW 191

int cmlq_copysign_afloat_afloat_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_COPYSIGN_AFLOAT_AFLOAT_KW 192

int cmlq_copysign_adouble_adouble_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_COPYSIGN_ADOUBLE_ADOUBLE_KW 193

int cmlq_cos_afloat_afloat_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_COS_AFLOAT_AFLOAT_KW 194

int cmlq_cos_adouble_adouble_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_COS_ADOUBLE_ADOUBLE_KW 195

int cmlq_cosh_afloat_afloat_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_COSH_AFLOAT_AFLOAT_KW 196

int cmlq_cosh_adouble_adouble_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_COSH_ADOUBLE_ADOUBLE_KW 197

int cmlq_divmod_afloat_afloat_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_DIVMOD_AFLOAT_AFLOAT_KW 198

int cmlq_divmod_adouble_adouble_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_DIVMOD_ADOUBLE_ADOUBLE_KW 199

int cmlq_floor_divide_afloat_afloat_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FLOOR_DIVIDE_AFLOAT_AFLOAT_KW 200

int cmlq_floor_divide_adouble_adouble_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FLOOR_DIVIDE_ADOUBLE_ADOUBLE_KW 201

int cmlq_fmax_afloat_afloat_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FMAX_AFLOAT_AFLOAT_KW 202

int cmlq_fmax_adouble_adouble_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FMAX_ADOUBLE_ADOUBLE_KW 203

int cmlq_fmax_acomplex_acomplex_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FMAX_ACOMPLEX_ACOMPLEX_KW 204

int cmlq_fmin_afloat_afloat_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FMIN_AFLOAT_AFLOAT_KW 205

int cmlq_fmin_adouble_adouble_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FMIN_ADOUBLE_ADOUBLE_KW 206

int cmlq_fmin_acomplex_acomplex_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FMIN_ACOMPLEX_ACOMPLEX_KW 207

int cmlq_gcd_aint_aint_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_GCD_AINT_AINT_KW 208

int cmlq_gcd_along_along_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_GCD_ALONG_ALONG_KW 209

int cmlq_sqrt_adouble_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SQRT_ADOUBLE_KW 210

int cmlq_sqrt_afloat_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_SQRT_AFLOAT_KW 211

int cmlq_expm1_afloat_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_EXPM1_AFLOAT_KW 212

int cmlq_expm1_adouble_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_EXPM1_ADOUBLE_KW 213

int cmlq_floor_afloat_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FLOOR_AFLOAT_KW 214

int cmlq_floor_adouble_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FLOOR_ADOUBLE_KW 215

int cmlq_frexp_afloat_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FREXP_AFLOAT_KW 216

int cmlq_frexp_adouble_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_FREXP_ADOUBLE_KW 217

int cmlq_invert_along_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_INVERT_ALONG_KW 218

int cmlq_invert_aint_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_INVERT_AINT_KW 219

int cmlq_isfinite_afloat_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ISFINITE_AFLOAT_KW 220

int cmlq_isfinite_adouble_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ISFINITE_ADOUBLE_KW 221

int cmlq_isfinite_along_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ISFINITE_ALONG_KW 222

int cmlq_isfinite_aint_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ISFINITE_AINT_KW 223

int cmlq_isfinite_abool_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ISFINITE_ABOOL_KW 224

int cmlq_isinf_afloat_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ISINF_AFLOAT_KW 225

int cmlq_isinf_adouble_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ISINF_ADOUBLE_KW 226

int cmlq_isinf_along_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ISINF_ALONG_KW 227

int cmlq_isinf_aint_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ISINF_AINT_KW 228

int cmlq_isinf_abool_kw(void *restrict external_cache_pointer, PyObject *restrict **stack_pointer_ptr);

#define SLOT_ISINF_ABOOL_KW 229

