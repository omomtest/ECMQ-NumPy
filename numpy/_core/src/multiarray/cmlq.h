#ifndef CMLQ_H
#define CMLQ_H

#define CMLQ_CACHE_SIZE 512
#define SUBSCRIPT_CACHE_SIZE 512
#include <mapping.h>
#include <stdbool.h>
typedef struct _CMLQCounter{
    union {
        struct {
            uint16_t backoff : 4;
            uint16_t value : 12;
        };
        uint16_t as_counter;  // For printf("%#x", ...)
    };
}CMLQCounter;

#define CMLQCOUNTER_WARMPUP_BACKOFF 1
#define CMLQCOUNTER_WARMUP_VALUE 1

#define CMLQCOUNTER_COOLDOWN_BACKOFF 0
#define CMLQCOUNTER_COOLDOWN_VALUE 10

/*Here we need functions:
1. Initialize: WARMUP;
2. advance : value--;
3. backoff : 
4. triger : value == 0?
*/

static inline CMLQCounter make_CMLQCounter(uint16_t value, uint16_t backoff){
    assert(value<=0xFFF);
    assert(backoff<=15);
    CMLQCounter result;
    result.value = value;
    result.backoff = backoff;
    return result;
}

static inline void advance_CMLQCounter(CMLQCounter* counter){
    counter->value --;
    return;
    //return make_CMLQCounter(counter.value-1,counter.backoff);
}

static inline void backoff_CMLQCounter(CMLQCounter* counter){
    if(counter->backoff<12){
        counter->value = (1<<(counter->backoff+1))-1;
        counter->backoff ++;
    }
    else {
        counter -> value = (1<<12)-1;
    }
    return;
    // if(counter.backoff<12){
    //     return make_CMLQCounter((1 << (counter.backoff + 1)) - 1, counter.backoff);
    // }
    // else {
    //     return make_CMLQCounter((1 << 12)-1,12);
    // }
}

static inline void cooldown_CMLQCounter(CMLQCounter* counter){
    counter->value = CMLQCOUNTER_COOLDOWN_VALUE;
    counter->backoff = CMLQCOUNTER_COOLDOWN_BACKOFF;
    return;
    //return make_CMLQCounter(CMLQCOUNTER_COOLDOWN_VALUE,CMLQCOUNTER_COOLDOWN_BACKOFF);
}

static inline bool CMLQCounter_triggered(CMLQCounter counter){
    return counter.value == 0;
}

typedef struct _CMLQIterCache {
    /* Initial fixed position data */
    npy_uint32 itflags;
    npy_uint8 ndim, nop;
    npy_int8 maskop;
    npy_intp itersize, iterstart, iterend;
    /* iterindex is only used if RANGED or BUFFERED is set */
    npy_intp iterindex;
} CMLQIterCache;

enum CMLQCacheState {
    UNUSED = 0,
    TRIVIAL,
    ITERATOR,
    BROADCAST,
    DISABLED
};

typedef struct _CMLQIteratorPathCache {
    NpyIter *cached_iter;
    NpyIter_IterNextFunc *iter_next;
    npy_intp *countptr;
    char **dataptr;
    npy_intp *strides;
} CMLQIteratorPathCache;

typedef struct _CMLQTrivialPathCache {
    union {
        npy_intp steps[9];
        npy_intp fixed_strides[3];
    };
    npy_intp count;
} CMLQTrivialPathCache;

typedef union _CMLQCacheData {
    CMLQTrivialPathCache trivial;
    CMLQIteratorPathCache iterator;
} CMLQCacheData;

typedef struct _CMLQCacheStatsElem {
    char *opname;
    _Py_CODEUNIT *instr_ptr;
    uint64_t op_exec_count;
    uint64_t trivial_cache_hits;
    uint64_t trivial_cache_misses;
    uint64_t iterator_cache_hits;
    uint64_t iterator_cache_misses;
    uint64_t result_cache_hits;
    uint64_t result_cache_misses;
    uint64_t refcnt_misses;
    uint64_t ndims_misses;
    uint64_t shape_misses;
    uint64_t temp_elision_hits;
    uint64_t trivial_case;
    uint64_t iterator_case;
    uint64_t left_type_misses;
    uint64_t right_type_misses;
    uint64_t exponent_type_misses;
    uint64_t ufunc_type_misses;
    uint64_t trivial_cache_collisions;
    uint64_t iterator_cache_collisions;
    uint64_t function_end_clear;
    uint64_t trivial_cache_init;
    uint64_t iterator_cache_init;
    enum CMLQCacheState last_state;
} CMLQCacheStatsElem;

typedef struct _CMLQLocalityCacheElem {
    enum CMLQCacheState state;
    PyArrayObject *result;
    CMLQCounter counter;
    union {
        CMLQTrivialPathCache trivial;
        CMLQIteratorPathCache iterator;
    };
#ifdef CMLQ_STATS
    CMLQCacheStatsElem stats;
#endif

} CMLQLocalityCacheElem;

#define RESULT_CACHE_WARMUP 6
#define IS_RESULT_CACHE_UNSTABLE(elem) 0

// a negative counter means we are still warming up the cache

typedef struct _CMLQSubscriptCacheElem {
    _Py_CODEUNIT *instr;
    npy_index_info *indices;
    int index_type;
    int index_num;
    int array_ndim;
} CMLQSubscriptCacheElem;

#endif  // CMLQ_H
