#ifndef ARCH_FLAGS_IS_DEF
#define ARCH_FLAGS_IS_DEF

#ifdef ENABLE_VECTO

#if __AVX2__ == 1

#define VEC_SIZE_CHAR   32
#define VEC_SIZE_INT     8
#define VEC_SIZE_FLOAT   8
#define VEC_SIZE_DOUBLE  4

#define AVX2 1

#elif __SSE__ == 1

#define VEC_SIZE_CHAR   16
#define VEC_SIZE_INT     4
#define VEC_SIZE_FLOAT   4
#define VEC_SIZE_DOUBLE  2

#define SSE 1

#endif

#endif

void arch_flags_print (void);

#endif