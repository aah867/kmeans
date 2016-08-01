/*
 * simd.h
 *
 *  Created on: Jun 18, 2016
 *      Author: hasib
 */

#ifndef SIMD_H_
#define SIMD_H_

#include <xmmintrin.h>	// for SSE
#include <emmintrin.h>	// for SSE2
#include <immintrin.h>	// for AVX
#include <smmintrin.h>
#include <iostream>

#define ALIGN 32
#define VECSIZE64 2
#define VECSIZE32 4

#define simd_add_i(x,y) _mm_add_epi32(x,y)
#define simd_sub_i(x,y) _mm_sub_epi32(x,y)
#define simd_mul_i(x,y) _mm_mul_epi32(x,y)
#define simd_div_i(x,y) _mm_div_epi32(x,y)
#define simd_max_i(x,y) _mm_max_epi32(x,y)
#define simd_load_i(x) _mm_load_epi32(x)
#define simd_loadu_i(x) _mm_loadu_si128(x)
#define simd_store_i(x,y) _mm_store_si128(x,y)
#define simd_storeu_i(x,y) _mm_storeu_si128(x,y)
#define simd_set1_i(x) _mm_set1_epi32(x)
#define simd_or_i(x,y) _mm_or_si128(x,y)
#define simd_gt_i(x,y) _mm_cmpgt_epi32(x,y)
#define simd_and_i(x,y) _mm_and_si128(x,y)
#define simd_andnot_i(x,y) _mm_andnot_si128(x,y)
#define simd_lt_i(x,y) _mm_cmplt_epi32(x,y)
#define simd_set_zero() _mm_setzero_si128()
#define simd_mul_ui(x,y) _mm_mul_epu32(x,y)
#define simd_div_ui(x,y) _mm_div_epu32(x,y)
#define simd_mul_lo(x,y) _mm_mullo_epi32(x,y)

#define simd_add(x,y) _mm_add_pd(x,y)
#define simd_sub(x,y) _mm_sub_pd(x,y)
#define simd_mul(x,y) _mm_mul_pd(x,y)
#define simd_div(x,y) _mm_div_pd(x,y)
#define simd_max(x,y) _mm_max_pd(x,y)
#define simd_load(x) _mm_load_pd(x)
#define simd_store(x,y) _mm_store_pd(x,y)
#define simd_set1(x) _mm_set1_pd(x)
#define simd_or(x,y) _mm_or_pd(x,y)
#define simd_gt(x,y) _mm_cmpgt_pd(x,y)
#define simd_and(x,y) _mm_and_pd(x,y)
#define simd_andnot(x,y) _mm_andnot_pd(x,y)

#define simd_add_f(x,y) _mm_add_ps(x,y)
#define simd_sub_f(x,y) _mm_sub_ps(x,y)
#define simd_mul_f(x,y) _mm_mul_ps(x,y)
#define simd_div_f(x,y) _mm_div_ps(x,y)
#define simd_max_f(x,y) _mm_max_ps(x,y)
#define simd_load_f(x) _mm_load_ps(x)
#define simd_store_f(x,y) _mm_store_ps(x,y)
#define simd_set1_f(x) _mm_set1_ps(x)
#define simd_or_f(x,y) _mm_or_ps(x,y)
#define simd_gt_f(x,y) _mm_cmpgt_ps(x,y)
#define simd_and_f(x,y) _mm_and_ps(x,y)
#define simd_andnot_f(x,y) _mm_andnot_ps(x,y)

static inline __m128i muly(const __m128i &a, const __m128i &b)
{
#ifdef __SSE4_1__  // modern CPU - use SSE 4.1
    return _mm_mullo_epi32(a, b);
#else               // old CPU - use SSE 2
    __m128i tmp1 = _mm_mul_epu32(a,b); /* mul 2,0*/
    __m128i tmp2 = _mm_mul_epu32( _mm_srli_si128(a,4), _mm_srli_si128(b,4)); /* mul 3,1 */
    return _mm_unpacklo_epi32(_mm_shuffle_epi32(tmp1, _MM_SHUFFLE (0,0,2,0)), _mm_shuffle_epi32(tmp2, _MM_SHUFFLE (0,0,2,0))); /* shuffle results to [63..0] and pack */
#endif
}

typedef __m128i simd_int;
typedef __m128 simd_float;
typedef __m128d simd_double;

#endif /* SIMD_H_ */
