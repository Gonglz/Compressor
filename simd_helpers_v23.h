/**
 * V23 SIMDnotefunction
 *
 * noteAVX2notecomputenote:
 * 1. Statscompute(mean/std/min/max)
 * 2. note
 * 3. note
 */

#ifndef SIMD_HELPERS_V23_H
#define SIMD_HELPERS_V23_H

#include <math.h>
#include <float.h>

// detectionAVX2note
#ifdef __AVX2__
#include <immintrin.h>
#define V23_SIMD_ENABLED 1
#define V23_SIMD_WIDTH 8  // AVX2note8notefloat
#else
#define V23_SIMD_ENABLED 0
#define V23_SIMD_WIDTH 1
#endif

/**
 * v23: AVX2noteabsnotecompute
 * note, noteSIMDnoterows
 */
static inline void compute_abs_arrays_simd(
    const float *input1,
    const float *input2,
    float *abs_output1,
    float *abs_output2,
    size_t size
) {
#ifdef __AVX2__
    if (size >= 32) {
        // note (0x7FFFFFFFnote8note)
        __m256 sign_mask = _mm256_set1_ps(-0.0f);
        size_t vec_count = size / 8;
        size_t vec_size = vec_count * 8;

        for (size_t i = 0; i < vec_size; i += 8) {
            // note8notefloat
            __m256 v1 = _mm256_loadu_ps(&input1[i]);
            __m256 v2 = _mm256_loadu_ps(&input2[i]);

            // note: abs(x) = x & ~sign_mask
            __m256 abs1 = _mm256_andnot_ps(sign_mask, v1);
            __m256 abs2 = _mm256_andnot_ps(sign_mask, v2);

            // noteresult
            _mm256_storeu_ps(&abs_output1[i], abs1);
            _mm256_storeu_ps(&abs_output2[i], abs2);
        }

        // note
        for (size_t i = vec_size; i < size; i++) {
            abs_output1[i] = fabsf(input1[i]);
            abs_output2[i] = fabsf(input2[i]);
        }
        return;
    }
#endif

    // noteimplement
    for (size_t i = 0; i < size; i++) {
        abs_output1[i] = fabsf(input1[i]);
        abs_output2[i] = fabsf(input2[i]);
    }
}

/**
 * v23: AVX2notesumcompute
 * notemeancompute, note4-5note
 */
static inline float compute_sum_simd(const float *data, size_t size) {
#ifdef __AVX2__
    if (size >= 32) {
        __m256 sum_vec = _mm256_setzero_ps();
        size_t vec_count = size / 8;
        size_t vec_size = vec_count * 8;

        for (size_t i = 0; i < vec_size; i += 8) {
            __m256 v = _mm256_loadu_ps(&data[i]);
            sum_vec = _mm256_add_ps(sum_vec, v);
        }

        // note: 8notelanenote
        float sum_arr[8];
        _mm256_storeu_ps(sum_arr, sum_vec);
        float sum = 0.0f;
        for (int i = 0; i < 8; i++) sum += sum_arr[i];

        // note
        for (size_t i = vec_size; i < size; i++) {
            sum += data[i];
        }

        return sum;
    }
#endif

    // note
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) sum += data[i];
    return sum;
}

/**
 * v23: AVX2note
 * normalized[i] = (data[i] - mean) / std
 */
static inline void normalize_array_simd(
    const float *data,
    float *normalized,
    size_t size,
    float mean,
    float std
) {
#ifdef __AVX2__
    if (size >= 32) {
        __m256 mean_vec = _mm256_set1_ps(mean);
        float inv_std = (std > 1e-8f)? (1.0f / std): 1.0f;
        __m256 inv_std_vec = _mm256_set1_ps(inv_std);

        size_t vec_count = size / 8;
        size_t vec_size = vec_count * 8;

        for (size_t i = 0; i < vec_size; i += 8) {
            __m256 v = _mm256_loadu_ps(&data[i]);

            // (v - mean) / std
            __m256 result = _mm256_sub_ps(v, mean_vec);
            result = _mm256_mul_ps(result, inv_std_vec);

            _mm256_storeu_ps(&normalized[i], result);
        }

        // note
        bool use_std = (std > 1e-8f);
        for (size_t i = vec_size; i < size; i++) {
            normalized[i] = data[i] - mean;
            if (use_std) normalized[i] *= inv_std;
        }
        return;
    }
#endif

    // note
    bool use_std = (std > 1e-8f);
    float inv_std = use_std? (1.0f / std): 1.0f;

    for (size_t i = 0; i < size; i++) {
        normalized[i] = data[i] - mean;
        if (use_std) normalized[i] *= inv_std;
    }
}

/**
 * v23: AVX2note
 * computenote
 */
static inline size_t count_sign_matches_simd(
    const float *data1,
    const float *data2,
    size_t size
) {
#ifdef __AVX2__
    if (size >= 32) {
        __m256 zero = _mm256_setzero_ps();
        size_t match_count = 0;
        size_t vec_count = size / 8;
        size_t vec_size = vec_count * 8;

        for (size_t i = 0; i < vec_size; i += 8) {
            __m256 v1 = _mm256_loadu_ps(&data1[i]);
            __m256 v2 = _mm256_loadu_ps(&data2[i]);

            // note: v > 0
            __m256 sign1 = _mm256_cmp_ps(v1, zero, _CMP_GT_OQ);
            __m256 sign2 = _mm256_cmp_ps(v2, zero, _CMP_GT_OQ);

            // XOR: note0, note1
            __m256 xor_result = _mm256_xor_ps(sign1, sign2);

            // note
            int mask = _mm256_movemask_ps(xor_result);
            // popcount: note
            match_count += (8 - __builtin_popcount(mask));
        }

        // note
        for (size_t i = vec_size; i < size; i++) {
            bool same_sign = ((data1[i] >= 0) == (data2[i] >= 0));
            if (same_sign) match_count++;
        }

        return match_count;
    }
#endif

    // note
    size_t match_count = 0;
    for (size_t i = 0; i < size; i++) {
        bool same_sign = ((data1[i] >= 0) == (data2[i] >= 0));
        if (same_sign) match_count++;
    }
    return match_count;
}

#endif // SIMD_HELPERS_V23_H
