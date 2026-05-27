/**
 * V23 SIMD优化辅助函数
 * 
 * 使用AVX2指令集优化纯计算部分：
 * 1. Stats计算（mean/std/min/max）
 * 2. 归一化操作
 * 3. 符号一致性检查
 */

#ifndef SIMD_HELPERS_V23_H
#define SIMD_HELPERS_V23_H

#include <math.h>
#include <float.h>

// 检测AVX2支持
#ifdef __AVX2__
#include <immintrin.h>
#define V23_SIMD_ENABLED 1
#define V23_SIMD_WIDTH 8  // AVX2一次处理8个float
#else
#define V23_SIMD_ENABLED 0
#define V23_SIMD_WIDTH 1
#endif

/**
 * v23: AVX2优化的abs数组计算
 * 同时对两个数组取绝对值，利用SIMD并行
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
        // 符号位掩码 (0x7FFFFFFF重复8次)
        __m256 sign_mask = _mm256_set1_ps(-0.0f);
        size_t vec_count = size / 8;
        size_t vec_size = vec_count * 8;
        
        for (size_t i = 0; i < vec_size; i += 8) {
            // 加载8个float
            __m256 v1 = _mm256_loadu_ps(&input1[i]);
            __m256 v2 = _mm256_loadu_ps(&input2[i]);
            
            // 取绝对值: abs(x) = x & ~sign_mask
            __m256 abs1 = _mm256_andnot_ps(sign_mask, v1);
            __m256 abs2 = _mm256_andnot_ps(sign_mask, v2);
            
            // 存储结果
            _mm256_storeu_ps(&abs_output1[i], abs1);
            _mm256_storeu_ps(&abs_output2[i], abs2);
        }
        
        // 处理剩余元素
        for (size_t i = vec_size; i < size; i++) {
            abs_output1[i] = fabsf(input1[i]);
            abs_output2[i] = fabsf(input2[i]);
        }
        return;
    }
#endif
    
    // 标量后备实现
    for (size_t i = 0; i < size; i++) {
        abs_output1[i] = fabsf(input1[i]);
        abs_output2[i] = fabsf(input2[i]);
    }
}

/**
 * v23: AVX2优化的sum计算
 * 用于mean计算，比标量循环快约4-5倍
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
        
        // 水平归约：8个lane相加
        float sum_arr[8];
        _mm256_storeu_ps(sum_arr, sum_vec);
        float sum = 0.0f;
        for (int i = 0; i < 8; i++) sum += sum_arr[i];
        
        // 剩余元素
        for (size_t i = vec_size; i < size; i++) {
            sum += data[i];
        }
        
        return sum;
    }
#endif
    
    // 标量后备
    float sum = 0.0f;
    for (size_t i = 0; i < size; i++) sum += data[i];
    return sum;
}

/**
 * v23: AVX2优化的归一化
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
        float inv_std = (std > 1e-8f) ? (1.0f / std) : 1.0f;
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
        
        // 剩余元素
        bool use_std = (std > 1e-8f);
        for (size_t i = vec_size; i < size; i++) {
            normalized[i] = data[i] - mean;
            if (use_std) normalized[i] *= inv_std;
        }
        return;
    }
#endif
    
    // 标量后备
    bool use_std = (std > 1e-8f);
    float inv_std = use_std ? (1.0f / std) : 1.0f;
    
    for (size_t i = 0; i < size; i++) {
        normalized[i] = data[i] - mean;
        if (use_std) normalized[i] *= inv_std;
    }
}

/**
 * v23: AVX2优化的符号比较
 * 计算两个数组中符号相同的元素个数
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
            
            // 判断符号：v > 0
            __m256 sign1 = _mm256_cmp_ps(v1, zero, _CMP_GT_OQ);
            __m256 sign2 = _mm256_cmp_ps(v2, zero, _CMP_GT_OQ);
            
            // XOR: 符号相同时为0，不同时为1
            __m256 xor_result = _mm256_xor_ps(sign1, sign2);
            
            // 检查每个元素
            int mask = _mm256_movemask_ps(xor_result);
            // popcount: 统计不匹配的位数
            match_count += (8 - __builtin_popcount(mask));
        }
        
        // 剩余元素
        for (size_t i = vec_size; i < size; i++) {
            bool same_sign = ((data1[i] >= 0) == (data2[i] >= 0));
            if (same_sign) match_count++;
        }
        
        return match_count;
    }
#endif
    
    // 标量后备
    size_t match_count = 0;
    for (size_t i = 0; i < size; i++) {
        bool same_sign = ((data1[i] >= 0) == (data2[i] >= 0));
        if (same_sign) match_count++;
    }
    return match_count;
}

#endif // SIMD_HELPERS_V23_H
