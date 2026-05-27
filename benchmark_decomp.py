#!/usr/bin/env python3
"""
性能对比测试：解压路径 OpenMP 优化（3轮）
测试 decompress_momentum_predicted_layer 的性能
"""

import sys
import os
import ctypes
import numpy as np
import time

def load_library(lib_path):
    return ctypes.CDLL(lib_path)

# C 结构体定义
class NDArray(ctypes.Structure):
    pass

NDArray._fields_ = [
    ("data", ctypes.c_void_p),
    ("shape", ctypes.POINTER(ctypes.c_size_t)),
    ("ndim", ctypes.c_size_t),
    ("total_size", ctypes.c_size_t),
    ("dtype", ctypes.c_int),
]

class CompressedLayerData(ctypes.Structure):
    pass

CompressedLayerData._fields_ = [
    ("type", ctypes.c_char * 32),
    ("codec", ctypes.c_char * 16),
    ("data", ctypes.POINTER(ctypes.c_uint8)),
    ("data_size", ctypes.c_size_t),
    ("bitmap", ctypes.POINTER(ctypes.c_uint8)),
    ("bitmap_size", ctypes.c_size_t),
    ("dominant_signs", ctypes.POINTER(ctypes.c_uint8)),
    ("dominant_signs_size", ctypes.c_size_t),
    ("shape", ctypes.c_size_t * 4),
    ("ndim", ctypes.c_size_t),
    ("original_dtype", ctypes.c_char * 16),
    ("stored_dtype", ctypes.c_char * 16),
    ("step", ctypes.c_int),
    ("num_predicted_kernels", ctypes.c_int),
    ("prediction_ratio", ctypes.c_float),
    ("sign_mismatch_ratio", ctypes.c_float),
    ("current_mean", ctypes.c_float),
    ("current_std", ctypes.c_float),
    ("prev_mean", ctypes.c_float),
    ("prev_std", ctypes.c_float),
    ("global_min", ctypes.c_float),
    ("global_max", ctypes.c_float),
    ("breakdown_stats_time", ctypes.c_double),
    ("breakdown_normalize_time", ctypes.c_double),
    ("breakdown_consistency_time", ctypes.c_double),
    ("breakdown_prediction_time", ctypes.c_double),
    ("breakdown_residual_compress_time", ctypes.c_double),
    ("breakdown_bitmap_compress_time", ctypes.c_double),
    ("breakdown_metadata_time", ctypes.c_double),
    ("breakdown_total_time", ctypes.c_double),
]

class CompressorConfig(ctypes.Structure):
    _fields_ = [
        ("momentum_lr", ctypes.c_float),
        ("consistency_threshold", ctypes.c_float),
        ("lossless_compressor", ctypes.c_char * 32),
        ("error_bounding_mode", ctypes.c_char * 16),
        ("error_bound", ctypes.c_float),
        ("sz3_lib_path", ctypes.c_char * 512),
        ("param_count_threshold", ctypes.c_size_t),
        ("max_history_length", ctypes.c_int),
    ]

def run_decomp_benchmark(lib_path, lib_name):
    """运行解压基准测试：3轮"""
    lib = load_library(lib_path)
    
    # 定义函数签名
    lib.momentum_compressor_create.argtypes = [ctypes.POINTER(CompressorConfig)]
    lib.momentum_compressor_create.restype = ctypes.c_void_p
    lib.momentum_compressor_destroy.argtypes = [ctypes.c_void_p]
    lib.momentum_compressor_set_client.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.momentum_compressor_compress_layer.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(NDArray)]
    lib.momentum_compressor_compress_layer.restype = ctypes.POINTER(CompressedLayerData)
    lib.momentum_compressor_decompress_layer.argtypes = [ctypes.c_void_p, ctypes.POINTER(CompressedLayerData), ctypes.c_char_p, ctypes.c_char_p]
    lib.momentum_compressor_decompress_layer.restype = ctypes.POINTER(NDArray)
    lib.ndarray_create.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_int]
    lib.ndarray_create.restype = ctypes.POINTER(NDArray)
    lib.ndarray_destroy.argtypes = [ctypes.POINTER(NDArray)]
    
    # 创建压缩器
    config = CompressorConfig()
    config.momentum_lr = 0.07
    config.consistency_threshold = 0.5
    config.lossless_compressor = b"zstd"
    config.error_bounding_mode = b"REL"
    config.error_bound = 1.0
    config.param_count_threshold = 1024
    config.max_history_length = 3
    
    compressor = lib.momentum_compressor_create(ctypes.byref(config))
    if not compressor:
        print(f"❌ {lib_name}: 创建压缩器失败")
        return None
    
    print(f"\n{'='*70}")
    print(f"📊 {lib_name} 解压性能测试 (3轮)")
    print(f"{'='*70}")
    
    # 大卷积层配置（触发动量预测）
    test_configs = [
        ("layer2.0.conv2", (128, 128, 3, 3)),   # 147.5K
        ("layer3.0.conv1", (256, 128, 3, 3)),   # 294.9K
        ("layer3.0.conv2", (256, 256, 3, 3)),   # 589.8K
        ("layer4.0.conv1", (512, 256, 3, 3)),   # 1.17M
        ("layer4.0.conv2", (512, 512, 3, 3)),   # 2.36M
    ]
    
    round_times = []
    
    for round_num in range(1, 4):
        print(f"\n🔄 Round {round_num}")
        lib.momentum_compressor_set_client(compressor, b"TestClient")
        
        round_start = time.time()
        decomp_time = 0.0
        
        for layer_name, shape in test_configs:
            # 第1步：创建原始数据并压缩
            grad_size = int(np.prod(shape))
            grad_data = np.random.randn(grad_size).astype(np.float32) * 0.01
            
            shape_array = (ctypes.c_size_t * len(shape))(*shape)
            ndarray = lib.ndarray_create(shape_array, len(shape), 0)
            
            if not ndarray:
                continue
            
            ctypes.memmove(ndarray.contents.data, grad_data.ctypes.data_as(ctypes.c_void_p), grad_data.nbytes)
            
            # 多轮压缩建立历史
            for step in range(3):
                compressed = lib.momentum_compressor_compress_layer(
                    compressor,
                    layer_name.encode('utf-8'),
                    ndarray
                )
                
                # 只在第3轮测试解压
                if step == 2 and compressed:
                    # 测试解压性能
                    decomp_start = time.time()
                    decompressed = lib.momentum_compressor_decompress_layer(
                        compressor,
                        compressed,
                        b"TestClient",
                        layer_name.encode('utf-8')
                    )
                    decomp_time += time.time() - decomp_start
                    
                    if decompressed:
                        lib.ndarray_destroy(decompressed)
                
                # 释放压缩数据
                if compressed:
                    if compressed.contents.data:
                        lib.free(compressed.contents.data)
                    if compressed.contents.bitmap:
                        lib.free(compressed.contents.bitmap)
                    if compressed.contents.dominant_signs:
                        lib.free(compressed.contents.dominant_signs)
                    lib.free(compressed)
            
            lib.ndarray_destroy(ndarray)
        
        round_time = time.time() - round_start
        round_times.append(round_time)
        
        print(f"  ✓ Round {round_num} 完成，总耗时: {round_time*1000:.1f} ms (解压: {decomp_time*1000:.1f} ms)")
    
    lib.momentum_compressor_destroy(compressor)
    
    if round_times:
        avg_time = np.mean(round_times)
        std_time = np.std(round_times)
        print(f"\n📈 统计结果:")
        print(f"  平均总耗时: {avg_time*1000:.1f} ms")
        print(f"  标准差: {std_time*1000:.1f} ms (±{std_time/avg_time*100:.1f}%)")
        print(f"  三轮时间: [{round_times[0]*1000:.1f}, {round_times[1]*1000:.1f}, {round_times[2]*1000:.1f}] ms")
        return {
            'lib': lib_name,
            'avg_time': avg_time,
            'std_time': std_time,
            'times': round_times,
        }
    
    return None

def main():
    print("\n" + "="*70)
    print("🚀 MomentumCompressor 解压性能对比（3轮）")
    print("="*70)
    print(f"📊 测试对象: ResNet50 中等~大卷积层（5个，触发动量预测）")
    print(f"🧵 线程数: 8 (OpenMP)")
    
    # 设置环境变量
    os.environ['LD_LIBRARY_PATH'] = "/home/exouser/.appfl/.compressor/SZ3/lib:" + os.environ.get('LD_LIBRARY_PATH', '')
    os.environ['OMP_NUM_THREADS'] = '8'
    
    serial_lib = "/home/exouser/compressor/final/libmomentum_compressor_serial.so"
    omp_lib = "/home/exouser/compressor/final/libmomentum_compressor_omp.so"
    
    results = {}
    
    # 运行串行基准
    if os.path.exists(serial_lib):
        result = run_decomp_benchmark(serial_lib, "串行基准 (Serial)")
        if result:
            results['serial'] = result
    else:
        print(f"\n❌ 串行库不存在: {serial_lib}")
    
    # 运行OpenMP优化
    if os.path.exists(omp_lib):
        result = run_decomp_benchmark(omp_lib, "OpenMP优化 (v8)")
        if result:
            results['omp'] = result
    else:
        print(f"\n❌ OpenMP库不存在: {omp_lib}")
    
    # 对比结果
    if 'serial' in results and 'omp' in results:
        serial = results['serial']
        omp = results['omp']
        
        speedup = serial['avg_time'] / omp['avg_time']
        improvement = (serial['avg_time'] - omp['avg_time']) / serial['avg_time'] * 100
        
        print(f"\n" + "="*70)
        print("📊 解压性能对比总结")
        print("="*70)
        print(f"\n{'版本':<20} {'平均耗时':<15} {'标准差':<15}")
        print("-" * 70)
        print(f"{'串行基准':<20} {serial['avg_time']*1000:>8.1f} ms  {serial['std_time']*1000:>8.1f} ms")
        print(f"{'OpenMP优化':<20} {omp['avg_time']*1000:>8.1f} ms  {omp['std_time']*1000:>8.1f} ms")
        print("-" * 70)
        print(f"\n🎯 性能提升:")
        print(f"  时间节省: {(serial['avg_time'] - omp['avg_time'])*1000:.1f} ms")
        print(f"  加速比: {speedup:.2f}x")
        print(f"  改进率: {improvement:.1f}%")
        
        if improvement > 0:
            print(f"\n✅ 解压路径 OpenMP 优化有效！")
        else:
            print(f"\n⚠️  解压路径性能未改进（可能工作负载较小）")
        
        print("\n" + "="*70)

if __name__ == '__main__':
    main()
