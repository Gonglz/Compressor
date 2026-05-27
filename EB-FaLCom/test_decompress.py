#!/usr/bin/env python3
"""
Simple decompression benchmark for C implementation
Tests decompression speed on captured compressed data
"""
import os
import pickle
import time
import sys
from pathlib import Path

# Setup paths
sys.path.insert(0, '/home/exouser/compressor/final/EB-FaLCom/src')
os.environ['LD_LIBRARY_PATH'] = '/home/exouser/.appfl/.compressor/SZ3/lib'

from appfl.compressor import FalComC


def benchmark_decompress(captured_dir: str, round_num: int = 1):
    """Benchmark decompression on captured data"""
    
    print(f"🔬 Decompression Benchmark - Round {round_num}")
    print("=" * 80)
    
    # Find all decompress input files for this round
    pattern = f"round_{round_num}_decompress_*_input.pkl"
    files = sorted(Path(captured_dir).glob(pattern))
    
    if not files:
        print(f"❌ No files found matching {pattern}")
        return
    
    print(f"📁 Found {len(files)} test files")
    print()
    
    # Create compressor
    print("📦 Initializing C compressor...")
    config = {
        'lossless_compressor': 'blosc',
        'error_bound_mode': 'REL',
        'error_bound': 0.001,
        'momentum_lr': 0.07,
        'param_cutoff': 1024
    }
    compressor = FalComC(config)
    print()
    
    # Test each file
    total_compressed_size = 0
    total_decompressed_size = 0
    total_time = 0
    
    for i, filepath in enumerate(files, 1):
        print(f"{'=' * 80}")
        print(f"Test {i}/{len(files)}: {filepath.name}")
        print(f"{'=' * 80}")
        
        # Load compressed data
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        compressed_bytes = data['compressed_data']
        compressed_size_mb = len(compressed_bytes) / 1024 / 1024
        
        print(f"📦 Compressed size: {compressed_size_mb:.2f} MB")
        
        # Benchmark decompression
        print("⏱️  Decompressing...")
        start = time.time()
        decompressed = compressor.decompress_model(compressed_bytes)
        elapsed = time.time() - start
        
        # Calculate decompressed size
        decompressed_size = 0
        layer_count = 0
        for k, v in decompressed.items():
            if hasattr(v, 'numel'):
                decompressed_size += v.numel() * v.element_size()
                layer_count += 1
        
        decompressed_size_mb = decompressed_size / 1024 / 1024
        throughput = decompressed_size_mb / elapsed if elapsed > 0 else 0
        
        print(f"✓ Decompressed: {decompressed_size_mb:.2f} MB")
        print(f"✓ Layers: {layer_count}")
        print(f"✓ Time: {elapsed*1000:.2f} ms")
        print(f"✓ Throughput: {throughput:.2f} MB/s")
        print()
        
        total_compressed_size += len(compressed_bytes)
        total_decompressed_size += decompressed_size
        total_time += elapsed
    
    # Summary
    print()
    print("=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Tests completed: {len(files)}")
    print(f"Total compressed: {total_compressed_size/1024/1024:.2f} MB")
    print(f"Total decompressed: {total_decompressed_size/1024/1024:.2f} MB")
    print(f"Total time: {total_time*1000:.2f} ms")
    print(f"Average time per test: {(total_time/len(files))*1000:.2f} ms")
    print(f"Overall throughput: {(total_decompressed_size/1024/1024)/total_time:.2f} MB/s")
    print()
    
    # Show potential for optimization
    print("🎯 OpenMP Optimization Potential:")
    print(f"   - {layer_count} layers per test (many can be parallelized)")
    print(f"   - Current throughput: {(total_decompressed_size/1024/1024)/total_time:.2f} MB/s")
    print(f"   - Target with 4 threads: ~{4*(total_decompressed_size/1024/1024)/total_time:.2f} MB/s")
    print(f"   - Target with 8 threads: ~{8*(total_decompressed_size/1024/1024)/total_time:.2f} MB/s")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark C decompression')
    parser.add_argument('--captured-dir', type=str, 
                        default='./dataset/resnet18',
                        help='Directory with captured data')
    parser.add_argument('--round', type=int, default=1,
                        help='Round number to test')
    
    args = parser.parse_args()
    benchmark_decompress(args.captured_dir, args.round)
