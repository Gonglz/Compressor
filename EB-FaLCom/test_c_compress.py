#!/usr/bin/env python3
"""
Test C compression implementation with captured training data
"""
import os
import sys
import pickle
import time
import torch
import numpy as np
from pathlib import Path

# Setup paths
sys.path.insert(0, '/home/exouser/compressor/final/EB-FaLCom/src')
os.environ['LD_LIBRARY_PATH'] = '/home/exouser/.appfl/.compressor/SZ3/lib'

from appfl.compressor import FalComC


def load_model_from_capture(filepath):
    """Load model from captured pickle file"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Reconstruct PyTorch model
    model = {}
    for k, v in data['model'].items():
        if isinstance(v['data'], np.ndarray):
            model[k] = torch.from_numpy(v['data'])
        else:
            model[k] = v['data']
    
    return model, data


def test_compression(captured_dir, round_num=1):
    """Test C compression on captured training data"""
    
    print("🔬 C Compression Benchmark")
    print("=" * 80)
    print()
    
    # Find compress input files
    pattern = f"round_{round_num}_compress_*_input.pkl"
    files = sorted(Path(captured_dir).glob(pattern))
    
    if not files:
        print(f"❌ No files found matching {pattern}")
        return
    
    print(f"📁 Found {len(files)} test files")
    print()
    
    # Create C compressor
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
    total_original_size = 0
    total_compressed_size = 0
    total_time = 0
    results = []
    
    for i, filepath in enumerate(files):
        print(f"{'=' * 80}")
        print(f"Test {i+1}/{len(files)}: {filepath.name}")
        print(f"{'=' * 80}")
        
        # Load model
        model, metadata = load_model_from_capture(filepath)
        
        print(f"📊 Model info:")
        print(f"   Layers: {len(model)}")
        print(f"   Client: {metadata.get('client_id', 'N/A')}")
        
        # Calculate original size
        original_size = sum(
            v.numel() * v.element_size() if hasattr(v, 'numel')
            else len(v) if isinstance(v, bytes)
            else 0
            for v in model.values()
        )
        original_size_mb = original_size / 1024 / 1024
        print(f"   Size: {original_size_mb:.2f} MB")
        print()
        
        # Benchmark compression
        print("⏱️  Compressing with C implementation...")
        start = time.time()
        client_id = metadata.get('client_id', 0)
        if isinstance(client_id, int):
            client_id = f"Client{client_id}"
        compressed = compressor.compress_model(model, client_id=client_id)
        elapsed = time.time() - start
        
        compressed_size = len(compressed)
        compressed_size_mb = compressed_size / 1024 / 1024
        ratio = original_size / compressed_size if compressed_size > 0 else 0
        throughput = original_size_mb / elapsed if elapsed > 0 else 0
        
        print(f"✓ Compressed: {compressed_size_mb:.2f} MB")
        print(f"✓ Ratio: {ratio:.2f}x")
        print(f"✓ Time: {elapsed*1000:.2f} ms")
        print(f"✓ Throughput: {throughput:.2f} MB/s")
        print()
        
        results.append({
            'file': filepath.name,
            'original_mb': original_size_mb,
            'compressed_mb': compressed_size_mb,
            'ratio': ratio,
            'time_ms': elapsed * 1000,
            'throughput': throughput
        })
        
        total_original_size += original_size
        total_compressed_size += compressed_size
        total_time += elapsed
    
    # Summary
    print()
    print("=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Tests completed: {len(files)}")
    print(f"Total original: {total_original_size/1024/1024:.2f} MB")
    print(f"Total compressed: {total_compressed_size/1024/1024:.2f} MB")
    print(f"Overall ratio: {total_original_size/total_compressed_size:.2f}x")
    print(f"Total time: {total_time*1000:.2f} ms")
    print(f"Average time: {(total_time/len(files))*1000:.2f} ms per test")
    print(f"Overall throughput: {(total_original_size/1024/1024)/total_time:.2f} MB/s")
    print()
    
    # Show individual results
    print("=" * 80)
    print("📈 DETAILED RESULTS")
    print("=" * 80)
    print(f"{'File':<30} {'Orig(MB)':<10} {'Comp(MB)':<10} {'Ratio':<8} {'Time(ms)':<10} {'MB/s':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['file']:<30} {r['original_mb']:<10.2f} {r['compressed_mb']:<10.2f} "
              f"{r['ratio']:<8.2f} {r['time_ms']:<10.2f} {r['throughput']:<10.2f}")
    print()
    
    # Optimization potential
    print("=" * 80)
    print("🎯 OpenMP OPTIMIZATION POTENTIAL")
    print("=" * 80)
    current_throughput = (total_original_size/1024/1024)/total_time
    print(f"Current (single-threaded): {current_throughput:.2f} MB/s")
    print(f"Target with 2 threads:     {2*current_throughput:.2f} MB/s (2x speedup)")
    print(f"Target with 4 threads:     {4*current_throughput:.2f} MB/s (4x speedup)")
    print(f"Target with 8 threads:     {8*current_throughput:.2f} MB/s (8x speedup)")
    print()
    print("Next steps:")
    print("  1. Profile C code to find bottlenecks")
    print("  2. Add OpenMP directives to parallel loops")
    print("  3. Re-test with this script to measure speedup")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test C compression implementation')
    parser.add_argument('--captured-dir', type=str,
                        default='./dataset/resnet18',
                        help='Directory with captured data')
    parser.add_argument('--round', type=int, default=1,
                        help='Round number to test')
    
    args = parser.parse_args()
    test_compression(args.captured_dir, args.round)
