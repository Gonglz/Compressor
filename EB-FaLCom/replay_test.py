#!/usr/bin/env python3
"""
Replay captured data to test C implementation independently.

This allows fast iteration on C code without running full training pipeline.
"""

import os
import sys
import pickle
import time
import numpy as np
import torch
from collections import OrderedDict
from typing import List, Dict, Any

sys.path.insert(0, '/home/exouser/compressor/final/EB-FaLCom/src')

from appfl.compressor.FalCom import FalCom
from appfl.compressor.FalComC import FalComC
from omegaconf import OmegaConf


class ReplayTester:
    """Test compressor using captured data"""
    
    def __init__(self, captured_dir="./captured_data"):
        self.captured_dir = captured_dir
        self.test_results = []
        
    def load_captured_data(self, round_num=0) -> List[Dict[str, Any]]:
        """Load all captured data for a specific round"""
        files = sorted(os.listdir(self.captured_dir))
        round_files = [f for f in files if f.startswith(f"round_{round_num}_")]
        
        data = []
        for fname in round_files:
            if "_input.pkl" in fname:
                filepath = os.path.join(self.captured_dir, fname)
                with open(filepath, 'rb') as f:
                    data.append({
                        'file': fname,
                        'data': pickle.load(f),
                        'type': 'compress' if 'compress' in fname else 'decompress'
                    })
        
        return data
    
    def reconstruct_model(self, serialized_model: Dict) -> OrderedDict:
        """Reconstruct model from serialized format"""
        model = OrderedDict()
        for k, v in serialized_model.items():
            data = v['data']
            if isinstance(data, np.ndarray):
                model[k] = torch.from_numpy(data)
            else:
                model[k] = data
        return model
    
    def test_compression(self, compressor, model_data: Dict, name: str):
        """Test compression on captured data"""
        model = self.reconstruct_model(model_data['model'])
        kwargs = model_data.get('kwargs', {})
        
        print(f"\n{'='*60}")
        print(f"Testing {name} - Compression")
        print(f"{'='*60}")
        print(f"Layers: {len(model)}")
        print(f"Total params: {sum(v.numel() if hasattr(v, 'numel') else np.prod(v.shape) for v in model.values())}")
        
        # Measure compression
        start = time.time()
        compressed = compressor.compress_model(model, **kwargs)
        compress_time = time.time() - start
        
        # Calculate sizes
        original_size = sum(
            v.numel() * v.element_size() if hasattr(v, 'numel')
            else np.prod(v.shape) * v['data'].dtype.itemsize
            for v in model.values()
        )
        compressed_size = len(compressed)
        ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        result = {
            'name': name,
            'operation': 'compress',
            'time': compress_time,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'ratio': ratio,
            'throughput_mbps': (original_size / 1024 / 1024) / compress_time if compress_time > 0 else 0
        }
        
        print(f"✓ Original size: {original_size / 1024 / 1024:.2f} MB")
        print(f"✓ Compressed size: {compressed_size / 1024 / 1024:.2f} MB")
        print(f"✓ Compression ratio: {ratio:.2f}x")
        print(f"✓ Time: {compress_time*1000:.2f} ms")
        print(f"✓ Throughput: {result['throughput_mbps']:.2f} MB/s")
        
        self.test_results.append(result)
        return compressed
    
    def test_decompression(self, compressor, compressed_data: bytes, name: str):
        """Test decompression on captured data"""
        print(f"\n{'='*60}")
        print(f"Testing {name} - Decompression")
        print(f"{'='*60}")
        print(f"Compressed size: {len(compressed_data) / 1024 / 1024:.2f} MB")
        
        # Measure decompression
        start = time.time()
        decompressed = compressor.decompress_model(compressed_data)
        decompress_time = time.time() - start
        
        # Calculate size
        decompressed_size = sum(
            v.numel() * v.element_size() if hasattr(v, 'numel')
            else np.prod(v.shape) * v.dtype.itemsize
            for v in decompressed.values()
        )
        
        result = {
            'name': name,
            'operation': 'decompress',
            'time': decompress_time,
            'compressed_size': len(compressed_data),
            'decompressed_size': decompressed_size,
            'throughput_mbps': (decompressed_size / 1024 / 1024) / decompress_time if decompress_time > 0 else 0
        }
        
        print(f"✓ Decompressed size: {decompressed_size / 1024 / 1024:.2f} MB")
        print(f"✓ Time: {decompress_time*1000:.2f} ms")
        print(f"✓ Throughput: {result['throughput_mbps']:.2f} MB/s")
        
        self.test_results.append(result)
        return decompressed
    
    def compare_implementations(self, round_num=0):
        """Compare Python vs C implementations"""
        print(f"\n🔬 Testing Round {round_num}")
        print("="*80)
        
        # Load captured data
        captured = self.load_captured_data(round_num)
        compress_data = [d for d in captured if d['type'] == 'compress']
        
        if not compress_data:
            print(f"❌ No compression data found for round {round_num}")
            return
        
        # Use first compression sample
        sample = compress_data[0]['data']
        
        # Create compressors
        config = OmegaConf.create({
            'momentum_lr': 0.07,
            'consistency_threshold': 0.5,
            'param_cutoff': 1024,
            'lossless_compressor': 'blosc',
            'sz_config': {
                'error_bounding_mode': 'REL',
                'error_bound': 1e-3
            }
        })
        
        print("\n📦 Creating compressors...")
        py_compressor = FalCom(config)
        c_compressor = FalComC(config)
        
        # Test Python implementation
        print("\n" + "="*80)
        print("PYTHON IMPLEMENTATION")
        print("="*80)
        py_compressed = self.test_compression(py_compressor, sample, "Python")
        py_decompressed = self.test_decompression(py_compressor, py_compressed, "Python")
        
        # Test C implementation
        print("\n" + "="*80)
        print("C IMPLEMENTATION")
        print("="*80)
        c_compressed = self.test_compression(c_compressor, sample, "C")
        c_decompressed = self.test_decompression(c_compressor, c_compressed, "C")
        
        # Compare results
        self.print_comparison()
    
    def print_comparison(self):
        """Print performance comparison table"""
        print("\n" + "="*80)
        print("PERFORMANCE COMPARISON")
        print("="*80)
        
        py_compress = [r for r in self.test_results if r['name'] == 'Python' and r['operation'] == 'compress'][0]
        c_compress = [r for r in self.test_results if r['name'] == 'C' and r['operation'] == 'compress'][0]
        py_decompress = [r for r in self.test_results if r['name'] == 'Python' and r['operation'] == 'decompress'][0]
        c_decompress = [r for r in self.test_results if r['name'] == 'C' and r['operation'] == 'decompress'][0]
        
        print(f"\n{'Operation':<20} {'Python':<15} {'C':<15} {'Speedup':<10}")
        print("-" * 65)
        
        # Compression time
        speedup = py_compress['time'] / c_compress['time'] if c_compress['time'] > 0 else 0
        print(f"{'Compress Time':<20} {py_compress['time']*1000:>10.2f} ms {c_compress['time']*1000:>10.2f} ms {speedup:>8.2f}x")
        
        # Decompression time
        speedup = py_decompress['time'] / c_decompress['time'] if c_decompress['time'] > 0 else 0
        print(f"{'Decompress Time':<20} {py_decompress['time']*1000:>10.2f} ms {c_decompress['time']*1000:>10.2f} ms {speedup:>8.2f}x")
        
        # Compression ratio
        print(f"\n{'Compression Ratio':<20} {py_compress['ratio']:>10.2f}x {c_compress['ratio']:>10.2f}x")
        
        # Throughput
        print(f"\n{'Compress Throughput':<20} {py_compress['throughput_mbps']:>10.2f} MB/s {c_compress['throughput_mbps']:>10.2f} MB/s")
        print(f"{'Decompress Throughput':<20} {py_decompress['throughput_mbps']:>10.2f} MB/s {c_decompress['throughput_mbps']:>10.2f} MB/s")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--round", type=int, default=0, help="Round number to test")
    parser.add_argument("--captured-dir", type=str, default="./captured_data")
    args = parser.parse_args()
    
    # Set environment
    os.environ['LD_LIBRARY_PATH'] = '/home/exouser/.appfl/.compressor/SZ3/lib'
    
    tester = ReplayTester(args.captured_dir)
    tester.compare_implementations(args.round)


if __name__ == "__main__":
    main()
