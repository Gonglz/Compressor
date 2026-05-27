import gzip
import lzma
import zlib
import zstd
import blosc
import torch
import pickle
import numpy as np
import heapq
from collections import Counter

from copy import deepcopy
from typing import Tuple, Union, List, Optional, Dict
from collections import OrderedDict
from omegaconf import DictConfig

from .base_compressor import BaseCompressor


class _BitWriter:
    def __init__(self):
        self._buffer = bytearray()
        self._current = 0
        self._nbits = 0

    def write_bit(self, bit: int) -> None:
        self._current = (self._current << 1) | (bit & 1)
        self._nbits += 1
        if self._nbits == 8:
            self._buffer.append(self._current)
            self._current = 0
            self._nbits = 0

    def finish(self) -> bytes:
        if self._nbits:
            self._current <<= 8 - self._nbits
            self._buffer.append(self._current)
            self._current = 0
            self._nbits = 0
        return bytes(self._buffer)


class _BitReader:
    def __init__(self, data: bytes):
        self._data = data
        self._index = 0
        self._current = 0
        self._nbits = 0

    def read_bit(self) -> int:
        if self._nbits == 0:
            if self._index >= len(self._data):
                raise ValueError("Insufficient bits for decoding.")
            self._current = self._data[self._index]
            self._index += 1
            self._nbits = 8
        bit = (self._current >> 7) & 1
        self._current = (self._current << 1) & 0xFF
        self._nbits -= 1
        return bit


class QSGDCompressor(BaseCompressor):
    """
    Quantized SGD compressor with Huffman coded magnitudes.
    Modified QSGD using L-infinity norm and Huffman encoding.
    
    Required config keys:
      - lossless_compressor: zstd | gzip | zlib | blosc | lzma (for non-weight params)
      - param_cutoff: minimum tensor length for QSGD
      - qsgd_level: number of quantization levels (s in the paper)
      - qsgd_stochastic (optional, default True): use stochastic rounding
      - qsgd_seed (optional, default None): random seed for stochastic rounding
    """

    def __init__(self, compressor_config: DictConfig):
        self.cfg = compressor_config
        self.lossless_compressor = compressor_config.get("lossless_compressor", "gzip")
        self.param_count_threshold = compressor_config.get("param_cutoff", 0)
        self.level = int(compressor_config.get("qsgd_level", 256))
        if self.level <= 0:
            raise ValueError("qsgd_level must be a positive integer.")
        self.stochastic = bool(compressor_config.get("qsgd_stochastic", True))
        seed = compressor_config.get("qsgd_seed", None)
        self.rng = np.random.default_rng(seed)

    def compress_model(
        self,
        model: Union[dict, OrderedDict, List[Union[dict, OrderedDict]]],
        batched: bool = False,
        client_id: Optional[str] = None,
    ) -> bytes:
        if batched:
            if isinstance(model, list):
                return pickle.dumps([self.compress_model(m) for m in model])
            if isinstance(model, (dict, OrderedDict)):
                return pickle.dumps(
                    OrderedDict(
                        (k, self.compress_model(v))
                        for k, v in model.items()
                    )
                )

        for _, value in model.items():
            is_nested = not isinstance(value, torch.Tensor)
            break

        if is_nested:
            compressed_models = OrderedDict()
            for key, weights in model.items():
                if isinstance(weights, (dict, OrderedDict)):
                    compressed_models[key] = self._compress_weights(weights)[0]
                else:
                    compressed_models[key] = weights
        else:
            compressed_models = self._compress_weights(model)[0]
        return pickle.dumps(compressed_models)

    def decompress_model(
        self,
        compressed_model: bytes,
        model: Union[dict, OrderedDict],
        batched: bool = False,
    ) -> Union[OrderedDict, dict, List[Union[OrderedDict, dict]]]:
        compressed_model = pickle.loads(compressed_model)

        if batched:
            if isinstance(compressed_model, list):
                return [
                    self.decompress_model(cm, model) for cm in compressed_model
                ]
            if isinstance(compressed_model, (dict, OrderedDict)):
                return OrderedDict(
                    (k, self.decompress_model(v, model))
                    for k, v in compressed_model.items()
                )

        for _, value in compressed_model.items():
            is_nested = not isinstance(value, bytes)
            break

        if is_nested:
            decompressed_model = OrderedDict()
            for key, value in compressed_model.items():
                if isinstance(value, (dict, OrderedDict)):
                    decompressed_model[key] = self._decompress_model(value, model)
                else:
                    decompressed_model[key] = value
        else:
            decompressed_model = self._decompress_model(compressed_model, model)
        return decompressed_model

    def _compress_weights(
        self, weights: Union[OrderedDict, dict]
    ) -> Tuple[Union[OrderedDict, dict], int]:
        if len(weights) == 0:
            return (weights, 0)
        for _, value in weights.items():
            if not isinstance(value, torch.Tensor):
                return (weights, 0)
            break

        compressed_weights = {}
        lossy_elements = 0

        for name, param in weights.items():
            param_flat = param.flatten().detach().cpu().numpy()
            if (
                "weight" in name
                and param_flat.size > self.param_count_threshold
            ):
                compressed_weights[name] = self._compress(param_flat)
                lossy_elements += param_flat.size
            else:
                compressed_weights[name] = self._lossless_compress(param_flat)
        return (compressed_weights, lossy_elements)

    def _compress(self, ori_data: np.ndarray) -> bytes:
        payload = self._qsgd_encode(ori_data)
        compressed = pickle.dumps(payload)
        
        # Debug information
        original_bytes = ori_data.nbytes
        compressed_bytes = len(compressed)
        ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 0
        # print(f"[QSGD] Original: {original_bytes}B → Compressed: {compressed_bytes}B (ratio: {ratio:.2f}x)")
        
        return compressed

    def _decompress_model(
        self,
        compressed_weights: Union[dict, OrderedDict],
        model: Union[dict, OrderedDict],
    ) -> Union[OrderedDict, dict]:
        if len(compressed_weights) == 0:
            return compressed_weights
        for _, value in compressed_weights.items():
            if not isinstance(value, bytes):
                return compressed_weights
            break

        decompressed_weights = OrderedDict()
        for name, param in model.state_dict().items():
            if (
                "weight" in name
                and param.numel() > self.param_count_threshold
            ):
                compressed_weights[name] = self._decompress(
                    cmp_data=compressed_weights[name],
                    ori_shape=(param.numel(),),
                    ori_dtype=np.float32,
                )
            else:
                compressed_weights[name] = self._lossless_decompress(
                    compressed_weights[name]
                )
            if param.shape == torch.Size([]):
                copy_arr = deepcopy(compressed_weights[name])
                copy_tensor = torch.from_numpy(copy_arr)
                decompressed_weights[name] = torch.tensor(copy_tensor).to(param.dtype)
            else:
                copy_arr = deepcopy(compressed_weights[name])
                copy_tensor = torch.from_numpy(copy_arr).to(param.dtype)
                decompressed_weights[name] = copy_tensor.reshape(param.shape)
        return decompressed_weights

    def _decompress(
        self, cmp_data: bytes, ori_shape: Tuple[int, ...], ori_dtype: np.dtype
    ) -> np.ndarray:
        payload = pickle.loads(cmp_data)
        numel = int(payload.get("numel", 0))
        if numel == 0:
            return np.zeros(ori_shape, dtype=ori_dtype)

        level = int(payload.get("level", self.level))
        if level <= 0:
            raise ValueError("Invalid quantization level in payload.")

        max_abs = float(payload.get("max_abs", 0.0))
        huffman_code = payload.get("huffman_code", {})
        
        # Decode levels with Huffman
        levels = np.asarray(
            self._huffman_decode(payload.get("levels", b""), numel, huffman_code),
            dtype=np.float32
        )

        sign_bytes = payload.get("signs", b"")
        sign_bits = np.unpackbits(np.frombuffer(sign_bytes, dtype=np.uint8))[:numel]
        signs = np.where(sign_bits > 0, 1.0, -1.0).astype(np.float32)

        # Dequantization: v_i = sign(v_i) * max_abs * (ξ_i / s)
        scale = max_abs / level if level else 0.0
        values = signs * levels * scale
        return values.astype(ori_dtype, copy=False).reshape(ori_shape)

    def _lossless_compress(self, data: np.ndarray) -> bytes:
        if self.lossless_compressor == "zstd":
            return zstd.compress(data, 10)
        if self.lossless_compressor == "gzip":
            return gzip.compress(data.tobytes())
        if self.lossless_compressor == "zlib":
            return zlib.compress(data.tobytes())
        if self.lossless_compressor == "blosc":
            return blosc.compress(data.tobytes(), typesize=4)
        if self.lossless_compressor == "lzma":
            return lzma.compress(data.tobytes())
        raise NotImplementedError

    def _lossless_decompress(self, data: bytes) -> np.ndarray:
        if self.lossless_compressor == "zstd":
            raw = zstd.decompress(data)
        elif self.lossless_compressor == "gzip":
            raw = gzip.decompress(data)
        elif self.lossless_compressor == "zlib":
            raw = zlib.decompress(data)
        elif self.lossless_compressor == "blosc":
            raw = blosc.decompress(data, as_bytearray=True)
        elif self.lossless_compressor == "lzma":
            raw = lzma.decompress(data)
        else:
            raise NotImplementedError
        return np.frombuffer(raw, dtype=np.float32)

    def _qsgd_encode(self, data: np.ndarray) -> dict:
        """
        QSGD encoding with L-infinity norm (max absolute value) normalization
        and Huffman encoding for the quantized levels.
        """
        flat = data.astype(np.float32, copy=False).flatten()
        numel = flat.size
        if numel == 0:
            return {
                "max_abs": np.float32(0.0),
                "numel": 0,
                "level": np.int32(self.level),
                "huffman_code": {},
                "levels": b"",
                "signs": b"",
            }

        # Use L-infinity norm (max absolute value)
        max_abs = np.max(np.abs(flat))
        
        if not np.isfinite(max_abs):
            raise ValueError("Encountered non-finite max_abs during QSGD compression.")

        if max_abs == 0.0 or self.level == 0:
            levels = np.zeros(numel, dtype=np.int32)
            signs_bool = np.ones(numel, dtype=np.uint8)
        else:
            # Quantization with max normalization: scaled = |v_i| / max_abs * s
            scaled = np.abs(flat) * (self.level / max_abs)
            
            if self.stochastic:
                base = np.floor(scaled).astype(np.int32)
                prob = scaled - base
                rnd = self.rng.random(numel)
                increments = (rnd < prob).astype(np.int32)
                levels = base + increments
            else:
                levels = np.round(scaled).astype(np.int32)
            
            levels = np.clip(levels, 0, self.level)
            signs_bool = (flat >= 0).astype(np.uint8)

        # Statistics for debugging
        unique, counts = np.unique(levels, return_counts=True)
        level_dist = dict(zip(unique.tolist(), counts.tolist()))
        bins_0_1 = sum(counts[unique <= 1]) if len(counts[unique <= 1]) > 0 else 0
        bins_0_10 = sum(counts[unique <= 10]) if len(counts[unique <= 10]) > 0 else 0
        
        # print(f"  Level stats: numel={numel}, max_abs={max_abs:.6f}")
        # print(f"  Distribution: 0-1={bins_0_1/numel*100:.1f}%, 0-10={bins_0_10/numel*100:.1f}%")
        # print(f"  Level range: [{np.min(levels)}, {np.max(levels)}]")

        # Encode signs as packed bits
        sign_bytes = np.packbits(signs_bool).tobytes()
        
        # Encode levels with Huffman
        levels_list = levels.tolist()
        level_bytes, huffman_code = self._huffman_encode(levels_list)
        
        
        total_compressed = len(level_bytes) + len(sign_bytes)

        return {
            "max_abs": np.float32(max_abs),
            "numel": np.int32(numel),
            "level": np.int32(self.level),
            "huffman_code": huffman_code,
            "levels": level_bytes,
            "signs": sign_bytes,
        }

    @staticmethod
    def _huffman_encode(levels: List[int]) -> Tuple[bytes, Dict[int, str]]:
        """
        Huffman encoding for quantization levels.
        Returns encoded bytes and the Huffman code table.
        """
        if not levels:
            return b"", {}
        
        # Handle single unique value case
        freq = Counter(levels)
        if len(freq) == 1:
            symbol = list(freq.keys())[0]
            # Use a dummy 1-bit code for single symbol
            huffman_code = {symbol: "0"}
            writer = _BitWriter()
            for _ in levels:
                writer.write_bit(0)
            return writer.finish(), huffman_code
        
        # Build Huffman tree using a min-heap
        heap = [[weight, [symbol, ""]] for symbol, weight in freq.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            # Assign '0' to left branch, '1' to right branch
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        
        # Extract Huffman codes
        huffman_code = {symbol: code for symbol, code in sorted(heapq.heappop(heap)[1:])}
        
        # Encode the data
        writer = _BitWriter()
        for level in levels:
            for bit in huffman_code[level]:
                writer.write_bit(int(bit))
        
        return writer.finish(), huffman_code
    
    @staticmethod
    def _huffman_decode(data: bytes, count: int, huffman_code: Dict[int, str]) -> List[int]:
        """
        Huffman decoding using the provided code table.
        """
        if count == 0:
            return []
        
        # Reverse the Huffman code table for decoding
        reverse_code = {code: symbol for symbol, code in huffman_code.items()}
        
        # Handle single symbol case
        if len(huffman_code) == 1:
            symbol = list(huffman_code.keys())[0]
            return [symbol] * count
        
        reader = _BitReader(data)
        decoded = []
        current_code = ""
        
        while len(decoded) < count:
            current_code += str(reader.read_bit())
            if current_code in reverse_code:
                decoded.append(reverse_code[current_code])
                current_code = ""
        
        return decoded