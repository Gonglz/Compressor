"""
FalComC - C implementation wrapper for FalCom compressor

This module provides a Python interface to the C-based momentum predictor compressor,
offering significant performance improvements over the pure Python implementation.
"""

import os
import sys
import ctypes
import csv
import time
import numpy as np
import torch
import logging
import pickle
import blosc
from typing import Dict, Any, List, Tuple, Union, Optional, OrderedDict as OrderedDictType
from collections import OrderedDict
from omegaconf import DictConfig

from appfl.compressor.base_compressor import BaseCompressor


MAX_NDIM = 8
APPFL_PROFILE_FIELDS = [
    "timestamp",
    "client_id",
    "layer_name",
    "input_type",
    "input_device",
    "dtype",
    "shape",
    "numel",
    "is_cuda",
    "gpu_to_cpu_numpy_ms",
    "ctypes_build_ms",
    "c_compress_ms",
    "payload_copy_ms",
    "layer_total_ms",
    "model_total_ms",
    "status",
]


# C structure definitions matching momentum_compressor.h
class NDArrayC(ctypes.Structure):
    """Matches NDArray struct in C"""
    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("shape", ctypes.POINTER(ctypes.c_size_t)),
        ("ndim", ctypes.c_size_t),
        ("total_size", ctypes.c_size_t),
        ("dtype", ctypes.c_int),  # DataType enum
    ]


class CompressorConfigC(ctypes.Structure):
    """Matches CompressorConfig in C header (momentum_compressor.h)"""
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


class CompressedLayerDataC(ctypes.Structure):
    """Matches CompressedLayerData in momentum_compressor.h"""
    _fields_ = [
        ("type", ctypes.c_char * 32),
        ("codec", ctypes.c_char * 16),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("data_size", ctypes.c_size_t),
        ("bitmap", ctypes.POINTER(ctypes.c_uint8)),
        ("bitmap_size", ctypes.c_size_t),
        ("dominant_signs", ctypes.POINTER(ctypes.c_uint8)),
        ("dominant_signs_size", ctypes.c_size_t),
        ("shape", ctypes.c_size_t * MAX_NDIM),
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


# DataType enum mapping
DTYPE_MAP = {
    np.dtype('float32'): 0,  # DTYPE_FLOAT32
    np.dtype('float64'): 1,  # DTYPE_FLOAT64
    np.dtype('int32'): 2,    # DTYPE_INT32
    np.dtype('int64'): 3,    # DTYPE_INT64
    np.dtype('uint8'): 4,    # DTYPE_UINT8
}

DTYPE_MAP_REVERSE = {v: k for k, v in DTYPE_MAP.items()}


class FalComC(BaseCompressor):
    """
    C-accelerated FalCom compressor.

    This is a drop-in replacement for FalCom.py that uses compiled C code
    for performance-critical operations while maintaining full API compatibility.
    """

    def __init__(self, compressor_config: DictConfig):
        super().__init__(compressor_config)
        self.config = compressor_config

        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.FalComC")
        if not self.logger.handlers:
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)

            file_handler = logging.FileHandler(os.path.join(output_dir, "falcom_c.log"), mode="a")
            formatter = logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Load C library
        self._load_c_library()

        # Initialize C compressor
        self._init_c_compressor()

        self.logger.info(f"✅ FalComC initialized with C backend")

    def _load_c_library(self):
        """Load the compiled C shared library"""
        # Try multiple possible library locations
        lib_name = "libmomentum_compressor.so"
        possible_paths = [
            os.path.join(os.path.dirname(__file__), lib_name),
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "final", lib_name),
            os.path.join("/home/exouser/compressor/final", lib_name),
            os.path.join(os.getcwd(), lib_name),
        ]

        self.lib = None
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    self.lib = ctypes.CDLL(path)
                    self.logger.info(f"✅ Loaded C library from: {path}")
                    break
                except Exception as e:
                    self.logger.warning(f"Failed to load {path}: {e}")

        if self.lib is None:
            raise RuntimeError(f"Could not load {lib_name}. Tried paths: {possible_paths}")

        # Define function signatures
        self._setup_function_signatures()

    def _setup_function_signatures(self):
        """Setup C function signatures for ctypes"""
        # momentum_compressor_create
        self.lib.momentum_compressor_create.argtypes = [ctypes.POINTER(CompressorConfigC)]
        self.lib.momentum_compressor_create.restype = ctypes.c_void_p

        # momentum_compressor_destroy
        self.lib.momentum_compressor_destroy.argtypes = [ctypes.c_void_p]
        self.lib.momentum_compressor_destroy.restype = None

        # momentum_compressor_set_client
        self.lib.momentum_compressor_set_client.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.momentum_compressor_set_client.restype = None

        # momentum_compressor_compress_layer
        self.lib.momentum_compressor_compress_layer.argtypes = [
            ctypes.c_void_p,  # compressor
            ctypes.c_char_p,  # layer_name
            ctypes.POINTER(NDArrayC),  # array
        ]
        self.lib.momentum_compressor_compress_layer.restype = ctypes.POINTER(CompressedLayerDataC)

        # momentum_compressor_decompress_layer
        self.lib.momentum_compressor_decompress_layer.argtypes = [
            ctypes.c_void_p,  # compressor
            ctypes.POINTER(CompressedLayerDataC),  # compressed
            ctypes.c_char_p,  # client_id
            ctypes.c_char_p,  # layer_name
        ]
        self.lib.momentum_compressor_decompress_layer.restype = ctypes.POINTER(NDArrayC)

        # Resource cleanup helpers
        self.lib.ndarray_destroy.argtypes = [ctypes.POINTER(NDArrayC)]
        self.lib.ndarray_destroy.restype = None
        self.lib.compressed_layer_data_free.argtypes = [ctypes.POINTER(CompressedLayerDataC)]
        self.lib.compressed_layer_data_free.restype = None

    def _init_c_compressor(self):
        """Initialize C compressor with configuration"""
        # Extract configuration
        self.momentum_lr = getattr(self.config, 'momentum_lr', 0.07)
        self.consistency_threshold = getattr(self.config, 'consistency_threshold', 0.5)
        self.param_cutoff = getattr(self.config, 'param_cutoff', 1024)
        self.lossless_compressor = getattr(self.config, 'lossless_compressor', 'blosc')

        sz_config = self.config.get("sz_config", {})
        self.error_bounding_mode = sz_config.get("error_bounding_mode", "REL")
        self.error_bound = float(sz_config.get("error_bound", 1e-3))

        # Find SZ3 library
        ext = ".dylib" if sys.platform.startswith("darwin") else ".so"
        possible_sz3_paths = [
            os.path.expanduser("~/.appfl/.compressor/SZ3/lib/libSZ3c" + ext),
            "/eagle/lc-mpi/ZhijingYe/FLComp/SZ_NP/lib64/libSZ3c" + ext,
        ]
        sz3_lib_path = ""
        for path in possible_sz3_paths:
            if os.path.exists(path):
                sz3_lib_path = path
                break

        # Create C config - order must match C struct exactly
        c_config = CompressorConfigC()
        c_config.momentum_lr = self.momentum_lr
        c_config.consistency_threshold = self.consistency_threshold
        c_config.lossless_compressor = self.lossless_compressor.encode('utf-8')
        c_config.error_bounding_mode = self.error_bounding_mode.encode('utf-8')
        c_config.error_bound = self.error_bound
        c_config.sz3_lib_path = sz3_lib_path.encode('utf-8')
        c_config.param_count_threshold = self.param_cutoff
        c_config.max_history_length = 3

        # Create C compressor instance
        self.c_compressor = self.lib.momentum_compressor_create(ctypes.byref(c_config))
        if not self.c_compressor:
            raise RuntimeError("Failed to create C compressor instance")

        # Current client ID
        self._current_client_id = None

    def _numpy_to_c_array(self, arr: np.ndarray) -> NDArrayC:
        """Convert numpy array to C NDArray structure"""
        c_array = NDArrayC()

        # Get dtype code
        if arr.dtype not in DTYPE_MAP:
            raise ValueError(f"Unsupported dtype: {arr.dtype}")

        # Ensure contiguous
        arr = np.ascontiguousarray(arr)

        c_array.dtype = DTYPE_MAP[arr.dtype]
        c_array.ndim = len(arr.shape)
        if c_array.ndim < 1 or c_array.ndim > MAX_NDIM:
            raise ValueError(f"Unsupported ndim: {c_array.ndim}")
        c_array.total_size = arr.size

        # Allocate and copy shape
        shape_buffer = (ctypes.c_size_t * c_array.ndim)(*arr.shape)
        c_array.shape = shape_buffer

        # Point to numpy data buffer
        c_array.data = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        # Keep Python-owned buffers alive while C reads this struct.
        c_array._shape_buffer = shape_buffer
        c_array._array_ref = arr

        return c_array

    def _c_array_to_numpy(self, c_array: ctypes.POINTER(NDArrayC)) -> np.ndarray:
        """Convert C NDArray to numpy array"""
        if not c_array:
            return None

        arr = c_array.contents
        dtype = DTYPE_MAP_REVERSE[arr.dtype]
        shape = tuple(arr.shape[i] for i in range(arr.ndim))

        # Create numpy array from C memory
        buffer = ctypes.cast(arr.data, ctypes.POINTER(ctypes.c_uint8 * (arr.total_size * dtype.itemsize)))
        np_arr = np.frombuffer(buffer.contents, dtype=dtype).reshape(shape)

        # Copy to ensure memory safety
        return np_arr.copy()

    @staticmethod
    def _decode_c_string(raw: bytes) -> str:
        """Decode a fixed-width C string field."""
        return bytes(raw).split(b"\0", 1)[0].decode("utf-8", errors="replace")

    @staticmethod
    def _ptr_to_bytes(ptr: ctypes.POINTER(ctypes.c_uint8), size: int) -> bytes:
        """Copy a C uint8_t buffer into Python-owned bytes."""
        if not ptr or size == 0:
            return b""
        return ctypes.string_at(ptr, size)

    @staticmethod
    def _bytes_to_ptr(blob: bytes) -> Tuple[Any, ctypes.POINTER(ctypes.c_uint8)]:
        """Create a ctypes buffer and pointer for C-owned struct reconstruction."""
        if not blob:
            return None, ctypes.POINTER(ctypes.c_uint8)()
        buf = (ctypes.c_uint8 * len(blob)).from_buffer_copy(blob)
        return buf, ctypes.cast(buf, ctypes.POINTER(ctypes.c_uint8))

    @staticmethod
    def _assign_c_string(target: CompressedLayerDataC, field: str, value: str, max_len: int):
        """Assign a Python string into a fixed-width char array field."""
        encoded = value.encode("utf-8")[:max_len - 1]
        setattr(target, field, encoded)

    def _compressed_layer_to_dict(self, compressed_ptr: ctypes.POINTER(CompressedLayerDataC)) -> Dict[str, Any]:
        """Copy a C CompressedLayerData into a pickle-safe Python dictionary."""
        compressed = compressed_ptr.contents
        return {
            "codec": "c_struct",
            "c_type": self._decode_c_string(compressed.type),
            "c_codec": self._decode_c_string(compressed.codec),
            "data": self._ptr_to_bytes(compressed.data, compressed.data_size),
            "bitmap": self._ptr_to_bytes(compressed.bitmap, compressed.bitmap_size),
            "dominant_signs": self._ptr_to_bytes(compressed.dominant_signs, compressed.dominant_signs_size),
            "shape": tuple(compressed.shape[i] for i in range(compressed.ndim)),
            "ndim": int(compressed.ndim),
            "original_dtype": self._decode_c_string(compressed.original_dtype),
            "stored_dtype": self._decode_c_string(compressed.stored_dtype),
            "step": int(compressed.step),
            "num_predicted_kernels": int(compressed.num_predicted_kernels),
            "prediction_ratio": float(compressed.prediction_ratio),
            "sign_mismatch_ratio": float(compressed.sign_mismatch_ratio),
            "current_mean": float(compressed.current_mean),
            "current_std": float(compressed.current_std),
            "prev_mean": float(compressed.prev_mean),
            "prev_std": float(compressed.prev_std),
            "global_min": float(compressed.global_min),
            "global_max": float(compressed.global_max),
            "breakdown_stats_time": float(compressed.breakdown_stats_time),
            "breakdown_normalize_time": float(compressed.breakdown_normalize_time),
            "breakdown_consistency_time": float(compressed.breakdown_consistency_time),
            "breakdown_prediction_time": float(compressed.breakdown_prediction_time),
            "breakdown_residual_compress_time": float(compressed.breakdown_residual_compress_time),
            "breakdown_bitmap_compress_time": float(compressed.breakdown_bitmap_compress_time),
            "breakdown_metadata_time": float(compressed.breakdown_metadata_time),
            "breakdown_total_time": float(compressed.breakdown_total_time),
        }

    def _compressed_dict_to_c(self, layer_data: Dict[str, Any]) -> Tuple[CompressedLayerDataC, List[Any]]:
        """Rebuild a C CompressedLayerData struct from a Python dictionary."""
        c_layer = CompressedLayerDataC()
        keepalive = []

        self._assign_c_string(c_layer, "type", layer_data["c_type"], 32)
        self._assign_c_string(c_layer, "codec", layer_data["c_codec"], 16)
        self._assign_c_string(c_layer, "original_dtype", layer_data["original_dtype"], 16)
        self._assign_c_string(c_layer, "stored_dtype", layer_data["stored_dtype"], 16)

        data_buf, c_layer.data = self._bytes_to_ptr(layer_data.get("data", b""))
        bitmap_buf, c_layer.bitmap = self._bytes_to_ptr(layer_data.get("bitmap", b""))
        signs_buf, c_layer.dominant_signs = self._bytes_to_ptr(layer_data.get("dominant_signs", b""))
        keepalive.extend(buf for buf in (data_buf, bitmap_buf, signs_buf) if buf is not None)

        c_layer.data_size = len(layer_data.get("data", b""))
        c_layer.bitmap_size = len(layer_data.get("bitmap", b""))
        c_layer.dominant_signs_size = len(layer_data.get("dominant_signs", b""))

        shape = tuple(layer_data.get("shape", ()))
        if len(shape) < 1 or len(shape) > MAX_NDIM:
            raise ValueError(f"Invalid compressed layer ndim: {len(shape)}")
        for i, dim in enumerate(shape):
            c_layer.shape[i] = int(dim)
        c_layer.ndim = len(shape)

        c_layer.step = int(layer_data.get("step", 0))
        c_layer.num_predicted_kernels = int(layer_data.get("num_predicted_kernels", 0))
        c_layer.prediction_ratio = float(layer_data.get("prediction_ratio", 0.0))
        c_layer.sign_mismatch_ratio = float(layer_data.get("sign_mismatch_ratio", 0.0))
        c_layer.current_mean = float(layer_data.get("current_mean", 0.0))
        c_layer.current_std = float(layer_data.get("current_std", 0.0))
        c_layer.prev_mean = float(layer_data.get("prev_mean", 0.0))
        c_layer.prev_std = float(layer_data.get("prev_std", 0.0))
        c_layer.global_min = float(layer_data.get("global_min", 0.0))
        c_layer.global_max = float(layer_data.get("global_max", 0.0))
        c_layer.breakdown_stats_time = float(layer_data.get("breakdown_stats_time", 0.0))
        c_layer.breakdown_normalize_time = float(layer_data.get("breakdown_normalize_time", 0.0))
        c_layer.breakdown_consistency_time = float(layer_data.get("breakdown_consistency_time", 0.0))
        c_layer.breakdown_prediction_time = float(layer_data.get("breakdown_prediction_time", 0.0))
        c_layer.breakdown_residual_compress_time = float(layer_data.get("breakdown_residual_compress_time", 0.0))
        c_layer.breakdown_bitmap_compress_time = float(layer_data.get("breakdown_bitmap_compress_time", 0.0))
        c_layer.breakdown_metadata_time = float(layer_data.get("breakdown_metadata_time", 0.0))
        c_layer.breakdown_total_time = float(layer_data.get("breakdown_total_time", 0.0))

        return c_layer, keepalive

    @staticmethod
    def _appfl_profile_enabled() -> bool:
        value = os.environ.get("FALCOM_APPFL_PROFILE", "")
        return value not in ("", "0", "false", "False", "FALSE", "off", "OFF")

    @staticmethod
    def _profile_csv_path() -> str:
        return os.environ.get("FALCOM_APPFL_PROFILE_CSV", "./falcom_appfl_profile.csv")

    @staticmethod
    def _cuda_synchronize_if_needed(obj: Any):
        if hasattr(obj, "is_cuda") and obj.is_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    @staticmethod
    def _profile_input_metadata(obj: Any) -> Dict[str, Any]:
        if hasattr(obj, "detach"):
            shape = tuple(int(x) for x in obj.shape)
            return {
                "input_type": type(obj).__name__,
                "input_device": str(obj.device),
                "dtype": str(obj.dtype),
                "shape": "x".join(str(x) for x in shape),
                "numel": int(obj.numel()),
                "is_cuda": bool(obj.is_cuda),
            }
        if isinstance(obj, np.ndarray):
            return {
                "input_type": type(obj).__name__,
                "input_device": "cpu",
                "dtype": str(obj.dtype),
                "shape": "x".join(str(x) for x in obj.shape),
                "numel": int(obj.size),
                "is_cuda": False,
            }
        return {
            "input_type": type(obj).__name__,
            "input_device": "unknown",
            "dtype": "unknown",
            "shape": "",
            "numel": 0,
            "is_cuda": False,
        }

    @staticmethod
    def _append_profile_rows(rows: List[Dict[str, Any]]):
        if not rows:
            return
        path = FalComC._profile_csv_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=APPFL_PROFILE_FIELDS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def set_client_id(self, client_id: str):
        """Set current client ID for per-client state management"""
        self._current_client_id = client_id
        if self.c_compressor:
            self.lib.momentum_compressor_set_client(
                self.c_compressor,
                client_id.encode('utf-8')
            )

    def compress_model(self,
                       model: Union[Dict, OrderedDictType[str, np.ndarray], List],
                       batched: bool = False,
                       client_id: Optional[str] = None) -> bytes:
        """
        Compress model using C implementation.

        Args:
            model: OrderedDict mapping layer names to numpy arrays
            batched: Whether to handle batched compression
            client_id: Client ID for per-client state management

        Returns:
            Compressed bytes
        """
        # C compression requires an explicit client context.
        if client_id is not None:
            self.set_client_id(client_id)
        elif self._current_client_id is None:
            self.set_client_id("Client1")

        # Handle batched mode
        if batched:
            if isinstance(model, list):
                return pickle.dumps([self.compress_model(m) for m in model])
            if isinstance(model, (dict, OrderedDict)):
                out = OrderedDict()
                for k, v in model.items():
                    out[k] = self.compress_model(v)
                return pickle.dumps(out)

        compressed_layers = {}
        profile_enabled = self._appfl_profile_enabled()
        profile_rows: List[Dict[str, Any]] = []
        model_start = time.perf_counter()

        for layer_name, array in model.items():
            layer_start = time.perf_counter()
            profile_meta = self._profile_input_metadata(array) if profile_enabled else {}
            gpu_to_cpu_numpy_ms = 0.0
            ctypes_build_ms = 0.0
            c_compress_ms = 0.0
            payload_copy_ms = 0.0
            status = "ok"

            # Convert PyTorch tensors to NumPy if needed
            if hasattr(array, 'detach'):  # torch.Tensor
                tensor_ref = array
                convert_start = time.perf_counter()
                self._cuda_synchronize_if_needed(tensor_ref)
                array = array.detach().cpu().numpy()
                self._cuda_synchronize_if_needed(tensor_ref)
                gpu_to_cpu_numpy_ms = (time.perf_counter() - convert_start) * 1000.0

            # Try C compression first
            try:
                # Convert to C array
                ctypes_start = time.perf_counter()
                c_array = self._numpy_to_c_array(array)
                ctypes_build_ms = (time.perf_counter() - ctypes_start) * 1000.0

                # Compress using C
                c_start = time.perf_counter()
                compressed_ptr = self.lib.momentum_compressor_compress_layer(
                    self.c_compressor,
                    layer_name.encode('utf-8'),
                    ctypes.byref(c_array)
                )
                c_compress_ms = (time.perf_counter() - c_start) * 1000.0

                if compressed_ptr:
                    try:
                        payload_start = time.perf_counter()
                        compressed_layers[layer_name] = self._compressed_layer_to_dict(compressed_ptr)
                        payload_copy_ms = (time.perf_counter() - payload_start) * 1000.0
                    finally:
                        self.lib.compressed_layer_data_free(compressed_ptr)
                else:
                    # Fall back to pickle for unsupported layers
                    status = "pickle_fallback"
                    compressed_layers[layer_name] = {
                        'codec': 'pickle',
                        'data': pickle.dumps(array)
                    }
            except Exception as e:
                # Fall back to pickle on any error
                status = "exception_fallback"
                self.logger.warning(f"C compression failed for {layer_name}, using pickle: {e}")
                compressed_layers[layer_name] = {
                    'codec': 'pickle',
                    'data': pickle.dumps(array)
                }
            finally:
                if profile_enabled:
                    layer_total_ms = (time.perf_counter() - layer_start) * 1000.0
                    row = {
                        "timestamp": time.time(),
                        "client_id": self._current_client_id,
                        "layer_name": layer_name,
                        **profile_meta,
                        "gpu_to_cpu_numpy_ms": f"{gpu_to_cpu_numpy_ms:.6f}",
                        "ctypes_build_ms": f"{ctypes_build_ms:.6f}",
                        "c_compress_ms": f"{c_compress_ms:.6f}",
                        "payload_copy_ms": f"{payload_copy_ms:.6f}",
                        "layer_total_ms": f"{layer_total_ms:.6f}",
                        "model_total_ms": "",
                        "status": status,
                    }
                    profile_rows.append(row)

        # Serialize compressed layers
        result = pickle.dumps(compressed_layers)
        if profile_enabled:
            model_total_ms = (time.perf_counter() - model_start) * 1000.0
            for row in profile_rows:
                row["model_total_ms"] = f"{model_total_ms:.6f}"
            self._append_profile_rows(profile_rows)
        return result

    def decompress_model(self,
                         compressed_data: bytes,
                         model: Any = None,
                         batched: bool = False,
                         client_id: Optional[str] = None) -> Union[OrderedDictType[str, np.ndarray], List]:
        """
        Decompress model using C implementation.

        Args:
            compressed_data: Compressed bytes
            model: Optional model template (unused in C version)
            batched: Whether to handle batched decompression
            client_id: Client ID for per-client state management

        Returns:
            OrderedDict mapping layer names to numpy arrays
        """
        # C decompression requires a client ID for history and prediction memory.
        if client_id is not None:
            self.set_client_id(client_id)
        elif self._current_client_id is None:
            self.set_client_id("Client1")

        # Handle batched mode
        if batched:
            data = pickle.loads(compressed_data)
            if isinstance(data, list):
                return [self.decompress_model(d) for d in data]
            if isinstance(data, (dict, OrderedDict)):
                out = OrderedDict()
                for k, v in data.items():
                    out[k] = self.decompress_model(v)
                return out
        # Deserialize
        compressed_layers = pickle.loads(compressed_data)

        decompressed_model = OrderedDict()

        for layer_name, layer_data in compressed_layers.items():
            # Check codec type
            if isinstance(layer_data, dict):
                codec = layer_data.get('codec', 'c_struct')
                compressed_bytes = layer_data.get('data', b'')
            else:
                # Legacy bytes lack the C metadata required by the current ABI.
                codec = 'legacy_c_bytes'
                compressed_bytes = layer_data

            if codec == 'pickle':
                # Decompress with pickle
                array = pickle.loads(compressed_bytes)
                decompressed_model[layer_name] = torch.from_numpy(array)
            else:
                # Decompress using C
                try:
                    if codec != 'c_struct':
                        raise ValueError(f"Unsupported C layer codec format: {codec}")
                    c_layer, keepalive = self._compressed_dict_to_c(layer_data)
                    c_array_ptr = self.lib.momentum_compressor_decompress_layer(
                        self.c_compressor,
                        ctypes.byref(c_layer),
                        self._current_client_id.encode('utf-8'),
                        layer_name.encode('utf-8')
                    )

                    if c_array_ptr:
                        try:
                            # Convert to numpy
                            array = self._c_array_to_numpy(c_array_ptr)
                            # Convert back to torch.Tensor to match expected format
                            decompressed_model[layer_name] = torch.from_numpy(array)
                        finally:
                            self.lib.ndarray_destroy(c_array_ptr)
                    else:
                        self.logger.error(f"Failed to decompress layer: {layer_name}")
                except Exception as e:
                    self.logger.error(f"C decompression failed for {layer_name}: {e}")

        return decompressed_model

    def __del__(self):
        """Cleanup C resources"""
        if hasattr(self, 'c_compressor') and self.c_compressor:
            self.lib.momentum_compressor_destroy(self.c_compressor)
