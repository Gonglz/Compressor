import numpy as np
import torch
import pickle
import logging
import os
import sys
import pathlib
import hashlib
import blosc
from omegaconf import DictConfig
from collections import OrderedDict
from typing import Union, List, Dict, Any, Optional, Tuple
import zstd

from appfl.compressor.base_compressor import BaseCompressor
from appfl.compressor import pysz

FLOAT_DTYPES = (np.float32, np.float64)


def _is_float_dtype(dt) -> bool:
    try:
        return np.issubdtype(dt, np.floating)
    except Exception:
        return str(dt) in ["float32", "float64"]


def _conv_layer_key(shape) -> str:
    return f"conv_{tuple(shape)}"


class FalCom(BaseCompressor):

    def __init__(self, compressor_config: DictConfig):
        super().__init__(compressor_config)
        self.config = compressor_config

        # Logging
        self.logger = logging.getLogger(f"{__name__}.FalCom")
        if not self.logger.handlers:
            # Create output directory if it doesn't exist
            output_dir = "./output"
            os.makedirs(output_dir, exist_ok=True)
            
            file_handler = logging.FileHandler(os.path.join(output_dir, "falcom.log"), mode="a")
            formatter = logging.Formatter('[%(asctime)s] %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Momentum predictor params
        self.momentum_lr = compressor_config.get("momentum_lr", 0.07)
        self.consistency_threshold = compressor_config.get("consistency_threshold", 0.5)
        
        # 无损压缩器选择 (zstd, blosc, pickle)
        self.lossless_compressor = getattr(compressor_config, 'lossless_compressor', 'zstd')

        # self.logger.info(f"Initializing FalCom with momentum_lr={self.momentum_lr}, "
        #                  f"consistency_threshold={self.consistency_threshold}, "
        #                  f"lossless_compressor={self.lossless_compressor}")

        # SZ3 params
        self.sz_config = compressor_config.get("sz_config", {})
        self.error_bounding_mode = self.sz_config.get("error_bounding_mode", "REL")
        self.error_bound = float(self.sz_config.get("error_bound", 1))
        self.sz_error_mode_dict = {
            "ABS": 0, "REL": 1, "ABS_AND_REL": 2, "ABS_OR_REL": 3,
            "PSNR": 4, "NORM": 5, "PW_REL": 10,
        }
        ext = ".dylib" if sys.platform.startswith("darwin") else ".so"
        # Try to find SZ3 library in multiple locations
        possible_paths = [
            os.path.expanduser("~/.appfl/.compressor/SZ3/lib/libSZ3c" + ext),
            "/eagle/lc-mpi/ZhijingYe/FLComp/SZ_NP/lib64/libSZ3c" + ext,
        ]
        self.compressor_lib_path = None
        for path in possible_paths:
            if os.path.exists(path):
                self.compressor_lib_path = path
                break
        if not self.compressor_lib_path:
            self.compressor_lib_path = possible_paths[0]  # Default to user home directory
        # self.logger.info(f"Using SZ3 compressor with error_bound={self.error_bound}, "
        #                  f"error_bounding_mode={self.error_bounding_mode}")

        # Only large 'weight' use lossy
        self.param_count_threshold = getattr(compressor_config, 'param_count_threshold', 1024)

        # States
        self.gradient_history: Dict[str, Dict[str, List[np.ndarray]]] = {}
        self.prediction_memory: Dict[str, Dict[str, np.ndarray]] = {}
        self.step_count: Dict[str, int] = {}
        self._current_client_id: Optional[str] = None

        # Stats
        self.compression_stats = {
            "total_compressions": 0,
            "prediction_ratios": [],
            "sign_mismatch_ratios": []
        }

    # -------------------- helpers --------------------
    def _dtype_to_numpy(self, dtype_input):
        if isinstance(dtype_input, str):
            try:
                return np.dtype(dtype_input)
            except Exception:
                dtype_map = {
                    "float32": np.float32, "float64": np.float64,
                    "<class 'numpy.float32'>": np.float32, "<class 'numpy.float64'>": np.float64,
                }
                for k, v in dtype_map.items():
                    if k in dtype_input:
                        return v
                return np.float32
        else:
            return np.dtype(dtype_input)

    def _lossless_compress(self, arr: np.ndarray) -> Tuple[bytes, str]:
        """Apply lossless compression to the array."""
        if self.lossless_compressor == 'zstd':
            data_bytes = pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)
            compressed = zstd.compress(data_bytes, 10)
            return compressed, 'zstd'
        
        elif self.lossless_compressor == 'blosc':
            data_bytes = pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)
            compressed = blosc.compress(data_bytes, typesize=4)
            return compressed, 'blosc'
        
        else:  # pickle
            data_bytes = pickle.dumps(arr, protocol=pickle.HIGHEST_PROTOCOL)
            return data_bytes, 'pickle'

    def _lossless_decompress(self, compressed_data: bytes, codec: str) -> np.ndarray:
        """Apply lossless decompression based on the codec."""
        if codec == 'zstd':
            data_bytes = zstd.decompress(compressed_data)
            return pickle.loads(data_bytes)
        
        elif codec == 'blosc':
            data_bytes = blosc.decompress(compressed_data)
            return pickle.loads(data_bytes)
        
        else:  # pickle
            return pickle.loads(compressed_data)

    def _should_lossy(self, layer_name: str, arr: np.ndarray) -> bool:
        return ("weight" in layer_name) and \
               (arr.size > self.param_count_threshold) and \
               (arr.dtype in FLOAT_DTYPES)

    # -------------------- SZ3 wrappers --------------------
    def _compress_with_sz3(self, data: np.ndarray, error_mode, abs_bound, rel_bound ) -> Tuple[bytes, float]:
        try:
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            if data.dtype not in FLOAT_DTYPES:
                lossless_data, codec = self._lossless_compress(data)
                return lossless_data, 1.0
            if not data.flags['C_CONTIGUOUS']:
                data = np.ascontiguousarray(data)
            if not os.path.exists(self.compressor_lib_path):
                self.logger.warning(f"SZ3 lib not found at {self.compressor_lib_path}; fallback to lossless")
                lossless_data, codec = self._lossless_compress(data)
                return lossless_data, 1.0
            compressor = pysz.SZ(szpath=self.compressor_lib_path)
            compressed_arr, cmp_ratio = compressor.compress(
                data=data.flatten(),
                eb_mode=self.sz_error_mode_dict.get(error_mode, 0),
                eb_abs=abs_bound,
                eb_rel=rel_bound,
                eb_pwr=0,
            )

            return compressed_arr.tobytes(), cmp_ratio
        except Exception as e:
            self.logger.error(f"SZ3 compression failed: {e}")
            self.logger.error(f"Data dtype: {data.dtype}, shape: {data.shape}")
            self.logger.error(f"SZ3 library path: {self.compressor_lib_path}")
            lossless_data, codec = self._lossless_compress(data)
            return lossless_data, 1.0

    def _decompress_with_sz3(self, compressed_data: bytes, original_shape: Tuple[int, ...], original_dtype) -> np.ndarray:
        if not os.path.exists(self.compressor_lib_path):
            raise RuntimeError("SZ3 library not found; cannot SZ3-decompress.")
        compressor = pysz.SZ(szpath=self.compressor_lib_path)
        cmp_data = np.frombuffer(compressed_data, dtype=np.uint8)
        odt = self._dtype_to_numpy(original_dtype)
        decompressed_arr = compressor.decompress(
            data_cmpr=cmp_data,
            original_shape=(int(np.prod(original_shape)),),
            original_dtype=odt,
        )
        return decompressed_arr.reshape(original_shape)

    # -------------------- sign/consistency helpers --------------------
    def _compute_normalized_sign_consistency(self, kernel_signs: np.ndarray) -> float:
        positives = np.sum(kernel_signs > 0)
        negatives = np.sum(kernel_signs < 0)
        zeros = np.sum(kernel_signs == 0)
        majority_count = (positives + zeros) if positives >= negatives else (negatives + zeros)
        total = kernel_signs.size if kernel_signs.size > 0 else 1
        return float(((majority_count / total) - 0.5) * 2)

    def _get_dominant_sign(self, kernel_signs: np.ndarray) -> int:
        return 1 if np.sum(kernel_signs > 0) >= np.sum(kernel_signs < 0) else -1

    # -------------------- core per-layer compression --------------------
    def _create_compressed_data(self, gradient: np.ndarray, client_id: str, layer_name: str) -> Dict[str, Any]:
        grad_np = gradient.detach().cpu().numpy() if isinstance(gradient, torch.Tensor) else np.asarray(gradient)

        # init states
        if client_id not in self.gradient_history:
            self.gradient_history[client_id] = {}
            self.prediction_memory[client_id] = {}
            self.step_count[client_id] = {}
            # self.logger.info(f"Initialized new client state for {client_id}")
        if layer_name not in self.gradient_history[client_id]:
            self.gradient_history[client_id][layer_name] = []

        if client_id not in self.step_count:
            self.step_count[client_id] = {}
        if layer_name not in self.step_count[client_id]:
            self.step_count[client_id][layer_name] = 0
            
        self.step_count[client_id][layer_name] += 1
        current_step = self.step_count[client_id][layer_name]

        # Step 1: direct + codec 按 should_lossy
        if current_step == 1:
            self.gradient_history[client_id][layer_name].append(grad_np.copy())
            if self._should_lossy(layer_name, grad_np):
                data_bytes, _ = self._compress_with_sz3(grad_np, error_mode="REL", abs_bound=0, rel_bound=self.error_bound)
                codec = "sz3"
                stored_dtype = str(grad_np.dtype)
            else:
                data_bytes, codec = self._lossless_compress(grad_np)
                stored_dtype = str(grad_np.dtype)
            # self.logger.info(f"Client {client_id} layer {layer_name} step {current_step}: Direct compression (codec={codec})")
            return {
                "type": "direct",
                "codec": codec,
                "data": data_bytes,
                "step": current_step,
                "shape": grad_np.shape,
                "original_dtype": str(grad_np.dtype),
                "stored_dtype": stored_dtype
            }

        # Step >=2
        prev_reconstructed_grad = self.gradient_history[client_id][layer_name][-1]
        current_grad = grad_np

        if (current_grad.ndim == 4) and self._should_lossy(layer_name, current_grad):
            compressed_result = self._compress_conv_layer(current_grad, prev_reconstructed_grad, client_id, current_step)
        else:
            compressed_result = self._compress_generic_layer(current_grad, client_id, current_step, layer_name=layer_name)

        reconstructed_grad = self._simulate_reconstruction(compressed_result, current_grad, prev_reconstructed_grad, client_id)

        # if "prediction_ratio" in compressed_result:
        #     self.logger.info(f"Client {client_id} layer {layer_name} step {current_step}: "
        #                      f"Prediction ratio={compressed_result['prediction_ratio']:.3f}, "
        #                      f"Sign mismatch ratio={compressed_result.get('sign_mismatch_ratio', 0):.3f}")

        self.gradient_history[client_id][layer_name].append(reconstructed_grad)
        if len(self.gradient_history[client_id][layer_name]) > 3:
            self.gradient_history[client_id][layer_name].pop(0)

        return compressed_result
    def _simulate_reconstruction(self, compressed_result: dict, original_grad: np.ndarray,
                             prev_reconstructed: np.ndarray, client_id: str) -> np.ndarray:
        if compressed_result["type"] in ("direct", "direct_generic"):
            return original_grad.copy()

        if compressed_result["type"] == "momentum_predicted":
            shape = original_grad.shape
            out_ch, in_ch, h, w = shape

            bm_meta = compressed_result["bitmap_metadata"]
            bm_packed_bytes = zstd.decompress(compressed_result["bitmap"])
            bm_packed = np.frombuffer(bm_packed_bytes, dtype=np.uint8)
            bm_flat = np.unpackbits(bm_packed)[:bm_meta["original_size"]].astype(bool)
            bitmap = bm_flat.reshape(bm_meta["original_shape"])

            stored_dt = self._dtype_to_numpy(compressed_result.get("original_dtype", "float32"))
            if _is_float_dtype(stored_dt):
                residual_data = self._decompress_with_sz3(compressed_result["data"], shape, stored_dt)
            else:
                residual_data = pickle.loads(compressed_result["data"])

            dom_meta = compressed_result["dominant_signs_metadata"]
            dom_bytes = compressed_result["dominant_signs"]
            if compressed_result["num_predicted_kernels"] > 0 and len(dom_bytes) > 0:
                dom_packed_bytes = zstd.decompress(dom_bytes)
                dom_packed = np.frombuffer(dom_packed_bytes, dtype=np.uint8)
                dom_bits = np.unpackbits(dom_packed)[:dom_meta["original_size"]].astype(bool)
                dominant_signs = np.ones(dom_meta["original_size"], dtype=np.int8)
                dominant_signs[~dom_bits] = -1
            else:
                dominant_signs = np.array([], dtype=np.int8)

            reconstructed = residual_data.copy()

            cur = compressed_result.get("current_grad_stats", {})
            prv = compressed_result.get("prev_grad_stats", {})
            current_mean = cur.get("mean", 0.0)
            current_std = cur.get("std", 1.0)
            prev_mean = prv.get("mean", 0.0)
            prev_std = prv.get("std", 1.0)

            abs_prev_global = np.abs(prev_reconstructed)
            prev_normalized = abs_prev_global - prev_mean
            if prev_std > 1e-8:
                prev_normalized = prev_normalized / prev_std

            layer_key = _conv_layer_key(shape)
            if layer_key not in self.prediction_memory[client_id]:
                self.prediction_memory[client_id][layer_key] = np.zeros(shape)
            layer_memory = self.prediction_memory[client_id][layer_key]

            pred_mask = bitmap
            num_pred = int(np.count_nonzero(pred_mask))

            if num_pred > 0:
                prev_norm_sel = prev_normalized[pred_mask]
                memory_sel = layer_memory[pred_mask]
                pred_norm = (1 - self.momentum_lr) * memory_sel + self.momentum_lr * prev_norm_sel
                layer_memory[pred_mask] = pred_norm

                if current_std > 1e-8:
                    abs_pred = pred_norm * current_std + current_mean
                else:
                    abs_pred = pred_norm + current_mean
                abs_pred = np.abs(abs_pred)

                if len(dominant_signs) == num_pred:
                    dominant_sign_arr = dominant_signs.astype(np.float32)
                else:
                    dominant_sign_arr = np.ones(num_pred, dtype=np.float32)
                predicted_kernel = dominant_sign_arr[:, None, None] * abs_pred
                reconstructed[pred_mask] = residual_data[pred_mask] + predicted_kernel

                self.prediction_memory[client_id][layer_key] = layer_memory

            return reconstructed

        return original_grad.copy()

    def _compress_conv_layer(self, current_grad: np.ndarray, prev_grad: np.ndarray,
                         client_id: str, current_step: int) -> Dict[str, Any]:
        abs_err = self.error_bound * (np.max(current_grad) - np.min(current_grad))

        out_ch, in_ch, h, w = current_grad.shape
        kernel_size = h * w

        abs_current = np.abs(current_grad)
        abs_prev = np.abs(prev_grad)
        current_mean = float(np.mean(abs_current))
        current_std = float(np.std(abs_current))
        prev_mean = float(np.mean(abs_prev))
        prev_std = float(np.std(abs_prev))
        prev_normalized = abs_prev - prev_mean
        if prev_std > 1e-8:
            prev_normalized = prev_normalized / prev_std

        layer_key = _conv_layer_key(current_grad.shape)
        if layer_key not in self.prediction_memory[client_id]:
            self.prediction_memory[client_id][layer_key] = np.zeros_like(current_grad)
        layer_memory = self.prediction_memory[client_id][layer_key]

        kernel_signs_full = np.sign(current_grad)
        positives = np.sum(kernel_signs_full > 0, axis=(2, 3))
        negatives = np.sum(kernel_signs_full < 0, axis=(2, 3))
        zeros = np.sum(kernel_signs_full == 0, axis=(2, 3))

        majority_counts = np.where(positives >= negatives, positives + zeros, negatives + zeros)
        consistencies = ((majority_counts / kernel_size) - 0.5) * 2

        prediction_bitmap = consistencies >= self.consistency_threshold
        dominant_signs = np.where(positives >= negatives, 1, -1).astype(np.int8)

        residual_data = current_grad.copy()
        global_min = float(np.min(current_grad))
        global_max = float(np.max(current_grad))

        total_kernels = out_ch * in_ch
        predicted_kernels = int(np.count_nonzero(prediction_bitmap))
        sign_mismatch_count = 0
        total_predicted_elements = 0

        if predicted_kernels > 0:
            pred_mask = prediction_bitmap

            prev_norm_sel = prev_normalized[pred_mask]
            memory_sel = layer_memory[pred_mask]
            pred_norm = (1 - self.momentum_lr) * memory_sel + self.momentum_lr * prev_norm_sel
            layer_memory[pred_mask] = pred_norm

            if current_std > 1e-8:
                abs_pred = pred_norm * current_std + current_mean
            else:
                abs_pred = pred_norm + current_mean
            abs_pred = np.abs(abs_pred)

            dominant_sign_sel = dominant_signs[pred_mask].astype(np.float32)
            predicted_kernel = dominant_sign_sel[:, None, None] * abs_pred

            residual_block = current_grad[pred_mask] - predicted_kernel
            np.clip(residual_block, global_min, global_max, out=residual_block)
            residual_data[pred_mask] = residual_block

            pred_kernel_signs = np.sign(predicted_kernel)
            sign_matches = pred_kernel_signs == np.sign(current_grad[pred_mask])
            total_predicted_elements = sign_matches.size
            sign_mismatch_count = int(total_predicted_elements - np.count_nonzero(sign_matches))

            predicted_dominant_signs = (dominant_sign_sel > 0)
        else:
            predicted_dominant_signs = np.array([], dtype=bool)

        self.prediction_memory[client_id][layer_key] = layer_memory

        compression_data = residual_data.copy()
        if compression_data.dtype not in FLOAT_DTYPES:
            compression_data = compression_data.astype(np.float32)

        prediction_ratio = predicted_kernels / max(total_kernels, 1)
        sign_mismatch_ratio = sign_mismatch_count / max(total_predicted_elements, 1)

        compressed_residuals, res_cmp_ratio = self._compress_with_sz3(
            compression_data, error_mode="ABS", abs_bound=abs_err, rel_bound=0
        )

        bm_flat = prediction_bitmap.flatten()
        padding = (8 - len(bm_flat) % 8) % 8
        if padding > 0:
            bm_flat = np.append(bm_flat, np.zeros(padding, dtype=bool))
        bm_packed = np.packbits(bm_flat.astype(np.uint8))
        compressed_bitmap = zstd.compress(bm_packed.tobytes())
        bitmap_metadata = {
            "original_shape": prediction_bitmap.shape,
            "original_size": int(prediction_bitmap.size),
            "packed_size": int(len(bm_packed))
        }

        predicted_dominant_signs_list = predicted_dominant_signs.tolist()
        if predicted_dominant_signs_list:
            ds_bits = np.array(predicted_dominant_signs_list, dtype=bool)
            padding = (8 - len(ds_bits) % 8) % 8
            if padding > 0:
                ds_bits = np.append(ds_bits, np.zeros(padding, dtype=bool))
            ds_packed = np.packbits(ds_bits.astype(np.uint8))
            compressed_dominant_signs = zstd.compress(ds_packed.tobytes())
            dominant_signs_metadata = {
                "original_size": int(len(predicted_dominant_signs_list)),
                "packed_size": int(len(ds_packed))
            }
        else:
            compressed_dominant_signs = b''
            dominant_signs_metadata = {"original_size": 0, "packed_size": 0}

        return {
            "type": "momentum_predicted",
            "data": compressed_residuals,
            "bitmap": compressed_bitmap,
            "bitmap_metadata": bitmap_metadata,
            "dominant_signs": compressed_dominant_signs,
            "dominant_signs_metadata": dominant_signs_metadata,
            "num_predicted_kernels": len(predicted_dominant_signs_list),
            "step": current_step,
            "shape": current_grad.shape,
            "original_dtype": "float32" if compression_data.dtype == np.float32 else "float64",
            "prediction_ratio": prediction_ratio,
            "sign_mismatch_ratio": sign_mismatch_ratio,
            "current_grad_stats": {"mean": current_mean, "std": current_std},
            "prev_grad_stats": {"mean": prev_mean, "std": prev_std},
            "stats": {
                "total_kernels": total_kernels,
                "predicted_kernels": predicted_kernels,
                "sign_mismatch_count": sign_mismatch_count
            }
        }

    def _compress_generic_layer(self, current_grad: np.ndarray,
                                client_id: str, current_step: int,
                                layer_name: str) -> Dict[str, Any]:
        self.logger.debug(f"Using direct compression for generic layer with shape {current_grad.shape}")
        arr = np.asarray(current_grad)

        if self._should_lossy(layer_name, arr):
            stored_dtype = str(np.dtype(arr.dtype))
            data_bytes, cmp_ratio = self._compress_with_sz3(arr, error_mode="REL", abs_bound=0, rel_bound=self.error_bound)
            codec = "sz3"
        else:
            stored_dtype = str(np.dtype(arr.dtype))
            data_bytes, codec = self._lossless_compress(arr)

        return {
            "type": "direct_generic",
            "codec": codec,
            "data": data_bytes,
            "step": current_step,
            "shape": arr.shape,
            "original_dtype": str(arr.dtype),
            "stored_dtype": stored_dtype,
            "reason": "non_convolutional_layer_direct_compression"
        }

    # -------------------- model-level compress/decompress --------------------
    def compress_model(self,
                       model: Union[dict, OrderedDict, List[Union[dict, OrderedDict]]],
                       batched: bool = False,
                       client_id: Optional[str] = None) -> bytes:
        if batched:
            if isinstance(model, list):
                return self._lossless_compress([self.compress_model(m) for m in model])[0]
            if isinstance(model, (dict, OrderedDict)):
                out = OrderedDict()
                for k, v in model.items():
                    out[k] = self.compress_model(v)
                return self._lossless_compress(out)[0]

        model_params = model

        compressed_layers = OrderedDict()
        compression_metadata = {
            "compressor_type": "MomentumPredictorCompressor",
            "client_id": client_id,
            "layer_count": len(model_params),
            "config": {
                "momentum_lr": self.momentum_lr,
                "consistency_threshold": self.consistency_threshold,
                "error_bound": self.error_bound,
                "error_bounding_mode": self.error_bounding_mode,
                "param_cutoff": self.param_count_threshold,
            }
        }

        total_prediction_ratio = 0.0
        total_sign_mismatch_ratio = 0.0
        layer_count = 0

        for layer_name, layer_params in model_params.items():
            if isinstance(layer_params, torch.Tensor) and layer_params.ndim == 4 and self._should_lossy(layer_name, layer_params.detach().cpu().numpy()):
                compressed_layer = self._create_compressed_data(layer_params, client_id, layer_name)
                compressed_layers[layer_name] = compressed_layer
                if "prediction_ratio" in compressed_layer:
                    total_prediction_ratio += compressed_layer["prediction_ratio"]
                    layer_count += 1
                if "sign_mismatch_ratio" in compressed_layer:
                    total_sign_mismatch_ratio += compressed_layer["sign_mismatch_ratio"]

            elif isinstance(layer_params, torch.Tensor):
                compressed_layer = self._compress_generic_layer(
                    layer_params.detach().cpu().numpy(), client_id, current_step=1, layer_name=layer_name
                )
                compressed_layers[layer_name] = compressed_layer

            else:
                # 非 tensor → lossless compress
                data_bytes, codec = self._lossless_compress(layer_params)
                compressed_layers[layer_name] = {
                    "type": "direct",
                    "codec": codec,
                    "data": data_bytes,
                    "shape": None,
                    "original_dtype": None,
                    "stored_dtype": None
                }

        if layer_count > 0:
            avg_prediction_ratio = total_prediction_ratio / layer_count
            self.compression_stats["prediction_ratios"].append(avg_prediction_ratio)
            avg_sign_mismatch_ratio = total_sign_mismatch_ratio / layer_count
            self.compression_stats["sign_mismatch_ratios"].append(avg_sign_mismatch_ratio)
            # self.logger.info(f"Model compression complete - Average prediction ratio: {avg_prediction_ratio:.3f}, "
            #                  f"Average sign mismatch ratio: {avg_sign_mismatch_ratio:.3f}")

        self.compression_stats["total_compressions"] += 1
        final_data = {"compressed_layers": compressed_layers, "metadata": compression_metadata}
        final_bytes, codec = self._lossless_compress(final_data)
        # self.logger.info(f"Final compressed model size: {len(final_bytes)} bytes")
        return final_bytes

    def decompress_model(self,
                         compressed_model: bytes,
                         model: Union[dict, OrderedDict],
                         batched: bool = False) -> Union[OrderedDict, dict]:
        if batched:
            raise NotImplementedError("Batched decompression not yet implemented")

        try:
            compressed_data = self._lossless_decompress(compressed_model, 'zstd')
        except:
            try:
                compressed_data = self._lossless_decompress(compressed_model, 'blosc') 
            except:
                compressed_data = pickle.loads(compressed_model)
        compressed_layers = compressed_data["compressed_layers"]
        metadata = compressed_data["metadata"]

        client_id = metadata["client_id"]
        # self.logger.info(f"Decompressing model for client {client_id} with {len(compressed_layers)} layers")

        decompressed_model = OrderedDict()

        for layer_name, compressed_layer in compressed_layers.items():
            layer_type = compressed_layer["type"]
            self.logger.debug(f"Decompressing layer {layer_name} with type {layer_type}")

            try:
                if layer_type == "direct":
                    codec = compressed_layer.get("codec", "sz3")
                    shape = compressed_layer.get("shape", None)

                    if codec in ["pickle", "zstd", "blosc"]:
                        if codec == "pickle":
                            arr = pickle.loads(compressed_layer["data"])
                        else:
                            arr = self._lossless_decompress(compressed_layer["data"], codec)
                        
                        if shape is not None:
                            tensor = torch.as_tensor(arr)
                        else:
                            decompressed_model[layer_name] = arr
                            continue
                    else:
                        stored_dtype = compressed_layer.get("stored_dtype", "float32")
                        if shape is None:
                            raise ValueError("Missing shape for SZ3 direct layer")
                        arr = self._decompress_with_sz3(compressed_layer["data"], tuple(shape), stored_dtype)
                        tensor = torch.from_numpy(arr)

                    if client_id not in self.gradient_history:
                        self.gradient_history[client_id] = {}
                    if layer_name not in self.gradient_history[client_id]:
                        self.gradient_history[client_id][layer_name] = []
                    self.gradient_history[client_id][layer_name].append(
                        tensor.detach().cpu().numpy().copy()
                    )

                    if shape and isinstance(shape, (list, tuple)) and len(shape) == 4:
                        layer_key = _conv_layer_key(shape)
                        if client_id not in self.prediction_memory:
                            self.prediction_memory[client_id] = {}
                        if layer_key not in self.prediction_memory[client_id]:
                            self.prediction_memory[client_id][layer_key] = np.zeros(tuple(shape))

                    decompressed_model[layer_name] = tensor

                elif layer_type == "momentum_predicted":
                    decompressed_model[layer_name] = self._decompress_conv_layer(
                        compressed_layer, client_id, layer_name
                    )

                elif layer_type in ("momentum_predicted_generic", "direct_generic"):
                    codec = compressed_layer.get("codec", "sz3")
                    shape = tuple(compressed_layer["shape"])

                    if codec in ["pickle", "zstd", "blosc"]:
                        if codec == "pickle":
                            arr = pickle.loads(compressed_layer["data"])
                        else:
                            arr = self._lossless_decompress(compressed_layer["data"], codec)
                        tensor = torch.as_tensor(arr)
                    else:
                        stored_dtype = compressed_layer.get("stored_dtype", "float32")
                        arr = self._decompress_with_sz3(compressed_layer["data"], shape, stored_dtype)
                        tensor = torch.from_numpy(arr)

                    decompressed_model[layer_name] = tensor

                else:
                    raise ValueError(f"Unknown compression type: {layer_type}")

            except Exception as e:
                self.logger.error(f"Error decompressing layer {layer_name}: {str(e)}")
                raise

        # self.logger.info(f"Model decompression complete for client {client_id}")
        return decompressed_model

    # -------------------- conv/generic layer decompression --------------------
    def _decompress_conv_layer(self, compressed_layer: Dict[str, Any], client_id: str, layer_name: str) -> torch.Tensor:
        shape = tuple(compressed_layer["shape"])
        stored_dt = self._dtype_to_numpy(compressed_layer.get("original_dtype", "float32"))

        # residuals
        if _is_float_dtype(stored_dt):
            residual_data = self._decompress_with_sz3(compressed_layer["data"], shape, stored_dt)
        else:
            residual_data = pickle.loads(compressed_layer["data"])

        bm_meta = compressed_layer["bitmap_metadata"]
        bm_packed_bytes = zstd.decompress(compressed_layer["bitmap"])
        bm_packed = np.frombuffer(bm_packed_bytes, dtype=np.uint8)
        bm_flat = np.unpackbits(bm_packed).astype(bool)[:bm_meta["original_size"]]
        bitmap = bm_flat.reshape(bm_meta["original_shape"])

        dom_bytes = compressed_layer["dominant_signs"]
        dom_meta = compressed_layer["dominant_signs_metadata"]
        if compressed_layer["num_predicted_kernels"] > 0 and len(dom_bytes) > 0:
            dom_packed_bytes = zstd.decompress(dom_bytes)
            dom_packed = np.frombuffer(dom_packed_bytes, dtype=np.uint8)
            dom_bits = np.unpackbits(dom_packed).astype(bool)
            pred_dom_bits = dom_bits[:dom_meta["original_size"]]
        else:
            pred_dom_bits = np.array([], dtype=bool)

        if client_id not in self.gradient_history:
            self.gradient_history[client_id] = {}
            self.prediction_memory[client_id] = {}
        if layer_name not in self.gradient_history[client_id]:
            self.gradient_history[client_id][layer_name] = []
            self.logger.warning(f"No gradient history for client {client_id} layer {layer_name}, using direct reconstruction")
            return torch.from_numpy(residual_data).float()

        prev_reconstructed = self.gradient_history[client_id][layer_name][-1]
        if client_id not in self.prediction_memory:
            self.prediction_memory[client_id] = {}
        layer_key = _conv_layer_key(shape)
        if layer_key not in self.prediction_memory[client_id]:
            self.prediction_memory[client_id][layer_key] = np.zeros(shape)
        layer_memory = self.prediction_memory[client_id][layer_key]

        cur = compressed_layer.get("current_grad_stats", {})
        prv = compressed_layer.get("prev_grad_stats", {})
        current_mean = cur.get("mean", 0.0)
        current_std = cur.get("std", 1.0)
        prev_mean = prv.get("mean", 0.0)
        prev_std = prv.get("std", 1.0)

        abs_prev_global = np.abs(prev_reconstructed)
        prev_normalized = abs_prev_global - prev_mean
        if prev_std > 1e-8:
            prev_normalized = prev_normalized / prev_std

        reconstructed = residual_data.copy()

        pred_mask = bitmap
        num_pred = int(np.count_nonzero(pred_mask))

        if num_pred > 0:
            prev_norm_sel = prev_normalized[pred_mask]
            memory_sel = layer_memory[pred_mask]
            pred_norm = (1 - self.momentum_lr) * memory_sel + self.momentum_lr * prev_norm_sel
            layer_memory[pred_mask] = pred_norm

            if current_std > 1e-8:
                abs_pred = pred_norm * current_std + current_mean
            else:
                abs_pred = pred_norm + current_mean
            abs_pred = np.abs(abs_pred)

            if len(pred_dom_bits) == num_pred:
                dominant_sign_arr = np.where(pred_dom_bits, 1.0, -1.0)
            else:
                dominant_sign_arr = np.ones(num_pred, dtype=np.float32)

            predicted_kernel = dominant_sign_arr[:, None, None] * abs_pred
            reconstructed[pred_mask] = residual_data[pred_mask] + predicted_kernel

            self.prediction_memory[client_id][layer_key] = layer_memory

        self.gradient_history[client_id][layer_name].append(reconstructed.copy())
        if len(self.gradient_history[client_id][layer_name]) > 3:
            self.gradient_history[client_id][layer_name].pop(0)

        return torch.from_numpy(reconstructed).float()

    # -------------------- misc controls & stats --------------------
    def set_client_context(self, client_id: str):
        self._current_client_id = client_id
        if not client_id.startswith("Client"):
            client_parts = client_id.split("_")
            if len(client_parts) > 1 and client_parts[-1].isdigit():
                self._current_client_id = f"Client{client_parts[-1]}"

    def get_compression_stats(self) -> Dict[str, Any]:
        stats = self.compression_stats.copy()
        if stats["prediction_ratios"]:
            stats["avg_prediction_ratio"] = float(np.mean(stats["prediction_ratios"]))
            stats["std_prediction_ratio"] = float(np.std(stats["prediction_ratios"]))
        if stats["sign_mismatch_ratios"]:
            stats["avg_sign_mismatch_ratio"] = float(np.mean(stats["sign_mismatch_ratios"]))
        self.logger.info(f"Compression statistics: {stats}")
        return stats

    def reset_client_state(self, client_id: str):
        if client_id in self.gradient_history:
            del self.gradient_history[client_id]
        if client_id in self.prediction_memory:
            del self.prediction_memory[client_id]
        if client_id in self.step_count:
            del self.step_count[client_id]
        self.logger.info(f"Reset state for client: {client_id}")

    def reset_all_states(self):
        self.gradient_history.clear()
        self.prediction_memory.clear()
        self.step_count.clear()
        self.compression_stats = {
            "total_compressions": 0,
            "prediction_ratios": [],
            "sign_mismatch_ratios": []
        }
        self.logger.info("Reset all client states and statistics")

    def set_log_level(self, level: str):
        level_up = level.upper()
        if level_up == "DEBUG":
            self.logger.setLevel(logging.DEBUG)
        elif level_up == "INFO":
            self.logger.setLevel(logging.INFO)
        elif level_up == "WARNING":
            self.logger.setLevel(logging.WARNING)
        elif level_up == "ERROR":
            self.logger.setLevel(logging.ERROR)
        else:
            self.logger.warning(f"Unknown log level: {level}")
        self.logger.info(f"Log level set to: {level_up}")

    def get_detailed_stats(self) -> Dict[str, Any]:
        stats = self.get_compression_stats()
        client_stats = {}
        for client_id in self.gradient_history.keys():
            client_stats[client_id] = {
                "history_length": len(self.gradient_history[client_id]),
                "step_count": self.step_count.get(client_id, 0),
                "memory_layers": list(self.prediction_memory.get(client_id, {}).keys())
            }
        stats["client_details"] = client_stats
        return stats