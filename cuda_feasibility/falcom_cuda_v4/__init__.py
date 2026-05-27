"""Experimental FalCom CUDA v4 fused q8 closed-loop compressor."""

from .falcom_cuda_v4_wrapper import (  # noqa: F401
    CODEC_NAME,
    CODEC_VERSION,
    FalcomCudaV4State,
    compress_batch_cuda_v4,
    cuda_guard_status,
    cuda_v4_enabled,
    decompress_batch_cuda_v4,
    default_config,
    dumps_cuda_v4_layers,
    extension_available,
    guarded_all_cuda_enabled,
    layer_key,
    loads_cuda_v4_layers,
)
