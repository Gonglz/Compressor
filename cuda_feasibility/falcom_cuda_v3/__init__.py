"""Experimental FalCom CUDA v3 closed-loop compressor."""

from .falcom_cuda_v3_wrapper import (  # noqa: F401
    CODEC_VERSION,
    CODEC_NAME,
    FalcomCudaV3State,
    compress_batch_cuda_v3,
    decompress_batch_cuda_v3,
    dumps_cuda_v3_layers,
    cuda_v3_enabled,
    default_config,
    layer_key,
)
