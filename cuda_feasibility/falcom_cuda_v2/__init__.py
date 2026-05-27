"""Experimental FalCom CUDA v2 closed-loop compressor."""

from .falcom_cuda_v2_wrapper import (  # noqa: F401
    CODEC_VERSION,
    CODEC_NAME,
    FalcomCudaV2State,
    compress_batch_cuda_v2,
    decompress_batch_cuda_v2,
    dumps_cuda_v2_layers,
    cuda_v2_enabled,
    default_config,
    layer_key,
)
