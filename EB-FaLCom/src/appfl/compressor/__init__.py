from .compressor import Compressor
from .sz2_compressor import SZ2Compressor
from .sz3_compressor import SZ3Compressor
from .szx_compressor import SZxCompressor
from .zfp_compressor import ZFPCompressor
from .FalCom import FalCom
from .FalComC import FalComC
from .qsgd_compressor import QSGDCompressor

__all__ = [
    "Compressor",
    "SZ2Compressor",
    "SZ3Compressor",
    "SZxCompressor",
    "ZFPCompressor",
    "FalCom",
    "FalComC",
    "QSGDCompressor",
]