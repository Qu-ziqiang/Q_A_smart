"""
扩展管理模块初始化
"""

from .loader import setup_connection, verify_vector_extension

try:
    from ....config.platform_detector import PlatformDetector, auto_setup_driver
except Exception:  # pragma: no cover
    PlatformDetector = None
    auto_setup_driver = None

__all__ = [
    "setup_connection",
    "verify_vector_extension",
    "PlatformDetector",
    "auto_setup_driver",
]
