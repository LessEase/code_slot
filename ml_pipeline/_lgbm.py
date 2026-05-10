"""LightGBM import shim with a clearer, OS-aware error message.

The native LightGBM wheel links against OpenMP. On macOS this means the
``libomp.dylib`` runtime must be available, otherwise ``import lightgbm``
fails with a long ``dlopen`` traceback that buries the actual fix.

Importing ``lgb`` from this module yields the real ``lightgbm`` package
when it loads cleanly, and raises a single ``ImportError`` with install
instructions when it does not.
"""

import platform
import sys


def _install_hint() -> str:
    if sys.platform == "darwin":
        arch = platform.machine()
        brew = "/opt/homebrew" if arch == "arm64" else "/usr/local"
        return (
            "LightGBM on macOS requires the OpenMP runtime (libomp).\n"
            "Install it with:\n"
            "    brew install libomp\n"
            f"If Homebrew is installed in a non-standard prefix, ensure "
            f"'{brew}/opt/libomp/lib' is on DYLD_LIBRARY_PATH."
        )
    if sys.platform.startswith("linux"):
        return (
            "LightGBM requires libgomp. Install it with one of:\n"
            "    apt-get install libgomp1\n"
            "    yum install libgomp"
        )
    return "Reinstall lightgbm or ensure the OpenMP runtime is available."


try:
    import lightgbm as lgb  # noqa: F401  (re-exported)
except (OSError, ImportError) as exc:
    raise ImportError(
        f"Failed to load lightgbm: {exc}\n\n{_install_hint()}"
    ) from exc
