import sys
import subprocess

# Auto-install lightweight required packages if missing in the environment
_REQUIRED_PKGS = ["flatten_dict", "attrs", "tensorboard"]

for _pkg in _REQUIRED_PKGS:
    try:
        __import__(_pkg)
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pkg])
        except Exception:
            pass
