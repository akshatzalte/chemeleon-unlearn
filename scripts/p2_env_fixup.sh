#!/bin/sh
# One-off fix after `conda env create -f environment-pretrain.yml`.
# Importing pip-installed torch first loads the (older) system libstdc++, after which conda-forge
# extension modules (scipy, matplotlib, ...) fail with "CXXABI_1.3.15 not found". Preload the env's own
# libstdc++ at interpreter start-up via a .pth file so direct `$PY2 script.py` calls work without activation.
set -e
PY2=${PY2:-/home/akshatz/anaconda3/envs/chemeleon_pretrain/bin/python}
SP=$($PY2 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
echo "import ctypes, os, sys; ctypes.CDLL(os.path.join(sys.prefix, 'lib', 'libstdc++.so.6'), ctypes.RTLD_GLOBAL)" > "$SP/zz_preload_libstdcxx.pth"
