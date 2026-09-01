import os
import sys

# nothing under gpu/ is pip-installed, so the package is only importable
# if this directory is on the path — pytest loads this conftest before
# collecting gpu/tests/, which is early enough for module-level imports
GPU_DIR = os.path.dirname(os.path.abspath(__file__))
if GPU_DIR not in sys.path:
    sys.path.insert(0, GPU_DIR)
