import torch
print("Torch:", torch.__version__, "CUDA build:", torch.version.cuda)
print("CUDA disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Capacidad:", torch.cuda.get_device_capability())
    print("cuDNN habilitado:", torch.backends.cudnn.enabled)


import sys, torch, platform
print("Python exe:", sys.executable)
print("Python ver:", platform.python_version())
print("Torch     :", torch.__version__, "| CUDA build:", torch.version.cuda)
print("CUDA avail:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "Cap:", torch.cuda.get_device_capability())
