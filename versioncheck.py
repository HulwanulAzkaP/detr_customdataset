import torch

print("Is CUDA available?:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("PyTorch CUDA version:", torch.backends.cudnn.version())
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
