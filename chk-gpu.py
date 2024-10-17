import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow version:", tf.__version__)


import torch

print(f"PyTorch version: {torch.__version__}")
print(f"Is MPS (Metal Performance Shaders) available: {torch.backends.mps.is_available()}")
print(f"Is cuDNA available: {torch.backends.cudnn.is_available()}")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Test MPS
if torch.backends.mps.is_available():
    a = torch.ones(5, device=device)
    b = torch.ones(5, device=device)
    c = a + b
    print(c)