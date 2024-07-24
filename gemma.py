import torch

# Check if PyTorch is installed
print("PyTorch version:", torch.__version__)

# Check if CUDA is available and which device is being used
if torch.cuda.is_available():
    print("CUDA is available. Using GPU:", torch.cuda.get_device_name(0))
    # Create a tensor and move it to GPU to check if GPU is being used
    x = torch.rand(5, 3).cuda()
    print("Tensor on GPU:", x)
else:
    print("CUDA is not available. Using CPU.")
