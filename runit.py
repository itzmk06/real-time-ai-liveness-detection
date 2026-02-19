import numpy as np

# path to your .npy file
file_path = r"E:\Liveness-Detection-main\Liveness-Detection-main\Model (Using Dataset 600 Images)\backend\embeddings\embed_20251124_174826.npy"

# load the array
arr = np.load(file_path)

# print the array
print(arr)
print("Shape:", arr.shape)
print("Data type:", arr.dtype)
