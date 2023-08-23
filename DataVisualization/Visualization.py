import nibabel as nib
import matplotlib.pyplot as plt

# 加载NIfTI文件
nii_img = nib.load('Data/STS_01/STS_01_ct.nii')

# 获取图像数据和元数据
data = nii_img.get_fdata()
affine = nii_img.affine

# 可视化切片图像
num_slices = data.shape[-1]  # 获取切片数量
fig, axes = plt.subplots(1, num_slices, figsize=(12, 4))

for i in range(num_slices):
    axes[i].imshow(data[:, :, i].T, cmap='gray', origin='lower')
    axes[i].axis('off')
    axes[i].set_title(f"Slice {i+1}")

plt.tight_layout()
plt.show()