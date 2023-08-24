import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import uuid
import pydicom
import nibabel as nib
import numpy as np
from tqdm import tqdm
import open3d as o3d

path = "Data/sample_CT_dicom"


def DcmVisualization(path):
    # 读取DICOM图像
    image = sitk.ReadImage(path)
    print(image.GetSize())
    print(image.GetMetaData('Series'))
    # image = sitk.SmoothingRecursiveGaussian(image, 0.1)
    # 将图像转换为NumPy数组
    array = sitk.GetArrayFromImage(image)

    # 可视化图像
    plt.imshow(array[0], cmap="gray")  # 假设读取的DICOM包含多个切片，这里选择第一个切片进行可视化
    plt.show()


def DcmToNii(path):
    # 读取整个文件夹下的所有图片然后转换为nii
    reader = sitk.ImageSeriesReader()
    dicm_name = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicm_name)
    image = reader.Execute()
    sitk.WriteImage(image, path + "sample_CT.nii.gz")


def NiiVisualization(path,x=20,y=50,z=23):
    # 指定 NIfTI 文件路径
    # 使用 SimpleITK 库读取 NIfTI 文件
    image = sitk.ReadImage(path)
    print(image.GetSize())
    # 将 SimpleITK 图像转换为 NumPy 数组
    image_array = sitk.GetArrayFromImage(image)
    # 显示 NIfTI 图像
    postition = [x,y,z]
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(image_array[postition[0], :, :], cmap="gray")
    axs[0].axis("off")
    axs[1].imshow(image_array[:, postition[1], :], cmap="gray")
    axs[1].axis("off")
    axs[2].imshow(image_array[:, :, postition[2]], cmap="gray")
    axs[2].axis("off")
    plt.show()


def nii_to_dcm(nii_path, outPath):
    # 加载NIfTI图像
    os.makedirs(outPath, exist_ok=True)
    nii_image = sitk.ReadImage(nii_path)
    size = nii_image.GetSize()
    origin = nii_image.GetOrigin()
    spacing = nii_image.GetSpacing()
    direction = np.array(nii_image.GetDirection()).reshape(3, 3)
    image = sitk.GetArrayFromImage(nii_image)
    min_value = np.min(image)
    max_value = np.max(image)
    image = (image - min_value) / (max_value - min_value)#归一化
    for z in range(size[2]):
        image_slice = image[:,:,z] # 获取切片
        # 将图像切片转换为整数类型避免类型错误
        image_slice_int = (image_slice * np.iinfo(np.int16).max).astype(np.int16)
        
        dcm_image = sitk.GetImageFromArray(image_slice_int)
        # 设置DICOM文件的元数据
        dcm_image.SetSpacing(spacing)
        # 使用前两行作为方向矩阵
        dcm_direction = direction[0:2, 0:2].flatten()
        dcm_image.SetDirection(dcm_direction)
        dcm_image.SetMetaData("0000|0008", '123')  # Series Instance UID
        dcm_image.SetMetaData("0020|000d", 'study_instance_uid')   # Study Instance UID
        dcm_image.SetMetaData("0008|103e", 'series_Description')    # Series Description
        # Set Modality
        dcm_image.SetMetaData("0008|0060", 'modality')
        # 构造DICOM文件路径并保存
        dcm_path = os.path.join(outPath, f"slice_{z:03d}.dcm")
        sitk.WriteImage(dcm_image, dcm_path)

def qiege(path):
    image = sitk.GetArrayFromImage(sitk.ReadImage(path))
    image = image[0:100,0:100,0:100]
    plt.imshow(image)
    plt.show

nii_to_dcm("Data/STS_01/STS_01_ct_gtvt.nii.gz",'Data/STS_01/STS_01_ct_gtvt/')
# DcmVisualization('Data/sample_CT_dicom/1-002.dcm')