import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import numpy as np
from tqdm import tqdm
import nibabel as nib
import pydicom
from PIL import Image
path = "Data/sample_CT_dicom"


def DcmVisualization(path):
   # 读取单张DICOM图像
    image = sitk.ReadImage(path)

    # 将图像转换为NumPy数组
    array = sitk.GetArrayFromImage(image).reshape(512, 512)
    # 可视化图像
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.imshow(array, cmap='gray')
    ax.axis('off')
    plt.savefig('gray_image.png')
    image = Image.open('gray_image.png')

    # 转换为灰度图像
    gray_image = image.convert('L')

    # 保存为灰度图像
    gray_image.save('gray_image.png')


def DcmToNii(path):
    # 读取整个文件夹下的所有图片然后转换为nii
    reader = sitk.ImageSeriesReader()
    dicm_name = reader.GetGDCMSeriesFileNames(path)
    reader.SetFileNames(dicm_name)
    image = reader.Execute()
    sitk.WriteImage(image, path + "sample_CT.nii.gz")


def NiiVisualization(path, x=20, y=50, z=23):
    # 指定 NIfTI 文件路径
    # 使用 SimpleITK 库读取 NIfTI 文件
    image = sitk.ReadImage(path)
    print(image.GetSize())
    # 将 SimpleITK 图像转换为 NumPy 数组
    image_array = sitk.GetArrayFromImage(image)
    # 显示 NIfTI 图像
    postition = [x, y, z]
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
    image = (image - min_value) / (max_value - min_value)  # 归一化
    for z in range(size[2]):
        image_slice = image[:, :, z]  # 获取切片
        # 将图像切片转换为整数类型避免类型错误
        image_slice_int = (image_slice * np.iinfo(np.int16).max).astype(np.int16)
        # 创建一个空的DICOM对象
        dcm = pydicom.Dataset()
        # 设置DICOM的相关属性
        dcm.Rows = image_slice_int.shape[0]
        dcm.Columns = image_slice_int.shape[1]
        dcm.PixelData = image_slice_int.tobytes()
        dcm.is_little_endian = True
        dcm.is_implicit_VR = True
        # 设置Series属性
        dcm.SeriesDescription = "Your Series Description"
        dcm.SeriesNumber = 1
        dcm.SeriesInstanceUID = "Your Series Instance UID"
        # 设置文件路径和名称
        save_path = outPath + str(z) + ".dcm"
        # 保存DICOM文件
        pydicom.filewriter.dcmwrite(save_path, dcm)


def qiege(path):
    image = sitk.GetArrayFromImage(sitk.ReadImage(path))
    image = image[0:100, 0:100, 0:100]
    plt.imshow(image[0])
    plt.show()


def gauss(path):
    image = sitk.ReadImage(path)
    image = sitk.SmoothingRecursiveGaussian(image, 1)
    image = sitk.GetArrayFromImage(image)
    plt.imshow(image[0])
    plt.show()


def medianFiltering(path):
    image = sitk.ReadImage(path)
    median_filter = sitk.MedianImageFilter()
    median_filter.SetRadius([3, 3, 3])
    median_filtered_image = median_filter.Execute(image)
    image = sitk.GetArrayFromImage(median_filtered_image)
    plt.imshow(image[0])
    plt.show()


# nii_to_dcm("Data/STS_01/STS_01_ct_gtvt.nxii.gz",'Data/STS_01/STS_01_ct_gtvt/')
DcmVisualization("Data/sample_CT_dicom/1-011.dcm")
# NiiVisualization('Data/STS_01/STS_01_ct.nii.gz',22,50,23)
# qiege("Data/STS_01/STS_01_ct.nii.gz")
# DcmToNii("Data/sample_CT_dicom/")
# gauss("Data/STS_01/STS_01_ct.nii.gz")
# medianFiltering("Data/STS_01/STS_01_ct.nii.gz")
