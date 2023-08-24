import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import uuid
import pydicom
import nibabel as nib
path = "Data/sample_CT_dicom"


def DcmVisualization(path):
    # 读取DICOM图像
    image = sitk.ReadImage(path)

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


def NiiVisualization(path):
    # 指定 NIfTI 文件路径
    nii_file = "Data/STS_01/STS_01_ct.nii.gz"
    # 使用 SimpleITK 库读取 NIfTI 文件
    image = sitk.ReadImage(nii_file)
    print(image.GetSize())
    # 将 SimpleITK 图像转换为 NumPy 数组
    image_array = sitk.GetArrayFromImage(image)
    # 显示 NIfTI 图像
    postition = [20, 50, 23]
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(image_array[postition[0], :, :], cmap="gray")
    axs[0].axis("off")
    axs[1].imshow(image_array[:, postition[1], :], cmap="gray")
    axs[1].axis("off")
    axs[2].imshow(image_array[:, :, postition[2]], cmap="gray")
    axs[2].axis("off")
    plt.show()


def nii_to_dcm(nii_path, output_dir):
    # 加载NIfTI图像
    nii_image = sitk.ReadImage(nii_path)

    # 创建一个SimpleITK的ImageIO对象，指定为DICOM格式
    dicom_io = sitk.ImageIOFactory.CreateImageIO("DICOM")
    
    # 设置输出路径
    output_series_uid = str(uuid.uuid4())
    output_dicom_path = os.path.join(output_dir, output_series_uid)
    
    # 将NIfTI图像的空间信息应用于DICOM图像
    dicom_image = sitk.Cast(sitk.RescaleIntensity(nii_image), sitk.sitkInt16)
    dicom_image.SetDirection(nii_image.GetDirection())
    dicom_image.SetOrigin(nii_image.GetOrigin())
    dicom_image.SetSpacing(nii_image.GetSpacing())
    
    # 将DICOM图像信息写入ImageIO
    dicom_io.SetMetaDataDictionary(dicom_image.GetMetaDataKeys())
    dicom_io.SetMetaData(dicom_image.GetMetaData())
    
    # 保存DICOM图像
    dicom_io.WriteImage(dicom_image, output_dicom_path)

nii_path = 'Data/STS_01/STS_01_ct.nii.gz'  # 替换为你的NIfTI图像文件路径
output_dir = 'Data/sample_NiiToDCM'  # 替换为你想保存DICOM文件的目录路径
os.makedirs(output_dir, exist_ok=True)
nii_to_dcm(nii_path, output_dir)
