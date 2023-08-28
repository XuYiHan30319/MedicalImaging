import nibabel as nib
import matplotlib.pyplot as plt
import os

# dirpath = "/Users/blackcat/北大实习/Data/averaged-testing-images/"
# dir = os.listdir(dirpath)
# for file in dir:
#     # 读取NIfTI文件
#     path = dirpath + file
#     nifti_img = nib.load(path)
#     outPath = "/Users/blackcat/北大实习/Data/averaged-testing-images-png/" + file + "/"
#     os.makedirs(outPath)  # 输出路径
#     # 获取图像数据
#     img_data = nifti_img.get_fdata()
#     print(img_data.shape)
#     # 将图像数据保存为PNG图像
#     for i in range(img_data.shape[2]):
#         plt.imshow(img_data[:, :, i], cmap="gray")
#         plt.axis("off")
#         plt.savefig(outPath + str(i) + ".png", bbox_inches="tight", pad_inches=0)


# dirpath = "/Users/blackcat/北大实习/Data/averaged-training-images/"
# dir = os.listdir(dirpath)
# for file in dir:
#     # 读取NIfTI文件
#     path = dirpath + file
#     nifti_img = nib.load(path)
#     outPath = "/Users/blackcat/北大实习/Data/averaged-training-images-png/" + file + "/"
#     os.makedirs(outPath)  # 输出路径
#     # 获取图像数据
#     img_data = nifti_img.get_fdata()
#     print(img_data.shape)
#     # 将图像数据保存为PNG图像
#     for i in range(img_data.shape[2]):
#         plt.imshow(img_data[:, :, i], cmap="gray")
#         plt.axis("off")
#         plt.savefig(outPath + str(i) + ".png", bbox_inches="tight", pad_inches=0)

dirpath = "/Users/blackcat/北大实习/Data/averaged-training-labels/"
dir = os.listdir(dirpath)
for file in dir:
    # 读取NIfTI文件
    path = dirpath + file
    nifti_img = nib.load(path)
    outPath = "/Users/blackcat/北大实习/Data/averaged-training-labels-png/" + file + "/"
    os.makedirs(outPath)  # 输出路径
    # 获取图像数据
    img_data = nifti_img.get_fdata()
    print(img_data.shape)
    # 将图像数据保存为PNG图像
    for i in range(img_data.shape[2]):
        plt.imshow(img_data[:, :, i], cmap="gray")
        plt.axis("off")
        plt.savefig(outPath + str(i) + ".png", bbox_inches="tight", pad_inches=0)