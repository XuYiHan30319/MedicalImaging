from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# 图片加噪过程
def jiazao(x0, num_steps, noise_scale, a):
    xt = x0.copy()  # 初始化 xt 为 x0
    mean = 0  # 正态分布的均值
    stddev = 1  # 正态分布的标准差
    z = np.random.normal(mean, stddev, size=x0.shape)
    min_value = z.min()
    max_value = z.max()
    z = (z - min_value) / (max_value - min_value)
    xt = (a**0.5) * xt + ((1 - a) ** 0.5) * z
    print(xt.max())
    return xt


time = 1000  # 假设我们进行1000次加噪
beita = np.linspace(1e-4, 0.02, time)[1:-2]
aerfa = 1
for i in range(len(beita)):
    # if i <= 100:
    aerfa *= 1 - beita[i]
# 读取图片
image = Image.open("/Users/blackcat/Pictures/60103079_p0.jpg")  # 替换为你的图片文件路径
# 转换为 NumPy 数组
image_np = np.array(image)
min_value = image_np.min()
max_value = image_np.max()
image_np = (image_np - min_value) / (max_value - min_value)  # 转换到01之间
# 打印 NumPy 数组的形状（尺寸）

plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title("x0")

xt = jiazao(image_np, 100, 0.1, aerfa)
plt.subplot(1, 2, 2)
plt.imshow(xt)
plt.title("xt")
plt.show()
