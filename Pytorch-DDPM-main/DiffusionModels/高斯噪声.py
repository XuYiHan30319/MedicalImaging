# -*- coding: utf-8 -*-

from PIL import Image

from pylab import *

from numpy import *

import random

# 读取图片并转为数组
im = array(Image.open("/Users/blackcat/Pictures/2fbc224f29fef042e6e0555d44bc883132c0f06c_raw.jpg"))

# 设定高斯函数的偏移
means = 4

# 设定高斯函数的标准差
sigma = 25

# r通道
r = im[:, :, 0].flatten()

# g通道
g = im[:, :, 1].flatten()

# b通道
b = im[:, :, 2].flatten()


# 计算新的像素值
for i in range(im.shape[0] * im.shape[1]):
    r[i] = r[i] + random.gauss(0, sigma)

    g[i] = g[i] + random.gauss(0, sigma)

    b[i] = b[i] + random.gauss(0, sigma)


im[:, :, 0] = r.reshape([im.shape[0], im.shape[1]])

im[:, :, 1] = g.reshape([im.shape[0], im.shape[1]])

im[:, :, 2] = b.reshape([im.shape[0], im.shape[1]])

# 显示图像

imshow(im)

show()
