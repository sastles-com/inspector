import cv2
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

import csv
import pandas as pd
import os

from scipy import signal

# import streamlit as st
from PIL import Image




import plotly.graph_objs as go
import plotly
colors = plotly.colors.DEFAULT_PLOTLY_COLORS




_ratio = 0.1
_bias = 128
_kernel_size = 50
_kernel_pitch = 10
_mask_th = 4000000000.0  
_scratch_th = 0.03
_file_index = 0
_device_index = 0
_type_index = 0
_date_index = 0






def mosaic(src, ratio=0.1):
    small = cv2.resize(src, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_NEAREST)
    return cv2.resize(small, src.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)

def mosaic_area(src, x, y, width, height, ratio=0.1):
    dst = src.copy()
    dst[y:y + height, x:x + width] = mosaic(dst[y:y + height, x:x + width], ratio)
    return dst

# https://note.nkmk.me/python-opencv-mosaic/
def visualize_scratch(img, wsz, ptch):
    img2 = img.copy()
    h, w = img.shape[:2]

    imageArray = np.zeros((h, w))
    cnt = int(wsz / ptch)
    
    ave = np.abs(img - img.mean())
    for y in range(0, cnt):
        for x in range(0, cnt):
            # imageArray += mosaic_area(img, x, y, w, h, ratio = 0.005)
            # imageArray += mosaic_area(img, x, y, w, h, ratio = 0.05)
            imageArray += mosaic_area(img, x, y, w, h, ratio = 0.1)

    # print(imageArray.max(), imageArray.min(), imageArray.mean())
    return imageArray


def average(x, axis = None, bias=128):
    xmean = x.mean(axis=axis, keepdims=True)
    xstd  = np.std(x, axis=axis, keepdims=True)
    zscore = (x-xmean) + bias
    return zscore







# st.title('scratch inspector')

# file_expander = st.sidebar.beta_expander('File selection')
# path = st.text_input('画像のパスを入力してください', '/Users/katano/Documents/work/XD4/glue/scratch_inspector/data/input.png')
path = "/Users/katano/Documents/work/XD4/glue/scratch_inspector/data/input.png"

bgr = cv2.imread(path)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
# st.image(rgb, caption='original images : ', use_column_width = True)

img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
# left_column, right_column = st.beta_columns(2)


sobely = np.abs(cv2.Sobel(img,cv2.CV_64F,0,1,ksize=15))
# cv2.imwrite(save_path + 'sobely/' + filename, sobely)
ry = visualize_scratch(sobely, 50, 10)

sobelx = np.abs(cv2.Sobel(img,cv2.CV_64F,1,0,ksize=15))
# cv2.imwrite(save_path + 'sobelx/' + filename, sobelx)
rx = visualize_scratch(sobelx, 50, 10)


fig = plt.figure(facecolor="black")
fig.patch.set_alpha(0.0)
plt.rcParams['axes.edgecolor'] = 'grey'

plt.subplot(1,2,1)
plt.imshow(sobelx, cmap = 'gray')
plt.title('Sobel X', color="white")
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
plt.imshow(sobely, cmap = 'gray')
plt.title('Sobel Y', color="white")
plt.xticks([])
plt.yticks([])

# fig.show()
plt.show()
# plt.clf()


