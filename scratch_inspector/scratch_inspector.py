import cv2
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt

import csv
import pandas as pd
import os

from scipy import signal

import streamlit as st
# from bokeh.plotting import figure, show, output_file
# from PIL import Image

# # from bokeh.plotting import figure,show
# from bokeh.models import ColumnDataSource, CDSView, GroupFilter

# from bokeh import palettes
# from bokeh.core.properties import value
# from bokeh.io import export_png
# from bokeh.plotting import figure as bokeh_figure



import plotly.graph_objs as go
import plotly
colors = plotly.colors.DEFAULT_PLOTLY_COLORS






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


 

piano_df = pd.read_csv(filepath_or_buffer="piano_result.csv", encoding="utf_8", sep=",")
# piano_df.drop(['Unnamed: 0', 'Condition', 'Original', 'Others'], axis=1)
# piano_df.drop(['path'], axis=1)
piano_df = piano_df.drop(piano_df.columns[[0, 1, 6, 7, 9]], axis=1)

save_path = "/Users/katano/Documents/work/X50/piano/"
# df = pd.read_csv(filepath_or_buffer=save_path + "images/crop/filelist.txt", encoding="utf_8", sep=",")
# print(df)

# piano_df['path'] = df['path']
# st.dataframe(piano_df)
st.write('dataframe')
print('dfasdfa')
df = pd.read_csv(filepath_or_buffer=save_path + "grade_result.csv", encoding="utf_8", sep=",")
print(df)
# st.write(df['grade'])
st.dataframe(df, width=1000, height=200)

# histr = cv2.calcHist([img],[i],None,[256],[0,256])



# print(piano_df , piano_df.columns[[0, 1, 3, 4, 6]])
# print(piano_df )
# st.write(piano_df )

st.write('dataframe')

[dw,dh] = piano_df.shape
print(dw, dh)

pd.set_option("display.max_colwidth", 200)
# st.dataframe(piano_df)
st.dataframe(piano_df, width=1000, height=200)
# st.dataframe(piano_df, dw, dh)
 

st.title('scratch inspector')
# st.subheader('file list')
# st.subheader('サブヘッダー')
# st.write('文字列') # markdown


# color = ['red', 'green', 'yellow', 'blue']
# source = ColumnDataSource(df)
# bokeh_plot = figure(plot_height=500,plot_width=500,title='傷の数 x 傷の深さ', x_axis_label='num of scratch', y_axis_label='average of scratch depth')
# bokeh_plot.circle(x='num_scratch', y='average',source=source,legend_label='grade')
# st.bokeh_chart(bokeh_plot, use_container_width=True)

df0 = df[df['grade'] == 0]
df1 = df[df['grade'] == 1]
df2 = df[df['grade'] == 2]
df3 = df[df['grade'] == 3]
df4 = df[df['grade'] == 4]
# st.write(df0)
trace0 =  go.Scatter(x = df['num_scratch'][df['grade'] == 0], y = df['average'][df['grade'] == 0], mode='markers', name ='grade 0', text = df0['filename'])
trace1 =  go.Scatter(x = df['num_scratch'][df['grade'] == 1], y = df['average'][df['grade'] == 1], mode='markers', name ='grade 1', text = df1['filename'])
trace2 =  go.Scatter(x = df['num_scratch'][df['grade'] == 2], y = df['average'][df['grade'] == 2], mode='markers', name ='grade 2', text = df2['filename'])
trace3 =  go.Scatter(x = df['num_scratch'][df['grade'] == 3], y = df['average'][df['grade'] == 3], mode='markers', name ='grade 3', text = df3['filename'])
trace4 =  go.Scatter(x = df['num_scratch'][df['grade'] == 4], y = df['average'][df['grade'] == 4], mode='markers', name ='grade 4', text = df4['filename'])

layout = go.Layout(title=dict(text='<b style="color:yellow">scratch grade</b><br>', font=dict(size=32)), 
                    # autosize=True,
                    # showlegend=True,
                    # margin=dict(l=50, r=50, t=100, b=50, autoexpand=False), 
                    # line_shape="hv",
                    height=700,
                    width=700,
                    xaxis = dict(title="num of scratch", range = [5, 30], dtick=5, color='white', showgrid=False, mirror=True),   
                    yaxis = dict(title="depth of scratch", range = [-0.1, 0.5], dtick=0.1, color='white', showgrid=False, mirror=True, zeroline=False)
                    )
fig = dict(data = [trace0, trace1, trace2, trace3, trace4], layout = layout)
st.write(fig)

def update_point(trace, points, state):
    print(trace, points, state)

# fig.on_click(update_point)

# f = go.FigureWidget([go.Scatter()])




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



# st.sidebar.subheader('File selection')
file_expander = st.sidebar.beta_expander('File selection')
date_list = list(set(piano_df['Date']))
name_list = list(piano_df['SampleNumber'])


device_list = list(set(piano_df['Device']))
device_index = file_expander.selectbox('device', device_list, index = _device_index)

# list(map(str,user_list))
type_list = list(set(piano_df.query('Device=="{0}"'.format(device_index))['type']))
# type_list = list(map(str,list(set(piano_df.query('Device=="{0}"'.format(device_index))['type'])) ))
type_index = file_expander.selectbox('type', type_list, index = _type_index)


# date_df = piano_df.query('Device=="{0}" & type=="{1}"'.format(device_index, type_index))
# date_list = list(set(date_df['Date']))
# date_index = file_expander.selectbox('Date', date_list, index = _date_index)

st.dataframe(piano_df, width=1000, height=200)

# fn_df = date_df.query('Date=="{0}"'.format(date_index))
fn_df = piano_df.query('type=="{0}"'.format(str(type_index)))
print(fn_df)

filename_list = list(set(fn_df['SampleNumber']))

filename = st.sidebar.selectbox('filename', filename_list, index = _file_index)
th = st.sidebar.number_input('scratch threshold', 0.0, 1.0, 0.03)

# st.write(filename)
# path = piano_df['path'].iloc[_file_index]
path = save_path + 'images/crop/' + filename + '.JPG'
# st.write(path)
# img = np.array(Image.open(path))
# img = np.array(cv2.cvtColor(Image.open(path), cv2.COLOR_BGR2GRAY))
bgr = cv2.imread(path)
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
st.image(rgb, caption='original images : ' + filename, use_column_width = True)





img = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
left_column, right_column = st.beta_columns(2)



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

# ax = plt.gca()
# ax.set_facecolor('black')
st.pyplot(fig, use_column_width=True)

fig = plt.figure(facecolor="black")
fig.patch.set_alpha(0.0)
plt.rcParams['axes.edgecolor'] = 'grey'

plt.subplot(1,2,1)
plt.imshow(rx, cmap = 'viridis')
plt.title('enhance X', color="white")
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
plt.imshow(ry, cmap = 'viridis')
plt.title('enhance Y', color="white")
plt.xticks([])
plt.yticks([])

# ax = plt.gca()
# ax.set_facecolor('black')
st.pyplot(fig, use_column_width=True)
plt.clf()



# left_column.image(np.array(sobelx), caption='sobel-x', use_column_width=True)
# right_column.image(sobely, caption='sobel-y', use_column_width=True)

# left_column.image(rx, caption='enhance-x', use_column_width=True)
# right_column.image(ry, caption='enhance-y', use_column_width=True)

_mask_th = st.slider('mask rate', 2000000000, 20000000000, 4000000000, 100000000)

mask = np.array(ry - rx) / _mask_th         # 1.0
# mask = np.array(ry - rx) / 4000000000.0         # 1.0

mask[mask < 0] = 0
mask[mask > 1.0] = 1.0

histx = mask.mean(axis=0)
histy = mask.mean(axis=1)

x = np.arange(0,histy.shape[0])
maxid = signal.argrelmax(histy, order=10) 
# minid = signal.argrelmin(y, order=1) #最小値


cols = ['index', 'filename', 'num_scratch', 'grade']
data = pd.DataFrame(index=[], columns=cols)

filename = os.path.basename(path)
name = os.path.splitext(filename)[0]

series = piano_df[piano_df['SampleNumber'] == name]
print(series)
grade = series['Grade'].iloc[-1]

# datap = {index, filename, len(maxid[0]), sorted(histy[maxid], reverse=True)}
# dataap = pd.Series([filename, len(maxid[0]), grade], index=data.columns, name=index)
# X.append(sorted(histy[maxid], reverse=True))
# data = data.append(dataap)
# print('------', type(maxid), len(maxid[0]), dataap)




color = ('b','g')

fig = plt.figure(facecolor="black")
fig.patch.set_alpha(0.0)
plt.clf()
plt.rcParams['axes.edgecolor'] = 'grey'

plt.plot(histx, color = 'r', label='vertical')
plt.plot(x, histy, color = 'y', label='horizontal')
plt.plot(x[maxid], histy[maxid],'go',label='peak')
plt.ylim([0,1])
plt.title('peak search', color="white")

plt.xticks(x)
plt.yticks(histy)
plt.legend()

st.pyplot(fig, use_column_width=True)



