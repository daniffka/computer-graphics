import numpy as np
from PIL import Image, ImageOps
import math

###############################   1   #############################
###############################   2   #############################

def dotted_line(img_arr, x0, y0, x1, y1, color, count):
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        img_arr[y, x] = color

def dotted_line_fix(img_arr, x0, y0, x1, y1, color):
    count = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
    step = 1.0 / count
    for t in np.arange(0, 1, step):
        x = round((1.0 - t) * x0 + t * x1)
        y = round((1.0 - t) * y0 + t * y1)
        img_arr[y, x] = color

def x_loop_line_1(img_arr, x0, y0, x1, y1, color):
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        img_arr[y, x] = color

def x_loop_line_2(img_arr, x0, y0, x1, y1, color):
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        img_arr[y, x] = color

def x_loop_line_3(img_arr, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if (xchange):
            img_arr[x, y] = color
        else:
            img_arr[y, x] = color

def x_loop_line_x2(img_arr, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2.0 * (x1 - x0) * abs(y1 - y0) / (x1 - x0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if (xchange):
            img_arr[x, y] = color
        else:
            img_arr[y, x] = color
        derror += dy
        if (derror > 2.0 * (x1 - x0) * 0.5):
            derror -= 2.0 * (x1 - x0) * 1.0
            y += y_update

def bresenham(img_arr, x0, y0, x1, y1, color):
    xchange = False
    if (abs(x0 - x1) < abs(y0 - y1)):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        xchange = True
    if (x0 > x1):
        x0, x1 = x1, x0
        y0, y1 = y1, y0
    y = y0
    dy = 2 * abs(y1 - y0)
    derror = 0.0
    y_update = 1 if y1 > y0 else -1
    for x in range(x0, x1):
        t = (x - x0) / (x1 - x0)
        y = round((1.0 - t) * y0 + t * y1)
        if (xchange):
            img_arr[x, y] = color
        else:
            img_arr[y, x] = color
        derror += dy
        if (derror > (x1 - x0)):
            derror -= 2 * (x1 - x0)
        y += y_update

h = 2000
w = 2000
img_arr = np.zeros((h, w,3), dtype=np.uint8)

for i in range(13):
    x0 = 300
    y0 = 300
    x1 = int(300 + 95 * math.cos(i * 6.28 / 13))
    y1 = int(300 + 95 * math.sin(i * 6.28 / 13))
    color = 255
    count = 30
    #dotted_line(img_arr,x0,y0,x1,y1,color,count)
    #dotted_line_fix(img_arr, x0, y0, x1, y1, color)
    #x_loop_line_1(img_arr, x0, y0, x1, y1,color)
    #x_loop_line_2(img_arr, x0, y0, x1, y1,color)
    #x_loop_line_3(img_arr, x0, y0, x1, y1,color)
    #x_loop_line_x2(img_arr, x0, y0, x1, y1,color)
    #bresenham(img_arr, x0, y0, x1, y1, color)

###############################   3   #############################
###############################   4   #############################
'''
arr = []
file = open('model_1.obj')
for str in file:
    splitted_str = str.split()
    if (splitted_str[0] == 'v'):
        arr.append([float(splitted_str[1]), float(splitted_str[2]), float(splitted_str[3])])

for i in arr:
    img_arr[int(600 * i[0]) + 500, int(600 * i[1]) + 500] = 255

img = Image.fromarray(img_arr)
img.save('img14.png')
img.show()
'''
###############################   5   #############################
'''
arr = []
file = open('model_1.obj')
for str in file:
    splitted_str = str.split()
    if (splitted_str[0] == 'f'):
        arr.append([splitted_str[1].split('/')[0], splitted_str[2].split('/')[0], splitted_str[3].split('/')[0]])
        print(arr)
'''
###############################   6   #############################

file = open('model_1.obj')
v = []
f = []
for str in file:
    splitted_str = str.split()
    if (splitted_str[0] == 'v'):
        v.append([float(splitted_str[1]), float(splitted_str[2]), float(splitted_str[3])])
    if (splitted_str[0] == 'f'):
        f.append([int(splitted_str[1].split('/')[0]), int(splitted_str[2].split('/')[0]),
                  int(splitted_str[3].split('/')[0])])

for p in f:
    v0 = v[p[0] - 1]
    v1 = v[p[1] - 1]
    v2 = v[p[2] - 1]
    x0 = int(v0[0] * 10000 + 1000)
    y0 = int(v0[1] * 10000 + 1000)
    x1 = int(v1[0] * 10000 + 1000)
    y1 = int(v1[1] * 10000 + 1000)
    x2 = int(v2[0] * 10000 + 1000)
    y2 = int(v2[1] * 10000 + 1000)

    bresenham(img_arr, x0, y0, x1, y1, 255)
    bresenham(img_arr, x1, y1, x2, y2, 255)
    bresenham(img_arr, x2, y2, x0, y0, 255)

img = Image.fromarray(img_arr)
img = ImageOps.flip(img)
img.save('img15.png')
img.show()
