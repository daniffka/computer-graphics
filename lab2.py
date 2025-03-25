import numpy as np
from PIL import Image, ImageOps
import math

###############################   2 лаба    #############################

def baricentric_dots(x,y,x0,y0,x1,y1,x2,y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2

def triangle(img_arr,x0,y0,z0,x1,y1,z1,x2,y2,z2,color0,color1,color2):
    xmin = math.floor(min(x0, x1, x2))
    xmax = math.ceil(max(x0, x1, x2))
    ymin = math.floor(min(y0, y1, y2))
    ymax = math.ceil(max(y0, y1, y2))
    if (xmin < 0): xmin = 0
    if (ymin < 0): ymin = 0
    if (xmax > 2000): xmax = 2000
    if (ymax > 2000): ymax = 2000
    for i in range(int(xmin), int(xmax)):
        for j in range(int(ymin), int(ymax)):
            lambda0,lambda1,lambda2=baricentric_dots(i,j,x0,y0,x1,y1,x2,y2)
            if (lambda0>0) and (lambda1>0) and (lambda2>0):
                zc = lambda0 * z0 + lambda1 * z1 + lambda2 * z2
                if zc<=matrix[j][i]:
                    img_arr[j][i]=(color0,color1,color2)
                    matrix[j][i]=zc

l = [0, 0, 1]
def norm_scalar(n):
    global l
    return np.dot(l, n) / (math.sqrt((l[0] ** 2 + l[1] ** 2 + l[2] ** 2)) * math.sqrt((n[0] ** 2 + n[1] ** 2 + n[2] ** 2)))

def get_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2):
    return ((y1 - y2) * (z1 - z0) - (y1 - y0) * (z1 - z2), (z1 - z2) * (x1 - x0) - (x1 - x2) * (z1 - z0),
            (x1 - x2) * (y1 - y0) - (y1 - y2) * (x1 - x0))

h=2000
w=2000

matrix=np.full((h,w), np.inf, dtype=float)
img_arr = np.zeros((h,w,3), dtype=np.uint8)

file = open('../computer-graphics/model_1.obj')
v=[]
f=[]
for str in file:
    splitted_str=str.split()
    if (splitted_str[0]=='v'):
        v.append([float(splitted_str[1]),float(splitted_str[2]),float(splitted_str[3])])
    if (splitted_str[0]=='f'):
        f.append([int(splitted_str[1].split('/')[0]), int(splitted_str[2].split('/')[0]), int(splitted_str[3].split('/')[0])])

for p in f:
    v0 = v[p[0] - 1]
    v1 = v[p[1] - 1]
    v2 = v[p[2] - 1]
    x0 = v0[0]*10000+1000
    y0 = v0[1]*10000+1000
    x1 = v1[0]*10000+1000
    y1 = v1[1]*10000+1000
    x2 = v2[0]*10000+1000
    y2 = v2[1]*10000+1000
    z0 = v0[2]*10000+1000
    z1 = v1[2]*10000+1000
    z2 = v2[2]*10000+1000
    norm=get_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    triangle(img_arr,x0,y0,z0,x1,y1,z1,x2,y2,z2,-255*norm_scalar(norm),122,30)

img=Image.fromarray(img_arr, mode="RGB")
img=ImageOps.flip(img)
img.save('lab2.png')
img.show()







