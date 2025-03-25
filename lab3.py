import numpy as np
from PIL import Image, ImageOps
import math

###############################   3 лаба    #############################

def baricentric_dots(x,y,x0,y0,x1,y1,x2,y2):
    lambda0 = ((x - x2) * (y1 - y2) - (x1 - x2) * (y - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda1 = ((x0 - x2) * (y - y2) - (x - x2) * (y0 - y2)) / ((x0 - x2) * (y1 - y2) - (x1 - x2) * (y0 - y2))
    lambda2 = 1.0 - lambda0 - lambda1
    return lambda0, lambda1, lambda2

def triangle(img_arr,x0,y0,z0,x1,y1,z1,x2,y2,z2,color0,color1,color2):
    xmin, xmax, ymin, ymax = math.floor(min(x0, x1, x2)), math.ceil(max(x0, x1, x2)), math.floor(min(y0, y1, y2)), math.ceil(
        max(y0, y1, y2))
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

def rotate(a,b,g,x,y,z,tx,ty,tz):

    X=np.array([[1,0,0], [0,math.cos(a),math.sin(a)], [0, -math.sin(a), math.cos(a)]])
    Y=np.array([[math.cos(b), 0, math.sin(b)], [0,1,0], [-math.sin(b), 0, math.cos(b)]])
    Z=np.array([[math.cos(g), math.sin(g), 0], [-math.sin(g),math.cos(g), 0], [0,0,1]])

    R=np.dot(X,Y)
    R1=np.dot(R,Z)
    XYZ=np.array([x,y,z])
    tR=np.array([tx,ty,tz])
    return np.dot(R1,XYZ)+tR


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

for p in v:
    p[0],p[1],p[2]=rotate(0, 45, 15, p[0], p[1], p[2], 0, 0.01, 0.8)

normals=[]
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
    norm = get_normal(x0, y0, z0, x1, y1, z1, x2, y2, z2)
    normals.append(norm)

U0=w/2
V0=h/2
ax=8000
ay=8000
i=0
for p in f:
    v0 = v[p[0] - 1]
    v1 = v[p[1] - 1]
    v2 = v[p[2] - 1]

    z0 = v0[2]
    z1 = v1[2]
    z2 = v2[2]
    x0 = (v0[0]*ax)/z0+U0
    y0 = (v0[1]*ay)/z0+V0
    x1 = (v1[0]*ax)/z1+U0
    y1 = (v1[1]*ay)/z1+V0
    x2 = (v2[0]*ax)/z2+U0
    y2 = (v2[1]*ay)/z2+V0
    norma=normals[i]
    triangle(img_arr, x0, y0, z0, x1, y1, z1, x2, y2, z2, -255 * norm_scalar(norma), 122, 30)
    i=i+1
img=Image.fromarray(img_arr, mode="RGB")
img=ImageOps.flip(img)
img.save('lab3.png')
img.show()







