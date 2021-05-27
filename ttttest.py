import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import json
import cv2
import random
from tqdm import tqdm

pic = np.load('/Users/makangbo/Desktop/ceshipic.npy')
prpic = np.load('/Users/makangbo/Desktop/ceshiprpic.npy')
bbox = np.load('/Users/makangbo/Desktop/ceshibbox.npy')
skeleton = np.load('/Users/makangbo/Desktop/ceshiskeleton.npy')
t_skeleton = np.load('/Users/makangbo/Desktop/ceshiskele.npy')

print(pic.shape)
print(prpic.shape)
print(bbox)
print('==')
print(skeleton)
print('==')
print(t_skeleton)

'''
img =  np.array(prpic).astype('uint8')
img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)  
cv2.imshow('a',img)
cv2.waitKey(500)
'''

def show_skeleton(img,kpts,color=(255,128,128)):
    skelenton = [[2, 0], [0, 4], [0, 3], [4, 8], [8, 12], [3, 7], [7, 11], [0, 1], [1, 6],
                 [1, 5], [6, 10], [10, 14], [5, 9], [9, 13]]
    points_num = [num for num in range(1,16)]
    for sk in skelenton:
        pos1 = (int(kpts[sk[0]][0]), int(kpts[sk[0]][1]))
        pos2 = (int(kpts[sk[1]][0]), int(kpts[sk[1]][1]))
        if pos1[0] > 0 and pos1[1] > 0 and pos2[0] > 0 and pos2[1] > 0:
            cv2.line(img, pos1, pos2, color, 2, 8)
    for points in points_num:
        pos = (int(kpts[points-1][0]),int(kpts[points-1][1]))
        if pos[0] > 0 and pos[1] > 0 :
            cv2.circle(img, pos,4,(0,0,255),-1) #为肢体点画红色实心圆
    return img


skeleton_color = [(154, 194, 182),
                  (123, 151, 138),
                  (0,   208, 244),
                  (8,   131, 229),
                  (18,  87,  220)]  # 选择自己喜欢的颜色

kps = [[125.37481624, 29.11984701],
         [113.82179584, 82.78887795],
         [133.62368389,  13.42138084],
         [124.22854826 , 34.22850289],
         [105.77149833 , 27.77141551],
         [112.28572014,  96.45708053],
         [112.11427996,  93.14285054],
         [135.88322312 , 68.76421825],
         [ 87.30727389 , 31.22311439],
         [108.56065737 ,137.07388475],
         [135.72597343 ,120.19419471],
         [160.26325205 , 83.08358817],
         [107.13205441  ,42.84036682],
         [ 79.15493194 ,162.61007185],
         [140.28667777 ,161.49631475]]



color = random.choice(skeleton_color)
image = show_skeleton(pic, kps, color=color)
image2 = show_skeleton(prpic, kps, color=color)

cv2.imwrite("/Users/makangbo/Desktop/mpi_3.png", image)
cv2.imwrite("/Users/makangbo/Desktop/mpi_2.png", image2)
