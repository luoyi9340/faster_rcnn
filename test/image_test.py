# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2020年12月30日
'''
import matplotlib.pyplot as plot
import matplotlib.patches as mpathes
from PIL import Image
import numpy as np



file = "/Users/irenebritney/Desktop/vcode/dataset/num_letter/train/24f1663f-8045-4ade-b555-ae69988ee3fe.png"
img = Image.open(file)
img = img.resize((480, 180), Image.ANTIALIAS)
arr = np.asarray(img)
arr = arr / 255.
plot.figure()
plot.imshow(arr)

plot.figure()
arr2 = (arr - 0.5) / 2
print(arr2.shape)
plot.imshow(arr2)
plot.show()

