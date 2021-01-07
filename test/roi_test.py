# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2020年12月30日
'''
import matplotlib.pyplot as plot
import matplotlib.patches as mpathes
from PIL import Image
import numpy as np

import data.dataset_rois as rois
from utils.conf import DATASET


rois_creator = rois.RoisCreator()
# rois_creator.create()
file_anchors = rois_creator.test_create(file_name='159af410-cc08-41a8-8156-c563d831a0d0', 
                                        label_file_path=DATASET.get_label_train(), 
                                        count=10, train_positives_iou=0.6)
  
fa = file_anchors[0]
  
file = DATASET.get_in_train() + "/" + fa['file_name'] + ".png"
img = Image.open(file)
img = img.resize((480, 180), Image.ANTIALIAS)

 
fig = plot.figure()
ax = fig.add_subplot(1,1,1)
 
#    打印找到的anchors
print(len(fa['positives']))
for (iou, anchor, label) in fa['positives']:
    rect = plot.Rectangle((anchor[0] - anchor[2]/2, anchor[1] - anchor[3]/2), anchor[2], anchor[3], fill=False, edgecolor = 'red',linewidth=1)
    ax.add_patch(rect)
    pass
 
#    打印label
for label in fa['labels']:
    rect = plot.Rectangle((label[1], label[2]), label[3], label[4], fill=False, edgecolor='blue',linewidth=1)
    ax.add_patch(rect)
    pass
 
#    打印全部的anchors
# for anchor in rois_creator._original_anchors:
#     rect = plot.Rectangle((anchor[0]-anchor[2]/2, anchor[1]-anchor[3]/2), anchor[2], anchor[3], fill=False, edgecolor='green',linewidth=1)
#     ax.add_patch(rect)
# #     plot.scatter(anchor[0], anchor[1])
#     pass
 
plot.imshow(np.asarray(img))
plot.show()
