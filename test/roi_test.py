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



#    单独展示anchor结果
def show_anchors(fa, 
                 is_show_positive=True,
                 is_show_negative=True,
                 count_negative=100,
                 is_show_labels=True,
                 is_show_anchors=True):
    file = DATASET.get_in_train() + "/" + fa['file_name'] + ".png"
    img = Image.open(file)
    img = img.resize((480, 180), Image.ANTIALIAS)
     
    fig = plot.figure()
    ax = fig.add_subplot(1,1,1)
     
    #    打印找到的anchors
    if (is_show_positive):
        print(len(fa['positives']))
        for (iou, anchor, label) in fa['positives']:
            rect = plot.Rectangle((anchor[0] - anchor[2]/2, anchor[1] - anchor[3]/2), anchor[2], anchor[3], fill=False, edgecolor = 'red',linewidth=1)
            ax.add_patch(rect)
            pass
        pass
    #    打印找到的anchors
    if (is_show_negative):
        print(len(fa['negative']))
        i = 0
        for (iou, anchor) in fa['negative']:
            if (i > count_negative): break;
            i += 1
            rect = plot.Rectangle((anchor[0] - anchor[2]/2, anchor[1] - anchor[3]/2), anchor[2], anchor[3], fill=False, edgecolor = 'red',linewidth=1)
            ax.add_patch(rect)
            pass
        pass
    #    打印label
    if (is_show_labels):
        for label in fa['labels']:
            rect = plot.Rectangle((label[1], label[2]), label[3], label[4], fill=False, edgecolor='blue',linewidth=1)
            ax.add_patch(rect)
            pass
        pass
    #    打印全部的anchors
    if (is_show_anchors):
        for anchor in rois_creator._original_anchors:
            rect = plot.Rectangle((anchor[0]-anchor[2]/2, anchor[1]-anchor[3]/2), anchor[2], anchor[3], fill=False, edgecolor='green',linewidth=1)
            ax.add_patch(rect)
        #     plot.scatter(anchor[0], anchor[1])
            pass
        pass
     
    plot.imshow(np.asarray(img))
    plot.show()
    pass


#    打印每张图的基本信息
def show_msg(anchors_iterator):
    num_p, num_p_4l, num_p_5l, num_p_6l = 0, 0, 0, 0
    num, num_4l, num_5l, num_6l = 0, 0, 0, 0
    for fa in anchors_iterator:
        num += 1
        num_p += len(fa['positives'])
        if (len(fa['labels']) == 4):
            num_4l += 1
            num_p_4l += len(fa['positives'])
        elif (len(fa['labels']) == 5):
            num_5l += 1
            num_p_5l += len(fa['positives'])
        elif (len(fa['labels']) == 6):
            num_6l += 1
            num_p_6l += len(fa['positives'])

        print('file_name:', fa['file_name'], ' positives:', len(fa['positives']), ' negative:', len(fa['negative']), ' lables:', len(fa['labels']))
        pass
    print('avg_p:', num_p/num, ' avg_p_4l:', num_p_4l/num_4l, ' avg_p_5l:', num_p_5l/num_5l, ' avg_p_6l:', num_p_6l/num_6l)
    pass


rois_creator = rois.RoisCreator()
# rois_creator.create()
file_anchors = rois_creator.test_create(label_file_path=DATASET.get_label_train(), 
#                                         file_name='159af410-cc08-41a8-8156-c563d831a0d0', 
                                        count=100, 
                                        train_positives_iou=0.7,
                                        train_negative_iou=0.05)
fa = file_anchors[0]
show_anchors(fa, 
             is_show_positive=False, 
             is_show_negative=True, 
             is_show_labels=False, 
             is_show_anchors=False)

