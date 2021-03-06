# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2020年12月30日
'''
import sys
import os
#    取项目根目录
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)).split('faster_rcnn')[0]
ROOT_PATH = ROOT_PATH + "faster_rcnn"
sys.path.append(ROOT_PATH)

import matplotlib.pyplot as plot
from PIL import Image
import numpy as np
np.set_printoptions(suppress=True, threshold=16)

import data.dataset_rois as rois
import utils.conf as conf
import models.layers.rpn.preprocess as preprocess
import utils.alphabet as alphabet



#    单独展示anchor结果
def show_anchors(fa, 
                 is_show_positive=True,
                 is_show_negative=True,
                 count_negative=100,
                 is_show_labels=True,
                 is_show_anchors=True,
                 is_show_anchors_center=True):
    file = conf.DATASET.get_in_train() + "/" + fa['file_name'] + ".png"
    img = Image.open(file)
    img = img.resize((480, 180), Image.ANTIALIAS)
     
    fig = plot.figure()
    ax = fig.add_subplot(1,1,1)
     
    #    打印找到的anchors
    if (is_show_positive):
        print('positives:', len(fa['positives']))
        for (_, anchor, label) in fa['positives']:
            print(anchor[0], anchor[1], anchor[2], anchor[3])
            rect = plot.Rectangle((anchor[0] - anchor[2]/2, anchor[1] - anchor[3]/2), anchor[2], anchor[3], fill=False, edgecolor = 'red',linewidth=1)
            ax.add_patch(rect)
            pass
        pass
    #    打印找到的anchors
    if (is_show_negative):
        print('negative:', len(fa['negative']))
        i = 0
        for (_, anchor) in fa['negative']:
            if (i > count_negative): break;
            i += 1
            rect = plot.Rectangle((anchor[0] - anchor[2]/2, anchor[1] - anchor[3]/2), anchor[2], anchor[3], fill=False, edgecolor = 'green',linewidth=1)
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
            rect = plot.Rectangle((anchor[0]-anchor[2]/2, anchor[1]-anchor[3]/2), anchor[2], anchor[3], fill=False, edgecolor='yellow',linewidth=1)
            ax.add_patch(rect)
            pass
        pass
    
    #    打印全部anchor中心点
    if (is_show_anchors_center):
        prev_x, prev_y = None, None
        for anchor in rois_creator._original_anchors:
            if (prev_x == anchor[0] and prev_y == anchor[1]): continue
            plot.scatter(anchor[0], anchor[1])
            prev_x, prev_y = anchor[0], anchor[1]
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
    avg_p = num_p/num if num > 0 else 0
    avg_p_4l = num_p_4l/num_4l if num_4l > 0 else 0
    avg_p_5l = num_p_5l/num_5l if num_5l > 0 else 0
    avg_p_6l = num_p_6l/num_6l if num_6l > 0 else 0
    print('avg_p:', avg_p, ' avg_p_4l:', avg_p_4l, ' avg_p_5l:', avg_p_5l, ' avg_p_6l:', avg_p_6l)
    pass

roi_areas = [64, 68, 76, 80, 84, 92]
roi_scales = [0.8, 1]
rois_creator = rois.RoisCreator(roi_areas=roi_areas, roi_scales=roi_scales, cnns_feature_map_scaling=8)
# # # rois_creator.create()
file_anchors = rois_creator.test_create(label_file_path=conf.DATASET.get_label_train(),
#                                         file_name='7f1119ea-66f9-4a35-888e-9c9a95c297b5', 
                                        count=10, 
                                        train_positives_iou=0.65,
                                        train_negative_iou=0.3)
fa = file_anchors[2]
show_anchors(fa, 
             is_show_positive=True, 
             is_show_negative=False, 
             is_show_labels=True, 
             is_show_anchors=False,
             is_show_anchors_center=True)
# show_msg(file_anchors)



def show_rois(X, Y, show_P=True, show_N=True, show_L=True):
    '''
        @param x: 图片矩阵
        @param y: rois数据
                    [
                        [IoU, x, y, w, h, idx_w, idx_h, idx_area, idx_scales, vcode_index, x, y, w, h]
                        ...
                    ]
    '''
    fig = plot.figure()
    ax = fig.add_subplot(1,1,1)
     
    d = {}
    for y in Y:
        #    展示anchors
        #    展示负样本
        if (show_N and y[9] < 0):
            rect = plot.Rectangle((y[1] - y[3]/2, y[2] - y[4]/2), y[3], y[4], fill=False, edgecolor='green',linewidth=1)
            ax.add_patch(rect)
            pass
        #    展示正样本
        if (show_P and y[9] >= 0):
            rect = plot.Rectangle((y[1] - y[3]/2, y[2] - y[4]/2), y[3], y[4], fill=False, edgecolor='red',linewidth=1)
            ax.add_patch(rect)
            pass
        #    展示label
        if (show_L and y[9] >= 0 and d.get(y[9]) is None):
            d[y[9]] = 1
            print(alphabet.index_category(int(y[9])), y[10], y[11], y[12], y[13])
            rect = plot.Rectangle((y[10], y[11]), y[12], y[13], fill=False, edgecolor='blue',linewidth=1)
            ax.add_patch(rect)
            pass
        pass
     
    #    展示图片
    X = X / 255.
    plot.imshow(X)
    plot.show()
    pass
 
#    读一个已经生成好的rois.jsons试一下
# image_dir = conf.DATASET.get_in_train()
# count = conf.DATASET.get_count_train()
# rois_out = conf.ROIS.get_train_rois_out()
# is_rois_mutiple_file = False
# count_positives = 48
# count_negative = 48
# batch_size = conf.ROIS.get_batch_size()
# db = rois.read_rois_generator(count, rois_out, is_rois_mutiple_file, image_dir, count_positives, count_negative, batch_size, 
#                                    x_preprocess=None, 
#                                    y_preprocess=None)
# 
# show_idx = 1
# idx = 0
# for x, y in db:
#     if (idx >= show_idx):
#         show_rois(x, y, show_N=False)
#         break
#     idx += 1
#     pass






