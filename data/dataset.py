# -*- coding: utf-8 -*-  
'''
图片验证码数据集
    图片分为3种尺寸：
        480*180：4个字母
        600*180：5个字母
        720*180：6个字母
    标注文件为json格式：
        {fileName:'${fileName}', vcode='${vcode}', annos:[{key:'值', x:x, y:y, w:w, h:h}, {key:'值', x:x, y:y, w:w, h:h}...]}
        其中：
            fileName：文件名（不含png）
            vcode：验证码
            annos：矩形框信息
                key：矩形框中的文字
                x：左上w坐标
                y：左上h坐标
                w：矩形框宽度
                h：矩形框高度

@author: luoyi
Created on 2020年12月29日
'''
import json
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plot

from utils.conf import DATASET
from utils.alphabet import ALPHABET, category_index


#    标签文件迭代器
def label_file_iterator(label_file_path=DATASET.get_label_train(),
                        count=DATASET.get_count_train()):
    '''标签文件迭代器
        json格式
        {
            fileName:'${fileName}', 
            vcode='${vcode}', 
            annos:[
                    {key:'值', x:x, y:y, w:w, h:h}, 
                    {key:'值', x:x, y:y, w:w, h:h}...
                ]
        }
        @param label_file_path: 标签文件
        @param image_dir: 图片目录
    '''
    i = 0
    for line in open(label_file_path, mode='r', encoding='utf-8'):
        if (i >= count): break
        i += 1
        
        j = json.loads(line)
        
        file_name = j['fileName']
        vcode = j['vcode']
        annos = j['annos']
        labels = []
        for anno in annos:
            labels.append((anno['key'], anno['x'], anno['y'], anno['w'], anno['h']))
            pass
        
        yield file_name, vcode, labels
        pass
    pass

