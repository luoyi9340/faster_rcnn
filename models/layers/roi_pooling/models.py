# -*- coding: utf-8 -*-  
'''
poi_pooling层
    包装一个layer出来是为了让模型直接连续

@author: luoyi
Created on 2021年1月12日
'''
import tensorflow as tf

import utils.conf as conf
from models.layers.roi_pooling.preprocess import roi_pooling


#    poi pooling层
class ROIPooling(tf.keras.layers.Layer):
    def __init__(self, name=None, kernel_size=conf.FAST_RCNN.get_roipooling_kernel_size(), **kwargs):
        '''roi pooling层
            @param kernel_size: 输出的特征图
        '''
        super(ROIPooling, self).__init__(name=name, **kwargs)
        
        self.__kernel_size = kernel_size
        pass
    
    #    前向
    def call(self, x, y, **kwargs):
        return roi_pooling(x, y, roipooling_ksize=self.__kernel_size)
    
    pass