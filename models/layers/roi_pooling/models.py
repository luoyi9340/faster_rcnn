# -*- coding: utf-8 -*-  
'''
poi_pooling层

@author: luoyi
Created on 2021年1月12日
'''
import tensorflow as tf


#    poi pooling层
class ROIPooling(tf.keras.layers.Layer):
    def __init__(self, name=None, op='max', out_size=(7, 7), **kwargs):
        '''
            @param kernel_size: 输出的特征图
            @param op: 每个小区域执行的操作 max | avg
        '''
        super(ROIPooling, self).__init__(name=name, **kwargs)
        
        self.__out_size = out_size
        self.__op = op.lower()
        
        pass
    
    #    前向
    def call(self, x, training=None, mask=None):
        #    取h轴，w轴的切分系数
        (h_s, w_s) = self.__split_coefficient(x, self.__out_size)
        print(h_s, w_s)
        
        #    先切h轴，再切w轴
        x_h = tf.split(x, h_s, axis=1)
        idx_h, idx_w = 0, 0
        T = []
        for xh in x_h:
            idx_w = 0
            x_h = tf.split(xh, w_s, axis=2)
            t = []
            for xw in x_h:
                if (self.__op == 'avg'): t.append(tf.nn.avg_pool2d(xw, ksize=[h_s[idx_h], w_s[idx_w]], strides=[h_s[idx_h], w_s[idx_w]], padding='VALID'))
                else: t.append(tf.nn.max_pool2d(xw, ksize=[h_s[idx_h], w_s[idx_w]], strides=[h_s[idx_h], w_s[idx_w]], padding='VALID'))
                idx_w += 1
                pass
            T.append(t)
            idx_h += 1
            pass
        
        #    先拼w轴，再拼h轴
        res = []
        for t in T:
            res.append(tf.concat(t, axis=2))
            pass
        res = tf.concat(res, axis=1)
        
        #    验算：(res.shape[1], res.shape[2]) == out_shape
        if (res.shape[1] != self.__out_size[0] \
            or res.shape[2] != self.__out_size[1]):
            raise Exception(self.name + " out_size:" + str(self.__out_size) + " not equal y:" + str(res.shape))
            pass
        
        return res
    
    #    取h方向、w方向各自切分系数
    def __split_coefficient(self, x, out_size):
        #    取输入尺寸 输出尺寸
        (_, H, W, _) = x.shape
        (kh, kw) = out_size
        #    h轴切分系数（每个max_pooling的h）
        hu = round(H / kh)
        h_s = [hu for _ in range(kh)]
        h_s[-1] += H - kh * hu
        #    w轴切分系数（每个max_pooling的w）
        wu = round(W / kw)
        w_s = [wu for _ in range(kw)]
        w_s[-1] += W - kw * wu
        return (h_s, w_s)
    pass