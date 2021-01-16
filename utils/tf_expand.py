# -*- coding: utf-8 -*-  
'''
tf扩展

@author: luoyi
Created on 2021年1月13日
'''
import tensorflow as tf
import numpy as np

import utils.logger_factory as logf


log = logf.get_logger('log_name')



#    最近的N个batch的平均loss，比最近的M个batch的平均loss，大于阈值，连续出现times次。则调整学习率为之前的R倍
class AdjustLRWithBatchCallback(tf.keras.callbacks.Callback):
    def __init__(self, n_batch=100, m_batch=200, threshold=0.9999, times=5, adjust_rate=0.1):
        '''最近的M个batch的平均loss，比最近的M个batch的平均loss，大于阈值，连续出现times次。则调整学习率为之前的R倍
            @param n_batch: 最近的N个batch的平均loss
            @param m_batch: 最近的M个batch的平均loss
            @param threshold: 阈值（等于1表示只判断loss不下降的情况）
            @param times: 连续出现times次
            @param adjust_rate: 调整比例 new_lr = lr * adjust_rate
        '''
        self.__n_batch = n_batch
        self.__m_batch = m_batch
        self.__threshold = threshold
        self.__adjust_rate = adjust_rate
        self.__times = times
        
        self.__loss_his = []
        self.__loss_his_len = 0
        self.__continuation = 0                 #    连续出现几次
        pass
    
    #    每轮batch结束之后记录当前batch的loss
    def on_batch_end(self, batch, logs=None):
        #    记录loss
        self.__loss_his.append(logs['loss'])
        self.__loss_his_len += 1
        
        #    判断是否n和m是否够数了
        if (self.__loss_his_len > self.__m_batch):
            #    判断本次的 n_batch平均loss / m_batch平均loss 是否达到阈值
            if (self.__is_adjust_lr(loss_his=self.__loss_his, 
                                  n_batch=self.__n_batch, 
                                  m_batch=self.__m_batch, 
                                  threshold=self.__threshold)):
                self.__continuation += 1
                #    判断是否
                if (self.__continuation > self.__times):
                    self.__adjust_lr(adjust_rate=self.__adjust_rate)
                    pass
                pass
            else:
                self.__reset_according()
            pass
        pass
    
    #    判断是否调整学习率
    def __is_adjust_lr(self, loss_his=[], n_batch=10, m_batch=20, threshold=0.9):
        #    检测loss_his中最后n_batch个loss的平均 / 最后m_batch个loss的平均是否 > threshold
        loss_his = np.array(loss_his)
        last_n_loss = loss_his[n_batch:]
        lass_m_loss = loss_his[m_batch:]
        last_n_loss_mean = np.mean(last_n_loss)
        lass_m_loss_mean = np.mean(lass_m_loss)
        r = last_n_loss_mean / lass_m_loss_mean
        return r > threshold
    
    #    调整学习率为之前的adjust_rate倍
    def __adjust_lr(self, adjust_rate=0.1):
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        new_lr = lr * adjust_rate
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
        
        self.__reset_according()
        pass
    
    #    重置条件
    def __reset_according(self):
        #    清空loss_his，loss_his计数置零
        del self.__loss_his
        self.__loss_his = []
        self.__loss_his_len = 0
        
        #    连续次数置零
        self.__continuation = 0
        pass
        
    pass
