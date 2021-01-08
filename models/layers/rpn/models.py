# -*- coding: utf-8 -*-  
'''
RPN模型定义

@author: luoyi
Created on 2020年12月31日
'''
import tensorflow as tf

import utils.conf as conf


#    RPN网络
class RPNNet(tf.keras.layers.Layer):
    '''RPN网络
        ======================= CNNs =======================
        ----------------- layer cnns-------------------
        ResNet34 | ResNet50 | VGGNet16
        ======================= cls + reg =======================
        conv:[3*3*512] strides=1 padding=1 out=(上层输出)
        分为两支：
            分支1：
            ----------------- layer cls-------------------
            conv:[1*1* K*2] strides=1 padding=0 out=(上层输出特征图尺寸, K*2)
            active: Softmax
            loss: 多分类交叉熵
            分支2：
            ----------------- layer reg-------------------
            conv:[1*1* K*4] strides=1 padding=0 out=(上层输出特征图尺寸, K*4)
            loss: smooth L1
            
    '''
    def __init__(self, training=True, input_shape=(12, 30, 512), K=conf.RPN.get_K(), loss_lamda=conf.RPN.get_loss_lamda(), kernel_initializer = tf.initializers.HeNormal(), bias_initializer = tf.initializers.Zeros(), **kwargs):
        '''
            @param training: 本层是否参与训练
            @param input_shape: 输入尺寸
            @param K: 每个点anchor个数
        '''
        super(RPNNet, self).__init__(name='RPNNet', **kwargs)
        
        self.__training = training
        
        #    输入shape
        self.__input_shape = input_shape
        #    特征图每个点代表多少个anchor
        self.__K = K
        #    Lcls + λ*Lreg中间的λ
        self.__loss_lamda = loss_lamda
        
        self.__kernel_initializer = kernel_initializer
        self.__bias_initializer = bias_initializer
        
        #    计算输出形状
        #    (h, w, 2, K)为layer_cls输出
        #    (h, w, 4, K)为layer_reg输出
        self.__output_shape = (self.__input_shape[0], self.__input_shape[1], 6, self.__K)
        
        #    装配网络
        self.__assembling()
        
        pass
    
    #    装配网络
    def __assembling(self):
        #    上层输出通道数，本层原样输出
        filters = self.__input_shape[2]
        
        #    过3*3卷积
        self.__layer_conv = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(name=self.name + '_Conv1', filters=filters, kernel_size=(3, 3), strides=1, padding='same', input_shape=self.__input_shape, kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                tf.keras.layers.BatchNormalization(name=self.name + '_BN1', trainable=self.__training),
                tf.keras.layers.ReLU(name=self.name + '_ReLU1')
            ], name=self.name + '_layer_conv')
        
        #    cls分支
        filters_cls = self.__K * 2
        #    告诉softmax前K个值是前景概率，后K个值是背景概率。这里吧输出变为(batch_size, h, w, 2, K)，让softmax直接在-2维上做
        #    ps：瞬间明白了论文中为什么要加reshape，看来学东西最有效的办法就是自己弄一遍。。。
        target_shape_cls = (self.__input_shape[0], self.__input_shape[1], 2, self.__K)
        self.__layer_cls = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(name=self.name + '_cls_conv', filters=filters_cls, kernel_size=(1, 1), strides=1, padding='same', kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                tf.keras.layers.Reshape(target_shape=target_shape_cls),         #    将输出从(batch_size, h, w, K*2)reshape为(batch_size, h, w, 2, K)的格式
                                                                                #    下面的Softmax直接在-2维上做softmax函数
                                                                                #    结果即为第1个K为前景概率，第2个K为背景概率。(None, h, w, 0, i) + (None, h, w, 1, i) = 1
                                                                                #    否则(batch_size, h, w, K*2)的最后一维那K*2个结果加起来=1
                tf.keras.layers.Softmax(axis=-2)
                #    此时的输出为(batch_size, h, w, 2, K)
            ], name=self.name + '_layer_cls')
        
        #    reg分支
        filters_reg = self.__K * 4
        target_shape_reg = (self.__input_shape[0], self.__input_shape[1], 4, self.__K)
        self.__layer_reg = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(name=self.name + '_reg_conv', filters=filters_reg, kernel_size=(1, 1), strides=1, padding='same', kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                tf.keras.layers.Reshape(target_shape=target_shape_reg)
                #    此时的输出为(batch_size, h, w, 4, K)
            ], name=self.name + '_layer_reg')
        
        pass
    
    #    前向传播
    def call(self, x, training=None, mask=None):
        y = self.__layer_conv(x, training=training, mask=mask)
        #    分类前向结果
        y_cls = self.__layer_cls(y, training=training, mask=mask)
        #    回归前向结果
        y_reg = self.__layer_reg(y, training=training, mask=mask)
        
        #    返回值不支持tuple形式。这里先拼接成(batch_size, h, w, 2+4, K)形式，后面再拆分
#         return (y_cls, y_reg)
        y_res = tf.concat([y_cls, y_reg], axis=3)
        return y_res
    
    #    输出形状
    def get_output_shape(self):
        return self.__output_shape
    
    pass



