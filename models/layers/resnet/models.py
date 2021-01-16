# -*- coding: utf-8 -*-  
'''
ResNet网络
    - ResNet 34
    - ResNet 50

@author: luoyi
Created on 2020年12月31日
'''
import tensorflow as tf
import math

from utils.conf import CNNS
from models.layers.resnet.part import BasicBlock, Bottleneck


#    ResNet34网络结果
class ResNet34(tf.keras.layers.Layer):
    '''ResNet34网络结构（所有卷积核均不改变原图尺寸，要改变尺寸时追加max pooling）
        输入：180 * 480 * 3（h * w * c）
        base_channel_num=32
        ======================= 1层 =======================
        -----------------layer 1-------------------
        Conv:
            kernel_size=[3*3* base_channel_num] stride=2 padding=1 active=relu norm=bn
            out=[90 * 240 * base_channel_num]
            此时缩放比例为：2
        ======================= 2层 =======================
        -----------------BasicBlock 1*3-------------------
        Conv:[3*3* base_channel_num] stride=1 padding=1 active=relu norm=bn out=[90 * 240 * base_channel_num]
        Conv:[3*3* base_channel_num] stride=1 padding=1 norm=bn out=[90 * 240 * base_channel_num]   
        shortcut: out=[90 * 240 * base_channel_num]   
        active: relu
        times: 3（该层重复3次）
        ======================= 3层 =======================
        -----------------BasicBlock 2-------------------
        Conv:[3*3* base_channel_num*2] stride=2 padding=1 active=relu norm=bn out=[45 * 120 * base_channel_num*2]
        Conv:[3*3* base_channel_num*2] stride=1 padding=1 norm=bn out=[45 * 120 * base_channel_num*2]   
        shortcut: out=[45 * 120 * base_channel_num*2]   
        active: relu
        -----------------BasicBlock 2*3-------------------
        Conv:[3*3* base_channel_num*2] stride=1 padding=1 active=relu norm=bn out=[45 * 120 * base_channel_num*2]
        Conv:[3*3* base_channel_num*2] stride=1 padding=1 norm=bn out=[45 * 120 * base_channel_num*2]   
        shortcut: out=[45 * 120 * base_channel_num*2]   
        active: relu
        times: 3（该层重复3次）
        此时缩放比例为：4
        ======================= 4层 =======================
        -----------------BasicBlock 3-------------------
        Conv:[3*3* base_channel_num*4] stride=2 padding=1 active=relu norm=bn out=[23 * 60 * base_channel_num*4]
        Conv:[3*3*128] stride=1 padding=1 norm=bn out=[23 * 60 * base_channel_num*4]   
        shortcut: out=[23 * 60 * base_channel_num*4]   
        active: relu
        -----------------BasicBlock 3*5-------------------
        Conv:[3*3* base_channel_num*4] stride=1 padding=1 active=relu norm=bn out=[23 * 60 * base_channel_num*4]
        Conv:[3*3* base_channel_num*4] stride=1 padding=1 norm=bn out=[23 * 60 * base_channel_num*4]   
        shortcut: out=[23 * 60 * base_channel_num*4]   
        active: relu
        times: 5（该层重复5次）
        此时缩放比例为：8
        ======================= 5层 =======================
        -----------------BasicBlock 4-------------------
        Conv:[3*3* base_channel_num*8] stride=2 padding=1 active=relu norm=bn out=[12 * 30 * base_channel_num*8]
        Conv:[3*3* base_channel_num*8] stride=1 padding=1 norm=bn out=[12 * 30 * base_channel_num*8]   
        shortcut: out=[12 * 30 * base_channel_num*8]   
        active: relu
        -----------------BasicBlock 4*2-------------------
        Conv:[3*3* base_channel_num*8] stride=1 padding=1 active=relu norm=bn out=[12 * 30 * base_channel_num*8]
        Conv:[3*3* base_channel_num*8] stride=1 padding=1 norm=bn out=[12 * 30 * base_channel_num*8]   
        shortcut: out=[12 * 30 * base_channel_num*8]   
        active: relu
        times: 2（该层重复2次）
        此时缩放比例为：16
        -----------------layer 2-------------------
        没了，就这么直接输出给外面的网络
        
    '''
    def __init__(self, training=True, scaling=CNNS.get_feature_map_scaling(), base_channel_num=CNNS.get_base_channel_num(), kernel_initializer = tf.initializers.he_normal(), bias_initializer = tf.initializers.Zeros(), **kwargs):
        '''
            @param training: 网络是否参与训练
            @param scaling: 特征图缩放比例
            @param base_channel_num: 基础通道数（后层的通道数在此基础上 * n）
            @param kernel_initializer: 卷积核参数初始化方式（默认he_normal）
            @param bias_initializer: 偏置项初始化方式（默认zero）
        '''
        super(ResNet34, self).__init__(name='ResNet34', **kwargs)
        
        self.__kernel_initializer = kernel_initializer
        self.__bias_initializer = bias_initializer
        
        self.__training = training

        #    根据缩放比例计算网络层数（第1层与第2层缩放比例一致，所以加1）
        self.__num_layer = int(math.log(scaling, 2)) + 1
        self.__scaling = scaling
        
        #    装配网络
        self.__assembling(base_channel_num)
        
        pass
    
    #    装配网络
    def __assembling(self, base_channel_num):
        #    第1层
        filters_layer1 = base_channel_num * 1
        if (self.__num_layer >= 1):
            self.__layer1 = tf.keras.models.Sequential([
                    #    坑，padding放在入口处load_weights会出错。。。
#                     tf.keras.layers.ZeroPadding2D(1),
                    tf.keras.layers.Conv2D(name=self.name + '_layer1_conv1', filters=filters_layer1, kernel_size=(3, 3), strides=2, padding='valid', input_shape=(182, 482, 3), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    tf.keras.layers.BatchNormalization(name=self.name + '_BN1', trainable=self.__training),
                    tf.keras.layers.ReLU(name=self.name + '_ReLU1')
                ], name=self.name + '_layer1')
            pass
        self.output_shape_layer1 = (90, 240, filters_layer1)

        #    第2层
        filters_layer2 = base_channel_num * 1
        if (self.__num_layer >= 2):
            self.__layer2 = tf.keras.models.Sequential([
                    BasicBlock(name=self.name + '_layer2_BasicBlock1', filters=[filters_layer2, filters_layer2], strides=1, input_shape=(90, 240, filters_layer1), output_shape=(90, 240, filters_layer2), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    BasicBlock(name=self.name + '_layer2_BasicBlock2', filters=[filters_layer2, filters_layer2], strides=1, input_shape=(90, 240, filters_layer2), output_shape=(90, 240, filters_layer2), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    BasicBlock(name=self.name + '_layer2_BasicBlock3', filters=[filters_layer2, filters_layer2], strides=1, input_shape=(90, 240, filters_layer2), output_shape=(90, 240, filters_layer2), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training)
                ], name=self.name + '_layer2')
            pass
        self.output_shape_layer2 = (90, 240, filters_layer2)
        
        #    第3层
        filters_layer3 = base_channel_num * 2
        if (self.__num_layer >= 3):
            self.__layer3 = tf.keras.models.Sequential([
                    BasicBlock(name=self.name + '_layer3_BasicBlock1', filters=[filters_layer3, filters_layer3], strides=2, input_shape=(90, 240, filters_layer2), output_shape=(45, 120, filters_layer3), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    BasicBlock(name=self.name + '_layer3_BasicBlock2', filters=[filters_layer3, filters_layer3], strides=1, input_shape=(45, 120, filters_layer3), output_shape=(45, 120, filters_layer3), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    BasicBlock(name=self.name + '_layer3_BasicBlock3', filters=[filters_layer3, filters_layer3], strides=1, input_shape=(45, 120, filters_layer3), output_shape=(45, 120, filters_layer3), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    BasicBlock(name=self.name + '_layer3_BasicBlock4', filters=[filters_layer3, filters_layer3], strides=1, input_shape=(45, 120, filters_layer3), output_shape=(45, 120, filters_layer3), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training)
                ], name=self.name + '_layer3')
            pass
        self.output_shape_layer3 = (45, 120, filters_layer3)
        
        #    第4层
        filters_layer4 = base_channel_num * 4
        if (self.__num_layer >= 4):
            self.__layer4 = tf.keras.models.Sequential([
                    BasicBlock(name=self.name + '_layer4_BasicBlock1', filters=[filters_layer4, filters_layer4], strides=2, input_shape=(45, 120, filters_layer3), output_shape=(23, 60, filters_layer4), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    BasicBlock(name=self.name + '_layer4_BasicBlock2', filters=[filters_layer4, filters_layer4], strides=1, input_shape=(23, 60, filters_layer4), output_shape=(23, 60, filters_layer4), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    BasicBlock(name=self.name + '_layer4_BasicBlock3', filters=[filters_layer4, filters_layer4], strides=1, input_shape=(23, 60, filters_layer4), output_shape=(23, 60, filters_layer4), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    BasicBlock(name=self.name + '_layer4_BasicBlock4', filters=[filters_layer4, filters_layer4], strides=1, input_shape=(23, 60, filters_layer4), output_shape=(23, 60, filters_layer4), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    BasicBlock(name=self.name + '_layer4_BasicBlock5', filters=[filters_layer4, filters_layer4], strides=1, input_shape=(23, 60, filters_layer4), output_shape=(23, 60, filters_layer4), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    BasicBlock(name=self.name + '_layer4_BasicBlock6', filters=[filters_layer4, filters_layer4], strides=1, input_shape=(23, 60, filters_layer4), output_shape=(23, 60, filters_layer4), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training)
                ], name=self.name + '_layer4')
            pass
        self.output_shape_layer4 = (23, 60, filters_layer4)
        
        #    第5层
        filters_layer5 = base_channel_num * 8
        if (self.__num_layer >= 5):
            self.__layer5 = tf.keras.models.Sequential([
                    BasicBlock(name=self.name + '_layer5_BasicBlock1', filters=[filters_layer5, filters_layer5], strides=2, input_shape=(23, 60, filters_layer4), output_shape=(12, 30, filters_layer5), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    BasicBlock(name=self.name + '_layer5_BasicBlock2', filters=[filters_layer5, filters_layer5], strides=1, input_shape=(12, 30, filters_layer5), output_shape=(12, 30, filters_layer5), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    BasicBlock(name=self.name + '_layer5_BasicBlock3', filters=[filters_layer5, filters_layer5], strides=1, input_shape=(12, 30, filters_layer5), output_shape=(12, 30, filters_layer5), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training)
                ], name=self.name + '_layer5')
            pass
        self.output_shape_layer5 = (12, 30, filters_layer5)
        pass
    
    #    前向传播
    def call(self, x, training=None, mask=None):
        #    ZeroPadding放在入口处load_weights会出错，第一个padding只能放在这。。。
        x = tf.pad(x, paddings=[[0,0], [1,1], [1,1], [0,0]])
        
        if (self.__num_layer >= 1):
            y = self.__layer1(x, training=training, mask=mask)
        
        if (self.__num_layer >= 2):
            y = self.__layer2(y, training=training, mask=mask)
        
        if (self.__num_layer >= 3):
            y = self.__layer3(y, training=training, mask=mask)
            
        if (self.__num_layer >= 4):
            y = self.__layer4(y, training=training, mask=mask)
        
        if (self.__num_layer >= 5):
            y = self.__layer5(y, training=training, mask=mask)
        
        #    检测输出的y与缩放比例是否一致(减掉padding的2)
        x_h, x_w = x.shape[1]-2, x.shape[2]-2
        y_h, y_w = y.shape[1], y.shape[2]
        if (math.ceil(x_w / self.__scaling) != int(y_w)
            or math.ceil(x_h / self.__scaling) != int(y_h)):
            raise Exception(self.name + " scaling error. scaling:" + str(self.__scaling) + ". x.shape:" + str(x.shape) + " y.shape:" + str(y.shape))
      
        return y
    
    #    输出shape，根据缩放等级不同可能不一样
    def get_output_shape(self):
        if (self.__num_layer >= 5):
            return self.output_shape_layer5
        
        if (self.__num_layer >= 4):
            return self.output_shape_layer4
        
        if (self.__num_layer >= 3):
            return self.output_shape_layer3
            
        if (self.__num_layer >= 2):
            return self.output_shape_layer2
        
        if (self.__num_layer >= 1):
            return self.output_shape_layer1
        pass
    
    pass


#    ResNet50网络
class ResNet50(tf.keras.layers.Layer):
    '''ResNet50网络
        输入：180 * 480 * 3
        base_channel_num = 32
        ======================= 1层 =======================
        -----------------layer 1-------------------
        Conv:
            kernel_size=[3*3 * base_channel_num*1] stride=2 padding=1 active=relu norm=bn
            out=[90 * 240 * base_channel_num*1]
        ======================= 2层 =======================    
        -----------------Bottleneck 1*3-------------------
        Conv:[1*1*base_channel_num*1] stride=1 padding=0 active=relu norm=bn out=[90 * 240 * base_channel_num*1]
        Conv:[3*3*base_channel_num*1] stride=1 padding=1 active=relu norm=bn out=[90 * 240 * base_channel_num*1]
        Conv:[1*1*base_channel_num*4] stride=1 padding=0 norm=bn out=[49 * 49 * base_channel_num*4]
        shortcut: out=[90 * 240 * base_channel_num*4]   
        active: relu
        times: 3（该层重复3次）
        ======================= 3层 =======================
        -----------------Bottleneck 2-------------------
        Conv:[1*1*base_channel_num*2] stride=1 padding=0 active=relu norm=bn out=[45 * 120 * base_channel_num*2]
        Conv:[3*3*base_channel_num*2] stride=2 padding=1 active=relu norm=bn out=[45 * 120 * base_channel_num*2]
        Conv:[1*1*base_channel_num*8] stride=1 padding=0 norm=bn out=[45 * 120 * base_channel_num*8]
        shortcut: out=[45 * 120 * base_channel_num*8]   
        active: relu
        -----------------Bottleneck 2*3-------------------
        Conv:[1*1*base_channel_num*2] stride=1 padding=0 active=relu norm=bn out=[45 * 120 * base_channel_num*2]
        Conv:[3*3*base_channel_num*2] stride=1 padding=1 active=relu norm=bn out=[45 * 120 * base_channel_num*2]
        Conv:[1*1*base_channel_num*8] stride=1 padding=0 norm=bn out=[45 * 120 * base_channel_num*8]
        shortcut: out=[45 * 120 * base_channel_num*8]   
        active: relu
        times: 3（该层重复3次）
        ======================= 4层 =======================
        -----------------Bottleneck 3-------------------
        Conv:[1*1*base_channel_num*4] stride=1 padding=0 active=relu norm=bn out=[23 * 60 * base_channel_num*4]
        Conv:[3*3*base_channel_num*4] stride=2 padding=1 active=relu norm=bn out=[23 * 60 * base_channel_num*4]
        Conv:[1*1*base_channel_num*16] stride=1 padding=0 norm=bn out=[23 * 60 * base_channel_num*16]
        shortcut: out=[23 * 60 * base_channel_num*16]   
        active: relu
        -----------------Bottleneck 3*5-------------------
        Conv:[1*1*base_channel_num*4] stride=1 padding=0 active=relu norm=bn out=[23 * 60 * base_channel_num*4]
        Conv:[3*3*base_channel_num*4] stride=1 padding=1 active=relu norm=bn out=[23 * 60 * base_channel_num*4]
        Conv:[1*1*base_channel_num*16] stride=1 padding=0 norm=bn out=[13 * 13 * base_channel_num*16]
        shortcut: out=[23 * 60 * base_channel_num*16]   
        active: relu
        times: 5（该层重复5次）
        ======================= 5层 =======================
        -----------------Bottleneck 4-------------------
        Conv:[1*1*base_channel_num*8] stride=1 padding=0 active=relu norm=bn out=[12 * 30 * base_channel_num*8]
        Conv:[3*3*base_channel_num*8] stride=2 padding=1 active=relu norm=bn out=[12 * 30 * base_channel_num*8]
        Conv:[1*1*base_channel_num*32] stride=1 padding=0 norm=bn out=[12 * 30 * base_channel_num*32]
        shortcut: out=[12 * 30 * base_channel_num*32]   
        active: relu
        -----------------Bottleneck 4*2-------------------
        Conv:[1*1*channel_num*8] stride=1 padding=0 active=relu norm=bn out=[12 * 30 * channel_num*8]
        Conv:[3*3*channel_num*8] stride=1 padding=1 active=relu norm=bn out=[12 * 30 * channel_num*8]
        Conv:[1*1*channel_num*32] stride=1 padding=0 norm=bn out=[12 * 30 * channel_num*32]
        shortcut: out=[12 * 30 * channel_num*32]   
        active: relu
        times: 2（该层重复2次）
        -----------------layer 2-------------------
        没了，就这么直接输出给外面的网络
    '''
    def __init__(self, training=True, scaling=CNNS.get_feature_map_scaling(), base_channel_num=CNNS.get_base_channel_num(), kernel_initializer = tf.initializers.he_normal(), bias_initializer = tf.initializers.Zeros(), **kwargs):
        '''
            @param training: 网络是否参与训练
            @param scaling: 特征图缩放比例
            @param base_channel_num: 基础通道数（后层的通道数在此基础上 * n）
            @param kernel_initializer: 卷积核参数初始化方式（默认he_normal）
            @param bias_initializer: 偏置项初始化方式（默认zero）
        '''
        super(ResNet50, self).__init__(name='ResNet50', **kwargs)
        
        self.__kernel_initializer = kernel_initializer
        self.__bias_initializer = bias_initializer
        
        self.__training = training
        
        #    根据缩放比例计算网络层数（第1层与第2层缩放比例一致，所以加1）
        self.__num_layer = int(math.log(scaling, 2)) + 1
        self.__scaling = scaling
        
        #    装配网络
        self.__assembling(base_channel_num)
        
        pass
    #    装配网络
    def __assembling(self, base_channel_num=32):
        #    第1层
        filters_layer1 = base_channel_num * 1
        if (self.__num_layer >= 1):
            self.__layer1 = tf.keras.models.Sequential([
                    tf.keras.layers.ZeroPadding2D(1),
                    tf.keras.layers.Conv2D(name=self.name + '_layer1_conv1', filters=filters_layer1, kernel_size=(3, 3), strides=2, padding='valid', input_shape=(180, 480, 3), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    tf.keras.layers.BatchNormalization(name=self.name + '_BN1', trainable=self.__training),
                    tf.keras.layers.ReLU(name=self.name + '_ReLU1')
                ], name=self.name + '_layer1')
            pass
        self.output_shape_layer1 = (90, 240, filters_layer1)
        
        #    第2层
        filters_rdim_layer2 = base_channel_num * 1
        filters_layer2 = base_channel_num * 4
        if (self.__num_layer >= 2):
            self.__layer2 = tf.keras.models.Sequential([
                    Bottleneck(name=self.name + '_layer2_Bottleneck1', filters=[filters_rdim_layer2, filters_rdim_layer2, filters_layer2], strides=1, input_shape=(90, 240, filters_layer1), output_shape=(90, 240, filters_layer2), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    Bottleneck(name=self.name + '_layer2_Bottleneck2', filters=[filters_rdim_layer2, filters_rdim_layer2, filters_layer2], strides=1, input_shape=(90, 240, filters_layer2), output_shape=(90, 240, filters_layer2), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    Bottleneck(name=self.name + '_layer2_Bottleneck3', filters=[filters_rdim_layer2, filters_rdim_layer2, filters_layer2], strides=1, input_shape=(90, 240, filters_layer2), output_shape=(90, 240, filters_layer2), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training)
                ], name=self.name + '_layer2')
            pass
        self.output_shape_layer2 = (90, 240, filters_layer2)
        
        #    第3层
        filters_rdim_layer3 = base_channel_num * 2
        filters_layer3 = base_channel_num * 8
        if (self.__num_layer >= 3):
            self.__layer3 = tf.keras.models.Sequential([
                    Bottleneck(name=self.name + '_layer3_Bottleneck1', filters=[filters_rdim_layer3, filters_rdim_layer3, filters_layer3], strides=2, input_shape=(90, 240, filters_layer2), output_shape=(45, 120, filters_layer3), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    Bottleneck(name=self.name + '_layer3_Bottleneck2', filters=[filters_rdim_layer3, filters_rdim_layer3, filters_layer3], strides=1, input_shape=(45, 120, filters_layer3), output_shape=(45, 120, filters_layer3), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    Bottleneck(name=self.name + '_layer3_Bottleneck3', filters=[filters_rdim_layer3, filters_rdim_layer3, filters_layer3], strides=1, input_shape=(45, 120, filters_layer3), output_shape=(45, 120, filters_layer3), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    Bottleneck(name=self.name + '_layer3_Bottleneck4', filters=[filters_rdim_layer3, filters_rdim_layer3, filters_layer3], strides=1, input_shape=(45, 120, filters_layer3), output_shape=(45, 120, filters_layer3), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training)
                ], name='_layer3')
            pass
        self.output_shape_layer3 = (45, 120, filters_layer3)
        
        #    第4层
        filters_rdim_layer4 = base_channel_num * 4
        filters_layer4 = base_channel_num * 16
        if (self.__num_layer >= 4):
            self.__layer4 = tf.keras.models.Sequential([
                    Bottleneck(name=self.name + '_layer4_Bottleneck1', filters=[filters_rdim_layer4, filters_rdim_layer4, filters_layer4], strides=2, input_shape=(45, 120, filters_layer3), output_shape=(23, 60, filters_layer4), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    Bottleneck(name=self.name + '_layer4_Bottleneck2', filters=[filters_rdim_layer4, filters_rdim_layer4, filters_layer4], strides=1, input_shape=(23, 60, filters_layer4), output_shape=(23, 60, filters_layer4), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    Bottleneck(name=self.name + '_layer4_Bottleneck3', filters=[filters_rdim_layer4, filters_rdim_layer4, filters_layer4], strides=1, input_shape=(23, 60, filters_layer4), output_shape=(23, 60, filters_layer4), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    Bottleneck(name=self.name + '_layer4_Bottleneck4', filters=[filters_rdim_layer4, filters_rdim_layer4, filters_layer4], strides=1, input_shape=(23, 60, filters_layer4), output_shape=(23, 60, filters_layer4), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    Bottleneck(name=self.name + '_layer4_Bottleneck5', filters=[filters_rdim_layer4, filters_rdim_layer4, filters_layer4], strides=1, input_shape=(23, 60, filters_layer4), output_shape=(23, 60, filters_layer4), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    Bottleneck(name=self.name + '_layer4_Bottleneck6', filters=[filters_rdim_layer4, filters_rdim_layer4, filters_layer4], strides=1, input_shape=(23, 60, filters_layer4), output_shape=(23, 60, filters_layer4), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training)
                ], name=self.name + '_layer4')
            pass
        self.output_shape_layer4 = (23, 60, filters_layer4)
        
        #    第5层
        filters_rdim_layer5 = base_channel_num * 8
        filters_layer5 = base_channel_num * 32
        if (self.__num_layer >= 5):
            self.__layer5 = tf.keras.models.Sequential([
                    Bottleneck(name=self.name + '_layer5_Bottleneck1', filters=[filters_rdim_layer5, filters_rdim_layer5, filters_layer5], strides=2, input_shape=(23, 60, filters_layer4), output_shape=(12, 30, filters_layer5), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    Bottleneck(name=self.name + '_layer5_Bottleneck2', filters=[filters_rdim_layer5, filters_rdim_layer5, filters_layer5], strides=1, input_shape=(12, 30, filters_layer5), output_shape=(12, 30, filters_layer5), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training),
                    Bottleneck(name=self.name + '_layer5_Bottleneck3', filters=[filters_rdim_layer5, filters_rdim_layer5, filters_layer5], strides=1, input_shape=(12, 30, filters_layer5), output_shape=(12, 30, filters_layer5), kernel_initializer=self.__kernel_initializer, bias_initializer=self.__bias_initializer, trainable=self.__training)
                ], name=self.name + '_layer5')
            pass
        self.output_shape_layer5 = (13, 30, filters_layer5)
        pass
    #    前向传播
    def call(self, x, training=None, mask=None):
        if (self.__num_layer >= 1):
            y = self.__layer1(x, training=training, mask=mask)
        
        if (self.__num_layer >= 2):
            y = self.__layer2(y, training=training, mask=mask)
        
        if (self.__num_layer >= 3):
            y = self.__layer3(y, training=training, mask=mask)
            
        if (self.__num_layer >= 4):
            y = self.__layer4(y, training=training, mask=mask)
        
        if (self.__num_layer >= 5):
            y = self.__layer5(y, training=training, mask=mask)
        
        #    检测输出的y与缩放比例是否一致
        x_h, x_w = x.shape[1], x.shape[2]
        y_h, y_w = y.shape[1], y.shape[2]
        if (math.ceil(x_w / self.__scaling) != int(y_w)
            or math.ceil(x_h / self.__scaling) != int(y_h)):
            raise Exception(self.name + " scaling error. scaling:" + str(self.__scaling) + ". x.shape:" + str(x.shape) + " y.shape:" + str(y.shape))
        
        return y
    
    #    输出shape，根据缩放等级不同可能不一样
    def get_output_shape(self):
        if (self.__num_layer >= 5):
            return self.output_shape_layer5
        
        if (self.__num_layer >= 4):
            return self.output_shape_layer4
        
        if (self.__num_layer >= 3):
            return self.output_shape_layer3
            
        if (self.__num_layer >= 2):
            return self.output_shape_layer2
        
        if (self.__num_layer >= 1):
            return self.output_shape_layer1
        pass
    pass


