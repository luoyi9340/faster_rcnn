# -*- coding: utf-8 -*-  
'''
fast_rcnn层
    特征框提取 与 ROIPooling 不再这一层做
    这里从ROIPooling后面，从每个N*N*C的特征图开始做分类
    
    ... 特征框提取 + roi pooling ...
    输入：b * N * N * C  （N为roi pooling出来的尺寸，C为cnns出来的通道数，b为batch_size）
    --------------- FC ---------------
    注：此时的batch_size为每张图的建议框数 * 原batch_size
    Flatten out=[N*N*C]
    fc: units=4096 dropout=0.8 out=[4096] active=relu
    fc: units=4096 dropout=0.8 out=[4096] active=relu
    fc: units=4096 dropout=0.8 out=[4096] active=relu
    fc: units=4096 dropout=0.8 out=[4096] active=relu
    --------------- cls + reg ---------------
    fc_cls: units=42*2 active=softmax     每个建议框属于每个分类的概率
    fc_reg: units=42*4 active=None        每个建议框属于每个分类时的偏移比/缩放比
    --------------- loss_cls + loss_reg ---------------
    loss = loss_cls + λ * loss_reg
    loss_cls = 1/Ncls * ∑(m=1->Ncls)∑(i=1->42) (p[m,i] * log(d[m,i]))
                    p[m,i]为第m个建议框的真实分类，第i个为1，其他为0
                    d[m,i]为第m个建议框的预测分类，所以这里只要算真实分类的预测即可。其他的算出来外面也要*0
    loss_reg = 1/Nreg * ∑(m=1->Nreg)∑(t∈(x,y,w,h)) smootL1(t[m,*] - d[m,*])
                    t[m,*]为第m个样本的真实偏移比/缩放比
                    d[m,*]为第m个样本的预测偏移比/缩放比
    
    
@author: luoyi
Created on 2021年1月15日
'''
import tensorflow as tf

import utils.conf as conf
import utils.alphabet as alphabet


#    fast rcnn层
class FastRCNNLayer(tf.keras.layers.Layer):
    '''fast rcnn层网络结构
    '''
    def __init__(self, 
                 name='FastRCNN', 
                 training=True, 
                 fc_weights=conf.FAST_RCNN.get_fc_weights(),
                 fc_dropout=conf.FAST_RCNN.get_fc_dropout(),
                 kernel_initializer=tf.initializers.he_normal(), 
                 bias_initializer=tf.initializers.Zeros(), 
                 input_shape=(conf.FAST_RCNN.get_roipooling_kernel_size()[0], conf.FAST_RCNN.get_roipooling_kernel_size()[1], 256),
                 **kwargs):
        super(FastRCNNLayer, self).__init__(name=name, input_shape=input_shape, **kwargs)
        
        self.__input_shape = input_shape
        
        #    装配网络
        self.__assembling(fc_weights, fc_dropout, kernel_initializer, bias_initializer)
        self.trainable = training
        
        pass
    
    #    装配网络
    def __assembling(self, fc_weights, fc_dropout, kernel_initializer=tf.initializers.he_normal(), bias_initializer=tf.initializers.Zeros()):
        #    装配fc层
        self.__fc_layers = tf.keras.models.Sequential(name='fast_rcnn_fc_layers')
        self.__fc_layers.add(tf.keras.layers.Flatten(name='fast_rcnn_fc_layers_flatten', input_shape=self.__input_shape))
        i = 0
        for fc_weight in fc_weights:
            self.__fc_layers.add(tf.keras.layers.BatchNormalization(name='fast_rcnn_fc_layers_bn_' + str(i)))
            self.__fc_layers.add(tf.keras.layers.Dense(name='fast_rcnn_fc_layers_fc_' + str(i), 
                                                       units=fc_weight,
                                                       activation=tf.keras.activations.relu,
                                                       kernel_initializer=kernel_initializer,
                                                       bias_initializer=bias_initializer))
            #    最后一层不追加dropout
            if (fc_dropout < 1 and fc_dropout > 0 \
                and i < len(fc_weights) - 1):
                self.__fc_layers.add(tf.keras.layers.Dropout(name='fast_rcnn_fc_layers_dropout_' + str(i), rate=fc_dropout))
                pass
            i += 1
            pass
        
        #    装配cls分支
        cls_newshape = (1, len(alphabet.ALPHABET))
        self.__cls_layers = tf.keras.models.Sequential([
                tf.keras.layers.Dense(name='fast_rcnn_cls_layers_fc', 
                                      units=len(alphabet.ALPHABET), 
                                      kernel_initializer=tf.initializers.he_normal(), 
                                      bias_initializer=tf.initializers.Zeros()),
                tf.keras.layers.Softmax(),
                tf.keras.layers.Reshape(target_shape=cls_newshape)
            ], name='fast_rcnn_cls_layers')
        
        #    装配reg分支
        reg_newshape = (4, len(alphabet.ALPHABET))
        self.__reg_layers = tf.keras.models.Sequential([
                tf.keras.layers.Dense(name='fast_rcnn_reg_layers_fc',
                                      units=len(alphabet.ALPHABET) * 4, 
                                      kernel_initializer=tf.initializers.he_normal(), 
                                      bias_initializer=tf.initializers.Zeros()),
                tf.keras.layers.Reshape(target_shape=reg_newshape)
            ], name='fast_rcnn_reg_layers')
        pass
    
    def call(self, x, training=True, mask=None):
        #    过4个fc层
        y = self.__fc_layers(x, training=training, mask=mask)
        
        #    分别过cls和reg
        y_cls = self.__cls_layers(y)                #    shape=(batch_size, 1, 31)    此时的batch_size是原cnns的batch_size * proposal_every_image
        y_reg = self.__reg_layers(y)                #    shape=(batch_size, 4, 31)    此时的batch_size是原cnns的batch_size * proposal_every_image
        
        #    叠加为(batch_size, 5, 42)
        y = tf.concat([y_cls, y_reg], axis=1)
        
        return y
    pass
