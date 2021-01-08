# -*- coding: utf-8 -*-  
'''
RPN模型

@author: luoyi
Created on 2021年1月5日
'''
import tensorflow as tf

import models
import models.layers.rpn as rpn
import models.layers.resnet as resnet
import utils.conf as conf


#    RPN模型
class RPNModel(models.AModel):
    def __init__(self, cnns_name=conf.RPN.get_cnns(), 
                        learning_rate=conf.RPN.get_train_learning_rate(),
                        scaling=conf.CNNS.get_feature_map_scaling(), 
                        train_cnns=True,
                        train_rpn=True,
                        loss_lamda=10,
                        is_build=True,
                        input_shape=(None, conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT, 3),
                        **kwargs):
        '''
            @param cnns_name: 使用哪个cnns网络(resnet34 | resnet50)
            @param scaling: 特征图缩放比例
            @param train_cnns: cnns层是否参与训练
            @param train_rpn: rpn层是否参与训练
        '''
        self.__scaling = scaling
        
        self.__cnns_name = cnns_name
        self.cnns = None
        
        self.__train_cnns = train_cnns
        self.__train_rpn = train_rpn
        
        self.__loss_lamda = loss_lamda
        
        super(RPNModel, self).__init__(name='rpn', learning_rate=learning_rate, **kwargs)
        
        if (is_build):
            self._net.build(input_shape=input_shape)
            pass
        pass
    
    #    优化器
    def optimizer(self, net, learning_rate=0.9):
        return tf.optimizers.Adam(learning_rate=learning_rate)
    #    损失函数
    def loss(self):
        return rpn.RPNLoss(loss_lamda=self.__loss_lamda)
    #    评价函数
    def metrics(self):
        return [rpn.RPNMetricCls(), rpn.RPNMetricReg()]
    #    模型名称
    def model_name(self):
        return self.name + "_" + self.__cnns_name
    
    
    #    装配模型
    def assembling(self, net):
        #    选择CNNsNet
        if (self.__cnns_name == 'resnet34'):
            self.cnns = resnet.ResNet34(training=self.__train_cnns)
            pass
        #    默认resnet50
        else:
            self.cnns = resnet.ResNet50(training=self.__train_cnns)
            pass
        
        #    创建RPNNet
        self.rpn = rpn.RPNNet(training=self.__train_rpn, input_shape=self.cnns.get_output_shape())
        
        #    装配模型
        net.add(self.cnns)
        net.add(self.rpn)
        pass
    pass
