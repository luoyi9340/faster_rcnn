# -*- coding: utf-8 -*-  
'''
RPN模型

@author: luoyi
Created on 2021年1月5日
'''
import tensorflow as tf

import models
import utils.conf as conf
from models.layers.rpn.models import RPNNet
from models.layers.rpn.losses import RPNLoss
from models.layers.rpn.metrics import RPNMetricCls, RPNMetricReg
from models.layers.rpn.preprocess import takeout_sample_np, all_positives_from_fmaps
from models.layers.rpn.nms import nms
from models.layers.resnet.models import ResNet34, ResNet50



#    RPN模型
class RPNModel(models.AModel):
    def __init__(self, cnns_name=conf.RPN.get_cnns(), 
                        learning_rate=conf.RPN.get_train_learning_rate(),
                        scaling=conf.CNNS.get_feature_map_scaling(), 
                        K=conf.RPN.get_K(),
                        cnns_base_channel_num=conf.CNNS.get_base_channel_num(),
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
        self.__K = K
        self.__cnns_base_channel_num = cnns_base_channel_num
        
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
        return RPNLoss(loss_lamda=self.__loss_lamda)
    #    评价函数
    def metrics(self):
        return [RPNMetricCls(), RPNMetricReg()]
    #    模型名称
    def model_name(self):
        return self.name + "_" + self.__cnns_name
    
    
    #    装配模型
    def assembling(self, net):
        #    选择CNNsNet
        if (self.__cnns_name == 'resnet34'):
            self.cnns = ResNet34(training=self.__train_cnns, scaling=self.__scaling, base_channel_num=self.__cnns_base_channel_num)
            pass
        #    默认resnet50
        else:
            self.cnns = ResNet50(training=self.__train_cnns, scaling=self.__scaling, base_channel_num=self.__cnns_base_channel_num)
            pass
        
        #    创建RPNNet
        self.rpn = RPNNet(training=self.__train_rpn, input_shape=self.cnns.get_output_shape(), K=self.__K, loss_lamda=self.__loss_lamda)
        
        #    装配模型
        net.add(self.cnns)
        net.add(self.rpn)
        pass
    
    
    #    测试
    def test(self, X, batch_size=2):
        '''测试
            @param X: 测试数据(num, h, w, 3)
            @param batch_size: 批量
            @return: 特征图(num, h, w, 6, K)
        '''
        fmaps = self._net.predict(X, batch_size=batch_size)
        return fmaps
    #    统计分类数据
    def test_cls(self, fmaps, ymaps):
        '''统计分类数据
            @param fmaps: Tensor(num, h, w, 6, K) test函数返回的特征图
            @param ymaps: Numpy(num, h, w, 6, K) 与fmaps对应的标签特征图
            @return: TP, TN, FP, TN, P, N
        '''
        (fmaps_cls_p, fmaps_cls_n, _), (ymaps_cls_p, ymaps_cls_n, _) = takeout_sample_np(ymaps, fmaps)
        return RPNMetricCls().tp_tn_fp_tf_p_n(tf.convert_to_tensor(ymaps_cls_p, dtype=tf.float32), 
                                              tf.convert_to_tensor(fmaps_cls_p, dtype=tf.float32), 
                                              tf.convert_to_tensor(ymaps_cls_n, dtype=tf.float32), 
                                              tf.convert_to_tensor(fmaps_cls_n, dtype=tf.float32))
    #    计算回归的平均绝对误差
    def test_reg(self, fmaps, ymaps):
        '''计算回归的平均绝对误差
            @param fmaps: Tensor(num, h, w, 6, K) test函数返回的特征图
            @param ymaps: Numpy(num, h, w, 6, K) 与fmaps对应的标签特征图
            @return: MAE
        '''
        (_, _, fmaps_reg_p), (ymaps_cls_p, _, ymaps_reg_p) = takeout_sample_np(ymaps, fmaps)
        return RPNMetricReg().mean_abs_error(tf.convert_to_tensor(ymaps_reg_p, dtype=tf.float32), 
                                             tf.convert_to_tensor(fmaps_reg_p, dtype=tf.float32), 
                                             tf.convert_to_tensor(ymaps_cls_p, dtype=tf.float32))
    
    
    #    生成全部建议框
    def candidate_box_from_fmap(self, 
                                fmaps, 
                                threshold_prob=conf.RPN.get_nms_threshold_positives(), 
                                threshold_iou=conf.RPN.get_nms_threshold_iou(), 
                                K=conf.RPN.get_K(),
                                roi_areas = conf.RPN.get_roi_areas(),
                                roi_scales = conf.RPN.get_roi_scales()):
        '''根据模型输出的fmaps生成全部候选框（所有被判定为前景的anchor，过nms）
            @param fmaps: numpy(num, h, w, 6, K)
            @param threshold_prob: 判定为前景的阈值
            @param threshold_iou: NMS中用到的IoU阈值。超过此阈值的anchor会被判定为重叠的anchor过滤掉
            @param K: 特征图中每个像素点对应多少个anchor(roi_areas * roi_scales的组合)
            @param roi_areas: anchor面积比划分(1:1时的长宽值)
            @param roi_scales: anchor长宽比划分
            @return: [numpy(num, 6)...] 
                            [正样本概率, xl,yl(左上点), xr,yr(右下点), 区域面积]
        '''
        #    取fmaps中生成的所有被判定为前景的anchor
        anchors = all_positives_from_fmaps(fmaps, threshold=threshold_prob, K=K, roi_areas=roi_areas, roi_scales=roi_scales)
        return nms(anchors, threshold=threshold_iou)
    pass
