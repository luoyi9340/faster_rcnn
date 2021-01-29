# -*- coding: utf-8 -*-  
'''
fast_rcnn模型
    1 从拿到的y_true中

@author: luoyi
Created on 2021年1月26日
'''
import tensorflow as tf

import utils.conf as conf
from models.abstract_model import AModel
from models.layers.resnet.models import ResNet34, ResNet50
from models.layers.fast_rcnn.models import FastRCNNLayer
from models.layers.fast_rcnn.losses import FastRcnnLoss
from models.layers.fast_rcnn.metrics import FastRcnnMetricCls, FastRcnnMetricReg
from models.layers.roi_pooling.preprocess import roi_pooling



#    训练参数定义
def step_signature():
    train_step_signature = [
                tf.TensorSpec(shape=(conf.PROPOSALES.get_batch_size(), conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(conf.PROPOSALES.get_batch_size(), conf.PROPOSALES.get_proposal_every_image(), 9), dtype=tf.float32),
            ]
    return train_step_signature

#    fast_rcnn模型
class FastRcnnModel(AModel):
    '''fast_rcnn model
        自定义训练过程
    '''
    def __init__(self, 
                 name='FastRcnnModel', 
                 learning_rate=0.001,
                 cnns_name=conf.FAST_RCNN.get_cnns(),
                 scaling=conf.CNNS.get_feature_map_scaling(), 
                 cnns_base_channel_num=conf.CNNS.get_base_channel_num(),
                 fc_weights=conf.FAST_RCNN.get_fc_weights(),
                 fc_layers=conf.FAST_RCNN.get_fc_layers(),
                 fc_dropout=conf.FAST_RCNN.get_fc_dropout(),
                 roipooling_ksize=conf.FAST_RCNN.get_roipooling_kernel_size(),
                 loss_lamda=conf.FAST_RCNN.get_loss_lamda(),
                 train_cnns=True,
                 train_fast_rcnn=True,
                 is_build=False,
                 input_shape=(None, conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT, 3),
                 **kwargs):
        '''
            @param cnns_name: 使用的cnns层名称
            @param scaling: cnns层的缩放比例。据此算出cnns层的深度
            @param cnns_base_channel_num: cnns层的基础通道数，据此算出cnns层的通道深度
            @param fc_weights: fast_rcnn中fc层的参数维度
            @param fc_layers: fast_rcnn中fc层数量
            @param fc_dropout: fast_rcnn中fc层dropout比率
            @param roipooling_kernel_size: roipooling的ksize
            @param train_cnns: 是否训练cnns网络
            @param train_fast_rcnn: 是否训练fast_rcnn网络
        '''
        self.__scaling = scaling
        self.__cnns_base_channel_num = cnns_base_channel_num
        
        self.__cnns_name = cnns_name
        self.cnns = None
        self.rpn = None
        
        self.__fc_weights = fc_weights
        self.__fc_layers = fc_layers
        self.__fc_dropout = fc_dropout
        self.__roipooling_ksize = roipooling_ksize
        self.__loss_lamda = loss_lamda
        
        self.__train_cnns = train_cnns
        self.__train_fast_rcnn = train_fast_rcnn
        self.__learning_rate = learning_rate
        
        super(FastRcnnModel, self).__init__(name=name, learning_rate=learning_rate, **kwargs)

        if (is_build):
            self._net.build(input_shape=input_shape)
            pass
        pass
    
    #    装配网络
    def assembling(self, net):
        #    装配cnns
        if (self.__cnns_name == 'resnet34'):
            self.cnns = ResNet34(training=self.__train_cnns, 
                                 scaling=self.__scaling, 
                                 base_channel_num=self.__cnns_base_channel_num)
            pass
        else:
            self.cnns = ResNet50(training=self.__train_cnns, 
                                 scaling=self.__scaling, 
                                 base_channel_num=self.__cnns_base_channel_num)
            pass
        
        #    装配fast_rcnn
        self.fast_rcnn = FastRCNNLayer(training=self.__train_fast_rcnn,
                                       fc_weights=self.__fc_weights,
                                       fc_layers=self.__fc_layers,
                                       fc_dropout=self.__fc_dropout)
#         net.add(self.cnns)
#         net.add(self.fast_rcnn)
        pass
    
    #    优化器
    def optimizer(self, net, learning_rate=0.001):
        return tf.optimizers.Adam(learning_rate=learning_rate)
    #    损失函数
    def loss(self):
        return FastRcnnLoss(loss_lamda=self.__loss_lamda)
    #    评价函数
    def metrics(self):
        return [FastRcnnMetricCls(), FastRcnnMetricReg()]
    
    
    #    训练步骤
    @tf.function(input_signature=step_signature())
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            #    先过cnns
            fmaps = self.cnns(x)
            #    再过roipooling
            fmaps_proposals = roi_pooling(fmaps, y, roipooling_ksize=self.__roipooling_ksize)
            #    建议框过fast_rcnn
            y_pred = self.fast_rcnn(fmaps_proposals)
            #    计算loss
            loss = self._net.loss(y, y_pred)
            pass
        #    梯度更新
        cnns_tvs = self.cnns.trainable_variables
        fast_rcnn_tvs = self.fast_rcnn.trainable_variables
        tvs = cnns_tvs + fast_rcnn_tvs
        grads = tf.gradients(loss, tvs)
        self._net.optimizer.apply_gradients(zip(grads, tvs))
        #    计算评价指标
        logs = self.run_metrics(y, y_pred, tag='train')
        logs['train_loss'] = loss
        return logs
    
    
    #    验证步骤
    @tf.function(input_signature=step_signature())
    def val_step(self, x, y):
        #    先过cnns -> 再过roipooling -> 建议框过fast_rcnn -> 计算loss
        fmaps = self.cnns(x)
        fmaps_proposals = roi_pooling(fmaps, y, roipooling_ksize=self.__roipooling_ksize)
        y_pred = self.fast_rcnn(fmaps_proposals)
        loss = self._net.loss(y, y_pred)
        #    计算评价
        logs = self.run_metrics(y, y_pred, tag='val')
        logs['val_loss'] = loss
        return logs
    
    
    #    计算评价指标
    def run_metrics(self, y_true, y_pred, tag='train'):
        logs = {}
        for metric in self._metrics:
            metric(y_true, y_pred)
            logs[tag + '_' + metric.name] = metric.result()
            pass
        return logs
    
    
    #    一轮epoch后重置各种参数
    def reset_after_epoch(self):
        #    重置评价指标
        for metric in self._metrics:
            metric.reset_states()
            pass
        pass
    
    
    #    自定义训练过程
    def custom_train(self,
                     db_train=None,
                     db_val=None,
                     batch_size=conf.PROPOSALES.get_batch_size(),
                     epochs=conf.PROPOSALES.get_epochs(),
                     steps_per_epoch=100,
                     tensorboard_dir=conf.FAST_RCNN.get_tensorboard_dir()):
        '''自定义训练过程
            @param db_train: 训练集
            @param db_val: 验证集
            @param batch_size: 训练集batch_size
            @param epochs: 训练epochs
            @param steps_per_epoch: 每轮epochs要训练多少步
        '''
        callbacks = self.callback_list(batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch)
        tensorboard_dir = tensorboard_dir + "/" + self.model_name() + "_b" + str(batch_size) + "_lr" + str(self.__learning_rate)
        train_summary_writer = tf.summary.create_file_writer(tensorboard_dir + "/train")
        val_summary_writer = tf.summary.create_file_writer(tensorboard_dir + '/validation')
        
        #    训练开始
        callbacks.on_train_begin()
        db_iter = iter(db_train)
        for epoch in range(epochs):
            #    训练过程中记录的日志
            logs = {}
            #    epoch开始
            callbacks.on_epoch_begin(epoch, logs)
            
            for step in range(steps_per_epoch):
                x, y = db_iter.next()
                crt_step = epoch * steps_per_epoch + step
                print('crt_step:', crt_step)
                #    每轮训练batch开始
                callbacks.on_train_batch_begin(step, logs)
                #    执行训练步骤
                train_logs = self.train_step(x, y)
                logs.update(train_logs)
                #    记录需要观测的值
                for k,v in train_logs.items():
                    tf.summary.scalar(k, v, step=crt_step)
                    pass
                #   记录每次batch的lr
                tf.summary.scalar('lr', self._net.optimizer.lr, step=crt_step) 
                
                #    每轮训练batch结束
                callbacks.on_train_batch_end(step, logs)
                pass
            
            #    验证开始
            callbacks.on_test_begin(logs)
            step = 0
            val_loss = 0
            val_cls_acc = 0
            val_reg_mae = 0
            for x, y in db_val:
                #    每轮验证开始
                callbacks.on_test_batch_begin(step, logs)
                #    执行验证步骤
                val_logs = self.val_step(x, y)
                #    记录验证损失，验证评价指标
                val_loss += val_logs.get('val_loss')
                val_cls_acc += val_logs.get('val_FastRcnnMetricCls')
                val_reg_mae += val_logs.get('val_FastRcnnMetricReg')
                
                #    每轮验证结束
                callbacks.on_test_batch_end(step, logs)
                step += 1
                pass
            val_loss = val_loss / step
            tf.summary.scalar('val_loss', val_loss, step=epoch) 
            logs['val_loss'] = val_loss
            val_cls_acc = val_cls_acc / step
            tf.summary.scalar('val_cls_acc', val_cls_acc, step=epoch) 
            logs['val_cls_acc'] = val_cls_acc
            val_reg_mae = val_reg_mae / step
            tf.summary.scalar('val_reg_mae', val_reg_mae, step=epoch) 
            logs['val_reg_mae'] = val_reg_mae
            
            #    验证结束
            callbacks.on_test_end(logs)
            
            #    epoch结束
            callbacks.on_epoch_end(epoch, logs)
            
            #    重置各种参数
            self.reset_after_epoch()
            pass
        #    训练结束
        callbacks.on_train_end(logs)
        pass
    
    
    #    各种回调
    def callback_list(self, batch_size, epochs, steps_per_epoch):
        '''各种回调'''
        callback_list = self.callbacks(auto_save_weights_after_traind=True, 
                                       auto_save_weights_dir=conf.FAST_RCNN.get_save_weights_dir(),
                                       auto_learning_rate_schedule=True, 
                                       auto_tensorboard=True, 
                                       auto_tensorboard_dir=conf.FAST_RCNN.get_tensorboard_dir(), 
                                       batch_size=batch_size)
        callbacks = tf.keras.callbacks.CallbackList(callbacks=callback_list,
                                                    add_history=True,
                                                    add_progbar=1,
                                                    verbose=1,
                                                    epochs=epochs,
                                                    model=self._net,
                                                    steps=steps_per_epoch)
        return callbacks
    pass




