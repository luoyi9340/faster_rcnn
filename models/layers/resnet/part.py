# -*- coding: utf-8 -*-  
'''
ResNet残差网络-零件
    - BasicBlock
        输入：X
        Conv:[3*3*C1]
        Conv:[3*3*C1]
        输出：H(X) + out
            其中：H(X)为调整X.shape与out一致
        ReLU
        
    - Bottleneck
        输入：X
        Conv:[1*1*C1]
        Conv:[3*3*C1]
        Conv:[1*1*C2]
        输出：H(X) + out
            其中：H(X)为调整X.shape与out一致
        ReLU

Created on 2020年12月23日

@author: irenebritney
'''
import tensorflow as tf


#    BasicBlock结构
class BasicBlock(tf.keras.layers.Layer):
    '''BasicBlock结构
        输入：X
        Conv:[3*3*C1] strides=1|2 padding=1
        Conv:[3*3*C2] strides=1 padding=1
        输出：H(X) + out
            其中：H(X)为调整X.shape与out一致
        ReLU
    '''
    def __init__(self, name=None, filters=None, strides=1, input_shape=None, output_shape=None, kernel_initializer=None, bias_initializer=None, **kwargs):
        super(BasicBlock, self).__init__(name=name, **kwargs)
        
        
        #    规定输出格式
        self.__output_shape = output_shape
        
        #    定义两层3*3卷积
        self.__conv = tf.keras.models.Sequential([
                tf.keras.layers.ZeroPadding2D(padding=1),
                tf.keras.layers.Conv2D(name=name + '_Conv1', filters=filters[0], kernel_size=(3, 3), strides=strides, padding='valid', input_shape=input_shape, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
                
                tf.keras.layers.ZeroPadding2D(padding=1),
                tf.keras.layers.Conv2D(name=name + '_Conv2', filters=filters[1], kernel_size=(3, 3), strides=1, padding='valid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
                tf.keras.layers.BatchNormalization()
            ])
        
        #    定义downsample。有一点原则，特征图体积缩小必然伴随通道数增加。否则体积与通道数都不变
        #    而且卷积核一定是3*3的，所以可以用1*1卷积核和strides直接操作
        self.__strides = strides
        if (strides != 1):
            self.__downsample = tf.keras.layers.Conv2D(filters[1], kernel_size=(1, 1), strides=strides, trainable=False, padding='valid', kernel_initializer=tf.keras.initializers.ones(), bias_initializer=tf.keras.initializers.zeros())
            pass
        
        pass
    def call(self, X, training=None, mask=None):
        #    先过两层卷积
        y = self.__conv(X, training=training, mask=mask)
        
        #    如果strides != 1说明特征图尺寸和深度都发生了改变
        if (self.__strides != 1):
            X = self.__downsample(X)
            pass
        
        y = tf.keras.layers.add([X, y])
        
        #    如果定义了output_shape，则检测输出是否一致
        if (self.__output_shape is not None
            and self.__output_shape != tuple(filter(None, y.shape))):
                raise Exception(self.name + "outputshape:" + str(self.__output_shape) + " not equal y:" + str(y.shape))
        
        return tf.nn.relu(y)
    pass


#    Bottleneck结构
class Bottleneck(tf.keras.layers.Layer):
    '''BasicBlock结构
        输入：X
        Conv:[1*1*C1] stride=1 padding=0
        Conv:[3*3*C2] stride=1|2 padding=1
        Conv:[1*1*C3] stride=1 padding=0
        输出：H(X) + out
            其中：H(X)为调整X.shape与out一致
        ReLU
    '''
    def __init__(self, name=None, filters=None, strides=1, input_shape=None, output_shape=None, kernel_initializer=None, bias_initializer=None, **kwargs):
        super(Bottleneck, self).__init__(name=name, **kwargs)
        
        
        #    规定输入/输出格式
        self.__input_shape = input_shape
        self.__output_shape = output_shape
        
        #    定义1*1, 3*3, 1*1卷积
        self.__conv = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(name=name + '_Conv0', filters=filters[0], kernel_size=(1, 1), strides=1, padding='valid', input_shape=input_shape, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
                tf.keras.layers.BatchNormalization(name=name + '_BN0', ),
                tf.keras.layers.ReLU(name=name + '_ReLU0', ),
                
                tf.keras.layers.ZeroPadding2D(padding=1),
                tf.keras.layers.Conv2D(name=name + '_Conv1', filters=filters[1], kernel_size=(3, 3), strides=strides, padding='valid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
                tf.keras.layers.BatchNormalization(name=name + '_BN1', ),
                tf.keras.layers.ReLU(name=name + '_ReLU1', ),
                
                tf.keras.layers.Conv2D(name=name + '_Conv2', filters=filters[2], kernel_size=(1, 1), strides=1, padding='valid', kernel_initializer=kernel_initializer, bias_initializer=bias_initializer),
                tf.keras.layers.BatchNormalization(name=name + '_BN2', )
            ])
        
        #    定义downsample。有一点原则，特征图体积缩小必然伴随通道数增加。否则体积与通道数都不变
        #    而且卷积核一定是3*3的，所以可以用1*1卷积核和strides直接操作
        self.__strides = strides
        if (strides != 1
            or self.__input_shape != self.__output_shape):
            self.__downsample = tf.keras.layers.Conv2D(filters[2], kernel_size=(1, 1), strides=strides, padding='valid', trainable=False, kernel_initializer=tf.keras.initializers.ones(), bias_initializer=tf.keras.initializers.zeros())
            pass
        
        pass
    def call(self, X, training=None, mask=None):
        #    先过两层卷积
        y = self.__conv(X, training=training, mask=mask)
        
        #    如果strides != 1说明特征图尺寸和深度都发生了改变
        if (self.__strides != 1
            or self.__input_shape != self.__output_shape):
            X = self.__downsample(X)
            pass
        
        y = tf.keras.layers.add([X, y])
        
        #    如果定义了output_shape，则检测输出是否一致
        if (self.__output_shape is not None
            and self.__output_shape != tuple(filter(None, y.shape))):
                raise Exception(self.name + "outputshape:" + str(self.__output_shape) + " not equal y:" + str(y.shape))
        
        return tf.nn.relu(y)
    pass



