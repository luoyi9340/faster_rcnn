# -*- coding: utf-8 -*-  
'''
图片验证码数据集
    图片分为3种尺寸：
        480*180：4个字母
        600*180：5个字母
        720*180：6个字母
    标注文件为json格式：
        {fileName:'${fileName}', vcode='${vcode}', annos:[{key:'值', x:x, y:y, w:w, h:h}, {key:'值', x:x, y:y, w:w, h:h}...]}
        其中：
            fileName：文件名（不含png）
            vcode：验证码
            annos：矩形框信息
                key：矩形框中的文字
                x：左上w坐标
                y：左上h坐标
                w：矩形框宽度
                h：矩形框高度

@author: luoyi
Created on 2020年12月29日
'''
import json
import os
import numpy as np
import PIL
import tensorflow as tf
import collections

import data.part as part
import utils.conf as conf
import utils.alphabet as alphabet
import utils.logger_factory as logf


log = logf.get_logger('data_original')


#    标签文件迭代器
def label_file_iterator(label_file_path=conf.DATASET.get_label_train(),
                        count=conf.DATASET.get_count_train()):
    '''标签文件迭代器（注：该函数的标签坐标是原坐标，要压缩的话自行处理）
        json格式
        {
            fileName:'${fileName}', 
            vcode='${vcode}', 
            annos:[
                    {key:'值', x:x, y:y, w:w, h:h}, 
                    {key:'值', x:x, y:y, w:w, h:h}...
                ]
        }
        @param label_file_path: 标签文件
        @param count: 读取记录数，如果文件记录数不够则循环读取直到够数为止
        @return: file_name(str), vcode(str), labels(list:[key, x, y, w, h])
    '''
    i = 0
    while (i < count):
        for line in open(label_file_path, mode='r', encoding='utf-8'):
            if (i >= count): break
            i += 1
            j = json.loads(line)
            
            file_name = j['fileName']
            vcode = j['vcode']
            annos = j['annos']
            labels = []
            for anno in annos:
                labels.append((anno['key'], anno['x'], anno['y'], anno['w'], anno['h']))
                pass
            
            yield file_name, vcode, labels
            pass
        #    如果结束了i还等于0说明文件是空的
        assert (i > 0), 'file is empty. file_name:{}'.format(file_name)
        pass
    pass





#    根据is_mutiple_file和file_path取所有规则下的文件名
def get_fpaths(is_mutiple_file=False, file_path=None):
    '''根据is_mutiple_file和file_path取所有规则下的文件名
        单文件模式直接返回文件名
        多文件模式按照{file_path}0, {file_path}1这样的顺序往后取。直到后缀数字断了为止
        @param is_mutiple_file: 是否多文件模式
        @param file_path: 文件路径
    '''
    if (not is_mutiple_file): return [file_path]
    
    f_idx = 0
    fpaths = []
    while (os.path.exists(file_path + str(f_idx))):
        fpaths.append(file_path + str(f_idx))
        f_idx += 1
        pass
    return fpaths
#    打开文件（该方法会清空之前的文件）
def file_writer(json_out=conf.ROIS.get_train_rois_out()):
    mkdirs_and_remove_file(rois_out=json_out)
    
    fw = open(json_out, mode='w', encoding='utf-8')
    return fw
#    建上级目录，清空同名文件
def mkdirs_and_remove_file(rois_out=conf.ROIS.get_train_rois_out()):
    #    判断创建上级目录
    _dir = os.path.dirname(rois_out)
    if (not os.path.exists(_dir)):
        os.makedirs(_dir)
        pass
        
    #    若文件存在则删除重新写入
    if (os.path.exists(rois_out)):
        os.remove(rois_out)
        pass
    pass


#    读取numpy数据集，仅用于小批量数据（比如跑个test）
def load_XY_np(count=100,
               image_dir=None,
               label_fpath=None,
               is_label_mutiple_file=False,
               x_preprocess=lambda x:((x / 255.) - 0.5 ) * 2,
               y_preprocess=None):
    '''读取numpy数据集。（数据会被全部加载到内存，仅用于小批量数据比如跑个test）
        @param count: 读取样本个数（文件中样本数不够的话以实际文件中样本个数为准）
        @param image_dir: 图片目录
        @param label_fpath: 标签文件路径
        @param is_label_mutiple_file: 标签文件是否为多文件。若为多文件则文件名顺序参考rois_out
        @param x_preprocess: x后置处理（入参：numpy [图片像素矩阵]，默认归一到[-1,1]）
        @param y_preprocess: y后置处理（入参：numpy [['vcode', (vcode_idx, x,y,w,h), ...]]）
    '''
    X = []
    Y = []
    label_files = get_fpaths(is_label_mutiple_file, label_fpath)
    
    #    遍历所有文件，读取count个样本
    label_num = 0
    for fpath in label_files:
        for line in open(fpath, mode='r', encoding='utf-8'):
            if (label_num > count): break;
            label_num += 1
            #    标签json
            label = json.loads(line)

            #    读图片像素矩阵
            file_name = label['fileName']
            image = PIL.Image.open(image_dir + "/" + file_name + '.png', mode='r')
            image = image.resize((conf.IMAGE_WEIGHT, conf.IMAGE_HEIGHT),PIL.Image.ANTIALIAS)
            x = np.asarray(image, dtype=np.float32)
            X.append(x)
            
            #    读label数据
            vcode = label['vcode']
            annos = label['annos']
            y = []
            y.append(vcode)
            #    根据vcode长度对labels进行压缩
            compressible_scaling = 4 / len(vcode)
            for anno in annos:
                key = alphabet.category_index(anno['key'])
                label_x = anno['x'] * compressible_scaling
                label_y = anno['y']
                label_w = anno['w'] * compressible_scaling
                label_h = anno['h']
                y.append((key ,label_x ,label_y ,label_w ,label_h))
                pass
            Y.append(y)
            pass
        if (label_num > count): break;
        pass
    X, Y = np.array(X), np.array(Y)
    
    #    数据过前置处理
    if (x_preprocess): X = x_preprocess(X)
    if (y_preprocess): Y = y_preprocess(Y)
    return X, Y


#    y数据暂存队列（先进先出）。自定义layer无法直接拿到y数据，这里迭代时就将y暂存进队列，需要时自取
class OriginalCrtBatchQueue():
    '''暂存当前迭代的batch_size个y_true
        自定义的layer中拿不到y_true，但某些地方确实需要这部分数据（比如roi_pooling）
        这里暂存最后迭代的batch_size个y_true，先进先出迭代顺序
        在数据源不使用'打乱'的前提下可保证layer中拿到的y_true是与当前训练数据对应的y_true
    '''
    def __init__(self, 
                 batch_size=conf.DATASET.get_batch_size(), 
                 dtype=tf.float32, 
                 ymaps_shape=[6, 5],
                 ):
        '''
            @param batch_size: queue的大小 == 批量大小
            @param dtype: 数据类型
            @param ymaps_shape: y数据形状。内每一个数据dtype=上面
        '''
        self._queue = collections.deque(maxlen=batch_size)
        self._dtype = dtype
        
        #    先填充一堆无用数据，过build
        self.fill_empty(batch_size, ymaps_shape)
        pass
    #    先填充maxlen个无用数据，过build
    def fill_empty(self, maxlen, ymaps_shape):
        for _ in range(maxlen):
            self._queue.append(tf.ones(shape=ymaps_shape))
            pass
        pass  
    def push(self, y):
        self._queue.append(y)
        pass
    def crt_data(self):
        y = tf.convert_to_tensor(list(self._queue), dtype=self._dtype)
        return y
    
    #    默认值，初始化用
    @staticmethod
    def default(batch_size=conf.PROPOSALES.get_batch_size(), 
                      dtype=tf.float32, 
                      ymaps_shape=(conf.PROPOSALES.get_proposal_every_image(), 9)):
        queue = OriginalCrtBatchQueue(batch_size=batch_size, dtype=dtype, ymaps_shape=ymaps_shape)
        return queue
    pass


#    文件迭代器
def files_iterator(image_dir=conf.DATASET.get_in_train(),
                   count=conf.DATASET.get_count_train(),
                   y_path=conf.DATASET.get_label_train(),
                   y_mutiple=conf.DATASET.get_label_train_mutiple(),
                   x_preprocess=lambda x:((x / 255.) - 0.5) * 2,
                   y_preprocess=None,
                   y_queue=OriginalCrtBatchQueue.default(),
                   ):
    '''
        @param image_dir: 图片文件目录
        @param count: 每个文件读取多少张图片
        @param y_path: 标签文件路径
        @param y_mutiple: 标签文件是否多文件。多文件会从train.jsons0, train.jsons1...开始往后读，直到某个idx读不到为止。所以idx一定要连续
        @param x_preprocess: x数据前置处理。默认：缩放到[-1, 1]之间
        @param y_preprocess: 标签数据预处理。默认：什么都不做
    '''
    label_files = get_fpaths(y_mutiple, y_path)
    for fpath in label_files:
        readed = 0
        while (readed <= count):
            for line in open(fpath):
                readed += 1
                if (readed > count): break
                
                d = json.loads(line)
                #    图片信息
                fname = d['fileName']
                x = part.read_image(image_path=image_dir + "/" + fname + '.png', 
                                    resize_weight=conf.IMAGE_WEIGHT, 
                                    resize_height=conf.IMAGE_HEIGHT, 
                                    preprocess=x_preprocess)
                
                #    标注信息
                annos = d['annos']
#                 vcode = d['vcode']
#                 scaling = 4. / len(vcode)
                y = []
                for anno in annos:
                    vidx = alphabet.category_index(anno['key'])
                    anno_x = anno['x']
                    anno_y = anno['y']
                    anno_w = anno['w']
                    anno_h = anno['h']
                    #    左上点x坐标，宽度按照比例缩放（所有的高度统一不参与缩放）
#                     anno_x = anno_x * scaling
#                     anno_w = anno_w * scaling
                    y.append([vidx, anno_x, anno_y, anno_w, anno_h])
                    pass
                y = np.array(y, dtype=np.float32)
                #    如果不够6个追加[-1]
                if (y.shape[0] < 6):
                    y = np.concatenate([y, -np.ones(shape=[6 - y.shape[0],5], dtype=np.float32)], axis=0)
                    pass
                if (y_preprocess): y = y_preprocess(y)
                
                y_queue.push(y)
                yield x, y
                pass
            
            #    如果循环结束了readed还是==0，说明文件是空的。直接跳出循环
            if (readed == 0): 
                log.warn('file is empty, %s', fpath)
                break;
            pass
        pass
    pass


#    tensor迭代数据源
def tensor_iterator_db(image_dir=conf.DATASET.get_in_train(),
                       count=conf.DATASET.get_count_train(),
                       y_path=conf.DATASET.get_label_train(),
                       y_mutiple=conf.DATASET.get_label_train_mutiple(),
                       x_preprocess=lambda x:((x / 255.) - 0.5) * 2,
                       y_preprocess=None,
                       batch_size=conf.DATASET.get_batch_size(),
                       epochs=conf.DATASET.get_epochs(),
                       shuffle_buffer_rate=conf.DATASET.get_shuffle_buffer_rate(),
                       ):
    '''tensor迭代数据源
        @param image_dir: 图片文件目录
        @param count: 每个文件读取多少张图片
        @param y_path: 标签文件路径
        @param y_mutiple: 标签文件是否多文件。多文件会从train.jsons0, train.jsons1...开始往后读，直到某个idx读不到为止。所以idx一定要连续
        @param x_preprocess: x数据前置处理。默认：缩放到[-1, 1]之间
        @param y_preprocess: 标签数据预处理。默认：什么都不做
        @param batch_size: 批量大小
        @param epochs: 训练epoch轮数
        @param shuffle_buffer_rate: 打乱数据的buffer是batch_size的多少倍。<0表示不打乱
    '''
    x_shape = tf.TensorShape([conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT, 3])
    y_shape = tf.TensorShape([6, 5])        #    每张图最多6个验证码，vidx=-1表示填充值。[vidx, x,y, w,h]相对原图
    y_queue = OriginalCrtBatchQueue.default(batch_size=batch_size, dtype=tf.float32, ymaps_shape=y_shape)
    db = tf.data.Dataset.from_generator(generator=lambda :files_iterator(image_dir=image_dir,
                                                                         count=count,
                                                                         y_path=y_path,
                                                                         y_mutiple=y_mutiple,
                                                                         x_preprocess=x_preprocess,
                                                                         y_preprocess=y_preprocess,
                                                                         y_queue=y_queue,
                                                                         ), 
                                        output_types=(tf.float32, tf.float32), 
                                        output_shapes=(x_shape, y_shape))
    if (shuffle_buffer_rate > 0):
        db = db.shuffle(buffer_size=shuffle_buffer_rate * batch_size)
        pass
    if (batch_size): db = db.batch(batch_size)
    if (epochs): db = db.repeat(epochs)
    return db, y_queue


