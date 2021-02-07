# -*- coding: utf-8 -*-  
'''
建议框数据源

@author: luoyi
Created on 2021年1月24日
'''
import numpy as np
import itertools
import json
import os
import tensorflow as tf
import collections as collections

import utils.conf as conf
import utils.alphabet as alphabet
import utils.logger_factory as logf
import data.dataset as ds
import data.part as part


log = logf.get_logger('proposal_creator')


#    建议框生成器
class ProposalsCreator():
    def __init__(self, 
                 threshold_nms_prob=conf.RPN.get_nms_threshold_positives(), 
                 threshold_nms_iou=conf.RPN.get_nms_threshold_iou(),
                 proposal_iou=0.7, 
                 proposal_every_image=32, 
                 rpn_model=None):
        '''
            @param threshold_nms_prob: 非极大值抑制的前景概率（低于此概率的将被判负样本）
            @param threshold_nms_iou: 非极大值抑制时IoU比例（超过此比例的anchor会被判重叠而过滤掉）
            @param proposal_iou: 判定为有效建议框的anchor与label的IoU比例
            @param proposal_every_image: 每张图片有效建议框数量
            @param rpnModel: 生成建议框的rpn模型
        '''
        self.__threshold_nms_prob = threshold_nms_prob
        self.__threshold_nms_iou = threshold_nms_iou
        self.__proposal_iou = proposal_iou
        self.__proposal_every_image = proposal_every_image
        self.__rpn_model = rpn_model     
        pass
    
    
    #    生成单一文件的建议框
    def __create_single(self, 
                        img_path='', 
                        vcode='', 
                        labels=None, 
                        x_preprocess=None):
        '''生成单一文件的建议框
            @param img_path: 图片路径
            @param vcode: 验证码值
            @param labels: 标签
                            [[key所属分类, x左上点, y左上点, w, h]...]
            @param not_enough_preprocess: 当数据不够proposal_every_image时的处理方式
                                            参数：实际取到的proposales列表
            @return: x, proposals 
                        x: 图片矩阵
                        proposals list
                                    [
                                        [IoU得分，proposal左上/右下点坐标(相对原图)， 所属分类索引， 标签左上点坐标/长宽]
                                        [iou, xl,yl,xr,yr, vidx, x,y,w,h]
                                    ]
        '''
        #    读取图片文件
        x = part.read_image(image_path=img_path, resize_weight=conf.IMAGE_WEIGHT, resize_height=conf.IMAGE_HEIGHT, preprocess=x_preprocess)
        #    生成候选框
        fmaps = self.__rpn_model.test(np.expand_dims(x, axis=0))
        anchors = self.__rpn_model.candidate_box_from_fmap(fmaps=fmaps, 
                                                          threshold_prob=self.__threshold_nms_prob, 
                                                          threshold_iou=self.__threshold_nms_iou)
        anchors = anchors[0]
#         [正样本概率, xl,yl(左上点), xr,yr(右下点), 区域面积]
        rect_srcs = np.zeros(shape=(anchors.shape[0], 4))
        rect_srcs[:,0] = anchors[:,1]
        rect_srcs[:,1] = anchors[:,2]
        rect_srcs[:,2] = anchors[:,3]
        rect_srcs[:,3] = anchors[:,4]
        #    计算候选框与每个标签的IoU
        labels_arrs = []
        k_count = 0
        for label in labels:
            #    标签数据根据vcode长度做缩放
            compressible_scaling = 4 / len(vcode)
            label_x = label[1] * compressible_scaling
            label_y = label[2]
            label_w = label[3] * compressible_scaling
            label_h = label[4]
            
            rect_tag = (label_x, label_y, label_x + label_w, label_y + label_h)
            iou = part.iou_xlyl_xryr_np(rect_srcs, rect_tag)
            #    取anchors中感兴趣的部分(xl,yl, xr,yr)，并按IoU降序排列，并追加label信息
            #    提取感兴趣的部分(xl,yl, xr,yr)
            idx_ = iou > self.__proposal_iou
            label_anchprs = anchors[idx_][:, 1:5]
            iou = iou[idx_]
            iou = np.expand_dims(iou, axis=-1)
            label_anchprs = np.concatenate([iou, label_anchprs], axis=1)
            label_anchprs = label_anchprs[np.argsort(label_anchprs[:,0])[::-1]]
            #    追加标签信息(vidx, x,y, w,h)
            label_info = np.array([[alphabet.category_index(label[0]), label_x, label_y, label_w, label_h]])
            label_info = np.repeat(label_info, label_anchprs.shape[0], axis=0)
            label_anchprs = np.concatenate([label_anchprs, label_info], axis=1)
            labels_arrs.append(label_anchprs.tolist())
            k_count += 1
            pass
        #    每个label交替出现
        tuple_arrs = tuple(labels_arrs)
        alternate_zip = itertools.zip_longest(*tuple_arrs)
        res = []
        for r in alternate_zip:
            for i in range(k_count):
                if (r[i] is not None): res.append(r[i])
                pass
            pass
        
        #    如果数量超过了，截断。保留proposal_every_image*4的数据量，供配置用
        if (len(res) > self.__proposal_every_image * 4):
            res = res[:self.__proposal_every_image * 4]
            pass
        
        return x, res
    
    #    生成建议框
    def create(self, 
               proposals_out='',
               image_dir='', 
               label_path='', 
               is_mutiple_file=False, 
               count=100,
               x_preprocess=lambda x:((x /255.) - 0.5) * 2,
               log_interval=100):
        '''根据标签文件生成建议框，写入json文件
            @param proposals_out: 写入json文件路径
            @param image_dir: 图片文件目录
            @param label_path: 标签文件路径
            @param is_mutiple_file: 是否多文件
            @param count: 每个文件读取数量(不够的会循环该文件凑够数)
            @param x_preprocess: 图片矩阵后置处理
            @param not_enough_preprocess: 当数据量不够proposal_every_image时的处理方式
        '''
        #    根据标签文件路径和是否多文件读取全部标签文件列表
        label_fpaths = ds.get_fpaths(is_mutiple_file, file_path=label_path)
        fw = ds.file_writer(json_out=proposals_out)
        
        #    统计数量
        num_img = 0
        num_proposals = 0
        for label_fpath in label_fpaths:
            label_iter = ds.label_file_iterator(label_fpath, count)
            for img_name, vcode, labels in label_iter:
                img_path = image_dir + '/' + img_name + '.png'
                _, proposals = self.__create_single(img_path=img_path, 
                                                    vcode=vcode, 
                                                    labels=labels, 
                                                    x_preprocess=x_preprocess)
                
                #    判断当数据量不够proposal_every_image时的处理方式
                if (len(proposals) < self.__proposal_every_image):
                    log.warning('proposal count:{} less then proposal_every_image:{} file_name:{}'.format(len(proposals), self.__proposal_every_image, img_name))
                    continue
                
                d = {'file_name':img_name, 'proposals':proposals}
                j = json.dumps(d)
                fw.write(j + '\n')
                
                num_img += 1
                num_proposals += len(proposals)
                if (num_img % log_interval == 0):
                    log.info('create proposal progress. num_img:{} count:{}'.format(num_img, count))
                    pass
                pass
            pass
        
        fw.close()
        
        log.info('create proposal finished. num_img:{} avg_proposals:{}'.format(num_img, num_proposals/num_img))
        pass
    
    #    测试生成建议框
    def test_create(self, 
                    image_dir='', 
                    label_path='', 
                    is_mutiple_file=False, 
                    count=100,
                    x_preprocess=lambda x:((x /255.) - 0.5) * 2):
        '''根据标签文件生成建议框
            @param proposals_out: 写入json文件路径
            @param image_dir: 图片文件目录
            @param label_path: 标签文件路径
            @param is_mutiple_file: 是否多文件
            @param count: 每个文件读取数量(不够的会循环该文件凑够数)
            @param x_preprocess: 图片矩阵后置处理
            @param not_enough_preprocess: 取到的proposales不够proposal_every_image时的处理方式
            @return: 迭代器
                        图片名称(不含路径和.png)
                        proposals numpy(proposal_every_image*4, 10)
                                    [iou, xl,yl,xr,yr, vidx, x,y,w,h]
        '''
        #    根据标签文件路径和是否多文件读取全部标签文件列表
        label_fpaths = ds.get_fpaths(is_mutiple_file, file_path=label_path)
        
        for label_fpath in label_fpaths:
            label_iter = ds.label_file_iterator(label_fpath, count)
            for img_path, vcode, labels in label_iter:
                img_path = image_dir + '/' + img_path + '.png'
                x, proposals = self.__create_single(img_path=img_path, 
                                                    vcode=vcode, 
                                                    labels=labels, 
                                                    x_preprocess=x_preprocess)
                
                yield x, proposals
                pass
            pass
        
        pass
    pass



#    暂存当前迭代的batch_size个y_true
class ProposalsCrtBatchQueue():
    '''暂存当前迭代的batch_size个y_true
        自定义的layer中拿不到y_true，但某些地方确实需要这部分数据（比如roi_pooling）
        这里暂存最后迭代的batch_size个y_true，先进先出迭代顺序
        在数据源不使用'打乱'的前提下可保证layer中拿到的y_true是与当前训练数据对应的y_true
    '''
    def __init__(self, 
                 batch_size=conf.PROPOSALES.get_batch_size(), 
                 dtype=tf.float32, 
                 ymaps_shape=(conf.PROPOSALES.get_proposal_every_image(), 9)):
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
        queue = ProposalsCrtBatchQueue(batch_size=batch_size, dtype=dtype, ymaps_shape=ymaps_shape)
        return queue
    pass
#    全局位置占个坑
proposals_crt_batch = None


#    建议框jsons文件迭代器
def read_proposals_generator(image_dir=conf.DATASET.get_in_train(),
                             count=conf.DATASET.get_count_train(),
                             proposals_out=conf.PROPOSALES.get_train_proposal_out(),
                             is_proposal_mutiple_file=conf.DATASET.get_label_train_mutiple(),
                             proposal_every_image=conf.PROPOSALES.get_proposal_every_image(),
                             x_preprocess=lambda x:((x / 255.) - 0.5) * 2,
                             y_preprocess=None,
                             proposals_crt_batch_queue=None):
    '''建议框jsons文件迭代器
        @param image_dir: 图片目录
        @param count: 每个proposals_out文件取多少条记录
        @param proposals_out: 建议框jsons文件路径
        @param is_proposal_mutiple_file: 是否多文件(多文件会proposals.jsons0, proposals.jsons1...开始，直到某个自增文件名找不到为止)
        @param proposal_every_image: 每张图片保留多少建议框
        @param x_preprocess: 图片数据预处理
        @param y_preprocess: 标签数据预处理
        @param proposals_crt_batch_queue: 暂存y数据（先进先出顺序）
    '''
    label_files = ds.get_fpaths(is_proposal_mutiple_file, proposals_out)
    
    #    遍历所有文件和所有行
    for fpath in label_files:
        readed = 0
        #    从一个文件中读满count条记录为止。文件记录数不够就重复读
        while (readed <= count):
            for line in open(fpath, mode='r', encoding='utf-8'):
                readed += 1
                if (readed > count): break

                d = json.loads(line)
                
                #    读取图片数据
                file_name = d['file_name']
                x = part.read_image(image_dir + '/' + file_name + '.png', 
                                    resize_weight=conf.IMAGE_WEIGHT, 
                                    resize_height=conf.IMAGE_HEIGHT, 
                                    preprocess=x_preprocess)
                
                proposals = d['proposals']
                assert (len(proposals) >= proposal_every_image), 'proposals.count:{} less then conf.proposal_every_image:{} file_name:{}'.format(len(proposals), proposal_every_image, file_name)
                if (len(proposals) > proposal_every_image): proposals = proposals[:proposal_every_image]
                y = np.array(proposals)
                if (y_preprocess): y = y_preprocess(proposals)
                if (proposals_crt_batch_queue): proposals_crt_batch_queue.push(y)
                
                yield x, y
                pass
            
            #    如果循环结束了readed还是==0，说明文件是空的。直接跳出循环
            if (readed == 0): 
                log.warn('file is empty, %s', fpath)
                break;
            pass
        pass
    pass

#    建议框数据源
def fast_rcnn_tensor_db(image_dir=conf.DATASET.get_in_train(),
                       count=conf.DATASET.get_count_train(),
                       proposals_out=conf.PROPOSALES.get_train_proposal_out(),
                       is_proposal_mutiple_file=conf.DATASET.get_label_train_mutiple(),
                       proposal_every_image=conf.PROPOSALES.get_proposal_every_image(),
                       batch_size=conf.PROPOSALES.get_batch_size(),
                       epochs=conf.PROPOSALES.get_epochs(),
                       shuffle_buffer_rate=conf.PROPOSALES.get_shuffle_buffer_rate(),
                       ymaps_shape=(conf.PROPOSALES.get_proposal_every_image(), 9),
                       x_preprocess=lambda x:((x / 255.) - 0.5) * 2,
                       y_preprocess=None
                       ):
    '''建议框数据源
        @param image_dir: 图片目录
        @param count: 每个proposals_out文件取多少条记录
        @param proposals_out: 建议框jsons文件路径
        @param is_proposal_mutiple_file: 是否多文件(多文件会proposals.jsons0, proposals.jsons1...开始，直到某个自增文件名找不到为止)
        @param proposal_every_image: 每张图片保留多少建议框
        @param batch_size: db的batch_size
        @param epochs: db.repear
        @param shuffle_buffer_rate: 打乱数据的buffer是batch_size的多少倍，<0则表示不打乱
        @return: 数据源, 当前迭代的y暂存器
    '''
    #    x，y形状
    x_shape = tf.TensorShape([conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT, 3])
    y_shape = tf.TensorShape(ymaps_shape)
    #    y_true数据暂存，保存最后一个批次的y_true（过y_preprocess后，与y_shape形状相同的tensor对象）的数据
    #    list长度为batch_size，每个数据都是tensor对象。自定义layer中直接用此队列拿当前迭代y值
    #    与shuffle无法共用，不能保证打乱后的顺序与先进先出的顺序相同
    proposals_crt_batch_queue = ProposalsCrtBatchQueue(batch_size=batch_size, ymaps_shape=ymaps_shape)
    
    db = tf.data.Dataset.from_generator(generator=lambda :read_proposals_generator(image_dir=image_dir,
                                                                                   count=count,
                                                                                   proposals_out=proposals_out,
                                                                                   is_proposal_mutiple_file=is_proposal_mutiple_file,
                                                                                   proposal_every_image=proposal_every_image,
                                                                                   x_preprocess=x_preprocess,
                                                                                   y_preprocess=y_preprocess,
                                                                                   proposals_crt_batch_queue=proposals_crt_batch_queue), 
                                        output_types=(tf.float32, tf.float32), 
                                        output_shapes=(x_shape, y_shape))
    #    是否要打乱
    #    deque那边是先进先出，光打乱一边顺序就不对了
#     if (shuffle_buffer_rate > 0):
#         db = db.shuffle(buffer_size=shuffle_buffer_rate * batch_size)
#         pass
    if (batch_size): db = db.batch(batch_size)
    if (epochs): db = db.repeat(epochs)
    return db, proposals_crt_batch_queue





#    取总样本数
def total_samples(proposal_out=conf.PROPOSALES.get_train_proposal_out(), 
                    is_proposal_mutiple_file=conf.DATASET.get_label_train_mutiple(), 
                    count=conf.DATASET.get_count_train(),
                    proposal_every_image=conf.PROPOSALES.get_proposal_every_image()
                    ):
    '''根据is_rois_mutiple_file和count取数据总数
        total = 文件数 * 每个文件图片数(count) * 每个图片正负样本数(positives_every_image + negative_every_image)
        @param rois_out: roi标签文件
        @param is_rois_mutiple_file: 是否多文件模式
        @param count: 单文件总数
    '''
    #    如果非多文件模式，直接返回count
    if (not is_proposal_mutiple_file): return count
    
    #    多文件模式 count = count * 文件总数
    fcount = 0
    while (os.path.exists(proposal_out + str(fcount))):
        fcount += 1
        pass
    return fcount * count