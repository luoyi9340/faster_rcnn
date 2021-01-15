# -*- coding: utf-8 -*-  
'''
rois生成器
rois数据源

@author: luoyi
Created on 2021年1月5日
'''
import os
import json
import math
import tensorflow as tf
import numpy as np
import PIL
import concurrent.futures.thread as thread

import data.dataset as ds
import utils.conf as conf
import utils.alphabet as alphabet
import utils.logger_factory as logf


log = logf.get_logger('data_rois_creator')


#    生成rois
class RoisCreator():
    '''rois生成器
    '''
    def __init__(self, 
                 roi_areas=conf.RPN.get_roi_areas(),                                     #    roi框面积[32, 64, 128]
                 roi_scales=conf.RPN.get_roi_scales(),                                   #    roi框长宽比例[0.5, 1., 2.]
                 cnns_feature_map_scaling=conf.CNNS.get_feature_map_scaling(),           #    cnns层缩放比例，16
                 train_positives_iou=conf.RPN.get_train_positives_iou(),                 #    正样本IoU阈值
                 train_negative_iou=conf.RPN.get_train_negative_iou()                    #    负样本IoU阈值
                 ):
        self.__roi_areas = roi_areas
        self.__roi_scales = roi_scales
        self.__cnns_feature_map_scaling = cnns_feature_map_scaling
        self.__train_positives_iou = train_positives_iou
        self.__train_negative_iou = train_negative_iou
        
        #    生成所有可能的anchors
        self._original_anchors = self.__create_original_anchors(W=conf.IMAGE_WEIGHT, H=conf.IMAGE_HEIGHT, 
                                                                feature_map_scaling=self.__cnns_feature_map_scaling, 
                                                                roi_areas=self.__roi_areas, roi_scales=self.__roi_scales)
        pass
    
    #    从单文件中读取标签并操作
    def __create_from_file(self, label_file_path, count, rois_out, log_interval=100, thread_name='thread0'):
        '''从单文件中读取标签并操作
            @param fpath: 标签文件路径
            @param count: 每个文件读取记录数
            @param rois_out: rois.json文件路径
            @param log_interval: 每多少条记录打印一次日志
        '''
        fw = file_writer(rois_out=rois_out)
        #    迭代标签文件
        label_iterator = ds.label_file_iterator(label_file_path=label_file_path, count=count)
        #    统计信息
        num_labels, num_positives, num_negative = 0, 0, 0
        num = 0
        for file_name, vcode, labels in label_iterator:
            anchors = self.__create_anchors(file_name, vcode, labels, self.__train_positives_iou, self.__train_negative_iou)
            
            j = json.dumps(anchors)
            fw.write(j + "\n")
            num_labels += 1
            num_positives += len(anchors['positives'])
            num_negative += len(anchors['negative'])
            
            num += 1
            if (num % log_interval == 0):
                log.info('create rois ' + thread_name + '. num:%d total:%d avg_p:%f avg_n:%f', num, count, num_positives/num_labels, num_negative/num_labels)
                pass
            pass
        
        fw.close()
        return (num_labels, num_positives, num_negative)
    
    #    生成rois
    def create(self, 
               label_file_path=conf.DATASET.get_label_train(),
               rois_out=conf.ROIS.get_train_rois_out(),
               count=conf.DATASET.get_count_train(),
               log_interval=100,
               label_mutiple=False,
               max_workers=-1):
        '''生成rois
            @param label_file_path: 标签文件完整路径
            @param rois_out: rois.jsons输出目录
            @param count: 读取记录条数（多文件为每个文件读取记录条数）
            @param log_interval: 每多少条记录打一次log
            @param label_mutiple: label是否为多文件（多文件则文件名从*.jsons0开始一直往后，直到读不到为止）
        '''
        label_files = ds.get_fpaths(label_mutiple=label_mutiple, file_path=label_file_path)
        
        #    遍历文件，用单文件或线程池的方式跑
        num_labels, num_positives, num_negative = 0, 0, 0
        #    单线程模式
        if (max_workers <= 0):
            idx = 0
            for fpath in label_files:
                rois_out_fpath = rois_out
                if (len(label_files) > 1): rois_out_fpath = rois_out + str(idx)
                (num_a, num_p, num_n) = self.__create_from_file(fpath, count, rois_out_fpath, log_interval)
                num_labels += num_a
                num_positives += num_p
                num_negative += num_n
                idx += 1
            pass
        #    多线程模式
        else:
            threadPool = thread.ThreadPoolExecutor(max_workers=max_workers)
            futures = []
            idx = 0
            for fpath in label_files:
                label_fpath = label_file_path + str(idx)
                thread_name = 'thread' + str(idx)
                thread_rois_out = rois_out + str(idx)
                future = threadPool.submit(self.__create_from_file, label_fpath, count, thread_rois_out, log_interval, thread_name)
                futures.append(future)
                idx += 1
                pass
            #    拿每个线程执行结果
            for future in futures:
                (num_a, num_p, num_n) = future.result()
                num_labels += num_a
                num_positives += num_p
                num_negative += num_n
                pass
            threadPool.shutdown(5)
            pass
        return (num_labels, num_positives, num_negative)
        
        
    #    测试方法
    def test_create(self,
                    file_name=None,
                    label_file_path=conf.DATASET.get_label_train(),
                    count=conf.DATASET.get_count_train(),
                    train_positives_iou=conf.RPN.get_train_positives_iou(), 
                    train_negative_iou=conf.RPN.get_train_negative_iou()):
        #    迭代标签文件
        label_iterator = ds.label_file_iterator(label_file_path=label_file_path, count=count)
        file_anchors = []
        for fname, vcode, labels in label_iterator:
            if (file_name is not None and fname != file_name): continue
            anchors = self.__create_anchors(fname, vcode, labels, train_positives_iou, train_negative_iou)
            file_anchors.append(anchors)
            pass
        return file_anchors
    #    根据比例对label中的值进行压缩
    def __compressible_label(self, labels, cs):
        new_labels = []
        for label in labels:
            new_labels.append((label[0], label[1] * cs, label[2], label[3] * cs, label[4]))
            pass
        return new_labels
    #    生成一张图片的anchors
    def __create_anchors(self, file_name, vcode, labels, 
                         train_positives_iou=0.7, train_negative_iou=0.3):
        '''生成一张图片的anchors
            @param labels: [(v, x,y,w,h)...] 此时的x,y是左上点坐标
            @param positives_every_label: 每个标签最多保留多少个anchor
        '''
        #    根据vcode长度对labels进行压缩
        compressible_scaling = 4 / len(vcode)
        labels = self.__compressible_label(labels, compressible_scaling)
        #    {file_name:"文件名", vcode:"验证码", positives:[(iou, (x,y,w,h), (v, x,y,w,h))...], negative:[(iou, (x,y,w,h)), ...]}
        j = {'file_name':file_name, 'labels':labels, 'vcode':vcode, 'positives':[], 'negative':[]}
        
        negative = []                           #    负样本列表
        positives = []                          #    正样本列表
        label_rois = {}                         #    每个标签对应正样本列表 k:标签值,v:[正样本列表s]
        
        #    遍历原生anchors，并计算与所有标签的IoU。
        for anchor in self._original_anchors:
            #    计算与所有标签的IoU
            #    与所有标签的iou最大值和对应的标签
            max_iou = 0
            max_iou_label = None
            for label in labels:
                if (label_rois.get(label[0]) is None): label_rois[label[0]] = []
                
                iou = self.__iou(label, anchor)
                
                if (max_iou < iou): 
                    max_iou = iou
                    max_iou_label = label
                pass
            
            #    如果最大iou < train_negative_iou，则判定为负样本
            if (max_iou < train_negative_iou): negative.append((max_iou, anchor))
            #    如果最大iou > train_positives_iou，则判定为max_iou_label的正样本
            elif (max_iou > train_positives_iou): label_rois[max_iou_label[0]].append((max_iou, anchor, max_iou_label))
            pass
        
        #    label_rois中每个列表排序并取前positives_every_label个信息
        for (_, arr) in label_rois.items(): 
            arr.sort(key=lambda e:e[0], reverse=True)
            positives += arr
            pass
        
        j['positives'] = positives
        j['negative'] = negative
        
        return j
    #    生成一张图片指定标签的anchors
    def __create_anchors_by_label(self, label, positives_every_label):
        '''生成一张图片指定标签的anchors
            @param label: (v, x,y,w,h)此时的x,y是左上点做标
            @param positives_every_label: 每个标签最多保留多少个anchor
        '''
        anchors = []
        #    遍历所有原始anchors，并计算与label的IoU
        for anchor in self._original_anchors:
            iou = self.__iou(label, anchor)
            anchors.append((iou, anchor))
            pass
        
        #    anchors按iou降序，并取前positives_every_label个元素
        anchors.sort(key=lambda e:e[0], reverse=True)
        anchors = anchors[0 : positives_every_label]
        return anchors[1,]
    
    
    #    计算IoU
    def __iou(self, label, anchor):
        '''计算IoU
            @param label: (v, x,y,w,h) x,y为左上点坐标
            @param anchor: (x,y,w,h) x,y为中心点坐标
        '''
        label_x = label[1] + label[3]/2
        label_y = label[2] + label[4]/2
        label_w = label[3]
        label_h = label[4]
        
        anchor_x = anchor[0]
        anchor_y = anchor[1]
        anchor_w = anchor[2]
        anchor_h = anchor[3]
        
        #    根据label和anchor中心点坐标判断位置关系：相离 / 相交
        #    如果两个中心坐标距离超过两个半径之和，则判定为相离，直接返回0
        if (math.fabs(label_x - anchor_x)  > (label_w + anchor_w) / 2.) \
            or (math.fabs(label_y - anchor_y)  > (label_h + anchor_h) / 2.):
            return 0.
        #    计算交集面积(4个顶点x中位于中间的两个 和 4个顶点y中位于中间的两个就是交集坐标)
        arr_x = [label_x + label_w/2, label_x - label_w/2, anchor_x + anchor_w/2, anchor_x - anchor_w/2]
        arr_x.sort()
        arr_y = [label_y + label_h/2, label_y - label_h/2, anchor_y + anchor_h/2, anchor_y - anchor_h/2]
        arr_y.sort()
        area_intersection = math.fabs(arr_x[1] - arr_x[2]) * math.fabs(arr_y[1] - arr_y[2])
        #    计算并集面积
        area_union = label_w * label_h + anchor_w * anchor_h - area_intersection
        if (area_union == 0):
            print(area_union)
        return area_intersection / area_union
    
    
    #   按照给定的W，H，特征图缩放比例，面积，长宽比生成所有可能的anchor
    def __create_original_anchors(self, 
                                W=conf.IMAGE_WEIGHT, H=conf.IMAGE_HEIGHT,
                                feature_map_scaling=16,
                                roi_areas=[32, 64, 128],
                                roi_scales=[0.5, 1, 2]
                                ):
        #    根据W ,H和feature_map_scaling一次计算所有的中心点坐标
        w_u, h_u = feature_map_scaling, feature_map_scaling                                 #    按照缩放比例计算小区域长宽（特征图中每个点对应原图区域的长宽）
#         p_all = (W / feature_map_scaling) * (H / feature_map_scaling)                     #    总点数
        w, h = w_u / 2, h_u / 2                                                             #    第一个点坐标
        anchors = []
        count_point = 0
        idx_w, idx_h = 0, 0                                                                 #    中心点对应特征图的坐标
        while (w < W and h < H):
            count_point += 1
            #    生成3种不同尺度和3种不同长宽比的anchor
            idx_area, idx_scales = 0, 0
            for area in roi_areas:
                idx_scales = 0
                for scales in roi_scales:
                    anchor = self.__create_original_anchors_by_xy_areas_scaling(w, h, area, scales, idx_w, idx_h, idx_area, idx_scales)
                    idx_scales += 1
                    if (anchor is not None):
                        anchors.append(anchor)
                        pass
                    pass
                idx_area += 1
                pass
            
            w += w_u                                                                        #    w轴步长为每个区域的宽度
            idx_w += 1
            if (w >= W):
                w = 0
                h += h_u                                                                    #    h轴步长为每个区域的高度
                idx_w = 0
                idx_h += 1
                pass
            pass
        return anchors
    #    以x, y为中心生成对应尺度和比例的矩形框
    def __create_original_anchors_by_xy_areas_scaling(self, x, y, area, scales, idx_w, idx_h, idx_area, idx_scales):
        #    按照比例计算长和宽
        w = area * scales
        h = area / scales
        #    如果左上坐标越界则返回None
        if (x - w/2 < 0 or y - h/2 < 0): return None
        #    如果右下坐标越界则返回None
        if (x + w/2 > conf.IMAGE_WEIGHT or y + h/2 > conf.IMAGE_HEIGHT): return None
        return (x, y, w, h, idx_w, idx_h, idx_area, idx_scales)
    pass





#    读上面creator生成的数据，组成tensor数据源
#    建上级目录，清空同名文件
def mkdirs_rois_out_and_remove_file(rois_out=conf.ROIS.get_train_rois_out()):
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


#    打开文件（该方法会清空之前的文件）
def file_writer(rois_out=conf.ROIS.get_train_rois_out()):
    mkdirs_rois_out_and_remove_file(rois_out=rois_out)
    
    fw = open(rois_out, mode='w', encoding='utf-8')
    return fw


#    rpn网络单独训练数据集生成器
def read_rois_generator(count=conf.DATASET.get_count_train(),
                        rois_out=conf.ROIS.get_train_rois_out(), 
                        is_rois_mutiple_file=False,
                        image_dir=conf.DATASET.get_in_train(), 
                        count_positives=conf.RPN.get_train_positives_every_image(),
                        count_negative=conf.RPN.get_train_negative_every_image(),
                        x_preprocess=lambda x:((x / 255.) - 0.5 ) * 2, 
                        y_preprocess=None):
    '''rpn网络单独训练数据集生成器
        x: 180*480*3 图片像素
        y: [
                [IoU得分, 
                 anchor中心x坐标, anchor中心y坐标, anchor宽, anchor高, anchor对应feature_map像素点x坐标, anchor对应feature_map像素点y坐标, anchor对应第几个area, anchor对应第几个scales
                 验证码值的index, label左上x坐标, label左上y坐标, label宽, label高]
                [IoU, x, y, w, h, idx_w, idx_h, idx_area, idx_scales, vcode_index, x, y, w, h]
                ...前count_positives个为正样本...
                ...后count_negative个为负样本...
            ]
        @param count: 每个文件读取记录数。若文件记录数不够则循环此文件直到够数为止
        @param rois_out: rois.jsons文件完整路径
        @param is_rois_mutiple_file: rois_jsons是否多文件（多文件会从rois.jsons0开始rois.jsons1，rois.jsons2一直往上遍历，直到遍历不到为止）
        @param image_dir: 图片文件目录
        @param x_preprocess: x数据预处理（默认归到-1 ~ 1之间）
        @param y_preprocess: y数据预处理
    '''
    label_files = ds.get_fpaths(is_rois_mutiple_file, rois_out)
    
    #    遍历所有文件和所有行
    for fpath in label_files:
        readed = 0
        #    从一个文件中读满count条记录为止。文件记录数不够就重复读
        while (readed <= count):
            for line in open(fpath, mode='r', encoding='utf-8'):
                readed += 1
                if (readed > count): break
                d = json.loads(line)
                
                #    读取图片信息，并且归一化
                file_name = d['file_name']
                image = PIL.Image.open(image_dir + "/" + file_name + '.png', mode='r')
                image = image.resize((conf.IMAGE_WEIGHT, conf.IMAGE_HEIGHT),PIL.Image.ANTIALIAS)
                x = np.asarray(image, dtype=np.float32)
                #    x数据做前置处理（归一化在这里做）
                if (x_preprocess is not None):x = x_preprocess(x)
                
                
                #    读取训练数据信息
                #    取正样本，如果不足count_positives，用IoU=-1，其他全0补全
                positives = d['positives']
                if (len(positives) > count_positives): positives = positives[0: count_positives]
                #    整个拉直成(*, 14)维数组
                positives = [[a[0]] + a[1] + [alphabet.category_index(a[2][0])] + a[2][1:] for a in positives]
                #    如果不足如果不足count_positives，用IoU=-1，其他全0补全
                if (len(positives) < count_positives): positives = positives + [[-1, 0,0,0,0,0,0,0,0, -1,0,0,0,0] for _ in range(count_positives - len(positives))]
                positives = np.array(positives)
                
                #    取负样本，并给每个样本补默认lable，如果不足count_negative，用IoU=-1，其他全0补全
                negative = d['negative']
                if (len(negative) > count_negative): negative = negative[0: count_negative]
                #    整个拉直成(*, 9)维数组
                negative = [[a[0]] + a[1] for a in negative]
                #    补label标签
                negative = np.c_[negative, [[-1, 0,0,0,0] for _ in range(len(negative))]]
                #    如果不足count_negative，用IoU=-1，其他全0补全
                if (len(negative) < count_negative): negative = negative + [[-1, 0,0,0,0,0,0,0,0, -1,0,0,0,0] for _ in range(count_negative - len(positives))]
        
                y = np.vstack((positives, negative))
                #    y数据过前置处理
                if (y_preprocess is not None): y = y_preprocess(y)
                
                yield x, y
                pass
            #    如果循环结束了readed还是==0，说明文件是空的。直接跳出循环
            if (readed == 0): 
                log.warn('file is empty, %s', fpath)
                break;
            pass
        pass
    
    pass


#    rpn网络单独训练数据集
def rpn_train_db(image_dir=conf.DATASET.get_in_train(), 
                        count=conf.DATASET.get_count_train(),
                        rois_out=conf.ROIS.get_train_rois_out(), 
                        is_rois_mutiple_file=False,
                        count_positives=conf.RPN.get_train_positives_every_image(),
                        count_negative=conf.RPN.get_train_negative_every_image(),
                        batch_size=conf.RPN.get_train_batch_size(), 
                        ymaps_shape=(12, 30, 6, 15),
                        x_preprocess=None, 
                        y_preprocess=None):
    '''rpn网络单独训练数据集
        @param image_dir: 图片文件目录
        @param count: 每个文件读取多少条记录，文件中记录数不够会循环此文件直到够数为止
        @param rois_out: rois.jsons文件路径
        @param is_rois_mutiple_file: rois.jsons是否为多文件（多文件会从rois.jsons0开始直到遍历不到文件）
        @param count_positives: 每张图片多少个正样本
        @param count_negative: 每张图片多少个负样本
        @param batch_size: 数据集的batch_size
        @param ymaps_shape: 特征图尺寸（参照one_hot做法会把label数据提前做成特征图的尺寸给到模型）
        @param x_preprocess: 训练数据预处理
        @param y_preprocess: 标签数据预处理（做成特征图尺寸就是在这里做的. 参考:rpn.preprocess.preprocess_like_fmaps）
    '''
    #    训练数据shape和标签数据shape
    x_shape = tf.TensorShape([conf.IMAGE_HEIGHT, conf.IMAGE_WEIGHT, 3])
    y_shape = tf.TensorShape(ymaps_shape)
    db = tf.data.Dataset.from_generator(lambda:read_rois_generator(count=count,
                                                                   rois_out=rois_out, 
                                                                   is_rois_mutiple_file=is_rois_mutiple_file,
                                                                   image_dir=image_dir, 
                                                                   count_positives=count_positives,
                                                                   count_negative=count_negative,
                                                                   x_preprocess=x_preprocess, 
                                                                   y_preprocess=y_preprocess),
                                        output_types=(tf.float32, tf.float32),
                                        output_shapes=(x_shape, y_shape)).batch(batch_size)
    return db


#    根据is_rois_mutiple_file和count取数据总数
def total_sample(rois_out, is_rois_mutiple_file=False, count=100):
    '''根据is_rois_mutiple_file和count取数据总数
        count = count                  单文件模式
                count * 文件总数        多文件模式
        @param is_rois_mutiple_file: 是否多文件模式
        @param count: 单文件总数
        @param rois_out: roi标签文件
    '''
    #    如果非多文件模式，直接返回count
    if (not is_rois_mutiple_file): return count
    
    #    多文件模式 count = count * 文件总数
    fcount = 0
    while (os.path.exists(rois_out + str(fcount))):
        fcount += 1
        pass
    return fcount * count


