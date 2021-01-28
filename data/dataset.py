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

import utils.conf as conf
import utils.alphabet as alphabet


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