# -*- coding: utf-8 -*-  
'''
Created on 2020年12月15日

@author: irenebritney
'''
import yaml
import os
import sys


#    取项目根目录（其他一切相对目录在此基础上拼接）
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)).split('faster_rcnn')[0]
ROOT_PATH = ROOT_PATH + "faster_rcnn"

#    训练图片统一高度
IMAGE_HEIGHT = 180
#    训练图片基本宽度（根据vcode长度有所不同，每个字120。自己算吧）
IMAGE_BASE_WEIGHT = 120
IMAGE_WEIGHT = 4 * IMAGE_BASE_WEIGHT

#    最大验证码数
MAX_VCODE_NUM = 6


#    取配置文件目录
CONF_PATH = ROOT_PATH + "/resources/conf.yml"
#    加载conf.yml配置文件
def load_conf_yaml(yaml_path=CONF_PATH):
    print('加载配置文件:' + CONF_PATH)
    f = open(yaml_path, 'r', encoding='utf-8')
    fr = f.read()
    
#     c = yaml.load(fr, Loader=yaml.SafeLoader)
    c = yaml.safe_load(fr)
    
    #    读取letter相关配置项
    dataset = Dataset(c['dataset']['in_train'], c['dataset']['count_train'], c['dataset']['label_train'], c['dataset']['label_train_mutiple'],
                    c['dataset']['in_val'], c['dataset']['count_val'], c['dataset']['label_val'], c['dataset']['label_val_mutiple'],
                    c['dataset']['in_test'], c['dataset']['count_test'], c['dataset']['label_test'], c['dataset']['label_test_mutiple'])
    
    rois = Rois(c['rois']['positives_iou'],
                c['rois']['negative_iou'],
                c['rois']['positives_every_image'],
                c['rois']['negative_every_image'],
                c['rois']['shuffle_buffer_rate'],
                c['rois']['batch_size'],
                c['rois']['epochs'],
                c['rois']['roi_areas'],
                c['rois']['roi_scales'],
                c['rois']['train_rois_out'],
                c['rois']['train_max_workers'],
                c['rois']['val_rois_out'],
                c['rois']['val_max_workers'],
                c['rois']['test_rois_out'],
                c['rois']['test_max_workers'])
    
    proposales = Proposals(c['proposals']['proposal_iou'],
                           c['proposals']['proposal_every_image'],
                           c['proposals']['train_proposal_out'],
                           c['proposals']['val_proposal_out'],
                           c['proposals']['test_proposal_out'],
                           c['proposals']['shuffle_buffer_rate'],
                           c['proposals']['batch_size'],
                           c['proposals']['epochs'])
    
    rpn = Rpn(c['rpn']['train_learning_rate'],
              c['rpn']['loss_lamda'],
              c['rpn']['save_weights_dir'],
              c['rpn']['tensorboard_dir'],
              c['rpn']['cnns'],
              c['rpn']['nms_threshold_positives'],
              c['rpn']['nms_threshold_iou'])
    
    cnns = CNNs(c['cnns']['feature_map_scaling'],
                c['cnns']['base_channel_num'])
    
    context = Context(c['context']['name'],
                      c['context']['is_after_operation'])
    
    fast_rcnn = FastRcnn(c['fast_rcnn']['roipooling_kernel_size'],
                         c['fast_rcnn']['loss_lamda'],
                         c['fast_rcnn']['fc_weights'],
                         c['fast_rcnn']['fc_layers'],
                         c['fast_rcnn']['fc_dropout'],
                         c['fast_rcnn']['cnns'],
                         c['fast_rcnn']['save_weights_dir'],
                         c['fast_rcnn']['tensorboard_dir'])
    
    return c, dataset, rois, rpn, cnns, context, proposales, fast_rcnn


#    验证码识别数据集。为了与Java的风格保持一致
class Dataset:
    def __init__(self, 
                 in_train="", count_train=50000, label_train="", label_train_mutiple=False,
                 in_val="", count_val=10000, label_val="", label_val_mutiple=False,
                 in_test="", count_test=10000, label_test="", label_test_mutiple=False):
        self.__in_train = convert_to_abspath(in_train)
        self.__count_train = count_train
        self.__label_train = convert_to_abspath(label_train)
        self.__label_train_mutiple = label_train_mutiple
        
        self.__in_val = convert_to_abspath(in_val)
        self.__count_val = count_val
        self.__label_val = convert_to_abspath(label_val)
        self.__label_val_mutiple = label_val_mutiple
        
        self.__in_test = convert_to_abspath(in_test)
        self.__count_test = count_test
        self.__label_test = convert_to_abspath(label_test)
        self.__label_test_mutiple = label_test_mutiple
        pass
    def get_in_train(self): return convert_to_abspath(self.__in_train)
    def get_count_train(self): return self.__count_train
    def get_label_train(self): return convert_to_abspath(self.__label_train)
    def get_label_train_mutiple(self): return self.__label_train_mutiple
    
    def get_in_val(self): return convert_to_abspath(self.__in_val)
    def get_count_val(self): return self.__count_val
    def get_label_val(self): return convert_to_abspath(self.__label_val)
    def get_label_val_mutiple(self): return self.__label_val_mutiple   
    
    def get_in_test(self): return convert_to_abspath(self.__in_test)
    def get_count_test(self): return self.__count_test
    def get_label_test(self): return convert_to_abspath(self.__label_test)
    def get_label_test_mutiple(self): return self.__label_test_mutiple
    pass

#    RPN相关配置
class Rpn():
    def __init__(self, 
                 train_learning_rate=0.01,
                 loss_lamda=10,
                 save_weights_dir="models",
                 tensorboard_dir="logs/tensorboard",
                 cnns='reset_34',
                 nms_threshold_positives=0.5,
                 nms_threshold_iou=0.7):
        self.__train_learning_rate = train_learning_rate
        self.__loss_lamda = loss_lamda
        self.__save_weights_dir = save_weights_dir
        self.__tensorboard_dir = tensorboard_dir
        self.__cnns = cnns
        self.__nms_threshold_positives = nms_threshold_positives
        self.__nms_threshold_iou = nms_threshold_iou
        pass
    def get_train_learning_rate(self): return self.__train_learning_rate
    def get_loss_lamda(self): return self.__loss_lamda
    def get_save_weights_dir(self): return convert_to_abspath(self.__save_weights_dir)
    def get_tensorboard_dir(self): return convert_to_abspath(self.__tensorboard_dir)
    def get_cnns(self): return self.__cnns
    def get_nms_threshold_positives(self): return self.__nms_threshold_positives
    def get_nms_threshold_iou(self): return self.__nms_threshold_iou
    pass

#    ROIS相关配置
class Rois():
    def __init__(self, 
                 positives_iou=0.7,
                 negative_iou=0.3,
                 positives_every_image=64,
                 negative_every_image=64,
                 shuffle_buffer_rate=128,
                 batch_size=2,
                 epochs=5,
                 roi_areas=[64, 68, 76, 80, 84, 92],
                 roi_scales=[0.8, 1.],
                 train_rois_out='temp/rois/rois_train.jsons',
                 train_max_workers=-1,
                 val_rois_out='temp/rois/rois_val.jsons',
                 val_max_workers=-1,
                 test_rois_out='temp/rois/rois_test.jsons',
                 test_max_workers=-1):
        self.__positives_iou = positives_iou
        self.__negative_iou = negative_iou
        self.__positives_every_image = positives_every_image
        self.__negative_every_image = negative_every_image
        self.__shuffle_buffer_rate = shuffle_buffer_rate
        self.__batch_size = batch_size
        self.__epochs = epochs
        self.__roi_areas = roi_areas
        self.__roi_scales = roi_scales
        
        self.__train_rois_out = train_rois_out
        self.__train_max_workers = train_max_workers
        self.__val_rois_out = val_rois_out
        self.__val_max_workers = val_max_workers
        self.__test_rois_out = test_rois_out
        self.__test_max_workers = test_max_workers
        
        self.__K = len(roi_areas) * len(roi_scales)
        pass
    def get_positives_iou(self): return self.__positives_iou
    def get_negative_iou(self): return self.__negative_iou
    def get_positives_every_image(self): return self.__positives_every_image
    def get_negative_every_image(self): return self.__negative_every_image
    def get_shuffle_buffer_rate(self): return self.__shuffle_buffer_rate
    def get_batch_size(self): return self.__batch_size
    def get_epochs(self): return self.__epochs
    def get_roi_areas(self): return self.__roi_areas
    def get_roi_scales(self): return self.__roi_scales
    def get_K(self): return self.__K
    def get_train_rois_out(self): return convert_to_abspath(self.__train_rois_out)
    def get_train_max_workers(self): return self.__train_max_workers
    def get_val_rois_out(self): return convert_to_abspath(self.__val_rois_out)
    def get_val_max_workers(self): return self.__val_max_workers
    def get_test_rois_out(self): return convert_to_abspath(self.__test_rois_out)
    def get_test_max_workers(self): return self.__test_max_workers
    pass

#    Proposals相关配置
class Proposals():
    def __init__(self, 
                 proposal_iou=0.7, 
                 proposal_every_image=32, 
                 train_proposal_out='',
                 val_proposal_out='',
                 test_proposal_out='',
                 shuffle_buffer_rate=-1,
                 batch_size=4,
                 epochs=2):
        self.__proposal_iou = proposal_iou
        self.__proposal_every_image = proposal_every_image
        self.__train_proposal_out = train_proposal_out
        self.__val_proposal_out = val_proposal_out
        self.__test_proposal_out = test_proposal_out
        self.__shuffle_buffer_rate = shuffle_buffer_rate
        self.__batch_size = batch_size
        self.__epochs = epochs
        pass
    def get_proposal_iou(self): return self.__proposal_iou
    def get_proposal_every_image(self): return self.__proposal_every_image
    def get_train_proposal_out(self): return convert_to_abspath(self.__train_proposal_out)
    def get_val_proposal_out(self): return convert_to_abspath(self.__val_proposal_out)
    def get_test_proposal_out(self): return convert_to_abspath(self.__test_proposal_out)
    def get_shuffle_buffer_rate(self): return self.__shuffle_buffer_rate
    def get_batch_size(self): return self.__batch_size
    def get_epochs(self): return self.__epochs
    pass

#    CNNs相关配置
class CNNs():
    def __init__(self, feature_map_scaling=8, base_channel_num=32):
        self.__feature_map_scaling = feature_map_scaling
        self.__base_channel_num = base_channel_num
        pass
    def get_feature_map_scaling(self): return self.__feature_map_scaling
    def get_base_channel_num(self): return self.__base_channel_num
    pass

#    环境相关配置
class Context():
    def __init__(self, name='dev', 
                 is_after_operation=False):
        self.__name = name
        self.__is_after_operation = is_after_operation
        pass
    def get_name(self): return self.__name
    def get_is_after_operation(self): return self.__is_after_operation
    pass

#    fast_rcnn相关配置
class FastRcnn():
    def __init__(self, 
                 roipooling_kernel_size=[7, 7],
                 loss_lamda=1,
                 fc_weights=4096,
                 fc_layers=4,
                 fc_dropout=0.8,
                 cnns='resnet34',
                 save_weights_dir='temp/models/fast_rcnn',
                 tensorboard_dir='logs/tensorboard/fast_rcnn'):
        self.__roipooling_kernel_size = roipooling_kernel_size
        self.__loss_lamda = loss_lamda
        self.__fc_weights = fc_weights
        self.__fc_layers = fc_layers
        self.__fc_dropout = fc_dropout
        self.__cnns = cnns
        self.__save_weights_dir = save_weights_dir
        self.__tensorboard_dir = tensorboard_dir
        pass
    def get_roipooling_kernel_size(self): return self.__roipooling_kernel_size
    def get_loss_lamda(self): return self.__loss_lamda
    def get_fc_weights(self): return self.__fc_weights
    def get_fc_layers(self): return self.__fc_layers
    def get_fc_dropout(self): return self.__fc_dropout
    def get_cnns(self): return self.__cnns
    def get_save_weights_dir(self): return convert_to_abspath(self.__save_weights_dir)
    def get_tensorboard_dir(self): return convert_to_abspath(self.__tensorboard_dir)
    pass


#    取配置的绝对目录
def convert_to_abspath(path):
    '''取配置的绝对目录
        "/"开头的目录原样输出
        非"/"开头的目录开头追加项目根目录
    '''
    if (path.startswith("/")):
        return path
    else:
        return ROOT_PATH + "/" + path
    
#    检测文件所在上级目录是否存在，不存在则创建
def mkfiledir_ifnot_exises(filepath):
    '''检测log所在上级目录是否存在，不存在则创建
        @param filepath: 文件目录
    '''
    _dir = os.path.dirname(filepath)
    if (not os.path.exists(_dir)):
        os.makedirs(_dir)
    pass
#    检测目录是否存在，不存在则创建
def mkdir_ifnot_exises(_dir):
    '''检测log所在上级目录是否存在，不存在则创建
        @param dir: 目录
    '''
    if (not os.path.exists(_dir)):
        os.makedirs(_dir)
    pass

ALL_DICT, DATASET, ROIS, RPN, CNNS, CONTEXT, PROPOSALES, FAST_RCNN = load_conf_yaml()


#    写入配置文件
def write_conf(_dict, file_path):
    '''写入当前配置项的配置文件
        @param dict: 要写入的配置项字典
        @param file_path: 文件path
    '''
    file_path = convert_to_abspath(file_path)
    mkfiledir_ifnot_exises(file_path)
    
    #    存在同名文件先删除
    if (os.path.exists(file_path)):
        os.remove(file_path)
        pass
    
    fw = open(file_path, mode='w', encoding='utf-8')
    yaml.safe_dump(_dict, fw)
    fw.close()
    pass


#    追加sys.path
def append_sys_path(path):
    path = convert_to_abspath(path)
    sys.path.append(path)
    print(sys.path)
    pass
