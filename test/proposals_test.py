# -*- coding: utf-8 -*-  
'''
建议框测试

@author: luoyi
Created on 2021年1月24日
'''
import matplotlib.pyplot as plot
import numpy as np

import data.dataset_proposals as proposals
import utils.conf as conf
from models.rpn import RPNModel


#    加载之前训练的RPN网络
model_conf_fpath = conf.RPN.get_save_weights_dir() + '/conf_rpn_resnet34.yml'
model_fpath = conf.RPN.get_save_weights_dir() + '/rpn_resnet34.h5'
_, _, M_ROIS, M_RPN, M_CNNS, M_CTX, M_PROPOSALS = conf.load_conf_yaml(model_conf_fpath)
#    初始化RPN网络
rpn_model = RPNModel(cnns_name=M_RPN.get_cnns(), 
                         learning_rate=M_RPN.get_train_learning_rate(),
                         scaling=M_CNNS.get_feature_map_scaling(), 
                         K=M_ROIS.get_K(),
                         cnns_base_channel_num=M_CNNS.get_base_channel_num(),
                         train_cnns=True,
                         train_rpn=True,
                         loss_lamda=M_RPN.get_loss_lamda(),
                         is_build=True)
rpn_model.load_model_weight(model_fpath)
#    设置cnns不参与训练
rpn_model.cnns.trainable = False


#    建议框生成器
proposals_creator = proposals.ProposalsCreator(threshold_nms_prob=0.5,
                                               threshold_nms_iou=0.95,
                                               proposal_iou=0.725,
                                               proposal_every_image=conf.PROPOSALES.get_proposal_every_image(),
                                               rpn_model=rpn_model)


proposales_iter = proposals_creator.test_create(image_dir=conf.DATASET.get_in_train(), 
                                                label_path=conf.DATASET.get_label_train(), 
                                                is_mutiple_file=conf.DATASET.get_label_train_mutiple(), 
                                                count=conf.DATASET.get_count_train(), 
                                                x_preprocess=lambda x:((x / 255.) - 0.5) * 2, 
                                                not_enough_preprocess=None)

#    图片展示
def show_img(X, proposales, is_show_proposales=True, is_show_labels=True):
    X = X / 2. + 0.5
    
    fig = plot.figure()
    ax = fig.add_subplot(1,1,1)
    
    label_dict = {}
    for proposale in proposales:
        if (is_show_proposales):
            x = proposale[1]
            y = proposale[2]
            w = np.abs(proposale[3] - proposale[1])
            h = np.abs(proposale[4] - proposale[2])
            rect = plot.Rectangle((x, y), w, h, fill=False, edgecolor = 'red',linewidth=1)
            ax.add_patch(rect)
            pass
        
        if (is_show_labels):
            if (label_dict.get(proposale[5]) is None):
                label_dict[proposale[5]] = 1
                
                x = proposale[6]
                y = proposale[7]
                w = proposale[8]
                h = proposale[9]
                rect = plot.Rectangle((x, y), w, h, fill=False, edgecolor = 'blue',linewidth=1)
                ax.add_patch(rect)
                pass
            pass
        pass
    
    plot.imshow(X)
    plot.show()
    pass

#    打印平均每张图片能找到多少建议框
# i = 0
# for x, proposales in proposales_iter:
#     print('idx:', i, ' proposales.count:', proposales.shape[0])
#     i += 1
#     pass

#    在图片上展示找到的建议框
show_idx = 25
idx = 0
for x, proposales in proposales_iter:
    if (idx < show_idx): 
        idx += 1
        continue
         
    show_img(x, proposales, is_show_proposales=True, is_show_labels=True)
    break
    pass



