# -*- coding: utf-8 -*-  
'''
当前目录加入sys.path

@author: luoyi
Created on 2021年1月5日
'''
import sys
import os
#    取项目根目录
ROOT_PATH = os.path.abspath(os.path.dirname(__file__)).split('faster_rcnn')[0]
ROOT_PATH = ROOT_PATH + "faster_rcnn"
sys.path.append(ROOT_PATH)