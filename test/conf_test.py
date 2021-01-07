# -*- coding: utf-8 -*-  
'''
配置相关测试

@author: luoyi
Created on 2020年12月29日
'''

from utils.conf import RPN


print(RPN.get_roi_areas())
print(RPN.get_roi_scales())


s = "aaa{}aaa{}aaa"
print(s.format("bbb", "ccc"))