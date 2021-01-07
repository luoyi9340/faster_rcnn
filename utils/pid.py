# -*- coding: utf-8 -*-  
'''
在当前项目录下新建一个pid文件，写入自己的pid

@author: luoyi
Created on 2021年1月4日
'''
import os

import utils.conf as conf


FILE_PATH = conf.ROOT_PATH + "/pid"


#    检测当前目录下是否存在pid文件，若存在则删除
def unique_file(file_path=FILE_PATH):
    if (os.path.exists(file_path)):
        os.remove(file_path)
        pass
    pass


#    写入pid文件
def write_pid(file_path=FILE_PATH):
    #    取当前进程pid
    pid = os.getpid()
    #    唯一性检测
    unique_file(file_path)
    fw = open(file_path, mode='w', encoding='utf-8')
    fw.write(str(pid))
    fw.close()
    pass


write_pid()
