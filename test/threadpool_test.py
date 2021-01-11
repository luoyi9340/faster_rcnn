# -*- coding: utf-8 -*-  
'''
线程池测试

@author: luoyi
Created on 2021年1月11日
'''
import time
import threading
import concurrent.futures.thread as thread


def test(idx, name):
    if (lock.acquire(timeout=1)):
        print('idx:', idx, ' name:', name)
        lock.release()
        pass
    else:
        print('not get lock.')
    
    time.sleep(1)
    return idx

#    感觉有坑。。。
lock = threading.Lock()
tp = thread.ThreadPoolExecutor(max_workers=5)
fs = []
for idx in range(10):
    f = tp.submit(test, idx, 'thread_' + str(idx))
    fs.append(f)
    pass

