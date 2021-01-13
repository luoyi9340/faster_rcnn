# -*- coding: utf-8 -*-  
'''
线程池测试

@author: luoyi
Created on 2021年1月11日
'''
import time
import threading
import concurrent.futures.thread as thread
import multiprocessing



def test(idx, name):
    if (lock.acquire(timeout=1)):
        print('idx:', idx, ' name:', name)
        lock.release()
        pass
    else:
        print('not get lock.')
    
    time.sleep(1)
    return idx

class SubProcess():
    
    def __init__(self, process_num=5):
        self.__process_num = process_num
        pass
    
    def test_process(self, idx, name):
        print('idx:', idx, ' name:', name)
        time.sleep(1)
        pass
    
    def do_fun(self):
        pool = multiprocessing.Pool(processes=5)
        fs = []
        for idx in range(10):
        #     f = tp.submit(test, idx, 'thread_' + str(idx))
        #     fs.append(f)
            f = pool.apply(self.test_process, (idx, 'process_' + str(idx)))
            fs.append(f)
            pass
        
        pool.close()
        pool.join()
        return fs
    
    pass

if __name__ == '__main__':
    #    感觉有坑。。。
    lock = threading.Lock()
    # tp = thread.ThreadPoolExecutor(max_workers=5)
    # fs = []
    # for idx in range(10):
    #     f = tp.submit(test, idx, 'thread_' + str(idx))
    #     fs.append(f)
    #     pass
    sp = SubProcess()
    sp.do_fun()
    
    print('run over.')
    pass

