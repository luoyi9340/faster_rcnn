# -*- coding: utf-8 -*-  
'''
日志测试

@author: luoyi
Created on 2021年1月5日
'''
import datetime

import utils.logger_factory as logf



log = logf.get_logger('rpn_loss')
log.info('test logger. str:%s num:%d', 'aaa', 18)
log.warning('test warn msg1:{} msg2:{}'.format('aaa', 18))


start = datetime.datetime.now()
end = datetime.datetime.now()
print('time:' + str(end - start))