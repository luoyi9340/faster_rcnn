# -*- coding: utf-8 -*-  
'''

@author: luoyi
Created on 2020年12月30日
'''


class A():
    single_obj = None
    @staticmethod
    def single():
        if (A.single_obj is None):
            A.single_obj = A()
            pass
        return A.single_obj
    pass


print(id(A.single()))
print(id(A.single()))




