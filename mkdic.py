# -*- coding: UTF-8 -*-
import  chardet
import os
path = "../data/danzi-test"
#制作字典映射
dic = {}
for name in os.listdir(path):
    
    dic[name] = 1
with open("./dic.txt",'w') as f:
    for i in dic:
        #print  (chardet.detect(i))
        #print (i)
        if (i =='0x2e'):
            i ='.'
        if (i =='0x2f'):
            i ='/'

        f.write(i + "\n")
