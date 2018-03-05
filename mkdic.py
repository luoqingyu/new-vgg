# -*- coding: UTF-8 -*-
import  chardet
import os
#制作字典映射
dic = {}
f = open("/home/work/data/annotation_test.txt")             # 返回一个文件对象  
line = f.readline()             # 调用文件的 readline()方法  
while line:
    label = line.split('_')[-2]
    for i in label:
        
        dic[i]=1
    line=f.readline()
f.close() 
with open("./dic.txt",'w') as f:
    for i in dic:
        #print  (chardet.detect(i))
        #print (i)
        if (i =='0x2e'):
            i ='.'
        if (i =='0x2f'):
            i ='/'

        f.write(i + "\n")
