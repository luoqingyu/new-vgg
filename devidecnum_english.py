#-*- coding: UTF-8 -*-

'''
1、读取指定目录下的所有文件
2、读取指定文件，输出文件内容
3、创建一个文件并保存到指定目录
'''
import os
import  chardet
import  random
import shutil
import PIL.Image as Image


def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)

        print (path)
        print ( ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print (path)
        print  (' 目录已存在')
        return False





# 遍历指定目录，显示目录下的所有文件名
def eachFile(filepath):
    pathDir =  os.listdir(filepath)
    hero_dir=[]
    danzi_list = []
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        #print child.decode('gbk') # .decode('gbk')是解决中文显示乱码问题
        hero_dir.append(child)
        danzi_list.append(allDir)
    return  hero_dir,danzi_list

def eachFile1(filepath):
    dir_list = []
    name_list = []
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        name_list.append(allDir)
        child = os.path.join('%s%s' % (filepath+'/', allDir))



        dir_list.append(child)
    return  dir_list,name_list



if __name__ == '__main__':

    E_list=['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','J','K','L','Q',
            'W','E','R','T','Y','U','I','O','P','Z','X','C','V','B','N','M','a','b','c','d','e','f','g','h','j','k','l','Q',
            'w','e','r','t','y','u','i','o','p','z','x','c','v','b','n','m',]
    filePath,danzi_list = eachFile("../data/danzi/")
    for i in danzi_list:
        new_list=[]
        if i in E_list:
            new_list.append(i)
    print(new_list)

    for i in new_list:
        pic_dir, pic_name = eachFile1("../data2/danzi-test/"+str(i))
        mkdir('../../data/danzi/'+str(i))
        for j in pic_dir:
            fromImage = Image.open(j)
            j = j.replace('data', 'data2')
            fromImage.save(j)











