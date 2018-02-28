# -*- coding: UTF-8 -*-
import  os
import  random
import PIL.Image as Image
import datetime
class GenerateImages:
    def __init__(self,pic_size=['32','256'],max_word_num=8,src_path='',out_path='',if_random_space='False',random_space=[0,0],space=0):
        self.pic_size = pic_size
        self.max_word_num = max_word_num
        self.src_path = src_path
        self.out_path = out_path
        self.if_random_space = if_random_space
        self.random_space = random_space
        self.space=space
        self.img_list=self.get_imgList()
        self.mkdir(out_path)


    def get_imgList(self):
        list = []
        A_path, na_list = self.readEachFile(read_path=self.src_path)
        print(na_list)
        for child_list in A_path:
            img, name = self.readEachFile(child_list + '/')


            list.append(img)
        return  list




    def readEachFile(self,read_path):
        pathDir = os.listdir(read_path)
        Absolute_dir = []
        name_list = []
        for allDir in pathDir:

            child = os.path.join('%s%s' % (read_path, allDir))
            # print child.decode('gbk') # .decode('gbk')是解决中文显示乱码问题
            Absolute_dir.append(child)


            name_list.append(allDir)
        return Absolute_dir, name_list







    def creat_pic_each_file(self,f,file):
        for num in range(50):
            toImage = Image.new('RGBA', (256, 32), (255, 255, 255))
            ture_name = ''
            for i in range(self.max_word_num):
                zi = random.randrange(0, len(self.img_list))             #选择字
                if (zi < len(self.img_list)):
                    ziti = random.randrange(0, len(self.img_list[zi]))    #选择字体
                    if (self.img_list[zi][ziti].split("/")[-2]=='0x2e'):
                        ture_name = ture_name + '.'
                    elif (self.img_list[zi][ziti].split("/")[-2] == '0x2f'):
                        ture_name = ture_name + '/'
                    else:
                        ture_name = ture_name + self.img_list[zi][ziti].split("/")[-2]  #获取字的编码

                    # x =  random.randrange(0, len(list))
                    fromImage = Image.open(self.img_list[zi][ziti])
                    fromImage = fromImage.resize((32, 32), Image.ANTIALIAS)  # 先拼的图片不多，不用缩小

                    if (self.if_random_space==True):
                        self.space = random.randrange(self.random_space[0],self.random_space[1])
                        #print (self.space)
                    toImage.paste(fromImage, ((i) * 32+(self.space), 0))


            name=ture_name.replace('/','')                    #去除不可用的符号
            name = name.replace('.', '')
            file_ = file.replace('/', '_')
            pic_name = file_+str(num)+'_' + name + '.png'
            toImage.save(self.out_path+file+file_+str(num)+'_' + name + '.png')
            f.write(file+pic_name+' '+ture_name+'\n')

    def mkdir(self,path):  # 判断是否存在指定文件夹，不存在则创建
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
            print  (' 创建成功')
            return True
        else:
            # 如果目录存在则不创建，并提示目录已存在
            print (path)
            print  (' 目录已存在')
            return False

    def creatPic(self,num='1000'):
        txtName = self.out_path+"word_list.txt"
        f = open(txtName, "w")
        for file1 in range(int(100000 / 25000)):
            self.mkdir(self.out_path+str(file1))
            for file2 in range(5):
                self.mkdir(self.out_path+str(file1)+'/'+str(file2))
                self.creat_pic_each_file(f=f,file=str(file1)+'/'+str(file2)+'/')
        f.close()

if __name__ == '__main__':
    start = datetime.datetime.now()
    x= GenerateImages(src_path='../../data/danzi-train/',
                      out_path='../../data/train/',
                      if_random_space = True,#是否随机间隔
                      random_space=[-5,5],#随机间隔
                      space=0)              #固定间隔

    x.creatPic(num=25000)                     #生成数目
    end = datetime.datetime.now()
    print(end - start)


