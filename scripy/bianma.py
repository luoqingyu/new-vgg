ture_name = ''
for i in range(self.max_word_num):
    zi = random.randrange(0, len(self.img_list))             #选择字
    if (zi < len(self.img_list)):
                    ziti = random.randrange(0, len(self.img_list[zi]))    #选择字体
                    ture_name = ture_name + self.img_list[zi][ziti].split("/")[-2]  #获取字的编码