import os
import utils
fa = open("../data/train1/word_list.txt", 'r')
for line in fa.readlines():
    img_path = '../data/train1/'+line.split(" ")[-2]
    img_label = line.split(" ")[-1]
    img_label=img_label.replace('\n','')
    if len(img_path!=8):
        print(img_path)
    print('999')

