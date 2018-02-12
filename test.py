# -*- coding: UTF-8 -*-
import  os
import  random
import PIL.Image as Image
import datetime
import numpy as np
import utils
import tensorflow as tf
import time
class ReadData:
    def __init__(self,img_list,label_list,batch_size=128):
        self.image = []
        self.labels = []
        self.dataset = tf.data.Dataset.from_tensor_slices((img_list, label_list))
        self.dataset = self.dataset.map(self.parse_function)
        self.dataset = self.dataset.repeat()  # 不带参数为无限个epoch
        self.dataset = self.dataset.shuffle(buffer_size=20000)  # 缓冲区，随机缓存区
        self.batched_dataset = self.dataset.batch(batch_size)
        self.iterator = self.batched_dataset.make_initializable_iterator()
    def parse_function(self,filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=1)
        image_decoded = image_decoded / 255
        image_resize = tf.image.resize_images(image_decoded,[32,tf.shape(image_decoded)[1]])
        add = tf.zeros((32, 256-tf.shape(image_resize)[1],1))+image_decoded[-1][-1]
        im =tf.concat( [image_resize,add],1)
        #print(im.shape)
        return im, label
    def init_itetator(self,sess):
        sess.run(self.iterator.initializer)
    def get_nex_batch(self):
        return  self.iterator.get_next()



def parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string, channels=1)
    image_decoded = image_decoded / 255
    image_resize = tf.image.resize_images(image_decoded,[32,tf.shape(image_decoded)[1]])
    add = tf.zeros((32, 256-tf.shape(image_resize)[1],1))+image_decoded[-1][-1]
    im =tf.concat( [image_resize,add],1)
        #print(im.shape)
    return im, label
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    val_feeder = utils.DataIterator(data_dir='../data/test/', istrain=False)
    filename = val_feeder.image
    print(len(filename))
    label = val_feeder.labels


    dataset = tf.data.Dataset.from_tensor_slices((filename, label))
    dataset = dataset.map(parse_function)
    dataset = dataset.repeat()  # 不带参数为无限个epoch
    dataset = dataset.shuffle(buffer_size=20000)  # 缓冲区，随机缓存区
    batched_dataset = dataset.batch(128)
    iterator = batched_dataset.make_initializable_iterator()
    with tf.Session() as sess:
        sess.run(iterator.initializer)

        tf_train_data  = iterator.get_next()
        start_time = time.time()
        for i in range(1000):
            imgbatch, label_batch = sess.run(tf_train_data)
        print(time.time()-start_time)


