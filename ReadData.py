# -*- coding: UTF-8 -*-
import  os
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
        image_decoded = tf.image.decode_jpeg(image_string, channels=1)
        image_decoded = image_decoded / 255
        #image_resize = tf.image.resize_images(image_decoded,[32,tf.shape(image_decoded)[1]])
        add = tf.zeros((tf.shape(image_decoded)[0],256-tf.shape(image_decoded)[1],1))+image_decoded[-1][-1]
        im =tf.concat( [image_decoded,add],1)
        image_resize = tf.image.resize_images(im,[32,256])
        #print(im.shape)
        return image_resize, label
    def init_itetator(self,sess):
        sess.run(self.iterator.initializer)
    def get_nex_batch(self):
        return  self.iterator.get_next()
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    val_feeder = utils.DataIterator(data_dir='/home/work/data/', istrain=True)
    filename1 = val_feeder.image
    print(len(filename1))
    label1 = val_feeder.labels
    #print('234333333',label1.shapes())
    train_data = ReadData(filename1,label1)
    config = tf.ConfigProto(allow_soft_placement=False)
    with tf.Session(config=config) as sess:
        train_data.init_itetator(sess)
        #这行必须放在loop之外
        tf_train_data  =  train_data.get_nex_batch()
        start_time = time.time()
        for i in range(100):
            imgbatch, label_batch = sess.run(tf_train_data)
            #print(imgbatch)
            #print(label_batch)
            print(imgbatch.shape)
        print(time.time()-start_time)


