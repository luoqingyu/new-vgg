# -*- coding: UTF-8 -*-
import datetime
import logging
import os
import time
import sys
import numpy as np
import tensorflow as tf
from skimage import io
from skimage import transform
import cnn_lstm_otc_ocr
import utils
import helper
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"
#import matplotlib.pyplot as plt
FLAGS = utils.FLAGS

os.environ['CUDA_VISIBLE_DEVICES']=FLAGS.gpu_idex
logger = logging.getLogger('Traing for OCR using CNN+' + FLAGS.model +'+CTC')
logger.setLevel(logging.INFO)


def train(train_dir=None, val_dir=None, mode='train'):
    if FLAGS.model == 'lstm':
        model = cnn_lstm_otc_ocr.LSTMOCR(mode)
    else:
        print("no such model")
        sys.exit()

    #开始构建图
    model.build_graph()
    print('loading train data, please wait---------------------')
    train_feeder = utils.DataIterator(data_dir=FLAGS.train_dir,istrain=True)

    #########################read data###############################
    filename = train_feeder.image
    label = train_feeder.labels
    print(len(filename))

    def _parse_function(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_png(image_string, channels=1)
        image_decoded = image_decoded / 255
        #image_resize = tf.image.resize_images(image_decoded,[32,tf.shape(image_decoded)[1]])
        #add = tf.zeros((32, 256-tf.shape(image_resize)[1],1))+image_decoded[-1][-1]
        #im =tf.concat( [image_resize,add],1)
        #print(im.shape)
        return image_decoded, label

    dataset = tf.data.Dataset.from_tensor_slices((filename, label))
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat()  # 不带参数为无限个epoch
    dataset = dataset.shuffle(buffer_size=10000)  # 缓冲区，随机缓存区
    batched_dataset = dataset.batch(128)
    iterator = batched_dataset.make_initializable_iterator()

    ##################################end######################################



    print('get image: ', train_feeder.size)
    print('loading validation data, please wait---------------------')
    val_feeder = utils.DataIterator(data_dir=FLAGS.val_dir,istrain=False)
    #######################################################################
    filename1 = val_feeder.image
    label1 = val_feeder.labels

    dataset1 = tf.data.Dataset.from_tensor_slices((filename1, label1))
    dataset1 = dataset1.map(_parse_function)
    dataset1 = dataset1.repeat()  # 不带参数为无限个epoch
    dataset1 = dataset1.shuffle(buffer_size=10000)  # 缓冲区，随机缓存区
    batched_dataset1 = dataset1.batch(128)
    iterator1 = batched_dataset1.make_initializable_iterator()
    print('get image: ', val_feeder.size)

    num_train_samples = train_feeder.size  
    num_batches_per_epoch = int(num_train_samples / FLAGS.batch_size)  # 训练集一次epoch需要的batch数

    num_val_samples = val_feeder.size
    num_batches_per_epoch_val = int(num_val_samples / FLAGS.batch_size)  # 验证集一次epoch需要的batch数

    shuffle_idx_val = np.random.permutation(num_val_samples)
    ###########################data################################################

    with tf.device('/cpu:0'):

        config = tf.ConfigProto(allow_soft_placement=True)

        #######################read  data###################################






        with tf.Session(config=config) as sess:
            sess.run(iterator.initializer)
            sess.run(iterator1.initializer)
            train_data = iterator.get_next()
            test_data = iterator1.get_next()
            #全局变量初始化
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=100) #存储模型
            train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train', sess.graph) 

            #导入预训练模型
            if FLAGS.restore:
                ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                if ckpt:
                    # the global_step will restore sa well
                    saver.restore(sess, ckpt)
                    print('restore from the checkpoint{0}'.format(ckpt))
                else:
                    print("No checkpoint")

            print('=============================begin training=============================')
            accuracy_res = []
            epoch_res = []
            tmp_max = 0
            tmp_epoch = 0
            #print  FLAGS.num_epochs

            for cur_epoch in range(FLAGS.num_epochs):
                shuffle_idx = np.random.permutation(num_train_samples)
                train_cost = 0
                start_time = time.time()
                batch_time = time.time()
                for cur_batch in range(num_batches_per_epoch):


                    #获得这一轮batch数据的标号##############################

                    batch_inputs,batch_labels  = sess.run(train_data)


                    new_batch_labels = utils.sparse_tuple_from_label(batch_labels)  # 对了
                    batch_seq_len = np.asarray([16 for _ in batch_inputs], dtype=np.int64)




                    # batch_inputs,batch_seq_len,batch_labels=utils.gen_batch(FLAGS.batch_size)###############

                    feed = {model.inputs: batch_inputs,
                            model.labels: new_batch_labels,
                            model.seq_len: batch_seq_len}
                    #print  batch_labels

                    # if summary is needed
                    # batch_cost,step,train_summary,_ = sess.run([cost,global_step,merged_summay,optimizer],feed)

                    summary_str, batch_cost, step, _ = \
                        sess.run([model.merged_summay, model.cost, model.global_step,
                                  model.train_op], feed)



                    # calculate the cost
                    train_cost += batch_cost * FLAGS.batch_size

                    #print  train_cost
                    train_writer.add_summary(summary_str, step)

                    # save the checkpoint
                    if step % FLAGS.save_steps == 1:
                        if not os.path.isdir(FLAGS.checkpoint_dir):
                            os.mkdir(FLAGS.checkpoint_dir)
                        logger.info('save the checkpoint of{0}', format(step))
                        saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'ocr-model'),
                                   global_step=step)
                    if (cur_batch  ) % 10 == 1:
                        print('batch', cur_batch, ': time', time.time() - batch_time,'loss',batch_cost)
                        batch_time = time.time()
                    # train_err += the_err * FLAGS.batch_size
                    # do validation
                    if step % FLAGS.validation_steps == 0:
                        validation_start_time = time.time()
                        acc_batch_total = 0
                        lastbatch_err = 0
                        lr = 0
                        for j in range(num_batches_per_epoch_val):
                            batch_inputs, batch_labels = sess.run(test_data)
                            new_batch_labels = utils.sparse_tuple_from_label(batch_labels)  # 对了
                            batch_seq_len = np.asarray([16 for _ in batch_inputs], dtype=np.int64)
                            val_feed = {model.inputs: batch_inputs,
                                        model.labels: new_batch_labels,
                                        model.seq_len: batch_seq_len}


                            dense_decoded, lr = \
                                sess.run([model.dense_decoded, model.lrn_rate],
                                         val_feed)

                            acc = utils.accuracy_calculation(batch_labels.tolist(), dense_decoded,
                                                             ignore_value=-1, isPrint=True)
                            acc_batch_total += acc
                        accuracy = (acc_batch_total * FLAGS.batch_size) / num_val_samples
                        accuracy_res.append(accuracy)
                        epoch_res.append(cur_epoch)
                        if accuracy > tmp_max:
                            tmp_max = accuracy
                            tmp_epoch = cur_epoch
                        avg_train_cost = train_cost / ((cur_batch + 1) * FLAGS.batch_size)

                        # train_err /= num_train_samples
                        now = datetime.datetime.now()
                        log = "{}/{} {}:{}:{} Epoch {}/{}, " \
                              "max_accuracy = {:.3f},max_Epoch {},accuracy = {:.3f},acc_batch_total = {:.3f},avg_train_cost = {:.3f}, " \
                              " time = {:.3f},lr={:.8f}"

                        print(log.format(now.month, now.day, now.hour, now.minute, now.second,
                                         cur_epoch + 1, FLAGS.num_epochs, tmp_max,tmp_epoch, accuracy,acc_batch_total,avg_train_cost,
                                         time.time() - validation_start_time, lr))


def infer(img_path, mode='infer'):
    # imgList = load_img_path('/home/yang/Downloads/FILE/ml/imgs/image_contest_level_1_validate/')
    imgList = helper.load_img_path(img_path)
    print(imgList[:5])

    model = cnn_lstm_otc_ocr.LSTMOCR(mode)
    model.build_graph()

    total_steps = len(imgList) / FLAGS.batch_size

    config = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print('restore from ckpt{}'.format(ckpt))
        else:
            print('cannot restore')

        decoded_expression = []
        for curr_step in range(total_steps):

            imgs_input = []
            seq_len_input = []
            for img in imgList[curr_step * FLAGS.batch_size: (curr_step + 1) * FLAGS.batch_size]:
                
                im = io.imread(img,as_grey=True)
                im = transform.resize(im, (FLAGS.image_height, FLAGS.image_width, FLAGS.image_channel))
                

                def get_input_lens(seqs):
                    length = np.array([FLAGS.max_stepsize for _ in seqs], dtype=np.int64)

                    return seqs, length

                inp, seq_len = get_input_lens(np.array([im]))
                imgs_input.append(im)
                seq_len_input.append(seq_len)

            imgs_input = np.asarray(imgs_input)
            seq_len_input = np.asarray(seq_len_input)
            seq_len_input = np.reshape(seq_len_input, [-1])

            feed = {model.inputs: imgs_input,
                    model.seq_len: seq_len_input}
            dense_decoded_code = sess.run(model.dense_decoded, feed)

            for item in dense_decoded_code:
                expression = ''

                for i in item:
                    if i == -1:
                        expression += ''
                    else:
                        expression += utils.decode_maps[i]

                decoded_expression.append(expression)

        with open('./result.txt', 'a') as f:
            for code in decoded_expression:
                f.write(code + '\n')


def main(_):
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    elif FLAGS.num_gpus == 1:
        dev = '/gpu:' + FLAGS.gpu_idex      #1    1
    else:
        raise ValueError('Only support 0 or 1 gpu.')

    with tf.device(dev):
        if FLAGS.mode == 'train':
            train(FLAGS.train_dir, FLAGS.val_dir, FLAGS.mode)

        elif FLAGS.mode == 'infer':                #做出推断
            infer(FLAGS.infer_dir, FLAGS.mode)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    print("plese enter model_name,gpu_idex,log_dir at lesast")
    tf.app.run()
