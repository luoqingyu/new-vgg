# -*- coding: UTF-8 -*-
import os
import numpy as np
import tensorflow as tf
from skimage import io
from skimage import transform
maxPrintLen = 100
tf.app.flags.DEFINE_string('checkpoint_dir', '../checkpoint/lstm_3', 'the checkpoint dir')
tf.app.flags.DEFINE_integer('rnn_layers', 3 ,'number of rnn layers')
tf.app.flags.DEFINE_string('gpu_idex', '0' ,'index of gpu' )
tf.app.flags.DEFINE_string('model', 'lstm' , 'name of the rnn part')
tf.app.flags.DEFINE_string('log_dir', '../log/lstm_3', 'the logging dir')
tf.app.flags.DEFINE_string('infer_dir', '../data/infer/', 'the infer data dir')
tf.app.flags.DEFINE_boolean('restore',False, 'whether to restore from the latest checkpoint')
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, 'inital lr')
tf.app.flags.DEFINE_integer('image_height', 32, 'image height')
tf.app.flags.DEFINE_integer('image_width',256, 'image width')
tf.app.flags.DEFINE_integer('image_channel', 1, 'image channels as input')
tf.app.flags.DEFINE_integer('max_stepsize',32, 'max stepsize in lstm, as well as '                                             'the output channels of last layer in CNN')
tf.app.flags.DEFINE_integer('num_hidden',128, 'number of hidden units in lstm')
tf.app.flags.DEFINE_integer('num_epochs', 1000, 'maximum epochs')
tf.app.flags.DEFINE_integer('batch_size',128, 'the batch_size')
tf.app.flags.DEFINE_integer('save_steps',200, 'the step to save checkpoint')
tf.app.flags.DEFINE_integer('validation_steps',1000, 'the step to validation')
tf.app.flags.DEFINE_float('decay_rate', 0.98, 'the lr decay rate')
tf.app.flags.DEFINE_float('beta1', 0.9, 'parameter of adam optimizer beta1')
tf.app.flags.DEFINE_float('beta2', 0.999, 'adam parameter beta2')
tf.app.flags.DEFINE_integer('decay_steps', 10000, 'the lr decay_step for optimizer')
tf.app.flags.DEFINE_float('momentum', 0.9, 'the momentum')

#tf.app.flags.DEFINE_string('train_dir','../data/test/', 'the train data dir')
#tf.app.flags.DEFINE_string('val_dir','../data/test/', 'the val data dir')
tf.app.flags.DEFINE_string('train_dir','/home/work/data/', 'the train data dir')
tf.app.flags.DEFINE_string('val_dir','/home/work/data/', 'the val data dir')
tf.app.flags.DEFINE_string('mode', 'train', 'train, val or infer')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'num of gpus')

FLAGS = tf.app.flags.FLAGS

# num_batches_per_epoch = int(num_train_samples/FLAGS.batch_size)

encode_maps = {}
decode_maps = {}

with open("./dic.txt") as f:
    i = 1
    for line in f.readlines():
        line = line.replace('\n','')

        #line[1] = int(line[1])

        encode_maps[line] = i
        decode_maps[i] = line

        i += 1
    print  (encode_maps)
        
encode_maps[''] = 0
decode_maps[0] = ''

# 所有类 + blank + space
num_classes = i+1
print("num_classes:", num_classes)
class DataIterator:
    def __init__(self, data_dir,istrain=True):
        self.image = []
        self.labels = []
        num_train = 1000000
        num_val   = 10000
        if istrain:
            i=0
            #fa = open(data_dir+"word_list.txt", 'r')
            fa = open(data_dir+"annotation_train.txt", 'r')
            for line in fa.readlines():
                if i<num_train:
                    img_path = data_dir+line.split(" ")[-2]
                    img_label = line.split("_")[1]
                    if os.path.exists(img_path):
                        try:
                            im = io.imread(img_path,as_grey=True)
                            h = im.shape[0]
                            l=im.shape[1]    
                            if FLAGS.image_width-l>=0:
                                try:
                                    code = [encode_maps[c] for c in list(img_label)]
                                    if (len(code)<16):
                                        for add in range(16-len(code)):
                                            code.append(0)
                                #print(code)
                                        self.labels.append(code)
                                        self.image.append(img_path)
                                        i=i+1                
                
                                except:
                                    continue
                        except:
                            continue
                    
        else:
            i = 0
            fa = open(data_dir+"annotation_val.txt", 'r')
            # fa = open(data_dir+"word_list.txt", 'r')
            for line in fa.readlines():
                if i<num_val:
                    img_path = data_dir + line.split(" ")[-2]
                    img_label = line.split("_")[1]
                    #print(img_label)
                    #img_label = img_label.replace('\n', '')
                    if os.path.exists(img_path):
                        try:
                            im = io.imread(img_path,as_grey=True)
                            h = im.shape[0]
                            l=im.shape[1]
                            if FLAGS.image_width-l>=0:
                                try:
                                    code = [encode_maps[c] for c in list(img_label)]
                                    if (len(code)<16):
                                        for add in range(16-len(code)):
                                            code.append(0)
                                        self.labels.append(code)
                                        self.image.append(img_path)
                                        i=i+1
                                except:
                                    continue
                        except:
                            continue
                        



    @property
    def size(self):

        return len(self.labels)


    def the_label(self, indexs):
        labels = []
        for i in indexs:
            labels.append(self.labels[i])

        return labels

    def input_index_generate_batch(self, index=None):
        if index:
            image_batch = [self.image[i] for i in index]
            label_batch = [self.labels[i] for i in index]
        else:
            image_batch = self.image
            label_batch = self.labels

        def get_input_lens(sequences):
            # 64 is the output channels of the last layer of CNN
            lengths = np.asarray([FLAGS.max_stepsize for _ in sequences], dtype=np.int64)

            return sequences, lengths

        batch_inputs, batch_seq_len = get_input_lens(np.array(image_batch))
        batch_labels = sparse_tuple_from_label(label_batch)

        return batch_inputs, batch_seq_len, batch_labels


def accuracy_calculation(original_seq, decoded_seq, ignore_value=-1, isPrint=False):
    if len(original_seq) != len(decoded_seq):
        print('original lengths is different from the decoded_seq, please check again')
        return 0
    count = 0
    for i, origin_label in enumerate(original_seq):
        while(origin_label[-1]==0):
            origin_label.remove(0)
        decoded_label = [j for j in decoded_seq[i] if j != ignore_value]
        if isPrint and i < maxPrintLen:
            # print('seq{0:4d}: origin: {1} decoded:{2}'.format(i, origin_label, decoded_label))

            with open('./test.csv', 'a+') as f:
                f.write(str(origin_label) + '\t' + str(decoded_label))
                f.write('\n')

        if origin_label == decoded_label:
            count += 1

    return count * 1.0 / len(original_seq)


def sparse_tuple_from_label(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []
    #print('2222222222222',type(sequences))
    for n, seq in enumerate(sequences):
        while(seq[-1]==0):
            seq.remove(0)
        #print(seq)
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)
    indices =  np.asarray(indices, dtype=np.int64)
    values =   np.asarray(values, dtype=dtype)
    shape =    np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)
    return indices, values, shape


def eval_expression(encoded_list):
    """
    :param encoded_list:
    :return:
    """

    eval_rs = []
    for item in encoded_list:
        try:
            rs = str(eval(item))
            eval_rs.append(rs)
        except:
            eval_rs.append(item)
            continue

    with open('./result.txt') as f:
        for ith in range(len(encoded_list)):
            f.write(encoded_list[ith] + ' ' + eval_rs[ith] + '\n')

    return eval_rs
