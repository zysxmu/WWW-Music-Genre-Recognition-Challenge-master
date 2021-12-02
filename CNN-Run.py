# encoding: utf-8 
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt; 
from skimage import io
from PIL import Image, ImageDraw, ImageFont
import csv
import random
from xception import xception, xception_arg_scope
import tensorflow.contrib.slim as slim
import datetime
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step

###global
TOTAL_IMAGE_NUM_TRAIN = 0#总训练图片数量
TOTAL_IMAGE_NUM_DEV = 0#总测试图片数量

BATCH_SIZE_TRAIN = 64 #训练集的batch size
BATCH_SIZE_DEV = 64#测试集的batch size
PRINT_STEP = 1000
TOTAL_STEP = 0
EPOCH = 20#epoch数量
BATCH_COUNT_TRAIN = 0#train的batch的批数
BATCH_COUNT_DEV = 0#dev的batch的批数

keep_prob = 0.5#drop参数
###end


###initialization
def initialization():#读取data文件夹获取train数据，并确认有多少个图片
    
    global TOTAL_IMAGE_NUM_TRAIN
    global TOTAL_IMAGE_NUM_DEV
    global BATCH_SIZE_TRAIN
    global BATCH_SIZE_DEV
    global BATCH_COUNT_TRAIN
    global BATCH_COUNT_DEV
    global EPOCH
    global TOTAL_STEP

    f = open('train_splited_num.txt')
    TOTAL_IMAGE_NUM_TRAIN = int(f.read())
    f.close()
    f = open('dev_splited_num.txt')
    TOTAL_IMAGE_NUM_DEV = int(f.read())
    f.close()
    
    
    if (TOTAL_IMAGE_NUM_TRAIN % BATCH_SIZE_TRAIN) == 0:#得到训练集的batch个数
        BATCH_COUNT_TRAIN = int(TOTAL_IMAGE_NUM_TRAIN / BATCH_SIZE_TRAIN) 
    else:
        BATCH_COUNT_TRAIN = int(TOTAL_IMAGE_NUM_TRAIN / BATCH_SIZE_TRAIN) + 1
        
    if (TOTAL_IMAGE_NUM_DEV % BATCH_SIZE_DEV) == 0:#得到测试集的batch个数
        BATCH_COUNT_DEV = int(TOTAL_IMAGE_NUM_DEV / BATCH_SIZE_DEV) 
    else:
        BATCH_COUNT_DEV = int(TOTAL_IMAGE_NUM_DEV / BATCH_SIZE_DEV) + 1
        
    TOTAL_STEP = BATCH_COUNT_TRAIN*EPOCH#得到总的训练步骤
    
    with open('information.txt', 'a') as f:
        f.write("Train number:%d,dev number:%d,Total Step:%d\n"\
                %(TOTAL_IMAGE_NUM_TRAIN,TOTAL_IMAGE_NUM_DEV,TOTAL_STEP))
###end    

###channel 2
a = tf.constant(1,shape=[1,299])
b = tf.constant(2,shape=[1,299])
channel2 = tf.concat([a,b],axis=0)
for i in range(297):
    channel2 = tf.concat([channel2,tf.constant(i+3,shape=[1,299])],axis=0)
channel_var = tf.constant(299.0)
channel_5 = tf.constant(0.5)
channel2 = tf.divide(tf.cast(channel2,tf.float32),channel_var)
channel2 = tf.subtract(tf.cast(channel2,tf.float32),channel_5)
channel2 = tf.reshape(channel2, [299, 299,1])
###end

###data.api
def _parse_function(serialized_example):#TFrecord的解析函数
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [299, 299,1])
    ### 标准化
    var = tf.constant(93.13)
    constant5 = tf.constant(127.5)
    img = tf.subtract(tf.cast(img,tf.float32),var)
    img = tf.divide(img,constant5)
    ###end
    img = tf.concat([img,channel2],axis=2)
    ###get onehot label
    label = tf.cast(features['label'], tf.int32)
    onehot_y = tf.one_hot(label,16)
    return img, onehot_y


train_dataset = tf.contrib.data.TFRecordDataset(["./train.tfrecords"])
train_dataset = train_dataset.cache()
train_dataset = train_dataset.map(_parse_function,num_threads=6)
train_dataset = train_dataset.shuffle(buffer_size=20000)
train_dataset = train_dataset.batch(BATCH_SIZE_TRAIN)
train_dataset = train_dataset.repeat()#不设置上限 

dev_dataset = tf.contrib.data.TFRecordDataset(["./dev.tfrecords"])
dev_dataset = dev_dataset.cache()
dev_dataset = dev_dataset.map(_parse_function,num_threads=6)
dev_dataset = dev_dataset.batch(BATCH_SIZE_DEV)
dev_dataset = dev_dataset.repeat()#不设置上限

###feedable
handle = tf.placeholder(tf.string, shape=[])
iterator = tf.contrib.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
X,Y = iterator.get_next()

#Create 2 iterators
train_iterator = train_dataset.make_initializable_iterator()
dev_iterator = dev_dataset.make_initializable_iterator()
###end

###data.api end

########Is_training
Is_training = tf.placeholder(tf.bool)#标记是否为True
###end

    
###dropout
KEEP_PROB = tf.placeholder(tf.float32)
###dropout end


###网络定义从这里开始
with slim.arg_scope(xception_arg_scope()):
    Y_prediction,end_points = xception(X,
                     num_classes=16,
                     is_training=Is_training,
                     scope='xception',
                     keep_prob=KEEP_PROB)
###网络结构这里结束

    
###定义交叉熵和训练步骤，得到正确预测的结果
#cross_entropy = tf.reduce_mean(\
#                              tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Y_prediction)) # 定义交叉熵为loss函数
loss = tf.losses.softmax_cross_entropy(onehot_labels = Y, logits = Y_prediction)
total_loss = tf.losses.get_total_loss()
global_step = get_or_create_global_step()

lr = tf.train.exponential_decay(
        learning_rate = 0.0001,
        global_step = global_step,
        decay_steps = 20000,
        decay_rate = 0.1,
        staircase = True)


optimizer = tf.train.AdamOptimizer(learning_rate = lr)
train_op = slim.learning.create_train_op(total_loss, optimizer) 
correct_prediction = tf.equal(tf.argmax(Y_prediction,1), tf.argmax(Y,1))
###end

initialization()#初始化函数，得到一些特定信息

variables_to_restore = slim.get_variables_to_restore()
###saver
saver = tf.train.Saver(variables_to_restore,max_to_keep = 20)  # 保存所有的变量，最多保存10个
model_file=tf.train.ladev_checkpoint('./save/')#尝试加载上次最新的训练结果
###end

with tf.Session() as sess: 
    ###creat handle
    train_handle = sess.run(train_iterator.string_handle())
    dev_handle = sess.run(dev_iterator.string_handle())
    ###end
    
    ###考察是否要加载旧模型
    if model_file !=None:
        saver.restore(sess, model_file)
        with open('information.txt', 'a') as f:
            f.write("old training!\n")
    else:
        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
        sess.run(init_op)
        with open('information.txt', 'a') as f:
            f.write("new training!\n")
    ###end
    
    ###train用的变量
    total_loss_val = 0#保存总损失
    sum_correct_train = 0#计算train集的正确率
    sum_all_train = 0#计算遍历了多少张图片
    total_loss_eval = 0
    ###end
    
    ###dev用的变量
    dev_oldAccuracy = 0
    dev_newAccuracy = 0
 
    ###end
    
    step = 0
    starttime = datetime.datetime.now()#计时开始
    with open('information.txt', 'a') as f:
        f.write("Training begin!\n")
        
    #初始化dropout,train=true,初始化train迭代器
    step,_,_ = sess.run([global_step,train_iterator.initializer,dev_iterator.initializer])
    #all_model_checkpoint_paths: "cnnModel.ckpt-2000"
    while step < TOTAL_STEP:
        
        step+=1#step+1
        
        ###train
        _, loss_val, correct_pre,loss_eval = sess.run([train_op, total_loss, correct_prediction,loss],\
                                            feed_dict={handle: train_handle,Is_training: True,KEEP_PROB:keep_prob})#训练+计算损失
        sum_correct_train += np.sum(correct_pre == True)
        total_loss_val += loss_val#累积本次step的损失,contain softmax and regular 
        total_loss_eval += loss_eval#only for softmax
        sum_all_train += np.sum(correct_pre == True)+np.sum(correct_pre == False)
        ###end
        
        if step % PRINT_STEP == 0:#如果到一个输出step
            ###输出train的信息
            with open('information.txt', 'a') as f:
                f.write("step %d,total_loss: %f, train accuracy %g, correct_num:%d,all_train:%d, loss:%g\n"%(\
                                                             step,total_loss_val,sum_correct_train/sum_all_train,\
                                                                                                    sum_correct_train,sum_all_train,total_loss_eval))#输出总损失
            
            total_loss_val = 0#更新
            sum_correct_train = 0#更新
            sum_all_train = 0#更新
            total_loss_eval = 0
            ###end
            
            ###存储当前得到的模型
            with open('information.txt', 'a') as f:
                f.write("Save!Step:%d\n"%(step))
            saver.save(sess,'./save/cnnModel.ckpt', global_step = step)#存储当前得到的模型
            ###end
            
            
            ###输出dev的信息
            sum_correct_dev = 0#计算测试集的正确率
            dev_loss = 0
            ###初始化dropout=1.0
            for batch_index in range(BATCH_COUNT_DEV):
                
                dev_loss_eval,correct_pre = sess.run([loss,correct_prediction],\
                                       feed_dict={handle: dev_handle,Is_training:False,KEEP_PROB:1.0})
                sum_correct_dev += np.sum(correct_pre == True)
                dev_loss+=dev_loss_eval
                
            with open('information.txt', 'a') as f:#输出
                f.write("step %d, loss %g, correct:%d,devNum:%d,dev-data accuracy %g\n"%(step,sum_correct_dev,dev_loss,TOTAL_IMAGE_NUM_DEV,\
                                                     sum_correct_dev/TOTAL_IMAGE_NUM_DEV)) 
            ###end
            
            ###更新旧的准确度和新的准确度
            dev_oldAccuracy = dev_newAccuracy
            dev_newAccuracy = sum_correct_dev/TOTAL_IMAGE_NUM_DEV
            ###end
            
            ###输出本次使用的时间
            endtime = datetime.datetime.now()#本次step print结束
            with open('information.txt', 'a') as f:
                    f.write("step:%d time %f min\n"%(step,(endtime-starttime).seconds/60))  
            starttime = datetime.datetime.now()###下次step开始之前计时
            ###end
