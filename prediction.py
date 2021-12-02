# encoding: utf-8 
import tensorflow as tf
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io
from PIL import Image, ImageDraw, ImageFont
import csv
from xception import xception, xception_arg_scope
import tensorflow.contrib.slim as slim
import preprocess


TOTAL_IMAGE_NUM_FINAL_TEST = 0 #
BATCH_SIZE_FINAL_TEST = 64#测试集的batch size
PRINT_EPOCH = 1
BATCH_COUNT_FINAL_TEST = 0#test的batch的批数


#image的参数
IMAGE_HEIGHT = 299#长度
IMAGE_WIDTH = 299#宽度

keep_prob = 1.0#drop参数

def initialization():#读取data文件夹获取train数据，并确认有多少个图片

    global TOTAL_IMAGE_NUM_FINAL_TEST
    
    global TOTAL_IMAGE_NUM_FINAL_TEST
    global BATCH_SIZE_FINAL_TEST
    global BATCH_COUNT_FINAL_TEST
    f = open('./test_splited_num.txt')
    TOTAL_IMAGE_NUM_FINAL_TEST = int(f.read())
    f.close()

    if os.path.exists('./prediction-split-softmax.csv'):
        os.remove('./prediction-split-softmax.csv')
 
    if os.path.exists('./submit.csv'):
        os.remove('./submit.csv)
    
    if (TOTAL_IMAGE_NUM_FINAL_TEST % BATCH_SIZE_FINAL_TEST) == 0:#得到final test set的batch个数
        BATCH_COUNT_FINAL_TEST = int(TOTAL_IMAGE_NUM_FINAL_TEST / BATCH_SIZE_FINAL_TEST) 
    else:
        BATCH_COUNT_FINAL_TEST = int(TOTAL_IMAGE_NUM_FINAL_TEST / BATCH_SIZE_FINAL_TEST) + 1
    

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
                                           'name': tf.FixedLenFeature([], tf.string),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [299, 299,1])
    name = tf.decode_raw(features['name'], tf.uint8)
    ### 标准化
    var = tf.constant(93.13)
    constant5 = tf.constant(127.5)
    img = tf.subtract(tf.cast(img,tf.float32),var)
    img = tf.divide(img,constant5)
    ###end
    img = tf.concat([img,channel2],axis=2)
    return img, name


final_test_dataset = tf.contrib.data.TFRecordDataset(["./finaltest-relative.tfrecords"])
final_test_dataset = final_test_dataset.map(_parse_function,num_threads=6)
final_test_dataset = final_test_dataset.batch(BATCH_SIZE_FINAL_TEST)
final_test_dataset = final_test_dataset.repeat(1)

iterator = final_test_dataset.make_one_shot_iterator()
    
finaltest_init_op = iterator.make_initializer(final_test_dataset)

X,Y_name = iterator.get_next()
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
Y_softmax = tf.nn.softmax(Y_prediction)

    
initialization()#初始化函数，包括初始化训练集和测试集，得到训练集和测试集的个数，取出id对应的种类
preprocess.main()#creat TFrecord

variables_to_restore = slim.get_variables_to_restore()
###saver
saver = tf.train.Saver(variables_to_restore,max_to_keep = 1)  # 保存所有的变量，最多保存10个
model_file=tf.train.latest_checkpoint('./save/')#尝试加载上次最新的训练结果

with open("./prediction-split-softmax.csv", 'a', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["file_id","Blues","Classical","Country","Easy Listening",'Electronic','Experimental','Folk','Hip-Hop','Instrumental','International','Jazz','Old-Time / Historic','Pop','Rock','Soul-RnB',"Spoken"])
     
    
with tf.Session() as sess: 
    #加载最新的模型
    if model_file !=None:
        saver.restore(sess,model_file)
  
    
    sess.run(finaltest_init_op)###初始化出finaltest集和dropout
    
    for batch_index in range(BATCH_COUNT_FINAL_TEST):
        name,y_sof = sess.run([Y_name,Y_softmax],feed_dict={Is_training: False,KEEP_PROB:1.0}) 
        for i in range(name.shape[0]):
            
            name_str = ""
            for j in range(name.shape[1]):
                name_str+=(chr(name[i][j]))
                
            row = []
            row.append(name_str)
            for j in range(y_sof.shape[1]):
                row.append(y_sof[i][j])
            with open("./prediction-split-softmax.csv", 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(row)
                

vote = {}
count ={}
csv_reader = csv.reader(open('./prediction-split-softmax.csv', encoding='utf-8'))
i = 0
for row in csv_reader:
    i+=1
    if i==1:
        continue
    row[0] = row[0][0:row[0].find('-',-4)]
    if row[0] not in vote:
        vote[row[0]] = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for k in range(16):
            vote[row[0]][k]+=float(row[k+1])
        count[row[0]] = 1
    else:
        for k in range(16):
            vote[row[0]][k]+=float(row[k+1])
        count[row[0]] += 1
with open("./submit.csv", 'a', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["file_id","Blues","Classical","Country","Easy Listening",'Electronic','Experimental','Folk','Hip-Hop','Instrumental','International','Jazz','Old-Time / Historic','Pop','Rock','Soul-RnB',"Spoken"])
for name,value in vote.items():
    with open("./submit.csv", 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        row = [name,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
        for i in range(16):
            row[i+1] = vote[row[0]][i]/count[row[0]]
        writer.writerow(row)
