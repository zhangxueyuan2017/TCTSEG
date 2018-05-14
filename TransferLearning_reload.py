# Author : hellcat
# Time   : 18-4-23

"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
 
import numpy as np
np.set_printoptions(threshold=np.inf)
"""

import numpy as np
import tensorflow as tf
from TransforLearning import creat_image_lists
import os
import cv2
import Deep_Cut

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

segimage_path = 'E:\\TRANSFER\\CELL\\'   #分割后的图片保存路径
originfile_path = 'D:\\abnormal_test'     #源大图图片路径
labelsave_path = 'D:\\abnormal_labeled'       #标记后的图片的保存路径


if not os.path.exists(segimage_path):
    os.mkdir(segimage_path)
if not os.path.exists(labelsave_path):
    os.mkdir(labelsave_path)


ckpt = tf.train.get_checkpoint_state('E:\\TRANSFER\\model\\')
print(ckpt)
saver = tf.train.import_meta_graph(ckpt.model_checkpoint_path + '.meta')
saver.restore(sess, ckpt.model_checkpoint_path)

g = tf.get_default_graph()

    # 遍历目录
for (root0, dirs0, files0) in os.walk(originfile_path):
    for filename0 in files0:
        print('STARTING SEGMENTATING.....')
        filepath0 = os.path.join(root0,filename0)
        print(filepath0)        #单张大图片的路径
        ret = Deep_Cut.testWaterSeg(filepath0,segimage_path)  
for (root1, dirs1, files1) in os.walk(segimage_path):        
    for file1 in files1:
        # 载入图片
        print('STARTING RECOGNIZING.....')
        filepath = os.path.join(root1, file1)
        print(filepath)       #单张分割图片的路径
        image_data = open(os.path.join(root1, file1), 'rb').read()
        bottleneck = sess.run(g.get_tensor_by_name('import/pool_3/_reshape:0'),
                              feed_dict={g.get_tensor_by_name('import/DecodeJpeg/contents:0'): image_data})
        
        class_result = sess.run(g.get_tensor_by_name('final_training_ops/Softmax:0'),
                                feed_dict={g.get_tensor_by_name('BottleneckInputPlaceholder:0'): bottleneck})
        
        images_lists = creat_image_lists(10, 10)
        tf.logging.info(images_lists.keys())
        if np.squeeze(class_result)[1] > np.squeeze(class_result)[0]:            
            (filepath,tempfilename) = os.path.split(filepath)
            (filename,extension) = os.path.splitext(tempfilename)
            print(filename)
            list = filename.split('_')
            fname = list[0]
            sequence = list[1]
            x_site = list[2] 
            y_site =list[3]
            print(originfile_path + '\\' + fname + '\\' + sequence + '.jpg')
            img = cv2.imread(originfile_path + '\\' + fname + '\\' + sequence + '.jpg')
            cv2.rectangle(img, (int(x_site),int(y_site)), (int(x_site) + 300,int(y_site) + 300),(255,0,255),4)
            savefile_path = labelsave_path + '\\' + str(fname) + '\\' 
            print(savefile_path)
            if not os.path.exists(savefile_path):
                os.makedirs(savefile_path)
            cv2.imwrite(savefile_path + str(sequence) + '.jpg', img)
        print(np.squeeze(class_result))
