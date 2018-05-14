
# coding: utf-8

# In[55]:


import cv2
import os

sfile_path = 'D:\segmention_abnormal_300'
bfile_path = 'D:\\abnormal_selected'
save_path = 'D:\\abnormal_labeled'
for (root,dirs,files) in os.walk(file_path):
    for filename in files:    
        filepath = os.path.join(root,filename)
        (filepath,tempfilename) = os.path.split(filepath)
        (filename,extension) = os.path.splitext(tempfilename)
        print(filename)
        list = filename.split('_')
        fname = list[0]
        sequence = list[1]
        x_site = list[2] 
        y_site =list[3]
        print(bfile_path + '\\' + fname + '\\' + sequence + '.jpg')
        img = cv2.imread(bfile_path + '\\' + fname + '\\' + sequence + '.jpg')
        cv2.rectangle(img, (int(x_site),int(y_site)), (int(x_site) + 300,int(y_site) + 300),(255,0,255),4)
      #  cv2.imwrite(save_path + '\\' + str(fname) + '\\' + str(sequence) + '.jpg', img)
        savefile_path = 'D:\\abnormal_labeled' + '\\' + str(fname) + '\\' 
        print(savefile_path)
        if not os.path.exists(savefile_path):
            os.makedirs(savefile_path)
      #  print(savefile_path + str(sequence) + '.jpg')
        cv2.imwrite(savefile_path + str(sequence) + '.jpg', img)
         #   file = open(savefile_path, 'wr')
          #  file.write(savefile_path)
           # file.close()
       


