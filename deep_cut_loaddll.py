# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 15:52:44 2018

@author: admin
"""
import time
import os
import Deep_Cut

def load_deepcut(seg_path, save_dir):
    time.clock()
    seg_path = input("Please input your cell path: ") 
    print("Cell_Path is : ", seg_path)
    save_dir = "D:\\Deep_Cut\\"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for (root,dirs,files) in os.walk(seg_path):
        filename in files:
        filepath = os.path.join(root,filename)
        print(filepath)
        ret = Deep_Cut.testWaterSeg(filepath,save_dir)
     
print("THe total time is :%s" % time.clock())
        