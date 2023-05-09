# -*- coding: utf-8 -*-
"""
Transfer Learning Project

author: Hao Zhou

"""

import os, random, glob, h5py
import shutil

# This is to move acc/ and angles/ to train and val
# Split them into 2 parts

# def match():

#     folder = os.path.dirname(os.path.realpath(__file__))
#     folder = folder + os.sep

#     accfolder = folder + 'angles' + os.sep
#     accList = glob.glob(accfolder + '*.txt')
#     accList = [v.split(os.sep)[-1].split('.')[0] for v in accList]
#     # print(accList)


#     keyfolder = folder + 'keypoints' + os.sep
#     keyList = glob.glob(keyfolder + '*.h5')



#     for v in keyList:
#         file = v.split(os.sep)[-1].split('.')[0]
#         # print(file)
#         if file in accList:
#             continue
#         else:
#             os.remove(v)

def split(ratio=0.2, augtype='raw'):

    folder = os.path.dirname(os.path.realpath(__file__))
    folder = folder + os.sep

    if augtype != 'raw':
        accfolder = folder + augtype + os.sep


    accList = glob.glob(accfolder + '*.mat')

    accList = [v.split(os.sep)[-1].split('.')[0] for v in accList]

    import random
    random.seed(7)
    random.shuffle(accList)

    valNum = int(len(accList)*ratio)
    trainNum = len(accList) - valNum

    assert valNum+trainNum==len(accList), 'Train and val sets ize must match!'

    return accList[:trainNum], accList[trainNum:]



 

def move(data, augtype, train=True):
    folder = os.path.dirname(os.path.realpath(__file__))
    folder = folder + os.sep

    if augtype != 'raw':
        prefix_acc = folder + augtype + os.sep
        prefix_ang = folder + augtype + os.sep
    
    postfix_acc = '.mat'
    postfix_ang = '.txt'

    target = None

    if train:
        target = folder.replace("fromVideo" + os.sep, "data" + os.sep + "train" + os.sep)
    else:
        target = folder.replace("fromVideo" + os.sep, "data" + os.sep + "val" + os.sep)

    if not os.path.exists(target):
        os.makedirs(target)


    for file in data:
        shutil.copyfile(prefix_acc+file+postfix_acc, target+file+postfix_acc)
        shutil.copyfile(prefix_ang+file+postfix_ang, target+file+postfix_ang)   
        
if __name__ == '__main__':

    # match()

    ratio = 0.1

    for aug in ['dshift1', 'orient1']:
        train, val = split(ratio=ratio, augtype=aug )
        move(train, augtype=aug, train=True)
        move(val  , augtype=aug, train=False)