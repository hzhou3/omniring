# -*- coding: utf-8 -*-
"""
Transfer Learning Project

author: Hao Zhou

"""
# from torch.utils.data import DataLoader as loader
# in this data loader, we set a sequence length instead of treating one file as one "sentence"


import os, math, random
import cv2
import glob
import scipy.io
import json
import numpy as np

import torch
from torchtext.legacy import data
from torchtext.legacy.data import Field, RawField, Dataset#, Iterator 


from collections import Counter
from signTrans.Constants import (                       
                            UNK_TOKEN,
                            PAD_TOKEN,
                            SOS_TOKEN,
                            EOS_TOKEN,
                            )

# from signTrans.tokenizer import Tokenizer
from signTrans.batch import Batch
from signTrans.iter import make_iter
# from signTrans.vocabulary import Vocabulary
from signTrans.utils import load_cfg

cfg = load_cfg()
# trg_tokenizer = Tokenizer(lang='en')
n_videos_train = cfg['n_videos_train']
n_videos_val = cfg['n_videos_val']
n_videos_test = cfg['n_videos_test']
upper = cfg['upper']
lower = cfg['lower']

seqlen = cfg['seqlen']
dim_in = cfg['dim_in']
dim_out = cfg['dim_out']


# keep_angle = [  
#                 0,
#                 1,
#                 2,
#                 3,
#                 4,
#                 5,
#                 6
#                 ]

keep_angle = list(range(cfg["dim_out"]))
assert len(keep_angle) == cfg['dim_out'], 'Output dimension error when loading data'


# first 5 are acc/ second 5 are direction
# keep_dim_pip_train = [3, 6, 10, 14, 18, 22, 23, 24, 25, 26] # PIP

#####################################


keep_dim_mcp_train = []
keep_dim_mcp_test = []


train_finger = {1: [2,22],
                2: [5,23],
                3: [9,24],
                4: [13,25],
                5: [17,26]}

test_finger = { 1: [12,13,14, 15,16,17],
                2: [21,22,23, 24,25,26],
                3: [30,31,32, 33,34,35],
                4: [39,40,41, 42,43,44],
                5: [48,49,50, 51,52,53]}


for finger in cfg['finger_list']:
    keep_dim_mcp_train = keep_dim_mcp_train + train_finger[finger]
    keep_dim_mcp_test = keep_dim_mcp_test + test_finger[finger]


# print(keep_dim_mcp_train, keep_dim_mcp_test)

assert len(keep_dim_mcp_test) == cfg['dim_in'], 'Input error when loading data'
assert len(keep_dim_mcp_train)*3 == cfg['dim_in'], 'Input error when loading data'


# keep_dim_mcp_train = [2, 5, 9, 13, 17, 22, 23, 24, 25, 26] # MCP
# keep_dim_mcp_test = [
#                     # 3,4,5, 6,7,8, 
#                     12,13,14, 15,16,17,
#                     21,22,23, 24,25,26,
#                     30,31,32, 33,34,35,
#                     39,40,41, 42,43,44,
#                     48,49,50, 51,52,53
#                     ] # wcf + dire


#####################################

# keep_dim_mcp_train = [2, 5, 9, 13, 17, 22, 23, 24, 25, 26] # MCP
# keep_dim_mcp_test = [
#                     # 0,1,2, 6,7,8, 
#                     9,10,11, 15,16,17,
#                     18,19,20, 24,25,26,
#                     27,28,29, 33,34,35,
#                     36,37,38, 42,43,44,
#                     35,46,47, 51,52,53
#                     ] # acc + dire

#####################################

# keep_dim_mcp_train = [2,5,9,13,17] # MCP
# keep_dim_mcp_test = [
#                     # 0,1,2, 
#                     9,10,11,
#                     18,19,20,
#                     27,28,29,
#                     36,37,38,
#                     35,46,47,
#                     ] # acc 

#####################################


def checkNaN(data):
    index = []
    NaN = False
    length = len([i for i in keep_angle if i < 5 ])
    for i in range(data.shape[0]):
        isNaN = [math.isnan(data[i][k]) for k in range(length)]
        # print(isNaN)
        if True in isNaN:
            NaN = True
            continue
        index.append(i)

    return index, NaN





def readTestDataPair(name='test'):

    test_user = cfg['test_user']
    test_ori = cfg['test_ori']
    
    if name != 'test':
        test_user = cfg['train_user']
        test_ori  = cfg['train_ori']
    else:
        test_user = cfg['test_user']
        test_ori  = cfg['test_ori']

    name = 'test'




    folder = os.path.dirname(os.path.realpath(__file__))
    folder = folder + os.sep + 'data' + os.sep
    folder = folder + name + os.sep

    # print(folder)

    accList = glob.glob(folder + '*.mat')
    angList = glob.glob(folder + '*.txt')

    import random
    random.seed(7)
    random.shuffle(accList)

    raw = []


    for i, k in enumerate(accList):
        acc_file = k.split(os.sep)[-1].split('.')[0]
        ang_file = acc_file + '.txt'

        user = acc_file.split('_')[0]
        ori = acc_file.split('_')[1]

        # print(user, ori)

        # if int(user) == 17:
        #     continue

        if int(user) not in test_user:
            continue
        if int(ori) not in test_ori:
            continue

        acc = scipy.io.loadmat(folder + acc_file + '.mat')
        # print(acc)
        acc = np.array(acc['wcf'])
        acc = acc.transpose(1,0)[:,:-1]
        

        acc = acc[:, keep_dim_mcp_test]
        # print(acc.shape)


        angle = np.loadtxt(folder + ang_file, delimiter=',')
        smallLen = acc.shape[0] if acc.shape[0] < angle.shape[0] else angle.shape[0]
        angle = angle[:smallLen, 1:]

        # print(acc.shape, angle.shape)
        angle = angle[:, keep_angle]
        acc = acc[:smallLen, :]

        # print(acc.shape, angle.shape)
        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(2,1)
        # axs[0].imshow(np.tile(acc, 50).transpose(1,0))
        # axs[1].imshow(np.tile(angle, 200).transpose(1,0))
        # plt.show()

        index, NaN = checkNaN(angle)

        assert acc.shape[0] == angle.shape[0], "Acc and Angle dim must match."
        
        acc = acc[index,:]
        angle = angle[index,:]

        ################################

        # ### have to do this when draw CDF

        # randint = random.randint(lower, upper)

        # # maxLen = randint*seqlen
        # maxLen = seqlen

        # if smallLen <= maxLen:
        #     stats = dict()
        #     stats['id'] = acc_file
        #     acc = acc[:seqlen*(acc.shape[0]//seqlen), :].reshape(-1, seqlen * dim_in)
        #     angle = angle[:seqlen*(angle.shape[0]//seqlen), :].reshape(-1, seqlen * dim_out)
        #     # print(acc.shape, angle.shape)
        #     if acc.shape[0] == 0:
        #         continue
        #     stats['acc'] = acc
        #     stats['angle'] = angle
        #     raw.append(stats)

        # else:

        #     # print(smallLen, smallLen // maxLen)

        #     for clip in range((smallLen // maxLen) -1):
        #         # print(clip)

        #         acc_clip = acc[clip*maxLen:(clip+1)*maxLen, :]
        #         angle_clip = angle[clip*maxLen:(clip+1)*maxLen, :]
                
        #         # print(acc_clip.shape, angle_clip.shape)

        #         if acc_clip.shape[0] == 0:
        #             print(acc_clip.shape)
        #             continue

        #         stats = dict()
        #         stats['id'] = acc_file+'_'+str(clip)
                

        #         acc_clip = acc_clip[:seqlen*(acc_clip.shape[0]//seqlen), :]
        #         acc_clip = acc_clip.reshape(-1, seqlen * dim_in)
        #         angle_clip = angle_clip[:seqlen*(angle_clip.shape[0]//seqlen), :]
        #         # angle_clip = (angle_clip - np.min(angle_clip, axis=0)) / (np.max(angle_clip, axis=0) - np.min(angle_clip, axis=0))
        #         angle_clip = angle_clip.reshape(-1, seqlen * dim_out)

        #         if acc_clip.shape[0] == 0:
        #             continue
        #         stats['acc'] = acc_clip
                
        #         stats['angle'] = angle_clip
        #         raw.append(stats)
                    

        #########################################
        stats = dict()
        stats['id'] = acc_file
        acc = acc[:seqlen*(acc.shape[0]//seqlen), :].reshape(-1, seqlen * dim_in)
        angle = angle[:seqlen*(angle.shape[0]//seqlen), :].reshape(-1, seqlen * dim_out)
        # print(acc.shape, angle.shape)
        stats['acc'] = acc
        stats['angle'] = angle
        raw.append(stats)

        #########################################

    print('Loaded {0} sample pair from {1}_set'.format(len(raw), name))

    return raw






def readDataPair(name='train'):
    folder = os.path.dirname(os.path.realpath(__file__))
    folder = folder + os.sep + 'data' + os.sep
    folder = folder + name + os.sep

    # print(folder)

    accList = glob.glob(folder + '*.mat')
    angList = glob.glob(folder + '*.txt')

    import random
    random.seed(7)
    random.shuffle(accList)

    # if name == 'train':
    #     accList = random.sample(accList, 18*8)
    #     print(len(accList))

    raw = []


    for i, k in enumerate(accList):
        acc_file = k.split(os.sep)[-1].split('.')[0]

        # if acc_file.split('_')[-2] == 'r':
        #     continue


        ang_file = acc_file + '.txt'
        

        acc = scipy.io.loadmat(folder + acc_file + '.mat')
        # print(acc)
        acc = np.array(acc['video_acc'])
        acc = acc[:, keep_dim_mcp_train, :]

        swap = list(range(len(keep_dim_mcp_train)))

        assert len(swap) % 2 == 0, "must be odd length!"

        swap_new = []
        half = len(swap) // 2
        for i in range(half):
            swap_new.append(i)
            swap_new.append(i + half)

        # print(swap, swap_new)

        acc[:, swap_new, :] = acc[:, swap, :]
        acc = acc.reshape(acc.shape[0], -1)


        # angle = scipy.io.loadmat(folder + ang_file)
        # angle = np.array(acc['video_ang'])

        angle = np.loadtxt(folder + ang_file, delimiter=',')
        angle = angle[0:acc.shape[0], :]
        angle = angle[:, keep_angle]



        # import matplotlib.pyplot as plt
        # fig, axs = plt.subplots(1,2)
        # axs[0].imshow(acc)
        # axs[1].imshow(np.tile(angle, 10))
        # plt.show()



        index, NaN = checkNaN(angle)

        assert acc.shape[0] == angle.shape[0], "Acc and Angle dim must match."
        
        acc = acc[index,:]
        angle = angle[index,:]



        ##############################################
        
        # randint = random.randint(lower, upper)
        # # maxLen = randint*seqlen     
        # maxLen = seqlen   

        # smallLen = acc.shape[0] if acc.shape[0] < angle.shape[0] else angle.shape[0]

        # if smallLen <= maxLen:
        #     if acc.shape[0]//seqlen == 0:
        #         continue
        #     stats = dict()
        #     stats['id'] = acc_file
        #     acc = acc[:seqlen*(acc.shape[0]//seqlen), :].reshape(-1, seqlen * dim_in)
        #     angle = angle[:seqlen*(angle.shape[0]//seqlen), :].reshape(-1, seqlen * dim_out)
        #     # print(acc.shape, angle.shape)
        #     stats['acc'] = acc
        #     stats['angle'] = angle
        #     raw.append(stats)

        # else:

        #     # print(smallLen, smallLen // maxLen)

        #     for clip in range((smallLen // maxLen) -1):
        #         # print(clip)

        #         acc_clip = acc[clip*maxLen:(clip+1)*maxLen, :]
        #         angle_clip = angle[clip*maxLen:(clip+1)*maxLen, :]
                
        #         # print(acc_clip.shape, angle_clip.shape)

        #         if acc_clip.shape[0] == 0:
        #             # print(acc_clip.shape)
        #             continue

        #         stats = dict()
        #         stats['id'] = acc_file+'_'+str(clip)
                

        #         acc_clip = acc_clip[:seqlen*(acc_clip.shape[0]//seqlen), :]
        #         acc_clip = acc_clip.reshape(-1, seqlen * dim_in)
        #         angle_clip = angle_clip[:seqlen*(angle_clip.shape[0]//seqlen), :]
        #         # angle_clip = (angle_clip - np.min(angle_clip, axis=0)) / (np.max(angle_clip, axis=0) - np.min(angle_clip, axis=0))
        #         angle_clip = angle_clip.reshape(-1, seqlen * dim_out)


        #         stats['acc'] = acc_clip
        #         stats['angle'] = angle_clip
        #         raw.append(stats)
        ###############################################



        stats = dict()
        stats['id'] = acc_file
        acc = acc[:seqlen*(acc.shape[0]//seqlen), :].reshape(-1, seqlen * dim_in)
        angle = angle[:seqlen*(angle.shape[0]//seqlen), :].reshape(-1, seqlen * dim_out)
        # print(acc.shape, angle.shape)

        if acc.shape[0] == 0:
            continue
        
        stats['acc'] = acc
        stats['angle'] = angle
        raw.append(stats)

        ###############################################

    print('Loaded {0} sample pair from {1}_set'.format(len(raw), name))

    return raw



class video2imu(data.Dataset):
    
    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))
    
    def __init__(self,
                 type = 'val',
                 fields = None,
                 **kwargs):
        """
        dataset based on torchtext
        """
        
        fields = [
            ('id', fields[0]), 
            ('src', fields[1]),
            ('trg', fields[2])
            ]
  
        # read raw data
        if type == 'train' or type == 'val':
            # type = 'test'
            raw = readDataPair(type)
        else:
            raw = readTestDataPair(type)
        
        examples = []
        for s in raw:
            uid = s['id']
            src = torch.tensor(s['acc'], dtype=torch.float32)
            trg = torch.tensor(s['angle'], dtype=torch.float32)
            
            examples.append(
                data.Example.fromlist(
                        [
                            uid,
                            src,
                            trg,
                        ],
                        fields,
                    )
                )
        
        
        super().__init__(examples, fields, **kwargs)



def load_train(cfg: dict):

    min_fre = cfg['min_fre']
    
    src_pad_feature_size = cfg['dim_in']*cfg['seqlen']  # 5 fingers acc/dire for each axis
    trg_pad_feature_size = cfg['dim_out']*cfg['seqlen']  # 5 fingers angle + 2 wrist angles
    
    src_max_len = cfg['src_max_len']
    trg_max_len = cfg['trg_max_len']
 
    
 
    def tokenize_features(features):
        ft_list = torch.split(features, 1, dim=0)
        out = [ft.squeeze() for ft in ft_list]
        return  out

    def stack_features(features, something):
        # print(len(features))
        return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)
    
    
    id_field = RawField()
    
    src_field = Field(
        use_vocab=False,
        init_token=None,
        dtype=torch.float32,
        # preprocessing=tokenize_features,
        # tokenize=lambda features: features,
        batch_first=True,
        include_lengths=True,
        postprocessing=stack_features,
        pad_token=torch.zeros((src_pad_feature_size,)),
        )


    trg_field = Field(
        use_vocab=False,
        init_token=None,
        dtype=torch.float32,
        batch_first=True,
        include_lengths=True,
        postprocessing=stack_features,
        pad_token=torch.zeros((trg_pad_feature_size,)),
        )   


    train = video2imu(
        type='train',
        fields=(id_field, src_field, trg_field),
        filter_pred=lambda x: len(vars(x)["src"]) <= src_max_len
        and len(vars(x)["trg"]) <= trg_max_len
        )
    
    val = video2imu(
        type='val',
        fields=(id_field, src_field, trg_field),
        filter_pred=lambda x: len(vars(x)["src"]) <= src_max_len
        and len(vars(x)["trg"]) <= trg_max_len
        )

    return train, val, None, None, None 
    
    



def load_test(cfg: dict):

    min_fre = cfg['min_fre']
    
    src_pad_feature_size = cfg['dim_in']*cfg['seqlen']  # 5 fingers acc/dire for each axis
    trg_pad_feature_size = cfg['dim_out']*cfg['seqlen']  # 5 fingers angle + 2 wrist angles
    
    
    src_max_len = cfg['src_max_len']
    trg_max_len = cfg['trg_max_len']
 
    
 
    def tokenize_features(features):
        ft_list = torch.split(features, 1, dim=0)
        out = [ft.squeeze() for ft in ft_list]
        return  out

    def stack_features(features, something):
        # print(len(features))
        return torch.stack([torch.stack(ft, dim=0) for ft in features], dim=0)
    
    
    id_field = RawField()
    
    src_field = Field(
        use_vocab=False,
        init_token=None,
        dtype=torch.float32,
        # preprocessing=tokenize_features,
        # tokenize=lambda features: features,
        batch_first=True,
        include_lengths=True,
        postprocessing=stack_features,
        pad_token=torch.zeros((src_pad_feature_size,)),
        )


    trg_field = Field(
        use_vocab=False,
        init_token=None,
        dtype=torch.float32,
        batch_first=True,
        include_lengths=True,
        postprocessing=stack_features,
        pad_token=torch.zeros((trg_pad_feature_size,)),
        )   
  
    test = video2imu(
        type='test',
        fields=(id_field, src_field, trg_field),
        filter_pred=lambda x: len(vars(x)["src"]) <= src_max_len
        and len(vars(x)["trg"]) <= trg_max_len
        )

    return None, None, test, None, None 
        
if __name__ == '__main__':

    
    # train, val, _, _, _ = load_train(cfg)

    _, _, train, _, _ = load_test(cfg)

    # print(train.src)
    
   
    train_iter = make_iter(dataset = train,
                           batch_size = cfg['batch_size'],)

    # print(type(train_iter))

    # for i, v in enumerate(train_iter):
    #     print(i)
    #     print(v.src)
 
    
    for batch in iter(train_iter):
        
        batch = Batch(
                batch=batch,
                )
        
        print(batch.src_length)
    
    
