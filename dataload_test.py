

import os 
import glob
import scipy.io
import numpy as np
import math
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

    print(keep_dim_mcp_test)

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

        # if int(user) not in test_user:
        #     continue
        # if int(ori) not in test_ori:
        #     continue

        acc = scipy.io.loadmat(folder + acc_file + '.mat')
        # print(acc)
        acc = np.array(acc['wcf'])
        acc = acc.transpose(1,0)[:,:-1]
        

        acc = acc[:, keep_dim_mcp_test]
        # print(acc.shape)


        angle = np.loadtxt(folder + ang_file, delimiter=',')
        smallLen = acc.shape[0] if acc.shape[0] < angle.shape[0] else angle.shape[0]
        angle = angle[:smallLen, 1:]

        try:

            # print(acc.shape, angle.shape)
            angle = angle[:, keep_angle]
            acc = acc[:smallLen, :]

            # print(acc.shape, angle.shape)
        except:
            continue


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
                

        #         acc_clip = acc_clip[:seqlen*(acc_clip.shape[0]//seqlen), :].reshape(-1, seqlen * dim_in)
        #         angle_clip = angle_clip[:seqlen*(angle_clip.shape[0]//seqlen), :].reshape(-1, seqlen * dim_out)
        #         # print(acc_clip.shape, angle_clip.shape)
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
        print(acc.shape, angle.shape)
        stats['acc'] = acc
        stats['angle'] = angle
        raw.append(stats)

        #########################################

    print('Loaded {0} sample pair from {1}_set'.format(len(raw), name))

    return raw








if __name__ == "__main__":
    readTestDataPair(name='test')



