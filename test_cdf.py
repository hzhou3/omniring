# -*- coding: utf-8 -*-
"""
Transfer Learning Project

author: Hao Zhou

"""

import torch
from tqdm import tqdm

import numpy as np
import os, random, time

# from signTrans.Constants import (                       
#                             # UNK_TOKEN,
#                             PAD_TOKEN,
#                             SOS_TOKEN,
#                             EOS_TOKEN,
#                             )


from signTrans.Models import Transformer, Transformer2d

from signTrans.utils import (
                    load_cfg,
                    cal_cdf,
                    )


from signTrans.batch import Batch
from signTrans.iter import make_iter


from model_loader import load_model






def main():
    
    opt = load_cfg()
    
    # For reproducibility
    if opt['seed'] is not None:
        torch.manual_seed(opt['seed'])
        torch.backends.cudnn.benchmark = False
        # torch.set_deterministic(True)
        np.random.seed(opt['seed'])
        random.seed(opt['seed'])

    if not opt['output_dir']:
        print('No experiment result will be saved.')
        raise
        

    if not os.path.exists(opt['output_dir']):
        print("=== Please specify where model weights are ===")
        return


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opt['src_vocab_size'] = opt['dim_in'] * opt['seqlen']
    opt['trg_vocab_size'] = opt['dim_out'] * opt['seqlen']
    

    if opt['model_type'] == '2d':
        from video2imu_datasets_2d import load_train, load_test
        opt['src_is_text'] = False
        opt['src_vocab_size'] = opt['dim_in']  * opt['seqlen']
    
    if opt['model_type'] == 'Transformer_test':
        from video2imu_datasets_test import load_train, load_test
        opt['src_is_text'] = False
        opt['src_vocab_size'] = opt['dim_in']  * opt['seqlen']

              
    batch_size = opt['batch_size']
    # if opt['batch_size'] != 1:
    #     batch_size = 1


    _, _, testset, _, _ = load_test(opt)   
    # test_iter = make_iter(dataset = testset, batch_size=batch_size)
    test_iter = make_iter(dataset = testset, batch_size=batch_size, train=False)
 
    
    # import gc
    # gc.collect()
    # torch.cuda.empty_cache()
    
    
    model = load_model(opt, device)

    
    if torch.cuda.is_available():
        opt['cuda'] = True
    else:
        opt['cuda'] = False

    all_finger = []


    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for batch in tqdm(test_iter, mininterval=2, desc='  - (Test)', leave=False):

            batch = Batch(
                    batch=batch,
                    cuda=opt['cuda'],
                    )

            if opt['model_type'] == 'multiseq':
                src_seq_long = batch.src_long
                src_seq_short = batch.src_short
                trg_seq = batch.trg
                pred = model(src_seq_long, src_seq_short, trg_seq)
            else:
                src_seq = batch.src
                trg_seq = batch.trg
                pred = model(src_seq, trg_seq)

            abs_all_finger = cal_cdf(pred, trg_seq, opt)
            all_finger.append(abs_all_finger)

    print("total inference time {}".format(time.time() - start_time))
             

    all_finger = np.concatenate(all_finger, axis=0)

    print(all_finger.shape)

    

    # all_finger = all_finger.reshape(all_finger.shape[0] * opt['seqlen'], opt['dim_out'])

    # print(all_finger.shape)



    ###########################################

    ## get distance in mm

    if mm == '_mm':

        all_finger[:,5] = (all_finger[:, 5] + all_finger[:, 6]) / 2
        all_finger[:,6] = 0

        all_finger = np.deg2rad(all_finger)
        all_finger = all_finger[:,0:6]

        print(all_finger.shape)

        all_finger[:, 0] *= 31.57
        all_finger[:, 1] *= 39.78
        all_finger[:, 2] *= 44.63
        all_finger[:, 3] *= 41.37
        all_finger[:, 4] *= 32.74
        all_finger[:, 5] *= 46.22 

        print(all_finger.shape)



    ##########################################

    idx = all_finger.argsort(axis=0)
    all_finger = all_finger[idx, np.arange(idx.shape[1])]

    # cut = int(all_finger.shape[0] * opt['cut_ratio'])
    # print(all_finger.shape[0], cut, opt['cut_ratio'])
    # all_finger = all_finger[:cut, :]


    ##########################################
    

    # # average over each row
    all_finger = np.mean(all_finger, axis=1)
    print(all_finger.shape)

    print(np.percentile(all_finger, 10), np.percentile(all_finger, 50), np.percentile(all_finger, 90))
    
    print(float('%.2f' % np.percentile(all_finger, 50)), ' & ', \
        float('%.2f' % np.percentile(all_finger, 90)),  ' & ', \
        float('%.2f' % np.mean(all_finger)),  ' & ', \
        float('%.2f' % np.std(all_finger)),  ' & ', \
        float('%.2f' % np.median(np.absolute(all_finger - np.median(all_finger)))),  ' & ', \
        float('%.2f' % np.mean(np.absolute(all_finger - np.mean(all_finger)))))

    #########################################

    if opt['output_dir'] == 'results':
        fname = 'cdf.txt'
    else:
        fname = opt['output_dir'] + '.txt'

    path = os.path.join(opt['output_dir'], 'cdf')
    if not os.path.exists(path):
        os.makedirs(path)
    np.savetxt(path + os.sep + fname, all_finger, delimiter=',')









if __name__ == "__main__":

    # mm = '_mm'
    mm = ''

    main()