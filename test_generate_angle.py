# -*- coding: utf-8 -*-
"""
Transfer Learning Project

author: Hao Zhou

"""

import torch
from tqdm import tqdm

import numpy as np
import os, random, time



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

            abs_all_finger = cal_cdf(pred, trg_seq, cfg=opt, sort=False)
            # print(pred.shape, abs_all_finger.shape)
            all_finger.append(abs_all_finger)

    print("total inference time {}".format(time.time() - start_time))
             

    all_finger = np.concatenate(all_finger)
    print(all_finger.shape)


    path = os.path.join(opt['output_dir'], 'angle')
    if not os.path.exists(path):
        os.makedirs(path)
    np.savetxt(path + os.sep + 'angle.txt', all_finger, delimiter=',')


    idx = all_finger.argsort(axis=0)
    all_finger = all_finger[idx, np.arange(idx.shape[1])]

    # cut = int(all_finger.shape[0] * opt['cut_ratio'])
    # print(all_finger.shape[0], cut, opt['cut_ratio'])
    # all_finger = all_finger[:cut, :]


    ##########################################

    # # average over each row
    all_finger = np.mean(all_finger, axis=1)
    print(np.percentile(all_finger, 10), np.percentile(all_finger, 50), np.percentile(all_finger, 90))

    #########################################











if __name__ == "__main__":

    main()