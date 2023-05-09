# -*- coding: utf-8 -*-
"""
Transfer Learning Project

author: Hao Zhou

"""

import torch
from tqdm import tqdm

import numpy as np
import os, random

# from signTrans.Constants import (                       
#                             # UNK_TOKEN,
#                             PAD_TOKEN,
#                             SOS_TOKEN,
#                             EOS_TOKEN,
#                             )


from signTrans.Models import Transformer, Transformer2d

from signTrans.utils import (
                    load_cfg,
                    cal_stats,
                    cal_stats2,
                    get_pct,
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
    opt['trg_vocab_size'] = opt['dim_out']  * opt['seqlen']
    

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
    # _, testset, _, _, _ = load_train(opt) 

    test_iter = make_iter(dataset=testset, batch_size=batch_size, train=False)
 

    model = load_model(opt, device)

    # 
    # output = os.path.join(opt['output_dir'], 'translate.txt') 
    
    if torch.cuda.is_available():
        opt['cuda'] = True
    else:
        opt['cuda'] = False

    pct10, pct50, pct90 = [], [], []

    model.eval()
    diff_array = []

    # pred_array = []

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
                # pred_array.append(pred)


            # print(pred.shape)
            # print(pred[0, 0, 700:750])
            # print(pred[100, 0, 700:750])
            # break


            diff, _, _ = cal_stats2(pred, trg_seq, opt)
            diff_array.append(diff)


    diff_array = np.concatenate(diff_array, axis=0)
    print(diff_array.shape)
    a,b,c = get_pct(diff_array, opt)
    a = [ float('%.2f' % elem) for elem in a ]
    b = [ float('%.2f' % elem) for elem in b ]
    c = [ float('%.2f' % elem) for elem in c ]
    print('\n #', a, '\n #', b, '\n #', c) 
    mean = [np.mean(a), np.mean(b), np.mean(c)]
    mean = [ float('%.2f' % elem) for elem in mean ]
    print(' #', mean)
    # print('{.2f} {.2f} {.2f}'.format(np.mean(a), np.mean(b), np.mean(c)))
    # print(np.mean(a), np.mean(b), np.mean(c))




    # print(pred_array[0].shape)





if __name__ == "__main__":
    main()