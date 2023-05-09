# -*- coding: utf-8 -*-
"""
Transfer Learning Project

author: Hao Zhou

"""

import torch
from tqdm import tqdm

import numpy as np
import os, random, time



from signTrans.Models import (
                    Transformer, 
                    Transformer2d, 
                    Transformer_test,
                    )

from signTrans.utils import (
                    load_cfg,
                    cal_cdf,
                    )


from signTrans.batch import Batch
from signTrans.iter import make_iter





def load_model(opt, device, load=True):
    
    

    if opt['model_type'] == '1d':
        model = Transformer(
            
            opt['src_vocab_size'],
            opt['trg_vocab_size'],
            
            src_pad_idx=None,
            trg_pad_idx=None, 
            
            trg_emb_prj_weight_sharing=opt['proj_share_weight'],
            emb_src_trg_weight_sharing=opt['embs_share_weight'],
            
            d_k=opt['d_k'],
            d_v=opt['d_v'],
            
            d_model=opt['d_model'],
            d_word_vec=opt['d_model'],
            d_hidden=opt['d_hidden'],
            
            n_layers=opt['n_layers'],
            n_heads=opt['n_heads'],
            dropout=opt['dropout'],
            scale_emb_or_prj=opt['scale_emb_or_prj'],
            
            n_position=opt['n_position'],
            
            src_is_text=opt['src_is_text'],
            ).to(device)
    elif opt['model_type'] == '2d':

        model = Transformer2d(    
            opt['src_vocab_size'],
            opt['trg_vocab_size'],
            
            src_pad_idx=None,
            trg_pad_idx=None, 
            
            trg_emb_prj_weight_sharing=opt['proj_share_weight'],
            emb_src_trg_weight_sharing=opt['embs_share_weight'],
            
            d_k=opt['d_k'],
            d_v=opt['d_v'],
            
            d_model=opt['d_model'],
            d_word_vec=opt['d_model'],
            d_hidden=opt['d_hidden'],
            
            n_layers=opt['n_layers'],
            n_heads=opt['n_heads'],
            dropout=opt['dropout'],
            scale_emb_or_prj=opt['scale_emb_or_prj'],
            
            n_position=opt['n_position'],
            
            src_is_text=opt['src_is_text'],
            ).to(device)

    elif opt['model_type'] == 'Transformer_test':

        model = Transformer_test(    
            opt['src_vocab_size'],
            opt['trg_vocab_size'],
            
            src_pad_idx=None,
            trg_pad_idx=None, 
            
            trg_emb_prj_weight_sharing=opt['proj_share_weight'],
            emb_src_trg_weight_sharing=opt['embs_share_weight'],
            
            d_k=opt['d_k'],
            d_v=opt['d_v'],
            
            d_model=opt['d_model'],
            d_word_vec=opt['d_model'],
            d_hidden=opt['d_hidden'],
            
            n_layers=opt['n_layers'],
            n_heads=opt['n_heads'],
            dropout=opt['dropout'],
            scale_emb_or_prj=opt['scale_emb_or_prj'],
            
            n_position=opt['n_position'],
            
            src_is_text=opt['src_is_text'],
            ).to(device)

    
    if load:    
        model_path = os.path.join(opt['output_dir'], 'model.chkpt')
        checkpoint = torch.load(model_path, map_location=device) 
        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')
        return model
    else:
        print('[Info] Model created.')
        return model
