# -*- coding: utf-8 -*-
"""
Sign Language Recognition Project

author: Hao Zhou

"""

import torch

from signTrans.utils import load_cfg

cfg = load_cfg()


class Batch:
    """
    create a batch
    """

    def __init__(self,
                 batch,
                 src_dim: int = 512,
                 train: bool = True,
                 cuda: bool = False,
                 src_is_text: bool = True,
                 ):


        
        if cfg['model_type'] == 'multiseq': 

            self.src_long, self.src_length_long = batch.src_long
            self.src_short, self.src_length_short = batch.src_short
            self.trg, self.trg_length = batch.trg

            self.ids = None
            
            if src_is_text == False:
                self.ids = batch.id

        else:

            self.src, self.src_length = batch.src
            self.trg, self.trg_length = batch.trg
            
            self.ids = None
            
            
            if src_is_text == False:
                self.ids = batch.id
            #     self.src_mask = (self.src != torch.zeros(src_dim))[..., 0].unsqueeze(1)

            # True if data valid
            # self.src_mask = (self.src != pad_index).unsqueeze(1)
            

        
        if cuda:
            self.to_()
            
    def to_(self):

        if cfg['model_type'] == 'multiseq': 
                self.src_long = self.src_long.cuda()
                self.src_short = self.src_short.cuda()
                self.trg = self.trg.cuda()    
        else:
            self.src = self.src.cuda()
            self.trg = self.trg.cuda()