# -*- coding: utf-8 -*-
"""
Transfer Learning Project

author: Hao Zhou

"""


import math
import time
from tqdm import tqdm
import numpy as np
import random
import os

import torch

import torch.optim as optim

from signTrans.Models import Transformer, Transformer2d
from signTrans.Optim import ScheduledOptim


from signTrans.utils import (
                    load_cfg,
                    cal_stats,
                    cal_stats2,
                    get_pct,
                    cal_loss,
                    save_cfg,
                    )

from signTrans.batch import Batch
from signTrans.iter import make_iter



def print_performances(train_loss, val_loss):
    print('Training loss {t:.3f} & Validation loss {v:.3f}'.format(t=train_loss, v=val_loss))



def eval_epoch(epoch, model, validation_data, device, opt, update):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    total_loss = 0
    pct10, pct50, pct90 = [], [], []

    i = 0
    diff_array = []
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2, desc='  - (Validate)', leave=False):
            
            batch = Batch(
                    batch=batch,
                    cuda=opt['cuda'],
                    )


            src_seq = batch.src
            trg_seq = batch.trg

            # forward
            pred = model(src_seq, trg_seq)

            loss = cal_loss(pred, trg_seq, opt, epoch=epoch)

            total_loss += loss.item()

            diff, _, _ = cal_stats2(pred, trg_seq, opt)

            diff_array.append(diff)


    diff_array = np.concatenate(diff_array, axis=0)
    a,b,c = get_pct(diff_array, opt)

    # print(a,b,c)
    
    return total_loss, a, b, c



def train_epoch(epoch, model, training_data, optimizer, opt, device, update, smoothing):
    ''' Epoch operation in training phase'''

    model.train()

    total_loss = 0

    pct10, pct50, pct90 = [], [], []

    diff_array = []

    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)', leave=False):
        
        batch = Batch(
                batch=batch,
                cuda=opt['cuda'],
                )
        

        src_seq = batch.src
        trg_seq = batch.trg
        
        # forward
        optimizer.zero_grad()
        pred = model(src_seq, trg_seq)

        # print(src_seq.shape, trg_seq.shape, pred.shape)

        loss = cal_loss(pred, trg_seq, opt, epoch=epoch)
        loss.backward()

        diff, _, _ = cal_stats2(pred, trg_seq, opt)

        diff_array.append(diff)

        optimizer.step_and_update_lr()

        total_loss += loss.detach().item()


    diff_array = np.concatenate(diff_array, axis=0)
    a,b,c = get_pct(diff_array, opt)

    # print(a,b,c)


    # pct10 = np.concatenate(pct10)
    # pct50 = np.concatenate(pct50)
    # pct90 = np.concatenate(pct90)

    # a = np.median(pct10, axis=0)
    # b = np.median(pct50, axis=0)
    # c = np.median(pct90, axis=0)

    return total_loss, a, b, c


def train(model, training_data, validation_data, optimizer, device, opt=None):
    ''' Start training '''
    
    # Use tensorboard to plot curves,
    if opt['use_tb']:
        print("[Info] Use Tensorboard")
        from torch.utils.tensorboard import SummaryWriter
        tb_writer = SummaryWriter(log_dir=os.path.join(opt['output_dir'], 'tensorboard'))

    log_train_file = os.path.join(opt['output_dir'], 'train.log')
    log_valid_file = os.path.join(opt['output_dir'], 'valid.log')

    print('[Info] Training performance will be written to file: {} and {}'.format(
        log_train_file, log_valid_file))

    with open(log_train_file, 'w') as log_tf, open(log_valid_file, 'w') as log_vf:
        log_tf.write('epoch,loss,pct10,pct50,pct90\n')
        log_vf.write('epoch,loss,pct10,pct50,pct90\n')
    
    save_cfg(opt)
    
    valid_losses = []

    time_train = [] # in training time of each epoch in seconds


    for epoch_i in range(opt['epoch']):
        bestUpdate = False
        print('[ Epoch', epoch_i, ']')

        start_time = time.time()
   
        train_loss, train_pct10, train_pct50, train_pct90= train_epoch(
                                                                epoch_i,
                                                                model, 
                                                                training_data, 
                                                                optimizer, 
                                                                opt, 
                                                                device,
                                                                bestUpdate,
                                                                smoothing=opt['label_smoothing'],
                                                                )
        end_time = time.time()

        time_train.append(end_time - start_time)

        # print(np.sum(time_train))
        
        # # Current learning rate
        lr = optimizer._optimizer.param_groups[0]['lr']


        valid_loss, val_pct10, val_pct50, val_pct90 = eval_epoch(
                                                        epoch_i,
                                                        model,
                                                        validation_data,
                                                        device,
                                                        opt,
                                                        bestUpdate)

        
        print_performances(train_loss, valid_loss)


        valid_losses += [valid_loss]
        checkpoint = {'epoch': epoch_i, 'settings': None, 'model': model.state_dict()}

        if opt['save_mode'] == 'all':
            model_name = 'model_val_loss_{loss:3.3f}.chkpt'.format(loss=valid_loss)
            torch.save(checkpoint, model_name)
            
        elif opt['save_mode'] == 'best':
            model_name = 'model.chkpt'
            if valid_loss <= min(valid_losses) and min(valid_losses) <= 90:
                bestUpdate = True
                torch.save(checkpoint, os.path.join(opt['output_dir'], model_name))
                print('    - [Info] The checkpoint file has been updated.')

        with open(log_train_file, 'a') as log_tf, open(log_valid_file, 'a') as log_vf:
            log_tf.write('{epoch},{loss: 8.5f},{pct10},{pct50},{pct90}\n'.format(
                epoch=epoch_i, loss=train_loss,
                pct10=train_pct10, pct50=train_pct50,
                pct90=train_pct90))
            log_vf.write('{epoch},{loss: 8.5f},{pct10},{pct50},{pct90}\n'.format(
                epoch=epoch_i, loss=valid_loss,
                pct10=val_pct10, pct50=val_pct50,
                pct90=val_pct90))

        if opt['use_tb']:
            tb_writer.add_scalars('loss', {'train': train_loss, 'val': valid_loss}, epoch_i)
            tb_writer.add_scalar('learning_rate', lr, epoch_i)



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
        os.makedirs(opt['output_dir'])

    if opt['batch_size'] < 2048 and opt['warmup'] <= 4000:
        print('[Warning] The warmup steps may be not enough.\n'\
              '(sz_b, warmup) = (2048, 4000) is the official setting.\n'\
              'Using smaller batch w/o longer warmup may cause '\
              'the warmup stage ends with only little data trained.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)

        

    opt['src_vocab_size'] = opt['dim_in'] * opt['seqlen']
    opt['trg_vocab_size'] = opt['dim_out'] * opt['seqlen']
    
    if opt['dataset'] == 'video2imu':
        from video2imu_datasets_2d import load_train, load_test
        opt['src_is_text'] = False
        opt['src_vocab_size'] = opt['dim_in'] * opt['seqlen']

    trainset, valset, _, _, _ = load_train(opt)

    _, _, testset, _, _ = load_test(opt)
    # trainset = testset
    valset = testset
    
    batch_size = opt['batch_size']
        
    train_iter = make_iter(dataset = trainset, batch_size=batch_size)
    val_iter = make_iter(dataset = valset, batch_size=batch_size)
    
    from model_loader import load_model
    transformer = load_model(opt, device, load=False)

   
    
    # TODO: build a optimizer
    optimizer = ScheduledOptim(
        optim.Adam(transformer.parameters(), betas=(0.9, 0.99), eps=1e-09),
        opt['lr_mul'], opt['d_model'], opt['warmup'])

    train(transformer, train_iter, val_iter, optimizer, device, opt)


if __name__ == '__main__':
    main()




























