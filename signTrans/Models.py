''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
from signTrans.Layers import EncoderLayer, DecoderLayer
from signTrans.utils import load_cfg

cfg = load_cfg()
dim_in = cfg['dim_in']

# def get_pad_mask_txt(seq, pad_idx):
#     return (seq != pad_idx).unsqueeze(-2)


def get_pad_mask_img(seq, pad_dim):
    
    # print(seq.shape)
    # TODO: make it more general
    pad = torch.zeros(pad_dim).cuda()
    out = (seq != pad)[..., 0].unsqueeze(-2)
    del pad
    return out



def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self,
                 d_hid,
                 n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_src_vocab,
            d_word_vec,
            n_layers,
            n_heads,
            d_k,
            d_v,
            d_model,
            d_hidden,
            pad_idx,
            dropout=0.1,
            n_position=200, 
            scale_emb=False,
            src_is_text=True):

        super().__init__()
        
        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=pad_idx)
        
        if src_is_text == False:
            self.src_word_emb = nn.Sequential(
                                  nn.Linear(n_src_vocab,d_word_vec),
                                  nn.ReLU(),
                                  nn.LayerNorm(d_word_vec, eps=1e-6)
                                )

        
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_hidden, n_heads, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, src_mask, return_attns=False):

        enc_slf_attn_list = []

        # -- Forward
        enc_output = self.src_word_emb(src_seq)
        if self.scale_emb:
            enc_output *= self.d_model ** 0.5
        enc_output = self.dropout(self.position_enc(enc_output))
        enc_output = self.layer_norm(enc_output)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
            enc_slf_attn_list += [enc_slf_attn] if return_attns else []

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output,


class Transformer(nn.Module):
    ''' encoder just for  model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab,
            n_trg_vocab,
            src_pad_idx,
            trg_pad_idx,
            d_word_vec=512,
            d_model=512,
            d_hidden=2048,
            n_layers=6,
            n_heads=8,
            d_k=64,
            d_v=64,
            dropout=0.1,
            n_position=200,
            trg_emb_prj_weight_sharing=True,
            emb_src_trg_weight_sharing=True,
            src_is_text=True,
            scale_emb_or_prj='prj'):

        super().__init__()
        
        self.src_is_text = src_is_text

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        # In section 3.4 of paper "Attention Is All You Need", there is such detail:
        # "In our model, we share the same weight matrix between the two
        # embedding layers and the pre-softmax linear transformation...
        # In the embedding layers, we multiply those weights by \sqrt{d_model}".
        #
        # Options here:
        #   'emb': multiply \sqrt{d_model} to embedding output
        #   'prj': multiply (\sqrt{d_model} ^ -1) to linear projection output
        #   'none': no multiplication

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        
        self.d_model = d_model

        self.dropout = dropout

        self.n_trg_vocab = n_trg_vocab

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab,
            n_position=n_position,
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_hidden=d_hidden,
            n_layers=n_layers,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            pad_idx=src_pad_idx,
            dropout=dropout,
            scale_emb=scale_emb,
            src_is_text=src_is_text,
            )

        
        self.last_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        # TODO: move init to a separate function
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'


    def forward(self, 
                src_seq, 
                trg_seq):
    
        if self.src_is_text:
            src_mask = get_pad_mask_txt(src_seq, self.src_pad_idx)
        else:
            src_mask = get_pad_mask_img(src_seq, src_seq.size(-1))
                
        # trg_mask = get_pad_mask_txt(trg_seq, self.trg_pad_idx) & get_subsequent_mask(trg_seq)

        enc_output, *_ = self.encoder(src_seq, src_mask)
        
        # dec_output, *_ = self.decoder(trg_seq, trg_mask, enc_output, src_mask)
        
        # seq_logit = self.trg_word_prj(dec_output)
        
        # if self.scale_prj:
        #     seq_logit *= self.d_model ** -0.5

        # print(enc_output.shape)

        out = self.last_prj(enc_output)
            
        return out
        # return seq_logit.view(-1, seq_logit.size(2))







class Transformer2d(nn.Module):
    ''' encoder just for  model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab,
            n_trg_vocab,
            src_pad_idx,
            trg_pad_idx,
            d_word_vec=512,
            d_model=512,
            d_hidden=2048,
            n_layers=6,
            n_heads=8,
            d_k=64,
            d_v=64,
            dropout=0.1,
            n_position=200,
            trg_emb_prj_weight_sharing=True,
            emb_src_trg_weight_sharing=True,
            src_is_text=True,
            scale_emb_or_prj='prj'):

        super().__init__()
        
        self.src_is_text = src_is_text

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx

        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        
        self.d_model = d_model

        self.dropout = dropout

        self.n_trg_vocab = n_trg_vocab

        self.encoder1 = Encoder(
            n_src_vocab=n_src_vocab,
            n_position=n_position,
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_hidden=d_hidden,
            n_layers=n_layers,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            pad_idx=src_pad_idx,
            dropout=dropout,
            scale_emb=scale_emb,
            src_is_text=src_is_text,
            )

        if cfg['model_type'] == '2d':


            self.encoder2 = nn.Sequential(
                                  nn.Linear(cfg['seqlen']*cfg['dim_in']//(cfg['dim_in']//6), d_model),
                                  nn.LeakyReLU(),
                                  nn.LayerNorm(d_model, eps=1e-6),
                                  # nn.Linear(d_model, d_model),
                                  # nn.ReLU(),
                                  # nn.LayerNorm(d_model, eps=1e-6),
                            )

        self.encoder_corr = nn.Sequential(
                              nn.Linear(cfg['seqlen']*cfg['dim_in']//(cfg['dim_in']//6), d_model),
                              nn.LeakyReLU(),
                              nn.LayerNorm(d_model, eps=1e-6),
                        )

        # self.last_prj = nn.Linear(d_model, n_trg_vocab, bias=False)

        self.last_prj = nn.Sequential(
                              nn.Linear(d_model, d_model*2),
                              nn.LeakyReLU(),
                              nn.LayerNorm(d_model*2, eps=1e-6),
                              nn.Dropout(0.1),
                              
                              nn.Linear(d_model*2, d_model*4),
                              nn.LeakyReLU(),
                              nn.LayerNorm(d_model*4, eps=1e-6),
                              nn.Dropout(0.1),

                              nn.Linear(d_model*4, n_trg_vocab),
                        )
        self.last_prj2 = nn.Linear(d_model*2, n_trg_vocab, bias=False)


        # TODO: move init to a separate function
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

    def forward(self, 
                src_seq, 
                trg_seq):

        if cfg['finger_corr'] == False:
            src_mask = get_pad_mask_img(src_seq, src_seq.size(-1))
            src_seq2 = transform_seq(src_seq)

            enc_output, *_ = self.encoder1(src_seq, src_mask)
            enc_output2 = self.encoder2(src_seq2)

            reshape = nn.Sequential(
                        nn.Conv1d(cfg['dim_in'] // (cfg['dim_in']//6), enc_output.size(1), 1, padding='same'),
                        nn.LayerNorm(cfg['d_model'], eps=1e-6),
                    ).cuda()


            enc_output2 = reshape(enc_output2)
            cat = torch.cat((enc_output, enc_output2), axis=2)

            out = self.last_prj2(cat)        
                
            return out
        else:
            src_mask = get_pad_mask_img(src_seq, src_seq.size(-1))
            src_seq2 = transform_seq(src_seq)

            enc_output_src, *_ = self.encoder1(src_seq, src_mask)

            enc_output_corr = self.encoder_corr(transform_seq(src_seq))

            enc_output_corr = torch.mean(enc_output_corr, dim=1).unsqueeze(1)


            enc_output = torch.sum(torch.stack([cfg['para_src'] * enc_output_src, \
                                            cfg['para_corr'] * enc_output_corr]), axis=0)
            out = self.last_prj(enc_output)

                
            return out  




def transform_seq(data):
    '''
    Rearrange seq tensor to (batch, finger, seq_len)
    '''

    # print(data.shape)
    b, seq, size = data.shape

    # b, dim_in, seq*(size//dim_in)
    src_seq = data.view(b, seq*(size//dim_in), dim_in)
    src_seq = src_seq.permute(0, 2, 1)

    # print(src_seq.shape)

    sample = torch.arange(0, src_seq.size(2), step=seq, dtype=torch.long).cuda()

    src_seq = src_seq[:, :, sample]
    src_seq = torch.split(src_seq, cfg['dim_in'] // 6, dim=1)
    src_seq = torch.cat(src_seq, axis=2)

    # print(src_seq.shape)

    return src_seq






class Transformer_test(nn.Module):
    ''' encoder just for  model with attention mechanism. '''

    def __init__(
            self,
            n_src_vocab,
            n_trg_vocab,
            src_pad_idx,
            trg_pad_idx,
            d_word_vec=512,
            d_model=512,
            d_hidden=2048,
            n_layers=6,
            n_heads=8,
            d_k=64,
            d_v=64,
            dropout=0.1,
            n_position=200,
            trg_emb_prj_weight_sharing=True,
            emb_src_trg_weight_sharing=True,
            src_is_text=True,
            scale_emb_or_prj='prj'):

        super().__init__()
        
        self.src_is_text = src_is_text

        self.src_pad_idx, self.trg_pad_idx = src_pad_idx, trg_pad_idx


        assert scale_emb_or_prj in ['emb', 'prj', 'none']
        
        scale_emb = (scale_emb_or_prj == 'emb') if trg_emb_prj_weight_sharing else False
        
        self.scale_prj = (scale_emb_or_prj == 'prj') if trg_emb_prj_weight_sharing else False
        
        self.d_model = d_model

        self.dropout = dropout

        self.n_trg_vocab = n_trg_vocab

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab,
            n_position=n_position,
            d_word_vec=d_word_vec,
            d_model=d_model,
            d_hidden=d_hidden,
            n_layers=n_layers,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            pad_idx=src_pad_idx,
            dropout=dropout,
            scale_emb=scale_emb,
            src_is_text=src_is_text,
            )
        
        # self.last_prj = nn.Linear(d_model, n_trg_vocab, bias=False)
        self.last_prj = nn.Sequential(
                              nn.Linear(d_model, d_model*2),
                              nn.LeakyReLU(),
                              nn.LayerNorm(d_model*2, eps=1e-6),
                              # nn.Dropout(0.2),
                              
                              nn.Linear(d_model*2, d_model*4),
                              nn.LeakyReLU(),
                              nn.LayerNorm(d_model*4, eps=1e-6),
                              # nn.Dropout(0.2),

                              nn.Linear(d_model*4, n_trg_vocab),
                        )

        # TODO: move init to a separate function
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p) 

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'


    def forward(self, 
                src_seq, 
                trg_seq):
    
        if self.src_is_text:
            src_mask = get_pad_mask_txt(src_seq, self.src_pad_idx)
        else:
            src_mask = get_pad_mask_img(src_seq, src_seq.size(-1))

        # print(src_seq.size(), trg_seq.size())
                
        enc_output, *_ = self.encoder(src_seq, src_mask)

        # print(enc_output.size())

        out = self.last_prj(enc_output)
            
        return out