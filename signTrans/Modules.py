import torch
import torch.nn as nn
import torch.nn.functional as F
# import time

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        
        # time.sleep(20)
        
        if mask is not None:
            # import time
            attn = attn.masked_fill(mask == 0, -1e9)
            # print(torch.cuda.memory_summary(abbreviated=True))
            # time.sleep(20)
            
            
        score = F.softmax(attn, dim=-1)

        attn = self.dropout(score)
        output = torch.matmul(attn, v)

        return output, attn
