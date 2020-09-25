import torch.nn as nn

import sys


# sys.path.append("../")


class singleTransformerBlock(nn.Module):
    def __init__(self, features, n_head, num_enc, num_dec, ff_dim):
        super(singleTransformerBlock, self).__init__()
        self.transformer = nn.Transformer(features, n_head, num_enc, num_dec, ff_dim)

    def forward(self, src, tgt):
        out = self.transformer(src, tgt)
        return out
