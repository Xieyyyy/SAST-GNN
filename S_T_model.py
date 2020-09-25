import torch.nn as nn

from GAT_LSTM_Block import GAT_LSTM_Block
from GAT_Transformer_Block import GAT_Transformer_Block


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.linear1 = nn.Linear(args.input_dim, args.upper_T_hidden_dim)
        if args.temporal_module == "LSTM" and args.spatial_module == "GAT":
            self.st_Model = GAT_LSTM_Block(args)
        elif args.temporal_module == "Transformer" and args.spatial_module == "GAT":
            self.tgt_linear = nn.Linear(args.input_dim, args.upper_T_hidden_dim)
            self.st_Model = GAT_Transformer_Block(args)
        self.linear2 = nn.Linear(args.down_T_ftrs, 1)

    def forward(self, x, tgt, adj):
        x = self.linear1(x)
        if self.args.temporal_module == "LSTM" and self.args.spatial_module == "GAT":
            st_block_out = self.st_Model(x, adj)
        elif self.args.temporal_module == "Transformer" and self.args.spatial_module == "GAT":
            tgt = self.tgt_linear(tgt)
            st_block_out = self.st_Model(x, tgt, adj)
        ret = self.linear2(st_block_out)
        return ret
