import torch
import torch.nn as nn
import torch.nn.functional as F

from LSTM.LSTMblock import LSTMblock
from GAT.GAT import GAT


class GAT_LSTM_Block(nn.Module):
    def __init__(self, args):
        super(GAT_LSTM_Block, self).__init__()
        self.LSTM1 = LSTMblock(args.upper_T_ftrs, args.upper_T_hidden_dim, args.upper_T_num_layers,
                               dropout=args.dropout, args=args, n_heads=args.upper_T_head)
        self.residual_block1 = nn.Conv2d(args.n_his, args.n_pred, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1),
                                         bias=True)
        self.GAT = GAT(args.S_in_ftrs, args.S_out_ftrs, args.dropout, args.S_alpha, args.S_n_heads,
                       args.S_n_hidden)
        self.residual_block2 = nn.Linear(args.S_in_ftrs, args.S_out_ftrs)
        self.LSTM2 = nn.LSTM(args.down_T_ftrs, args.down_T_hidden_dim, args.down_T_num_layers,
                             dropout=args.dropout, batch_first=False)
        self.args = args
        self.ho_down = nn.Parameter(
            torch.zeros(size=(args.down_T_num_layers, args.node_num, args.down_T_hidden_dim)))
        nn.init.xavier_uniform_(self.ho_down.data, gain=1.414)
        self.co_down = nn.Parameter(
            torch.zeros(size=(args.down_T_num_layers, args.node_num, args.down_T_hidden_dim)))
        nn.init.xavier_uniform_(self.co_down.data, gain=1.414)

    def forward(self, x, adj):
        residual1 = x
        lstm_ret = torch.stack([self.LSTM1(sample)[0] for sample in x])
        lstm_ret = lstm_ret + residual1
        lstm_ret = self.residual_block1(lstm_ret)
        lstm_ret = F.dropout(lstm_ret, self.args.dropout, training=self.training)

        residual2 = lstm_ret
        GAT_ret_list = []
        for sample in lstm_ret:
            slice_ret_list = torch.stack([self.GAT(slice, adj) for slice in sample])
            GAT_ret_list.append(slice_ret_list)
        GAT_ret = torch.stack(GAT_ret_list)
        residual2 = self.residual_block2(residual2)
        GAT_ret = GAT_ret + residual2
        GAT_ret = F.dropout(GAT_ret, self.args.dropout, training=self.training)

        residual3 = GAT_ret
        lstm_down_ret = torch.stack([self.LSTM2(sample, (self.ho_down, self.co_down))[0] for sample in GAT_ret])
        lstm_down_ret = lstm_down_ret + residual3
        lstm_down_ret = F.dropout(lstm_down_ret, self.args.dropout, training=self.training)

        return lstm_down_ret
