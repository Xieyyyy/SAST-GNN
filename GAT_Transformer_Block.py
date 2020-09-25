import torch
import torch.nn as nn
import torch.nn.functional as F

from Transformer.singleTransformerBlock import singleTransformerBlock
from GAT.GAT import GAT


class GAT_Transformer_Block(nn.Module):
    def __init__(self, args):
        super(GAT_Transformer_Block, self).__init__()
        # self.Transformer1 = TransformerBlock(args.upper_T_ftrs, args.upper_T_enc_layers, args.upper_T_dec_layers,
        #                                      args.dropout, args.upper_T_ff_dim,
        #                                      int(args.upper_T_head / args.upper_T_unit),
        #                                      args.upper_T_unit, args)
        self.Transformer1 = singleTransformerBlock(args.upper_T_ftrs, args.upper_T_head, args.upper_T_enc_layers,
                                                   args.upper_T_dec_layers, args.upper_T_ff_dim)
        self.args = args
        self.linear_tgt = nn.Linear(args.upper_T_hidden_dim, args.S_out_ftrs)
        self.residual_block1 = nn.Conv2d(args.n_his, args.n_pred, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1),
                                         bias=True)
        self.GAT = GAT(args.S_in_ftrs, args.S_out_ftrs, args.dropout, args.S_alpha, args.S_n_heads,
                       args.S_n_hidden)
        self.residual_block2 = nn.Linear(args.S_in_ftrs, args.S_out_ftrs)
        # self.Transformer2 = nn.Transformer(args.down_T_ftrs, args.down_T_heads, args.down_T_enc_layers,
        #                                    args.down_T_dec_layers, args.down_T_ff_dim, args.dropout)
        self.Transformer2 = singleTransformerBlock(args.down_T_ftrs, args.down_T_heads, args.down_T_enc_layers,
                                                   args.down_T_dec_layers, args.down_T_ff_dim)

    def forward(self, src, tgt, adj):
        residual1 = src
        transformer_ret = torch.stack([self.Transformer1(sample[0], sample[1]) for sample in zip(src, tgt)])
        residual1 = self.residual_block1(residual1)
        transformer_ret = transformer_ret + residual1

        residual2 = transformer_ret
        GAT_ret_list = []
        for sample in transformer_ret:
            slice_ret_list = torch.stack([self.GAT(slice, adj) for slice in sample])
            GAT_ret_list.append(slice_ret_list)
        GAT_ret = torch.stack(GAT_ret_list)
        residual2 = self.residual_block2(residual2)
        GAT_ret = GAT_ret + residual2
        GAT_ret = F.dropout(GAT_ret, self.args.dropout, training=self.training)

        residual3 = GAT_ret
        tgt = self.linear_tgt(tgt)
        transformer_down_ret = torch.stack([self.Transformer2(sample[0], sample[1]) for sample in zip(GAT_ret, tgt)])
        transformer_down_ret = transformer_down_ret + residual3
        transformer_down_ret = F.dropout(transformer_down_ret, self.args.dropout, training=self.training)

        return transformer_down_ret
