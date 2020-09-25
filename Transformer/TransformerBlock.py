import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, features, num_of_encoder_layers, num_of_decoder_layers, dropout, dim_feedforward, n_heads,
                 n_units, args):
        super(TransformerBlock, self).__init__()
        self.Transformers = []
        for _ in range(n_units):
            self.Transformers.append(nn.Transformer(features, n_heads, num_of_encoder_layers, num_of_decoder_layers,
                                                    dim_feedforward, dropout))
        for idx, transformer_unit in enumerate(self.Transformers):
            self.add_module('transformer_{}'.format(idx), transformer_unit)
        self.conv_layer = nn.Conv2d(n_heads * args.n_pred, args.n_pred, kernel_size=(1, 1), padding=(0, 0),
                                    stride=(1, 1),
                                    bias=True)

    def forward(self, src, tgt):
        cat_features = []
        for trans in self.Transformers:
            current_ret = trans(src, tgt)
            cat_features.append(current_ret)
        trans_out = torch.cat(cat_features).unsqueeze(0)
        trans_out = self.conv_layer(trans_out)
        return trans_out.squeeze(0)
