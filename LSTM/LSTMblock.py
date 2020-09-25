import torch
import torch.nn as nn


class LSTMblock(nn.Module):
    def __init__(self, features, hidden_size, num_layers, dropout, args, n_heads, batch_first=False):
        super(LSTMblock, self).__init__()
        self.LSTMs = []
        self.hos = []
        self.cos = []
        for _ in range(n_heads):
            self.LSTMs.append(
                nn.LSTM(features, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=batch_first))
            ho = nn.Parameter(torch.zeros(size=(args.upper_T_num_layers, args.node_num, args.upper_T_hidden_dim)))
            co = nn.Parameter(torch.zeros(size=(args.upper_T_num_layers, args.node_num, args.upper_T_hidden_dim)))
            if args.cuda:
                ho = ho.to(args.device)
                co = co.to(args.device)
            nn.init.xavier_uniform_(ho.data, gain=1.414)
            self.hos.append(ho)
            nn.init.xavier_uniform_(co.data, gain=1.414)
            self.cos.append(co)

        for i, lstm in enumerate(self.LSTMs):
            self.add_module('lstm_{}'.format(i), lstm)
        self.conv_layer = nn.Conv2d(n_heads * args.n_his, args.n_his, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1),
                                    bias=True)

    def forward(self, x):
        cat_ftrs = []
        for idx, lstm in enumerate(self.LSTMs):
            cat_ftrs.append(lstm(x, (self.hos[idx], self.cos[idx]))[0])
        lstms_out = torch.cat(cat_ftrs)
        lstms_out = lstms_out.unsqueeze(0)
        lstm_out = self.conv_layer(lstms_out)
        return lstm_out
