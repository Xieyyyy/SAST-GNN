import torch
import torch.nn as nn
import torch.nn.functional as F

from GAT.GATLayer import GATLayer


class GAT(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, n_heads, n_hidden):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = []
        for _ in range(n_heads):
            self.attentions.append(GATLayer(in_features, n_hidden, dropout=dropout, alpha=alpha))
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_attn = GATLayer(n_hidden * n_heads, out_features, dropout=dropout, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        cat_features = []
        for att in self.attentions:
            cat_features.append(att(x, adj))
        x = torch.cat(cat_features, dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.out_attn(x, adj)
        x = F.elu(x)
        return F.log_softmax(x, dim=1)
