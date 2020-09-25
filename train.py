from utils.utils import *
from utils.data_utils import *
from utils.math_utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import datetime
import os

from S_T_model import Model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--temporal_module", type=str, default="Transformer", required=False)
parser.add_argument("--spatial_module", type=str, default="GAT", required=False)
parser.add_argument("--dataset", type=str, default="PEMS", required=False)
parser.add_argument("--use_cuda", type=bool, default=True, required=False)
parser.add_argument("--dropout", type=float, default=0.5, help="Dropout prob", required=False)
parser.add_argument("--n_train", type=int, default=31, required=False)
parser.add_argument("--n_val", type=int, default=9, required=False)
parser.add_argument("--n_test", type=int, default=4, required=False)
parser.add_argument("--n_his", type=int, default=12, required=False)
parser.add_argument("--n_pred", type=int, default=12, required=False)
parser.add_argument("--batch_size_train", type=int, default=1, required=False)
parser.add_argument("--batch_size_val", type=int, default=1, required=False)
# -----------------------------------------------
parser.add_argument('--input_dim', type=int, default=1, help='Dim of input data', required=False)
# -----------------------------------------
parser.add_argument("--upper_T_hidden_dim", type=int, default=512, help="Dim of hidden layer in LSTM1",
                    required=False)
parser.add_argument("--upper_T_num_layers", type=int, default=3, help="Layer num of upper LSTM", required=False)
parser.add_argument("--upper_T_ftrs", type=int, default=512, help="Dim of upper LSTM in ST block", required=False)
parser.add_argument("--upper_T_head", type=int, default=8, help="Num of head in LSTM", required=False)
parser.add_argument("--upper_T_enc_layers", type=int, default=3, help="Enc layer num of upper Transformer",
                    required=False)
parser.add_argument("--upper_T_dec_layers", type=int, default=3, help="Dec layer num of upper Transformer",
                    required=False)
parser.add_argument("--upper_T_ff_dim", type=int, default=512, help="The dim in FFD of transformer", required=False)
parser.add_argument("--upper_T_unit", type=int, default=1, help="Transformer unit num", required=False)
# --------------------------
parser.add_argument("--S_in_ftrs", type=int, default=512, help="Input dim of GAT in ST block", required=False)
parser.add_argument("--S_out_ftrs", type=int, default=64, help="Output dim of GAT in ST block", required=False)
parser.add_argument("--S_alpha", type=float, default=0.2, help="Neg slope of Leakyrelu in GAT in ST block",
                    required=False)
parser.add_argument("--S_n_heads", type=int, default=8, help="Num of attention head in GAT in ST block",
                    required=False)
parser.add_argument("--S_n_hidden", type=int, default=64, help="Dim of hidden layer in GAT", required=False)
# --------------------------
parser.add_argument("--down_T_ftrs", type=int, default=64, help="Dim of down LSTM in ST block", required=False)
parser.add_argument("--down_T_hidden_dim", type=int, default=64, help="Dim of hidden layer in LSTM2",
                    required=False)
parser.add_argument("--down_T_num_layers", type=int, default=3, help="Layer num of down LSTM", required=False)
parser.add_argument("--down_T_heads", type=int, default=8, help="Down transformer heads", required=False)
parser.add_argument("--down_T_enc_layers", type=int, default=2, help="Down transformer enc layers", required=False)
parser.add_argument("--down_T_dec_layers", type=int, default=2, help="Down transformer dec layers", required=False)
parser.add_argument("--down_T_ff_dim", type=int, default=128, help="Down transformer ffd dim", required=False)

args = parser.parse_args()

args.cuda = args.use_cuda and torch.cuda.is_available()
args.device = torch.device("cuda:2")
args.record_pth = "../record/record_sin_tr_1212pems.txt"
args.model_pth = "../model/model_sin_tr_1212pems.pkl"
best_perf = [300, 300, 300]

if args.dataset == 'PEMS':
    args.node_num = 228
    adj = weight_matrix("../data/W_228.csv")
    adj = torch.Tensor(adj)

    dataset = data_gen("../data/V_228.csv", (args.n_train, args.n_val, args.n_test), args.node_num, is_csv=True,
                       n_frame=args.n_his + args.n_pred)
    model = Model(args)
    if args.cuda:
        adj = adj.to(args.device)
        model = model.to(args.device)

if args.dataset == "Bay":
    args.node_num = 325
    adj = load_bay_graph("../data/adj_mx_bay.pkl")[2]
    adj = torch.Tensor(adj)
    args.n_train, args.n_val, args.n_test = 140, 5, 35
    dataset = data_gen("../data/pems-bay.h5", (args.n_train, args.n_val, args.n_test), args.node_num, is_csv=False,
                       n_frame=args.n_his + args.n_pred)
    model = Model(args)
    if args.cuda:
        adj = adj.to(args.device)
        model = model.to(args.device)

start_time = datetime.datetime.now()

optimizer = optim.Adam(model.parameters(), lr=0.002)
criterionl1 = nn.L1Loss(reduction='mean')
criterionl2 = nn.MSELoss(reduction='mean')

with open(args.record_pth, "a") as f:
    f.write(str(args))


def train(X, tgt, y, adj):
    model.train()
    optimizer.zero_grad()
    out = model(X, tgt, adj)
    loss = criterionl1(out, y) + torch.sqrt(criterionl2(out, y))
    loss.backward()
    optimizer.step()
    return loss


def val(best_perf):
    print("validation-------------------------")
    print(args)
    eval_list = []
    batches = gen_batch(dataset.get_data('val'), args.batch_size_val, dynamic_batch=True, shuffle=True)
    for idx, y_batch in enumerate(batches):
        y_batch = torch.Tensor(y_batch)
        if args.cuda:
            y_batch = y_batch.to(args.device)
        with torch.no_grad():
            out = model(y_batch[:, :args.n_his, :, :], y_batch[:, args.n_his - args.n_pred:args.n_his, :, :], adj)
        cur_val = evaluation(y_batch[:, args.n_his:, :, :].cpu().numpy(), out.cpu().numpy(), dataset.get_stats())
        eval_list.append(cur_val)
    eval_list = np.asarray(eval_list)
    ave_eval_list = np.mean(eval_list, axis=0)
    print("MAPE:", ave_eval_list[0], ",MAE:", ave_eval_list[1], ",RMSE:", ave_eval_list[2])
    now_time = datetime.datetime.now()
    print("time:", now_time - start_time)

    with open(args.record_pth, "a") as f:
        f.write(
            "epoch:" + str(epoch) + ",idx:" + str(idx) + ",MAPE:" + str(ave_eval_list[0]) + ",MAE:" + str(
                ave_eval_list[1]) + ",RMSE:" + str(
                ave_eval_list[2]) + ",time:" + str(now_time - start_time) + "\n")

    print("end--------------------")
    return ave_eval_list


def save(best_perf, current_perf):
    if best_perf[0] > current_perf[0] and best_perf[1] > current_perf[1] and best_perf[2] > current_perf[2]:
        if os.path.exists(args.model_pth):
            print("remove existed model")
            os.remove(args.model_pth)
        torch.save(model, args.model_pth)
        best_perf = current_perf
    return best_perf


if __name__ == '__main__':
    losses = []
    for epoch in range(5):
        batches = gen_batch(dataset.get_data('train'), args.batch_size_train, dynamic_batch=True, shuffle=True)
        for idx, x_batch in enumerate(batches):
            x_batch = torch.Tensor(x_batch)
            if args.cuda:
                x_batch = x_batch.to(args.device)
            loss = train(x_batch[:, :args.n_his, :, :], x_batch[:, args.n_his - args.n_pred:args.n_his, :, :],
                         x_batch[:, args.n_his:, :, :], adj)
            losses.append(loss.item())
            if idx % 100 == 0:
                loss_record = np.asarray(losses)
                print(epoch, ":", idx, ":", np.mean(loss_record), ":", np.min(loss_record))
                losses.clear()
            if idx % 1000 == 0:
                ave_perf_list = val(best_perf)
                ave_perf_list = list(ave_perf_list)
                best_perf = save(best_perf, ave_perf_list)
