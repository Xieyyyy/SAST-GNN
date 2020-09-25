import numpy as np


def z_score(x, mean, std):
    '''
    Z-score做归一化
    '''
    return (x - mean) / std


def z_inverse(x, mean, std):
    '''
    逆归一化
    '''
    return x * std + mean


def MAPE(v, v_):
    '''
    百分比绝对误差
    '''
    return np.mean(np.abs(v_ - v) / (v + 1e-5))


def RMSE(v, v_):
    '''
    RMSE
    '''
    return np.sqrt(np.mean((v_ - v) ** 2))


def MAE(v, v_):
    '''
    MAE
    '''
    return np.mean(np.abs(v_ - v))


def evaluation(y, y_, x_stats):
    '''
    评价函数（MAPE,MAE,RMS）
    '''
    dim = len(y_.shape)

    if dim == 3:
        # single_step case
        v = z_inverse(y, x_stats['mean'], x_stats['std'])
        v_ = z_inverse(y_, x_stats['mean'], x_stats['std'])
        return np.array([MAPE(v, v_), MAE(v, v_), RMSE(v, v_)])
    else:
        # multi_step case
        tmp_list = []
        # y -> [time_step, batch_size, n_route, 1]
        # y = np.swapaxes(y, 0, 1)
        # recursively call
        for i in range(y_.shape[0]):
            tmp_res = evaluation(y[i], y_[i], x_stats)
            tmp_list.append(tmp_res)
        return np.concatenate(tmp_list, axis=0)
