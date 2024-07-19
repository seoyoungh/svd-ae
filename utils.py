import numpy as np
import random
from collections import defaultdict
import torch
import scipy.sparse as sp
import time 
import os

def get_common_path(hyper_params):
    ret = "{}_{}_".format(
        hyper_params['dataset'], hyper_params['model']
    )
    if hyper_params['model'] == 'svd-ae': ret += "k_{}_".format(hyper_params['k'])
    else:
        if hyper_params['grid_search_lamda']: ret += "grid_search_lamda_"
        else: ret += "lamda_{}_".format(hyper_params['lamda'])
    
    ret += "seed_{}".format(hyper_params['seed'])
    return ret

def get_item_count_map(data):
    item_count = defaultdict(int)
    for u, i, r in data.data['train']: item_count[i] += 1
    return item_count

def get_item_propensity(hyper_params, data, A = 0.55, B = 1.5):
    item_freq_map = get_item_count_map(data)
    item_freq = [ item_freq_map[i] for i in range(hyper_params['num_items']) ]
    num_instances = hyper_params['num_interactions']

    C = (np.log(num_instances)-1)*np.power(B+1, A)
    wts = 1.0 + C*np.power(np.array(item_freq)+B, -A)
    return np.ravel(wts)

def file_write(log_file, s, dont_print=False):
    if dont_print == False: print(s)
    if log_file is None: return
    f = open(log_file, 'a')
    f.write(s+'\n')
    f.close()

def log_end_epoch(hyper_params, metrics, step, time_elpased, metrics_on = '(TEST)', dont_print = False):
    string2 = ""
    for m in metrics: string2 += " | " + m + ' = ' + str("{:2.4f}".format(metrics[m]))
    string2 += ' ' + metrics_on

    if hyper_params['model'] == 'svd-ae':
        ss  = '| end of step {:4d} | time = {:5.2f}'.format(step, time_elpased)
    else:
        ss  = '| end of step {:4d} | time = {:5.2f} | best lambda = {}'.format(step, time_elpased, hyper_params['lamda'])

    ss += string2
    file_write(hyper_params['log_file'], ss, dont_print = dont_print)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

def get_file_name(dataset, k, path):
    ut = f"{dataset}-{k}-ut.npy"
    s = f"{dataset}-{k}-s.npy"
    vt = f"{dataset}-{k}-vt.npy"
    file_list = [ut, s, vt]
    file_list = [os.path.join(path, file) for file in file_list]
    return file_list

def preprocess_ease(adj_mat, device):
    start = time.time()
    adj_mat =  adj_mat
    item_adj = adj_mat.T @ adj_mat
    adj_mat = convert_sp_mat_to_sp_tensor(adj_mat).to_dense()
    item_adj = convert_sp_mat_to_sp_tensor(item_adj).to_dense()
    end = time.time()
    print('Pre-processing time: ', end-start)
    return adj_mat, item_adj

def preprocess_svd(LOAD, dataset, adj_mat, k, path, device):
    # start = time.time()
    file_list = get_file_name(dataset, k, path)
    rowsum = np.array(adj_mat.sum(axis=1))
    rowsum = np.where(rowsum == 0.0, 1.0, rowsum) # Do not divide by zero
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    norm_adj = d_mat.dot(adj_mat)
    colsum = np.array(adj_mat.sum(axis=0))
    colsum = np.where(colsum == 0.0, 1.0, colsum) # Do not divide by zero 
    d_inv = np.power(colsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_i = sp.diags(d_inv)
    d_mat_i_inv = sp.diags(1/d_inv)
    norm_adj = norm_adj.dot(d_mat_i)
    norm_adj = norm_adj.tocsc()
    adj_mat = convert_sp_mat_to_sp_tensor(adj_mat)
    norm_adj = convert_sp_mat_to_sp_tensor(norm_adj)

    if LOAD:
        cond = os.path.isfile(file_list[0]) & os.path.isfile(file_list[1]) & os.path.isfile(file_list[2])
        if cond:
            print("Load pre-calculated eigenvectors and eigenvalues!")
            ut, s, vt = np.load(file_list[0]), np.load(file_list[1]), np.load(file_list[2])
        else:
            print("Saved numpy files don't exist!")
            exit()
    else:
        start = time.time()
        ut, s, vt = torch.svd_lowrank(norm_adj, q=k, niter=2, M=None)
        end = time.time()
        if not os.path.isdir(path):
	        os.makedirs(path)
        np.save(file_list[0], ut.cpu().numpy())
        np.save(file_list[1], s.cpu().numpy())
        np.save(file_list[2], vt.cpu().numpy())

    norm_adj = norm_adj.to_dense() 
    ut = torch.FloatTensor(ut)
    s = torch.FloatTensor(s)
    vt = torch.FloatTensor(vt)
    # end = time.time()
    print('Pre-processing time: ', end-start)
    return adj_mat, norm_adj, ut, s, vt