import jax
import functools
from jax import scipy as sp
from jax import numpy as jnp
from neural_tangents import stax
import torch
from torch import nn
import numpy as np

def make_kernelized_rr_forward(hyper_params):
    _, _, kernel_fn = FullyConnectedNetwork(
        depth=hyper_params['depth'],
        num_classes=hyper_params['num_items']
    )
    # NOTE: Un-comment this if the dataset size is very big (didn't need it for experiments in the paper)
    # kernel_fn = nt.batch(kernel_fn, batch_size=128)
    kernel_fn = functools.partial(kernel_fn, get='ntk')

    @jax.jit
    def kernelized_rr_forward(X_train, X_predict, reg=0.1):
        K_train = kernel_fn(X_train, X_train) # user * user
        K_predict = kernel_fn(X_predict, X_train) # user * user
        K_reg = (K_train + jnp.abs(reg) * jnp.trace(K_train) * jnp.eye(K_train.shape[0]) / K_train.shape[0]) # user * user
        return jnp.dot(K_predict, sp.linalg.solve(K_reg, X_train, sym_pos=True))
        # sp.linalg.solve(K_reg, X_train, sym_pos=True)) -> user * item

    return kernelized_rr_forward, kernel_fn

def FullyConnectedNetwork( 
    depth,
    W_std = 2 ** 0.5, 
    b_std = 0.1,
    num_classes = 10,
    parameterization = 'ntk'
):
    activation_fn = stax.Relu()
    dense = functools.partial(stax.Dense, W_std=W_std, b_std=b_std, parameterization=parameterization)

    layers = [stax.Flatten()]
    # NOTE: setting width = 1024 doesn't matter as the NTK parameterization will stretch this till \infty
    for _ in range(depth): layers += [dense(1024), activation_fn] 
    layers += [stax.Dense(num_classes, W_std=W_std, b_std=b_std, parameterization=parameterization)]

    return stax.serial(*layers)

class EASE(nn.Module):
    def __init__(self, adj_mat, item_adj, device='cuda:0'):
        super(EASE, self).__init__()
        self.adj_mat = adj_mat.to(device)
        self.item_adj = item_adj.to(device)

    def forward(self, lambda_):
        G = self.item_adj
        diagIndices = np.diag_indices(G.shape[0])
        G[diagIndices] += lambda_
        P = torch.inverse(G)
        B = P / (-torch.diag(P))
        B[diagIndices] = 0
        rating = torch.mm(self.adj_mat, B)

        return rating

class SVD_AE(nn.Module):
    def __init__(self, adj_mat, norm_adj, user_sv, item_sv, device='cuda:0'):
        super(SVD_AE, self).__init__()
        self.adj_mat = adj_mat.to(device)
        self.norm_adj = norm_adj.to(device)
        self.user_sv = user_sv.to(device) # (K, M)
        self.item_sv = item_sv.to(device) # (K, N)

    def forward(self, lambda_mat):
        A = self.item_sv @ (torch.diag(1/lambda_mat)) @ self.user_sv.T
        rating = torch.mm(self.norm_adj, A @ self.adj_mat.to_dense())
        # torch.inverse(torch.diag(lambda_mat))
        return rating