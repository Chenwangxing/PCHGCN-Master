import math
import torch
import numpy as np

def ade(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N*T)
        
    return sum_all/All


def fde(predAll,targetAll,count_):
    All = len(predAll)
    sum_all = 0 
    for s in range(All):
        pred = np.swapaxes(predAll[s][:,:count_[s],:],0,1)
        target = np.swapaxes(targetAll[s][:,:count_[s],:],0,1)
        N = pred.shape[0]
        T = pred.shape[1]
        sum_ = 0 
        for i in range(N):
            for t in range(T-1,T):
                sum_+=math.sqrt((pred[i,t,0] - target[i,t,0])**2+(pred[i,t,1] - target[i,t,1])**2)
        sum_all += sum_/(N)

    return sum_all/All


def seq_to_nodes(seq_,max_nodes = 88):
    seq_ = seq_.squeeze()
    seq_len = seq_.shape[2]
    
    V = np.zeros((seq_len,max_nodes,2))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_[h]
            
    return V.squeeze()

def nodes_rel_to_nodes_abs(nodes,init_node):
    nodes_ = np.zeros_like(nodes)
    for s in range(nodes.shape[0]):
        for ped in range(nodes.shape[1]):
            nodes_[s,ped,:] = np.sum(nodes[:s+1,ped,:],axis=0) + init_node[ped,:]

    return nodes_.squeeze()

def closer_to_zero(current,new_v):
    dec =  min([(abs(current),current),(abs(new_v),new_v)])[1]
    if dec != current:
        return True
    else: 
        return False
        

def bivariate_loss(V_pred, V_trgt, CVM):
    # V_pred=[20, 12-Tpred, N, 2]   V_trgt=[12-Tpred, N, 2]   CVM==[1, 1, N, 2]
    V_D = V_trgt.unsqueeze(dim=0) - CVM
    #############################################
    error_displacement = (V_pred - V_D).norm(p=2, dim=-1)  # [20, Tpred, N]
    loss_ade = error_displacement.mean(dim=-2).mean()
    loss_fde = error_displacement[:, -1, :].mean()
    #############################################
    loss_euclidean_ade = error_displacement.mean(dim=-2).min(dim=0)[0].mean()
    loss_euclidean_fde = error_displacement[:, -1, :].min(dim=0)[0].mean()
    #############################################
    loss_euclidean_ade = loss_euclidean_ade + 0.01 * loss_ade
    loss_euclidean_fde = loss_euclidean_fde + 0.01 * loss_fde
    # return result
    return loss_euclidean_ade + loss_euclidean_fde
