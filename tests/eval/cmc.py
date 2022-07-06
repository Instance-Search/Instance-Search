import numpy as np
import os
import scipy.misc
from glob import glob
from PIL import Image
import sys
import copy
from scipy.spatial.distance import cdist

# def roc_curve(label, predict):
#     """For each thrd, we calculate the fpr, tpr and thrds."""
#     fpr, tpr, thrds = [], [], []

#     print(predict.shape)
#     fda

def GetROC(sims_in,prb_labels,gal_labels,separate_camera=True, all_pos_pair_num=100):
    sims = 1 - sims_in
    assert sims.shape[0]==prb_labels.shape[0]
    assert sims.shape[1]==gal_labels.shape[0]

    sims_label = np.equal(np.expand_dims(prb_labels[:,0], 1), np.expand_dims(gal_labels[:,0], 0)) * 1.0

    if separate_camera:
        mask = 1 - np.equal(np.expand_dims(prb_labels[:,1], 1), np.expand_dims(gal_labels[:,1], 0)) * 1.0
        sims = sims * mask
        sims_label = sims_label * mask

    # fpr, tpr, thrds = roc_curve(sims_label.flatten(), sims.flatten())

    thrds = np.arange(0.5,0.96,0.02)
    # thrds = np.arange(0.5,1.0,0.02)
    pec_list = []
    for thrd in thrds:
        sim1 = np.where(sims>=thrd,sims,0)
        sim2 = sims_label

        m1 = np.where(sim1>0,1,0)
        m2 = np.where(sim2>0,1,0)
        tp = np.sum(m1*m2)
        pec = float(tp)/(np.sum(m1)+1e-6)
        pec_list.append(pec)
        rec = float(tp)/(np.sum(m2)+1e-6)
        print('thrd %.3f: p-%.4f r-%.4f' %(thrd,pec,rec))
    get_recall_at_precision(0.9, pec_list, sims, sims_label, thrds, all_pos_pair_num)
    get_recall_at_precision(0.99, pec_list, sims, sims_label, thrds, all_pos_pair_num)
    return

def get_positive_pair_num(prb_labels,gal_labels,separate_camera=True):

    sims_label = np.equal(np.expand_dims(prb_labels[:,0], 1), np.expand_dims(gal_labels[:,0], 0)) * 1.0

    if separate_camera:
        mask = 1 - np.equal(np.expand_dims(prb_labels[:,1], 1), np.expand_dims(gal_labels[:,1], 0)) * 1.0
        sims_label = sims_label * mask
    return np.sum(sims_label)

def get_recall_at_precision(fixed_precision, pec_list, sims, sims_label, thrds, all_pos_pair_num):
    """Get recall at precision."""

    pec_array = np.array(pec_list)
    left_array = np.argwhere(pec_array < fixed_precision)
    right_array = np.argwhere(pec_array > fixed_precision)
    index_0 = left_array[-1][0]
    index_1 = right_array[0][0]
    new_thrds = np.arange(thrds[index_0], thrds[index_1],0.001)
    new_pec_list = []
    new_rec_list = []
    for thrd in new_thrds:
        sim1 = np.where(sims>=thrd,sims,0)
        sim2 = sims_label

        m1 = np.where(sim1>0,1,0)
        m2 = np.where(sim2>0,1,0)
        tp = np.sum(m1*m2)
        pec = float(tp)/(np.sum(m1)+1e-6)
        new_pec_list.append(pec)
        rec = float(tp)/(all_pos_pair_num+1e-6)
        new_rec_list.append(rec)
    diff = np.abs(np.array(new_pec_list)-fixed_precision)
    best_index = np.argmin(diff)
    print(f'precision: {new_pec_list[best_index]}, recall: {new_rec_list[best_index]}, thrd: {new_thrds[best_index]}')

def GetMAP(sims,prb_labels,gal_labels,separate_camera=True):
    assert sims.shape[0]==prb_labels.shape[0]
    assert sims.shape[1]==gal_labels.shape[0]

    idxs = np.argsort(sims)

    min_dist = np.mean([sims[i, idxs[i][0]] for i in range(prb_labels.shape[0])])
    if min_dist < 10e-4:
        print('largest sim between gallery and probe too high, may be duplicated?')
        print('Remove the top-1 prediction!')
        idxs = idxs[:, 1:]

    ids_map = 0.0
    querynum = 0
    for i in range(0,prb_labels.shape[0]):
        #find the same id
        valid1 = (np.where(gal_labels[:,0]==prb_labels[i,0]))[0]
        #find the same id in same camera
        valid2 = valid1[gal_labels[:,1][valid1]==prb_labels[i,1]]
        if separate_camera:
            valid = set(valid1)-set(valid2)
            idx = idxs[i,]
            idxx = np.in1d(idx,valid2)
            idx = idx[idxx!=1]
        else:
            valid = set(valid1)
            idx = idxs[i,]

        if len(valid) > 0:
            querynum += 1

        #find all same ids with probability
        count=0
        id_map=0.0
        for j in range(0,len(idx)):
            if idx[j] in valid:
                count += 1
                id_map += count/float(j+1)
        ids_map += id_map/(count+0.00001)
    print(f'valid query num: {querynum}')
    ids_map = ids_map/querynum
    return ids_map

def GetRanks(sims,prb_labels,gal_labels,top=1,separate_camera=True):
    assert sims.shape[0]==prb_labels.shape[0]
    assert sims.shape[1]==gal_labels.shape[0]
    
    idxs = np.argsort(sims
)
    min_dist = np.mean([sims[i, idxs[i][0]] for i in range(prb_labels.shape[0])])
    if min_dist < 10e-4:
        print('largest sim between gallery and probe too high, may be duplicated?')
        print('Remove the top-1 prediction!')
        idxs = idxs[:, 1:]

    querynum = 0
    count = np.zeros((prb_labels.shape[0],int(top/5)+1))
    for i in range(0,prb_labels.shape[0]):
        #find the same id
        valid1 = (np.where(gal_labels[:,0]==prb_labels[i,0]))[0]
        #find the same id in same camera
        valid2 = valid1[gal_labels[:,1][valid1]==prb_labels[i,1]]
        if separate_camera:
            valid = set(valid1)-set(valid2)
            idx = idxs[i,]
            idxx = np.in1d(idx,valid2)
            idx = idx[idxx!=1]
        else:
            valid = set(valid1)
            idx = idxs[i,]

        if len(valid) > 0:
            querynum += 1

        #find topk candidates
        for j in np.arange(0,top+1,5):
            if j==0:
                topk=1;
            else:
                topk=j;
            if valid&set(idx[:topk])!=set([]):
                count[i,int(j/5)] += 1;
    ranks = np.sum(count,axis=0)/querynum
    return ranks

def ComputeEuclid(array1,array2,fg_sqrt=True,fg_norm=False):
    #array1:[m1,n],array2:[m2,n]
    assert array1.shape[1]==array2.shape[1];
    # norm
    if fg_norm:
        array1 = array1/np.linalg.norm(array1,ord=2,axis=1,keepdims=True)
        array2 = array2/np.linalg.norm(array2,ord=2,axis=1,keepdims=True)
    # shape [m1, 1]
    square1 = np.sum(np.square(array1), axis=1)[..., np.newaxis]
    # shape [1, m2]
    square2 = np.sum(np.square(array2), axis=1)[np.newaxis, ...]
    #print 'array1,array2 shape:',array1.shape,array2.shape
    squared_dist = - 2 * np.matmul(array1, array2.T) + square1 + square2
    squared_dist[squared_dist < 0] = 0
    #shape [m1,m2]
    if fg_sqrt:
        dist = np.sqrt(squared_dist)
        #print('[test] using sqrt for distance')
    else:
        dist = squared_dist
        #print('[test] not using sqrt for distance')
    sim = 1-dist**2/2.
    return 1-sim

def ComputeCosine(array1,array2,fg_norm=False):
    # norm
    if fg_norm:
        array1 = array1/np.linalg.norm(array1,ord=2,axis=1,keepdims=True)
        array2 = array2/np.linalg.norm(array2,ord=2,axis=1,keepdims=True)
    narr1 = np.linalg.norm(array1,ord=2,axis=1,keepdims=True)
    narr2 = np.linalg.norm(array2,ord=2,axis=1,keepdims=True)
    sim = np.matmul(array1,array2.T)/np.matmul(narr1,narr2.T)
    sim = 0.5+0.5*sim
    return 1-sim

def ReRank(probFea,galFea,k1=20,k2=6,lambda_value=0.3):

    query_num = probFea.shape[0]
    all_num = query_num + galFea.shape[0]    
    feat = np.append(probFea,galFea,axis = 0)
    feat = feat.astype(np.float32)
    print('computing original distance')
    
    original_dist = cdist(feat,feat).astype(np.float32)  
    original_dist = np.power(original_dist,2).astype(np.float32)
    del feat    
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist/np.max(original_dist,axis = 0))
    V = np.zeros_like(original_dist).astype(np.float32)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    print('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i,:k1+1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = np.where(backward_k_neigh_index==i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate,:int(np.around(k1/2.))+1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,:int(np.around(k1/2.))+1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index,k_reciprocal_index))> 2./3*len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index,candidate_k_reciprocal_index)
            
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i,k_reciprocal_expansion_index])
        V[i,k_reciprocal_expansion_index] = weight/np.sum(weight)
    original_dist = original_dist[:query_num,]    
    if k2 != 1:
        V_qe = np.zeros_like(V,dtype=np.float32)
        for i in range(all_num):
            V_qe[i,:] = np.mean(V[initial_rank[i,:k2],:],axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:,i] != 0)[0])
    
    jaccard_dist = np.zeros_like(original_dist,dtype = np.float32)

    
    for i in range(query_num):
        temp_min = np.zeros(shape=[1,gallery_num],dtype=np.float32)
        indNonZero = np.where(V[i,:] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0,indImages[j]] = temp_min[0,indImages[j]]+ np.minimum(V[i,indNonZero[j]],V[indImages[j],indNonZero[j]])
        jaccard_dist[i] = 1-temp_min/(2-temp_min)
    
    final_dist = jaccard_dist*(1-lambda_value) + original_dist*lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num,query_num:]
    return final_dist
