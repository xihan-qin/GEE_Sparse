# -*- coding: utf-8 -*-
"""
    Skip using the weight matrix, use nk directly to calcualte embedding.
"""
################################################################################
import sys
import numpy as np
import copy
from numpy import linalg as LA
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
import time
# for sparse matrix
from scipy import sparse
#early stop
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
################################################################################

# invalide devide resutls will be nan
np.seterr(divide='ignore', invalid='ignore')

############------------graph_encoder_embed_start----------------###############
class GraphEncoderEmbed_sparse:
  def run(self, X, Y, n, **kwargs):
    defaultKwargs = {'EdgeList': False, 'DiagA': True, 'Laplacian': False, 'Correlation': True, "Weight": 0} # weight option: {0: 1/nk, 1 : nk/n, 2: one-hot}
    kwargs = { **defaultKwargs, **kwargs}

    X = X.copy()
    Y = Y.copy()

    if kwargs['EdgeList']:
      size_flag = self.edge_list_size    
      X = self.Edge_to_Sparse(X, n, size_flag)
    
    emb_strat = time.time()

    if kwargs['DiagA']:
      X = self.Diagonal(X, n)

    if kwargs['Laplacian']:
      X = self.Laplacian(X, n)
  
    
    w_flag = kwargs['Weight']
    W = None
    Z = self.Basic_2(X, Y, n, w_flag)


    if kwargs['Correlation']:
      Z = self.Correlation(Z)
    
    emb_end = time.time()
    emb_time = emb_end - emb_strat
    
    return Z, W, emb_time

  def Basic_2(self, X, Y, n, w_flag):
    """
      graph embedding basic function
      input X is sparse csr matrix of adjacency matrix
      -- if there is a connection between node i and node j:
      ---- X(i,j) = 1, no edge weight
      ---- X(i,j) = edge weight.
      -- if there is no connection between node i and node j:
      ---- X(i,j) = 0, 
      ---- note there is no storage for this in sparse matrix. 
      ---- No storage means 0 in sparse matrix.
      input Y is numpy array with size (n,1):
      -- value -1 indicate no lable
      -- value >=0 indicate real label
      input train_idx: a list of indices of input X for training set 
    """
    k = Y[:,0].max() + 1
    
    #nk: 1*n array, contains the number of observations in each class
    nk = np.zeros((1,k))
    for i in range(k):
      nk[0,i] = np.count_nonzero(Y[:,0]==i)
    
    
    Z = sparse.dok_matrix((n, k), dtype=np.float32)
    # X is csr sparse, go over the rows
    # if direct iterate, will get (0, col_idx), weight -- note the first is always 0
    # but cannot direct extract from  (0, col_idx), weight, because it is a class with attribute array and data to store them seperately
    # try .__dict__ to see all attributes
    row_id = -1
    for row in X:
        # two arrays store col_idx and edge_weight differently
        row_id += 1
        col_ids = row.indices
        weights = row.data
        for i in range(row.indptr[0], row.indptr[1]):
          col_id = col_ids[i]
          edg_w  = weights[i]   
          k_i = Y[col_id,0]
          # label -1 indicate unkown nodes
          if k_i >=0:
            Z[row_id,k_i] += 1/nk[0,k_i] * edg_w  
    Z = sparse.csr_matrix(Z)
    return Z


  def Diagonal(self, X, n):
    """
      input X is sparse csr matrix of adjacency matrix
      return a sparse csr matrix of X matrix with 1s on the diagonal
    """
    I = sparse.identity(n)
    X = X + I
    return X


  def Laplacian(self, X, n):
    """
      input X is sparse csr matrix of adjacency matrix
      return a sparse csr matrix of Laplacian normalization of X matrix
    """
    X_sparse = sparse.csr_matrix(X)
    # get an array of degrees
    dig = X_sparse.sum(axis=0).A1
    # diagonal sparse matrix of D
    D = sparse.diags(dig,0)
    _D = D.power(-0.5)
    # D^-0.5 x A x D^-0.5
    L = _D.dot(X_sparse.dot(_D)) 

    # _L = _D.dot(X_sparse.dot(_D))    
    # # L = I - D^-0.5 x A x D^-0.5
    # I = sparse.identity(n)
    # L = I - _L   

    return L
  
  def Correlation(self, Z):
    """
      input Z is sparse csr matrix of embedding matrix from the basic function
      return normalized Z sparse matrix
      Calculation:
      Calculate each row's 2-norm (Euclidean distance). 
      e.g.row_x: [ele_i,ele_j,ele_k]. norm2 = sqr(sum(ele_i^2+ele_i^2+ele_i^2))
      then divide each element by their row norm
      e.g. [ele_i/norm2,ele_j/norm2,ele_k/norm2] 
    """
    # 2-norm
    row_norm = sparse.linalg.norm(Z, axis = 1)

    # row division to get the normalized Z
    diag = np.nan_to_num(1/row_norm)
    N = sparse.diags(diag,0)
    Z = N.dot(Z)

    return Z
  
  def edge_list_size(self, X):
    """
      set default edge list size as S3.
      If find X only has 2 columns, 
      return a flag "S2" indicating this is S2 edge list
    """
    if X.shape[1] == 2:
      return "S2"
    else:
      return "S3"
    
  def Edge_to_Sparse(self, X, n, size_flag):
    """
      input X is an edge list.
      Note for X, the edge list: 
      it is assumed there is no duplication of one connection
      e.g. connection between node i and node j, 
      there is only one row for this connection. 
      either (node_i, node_j, edge_w), or(node_j, node_i, edge_w)
      Only one of them. 
      If there are duplication in your edge list, please remove them before run.

      For S2 edge list (e.g. node_i, node_j per row), add one to all connections
      return a sparse csr matrix of S3 edge list
    """   
    #Build an empty sparse matrix. 
    X_new = sparse.dok_matrix((n, n), dtype=np.float32)

    for row in X:
      if size_flag == "S2":
        [node_i, node_j] = row
        node_i = int(node_i)
        node_j = int(node_j)
        X_new[node_i, node_j] = 1
        X_new[node_j, node_i] = 1
      else:
        [node_i, node_j, weight] = row
        node_i = int(node_i)
        node_j = int(node_j)
        X_new[node_i, node_j] = weight
        X_new[node_j, node_i] = weight
    
    X_new = sparse.csr_matrix(X_new)

    return X_new


############------------graph_encoder_embed_end------------------###############