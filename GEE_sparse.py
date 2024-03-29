# -*- coding: utf-8 -*-
################################################################################
import numpy as np
import time
# for sparse matrix
from scipy import sparse
################################################################################

# invalide devide resutls will be nan
np.seterr(divide='ignore', invalid='ignore')

############------------graph_encoder_embed_start----------------###############
class GraphEncoderEmbed_sparse:
  def run(self, X, Y, n, **kwargs):
    defaultKwargs = {'EdgeList': False, 'DiagA': True, 'Laplacian': False, 'Correlation': True} 
    kwargs = { **defaultKwargs, **kwargs}

    X = X.copy()
    Y = Y.copy()

    if kwargs['EdgeList']:
      size_flag = self.edge_list_size    
      X = self.Edge_to_Sparse(X, n, size_flag)
    
    total_emb_strat = time.time()
    
    if kwargs['DiagA']:
      X = self.Diagonal(X, n)

    if kwargs['Laplacian']:
      X = self.Laplacian(X, n)

    Z, W = self.Basic(X, Y, n)

    if kwargs['Correlation']:
      Z = self.Correlation(Z)

    total_emb_end = time.time()
    total_emb_time = total_emb_end - total_emb_strat

    return Z, W, total_emb_time

  def Basic(self, X, Y, n):
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
    """
    # assign k to the max along the first column
    # Note for python, label Y starts from 0. Python index starts from 0. thus size k should be max + 1
    k = Y[:,0].max() + 1

    W = self.get_W(Y, n, k)
    Z = X.dot(W)  
    return Z, W

  def get_W(self, Y, n, k):
    # W: sparse matrix for encoder marix. 
    W = sparse.dok_matrix((n, k), dtype=np.float32)

    #nk: 1*n array, contains the number of observations in each class
    nk = np.zeros((1,k))
    for i in range(k):
      nk[0,i] = np.count_nonzero(Y[:,0]==i)

    #follow the paper: W[i,k] = {1/nk if Yi==k, otherwise 0}
    for i in range(n):
      k_i = Y[i,0]
      if k_i >=0:
        W[i,k_i] = 1/nk[0,k_i]

    return W

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

