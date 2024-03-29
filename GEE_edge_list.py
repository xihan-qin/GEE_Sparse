# -*- coding: utf-8 -*-

################################################################################
import numpy as np
from numpy import linalg as LA
import time
################################################################################
# invalide devide resutls will be nan
np.seterr(divide='ignore', invalid='ignore')

############------------graph_encoder_embed_start----------------###############
class GraphEncoderEmbed_Edge:
  def run(self, X, Y, n, **kwargs):
    defaultKwargs = {'DiagA': True, 'Laplacian': False, 'Correlation': True} 
    kwargs = { **defaultKwargs, **kwargs}

    X = X.copy()
    Y = Y.copy()

    X = self.to_s3_list(X)

    total_emb_strat = time.time()

    if kwargs['DiagA']:
      X = self.Diagonal(X, n)

    if kwargs['Laplacian']:
      X = self.Laplacian(X, n)

    Z, W = self.Basic(X, Y, n)

    if kwargs['Correlation']:
      Z = self.Correlation(Z, n)

    total_emb_end = time.time()
    total_emb_time = total_emb_end - total_emb_strat

    return Z, W, total_emb_time
  
  def Basic(self, X, Y, n):
    """
      graph embedding basic function
      input X is S3 edge list
      input Y is numpy array with size (n,1):
      -- value -1 indicate no lable
      -- value >=0 indicate real label
      input n: number of vertices
    """
    # assign k to the max along the first column
    # Note for python, label Y starts from 0. Python index starts from 0. thus size k should be max + 1
    k = Y[:,0].max() + 1

    W = self.get_W(Y, n, k)
    Z = np.zeros((n,k))
    for row in X:
      [v_i, v_j, edg_i_j] = row
      v_i = int(v_i)
      v_j = int(v_j)

      label_i = Y[v_i][0]
      label_j = Y[v_j][0]

      if label_j >= 0:
        Z[v_i, label_j] = Z[v_i, label_j] + W[v_j, label_j]*edg_i_j
      if (label_i >= 0) and (v_i != v_j):
        Z[v_j, label_i] = Z[v_j, label_i] + W[v_i, label_i]*edg_i_j

    return Z, W

  def get_W(self, Y, n, k):
    W = np.zeros((n,k))
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
    # add self-loop to edg list -- add 1 connection for each (i,i)
    self_loops = np.column_stack((np.arange(n), np.arange(n), np.ones(n)))
    # faster than vstack --  adding the second to the bottom
    X = np.concatenate((X,self_loops), axis = 0)
    return X

  def Laplacian(self, X, n):
    s = X.shape[0] # get the row number of the edg list

    D = np.zeros((n,1))
    for row in X:
      [v_i, v_j, edg_i_j] = row
      v_i = int(v_i)
      v_j = int(v_j)
      D[v_i] = D[v_i] + edg_i_j
      if v_i != v_j:
        D[v_j] = D[v_j] + edg_i_j

    D = np.power(D, -0.5)

    for i in range(s):
      X[i,2] = X[i,2] * D[int(X[i,0])] * D[int(X[i,1])]

    return X

  def Correlation(self, Z, n):
    """
      Calculate each row's 2-norm (Euclidean distance).
      e.g.row_x: [ele_i,ele_j,ele_k]. norm2 = sqr(sum(ele_i^2+ele_i^2+ele_i^2))
      then divide each element by their row norm
      e.g. [ele_i/norm2,ele_j/norm2,ele_k/norm2]
    """
    row_norm = LA.norm(Z, axis = 1)
    reshape_row_norm = np.reshape(row_norm, (n,1))
    Z = np.nan_to_num(Z/reshape_row_norm)

    return Z

  def to_s3_list(self,X):
    """
      if input X only has 2 columns, make it into s3 edge list.
      this function will return a s3 edge list
      [node_i, node_j, weight]...
    """
    s = X.shape[0] # get the row number of the edg list
    if X.shape[1] == 2:
      # enlarge the edgelist to s*3 by adding 1 to the thrid position as adj(i,j)
      X = np.insert(X, 1, np.ones(s,1))

    return X

############------------graph_encoder_embed_end------------------###############
