# -*- coding: utf-8 -*-
################################################################################
import numpy as np
import copy
import math
import os
################################################################################
class Model:
  def __init__(self, n):
    """
      Initialize the class
      n: the number of nodes
      d: initialize the number of class categories/communities to None
      X: initialize the adjacency matrix or edge list to None
      Y: initialize the classes/labels to None
    """
    self.n = n
    self.d = None
    self.X = None
    self.Y = None
    
  def summary(self):
    """
      print the adjacency matrix and Labels
    """
    info = self.info
    n = self.n
    d = self.d
    Y = self.Y
    X = self.X
    if type(Y) == np.ndarray:
      items = (
          ("Info:", info),
          ("n:", type(n), n),
          ("d:", type(d), d),                    
          ("X:", X.shape, X),
          ("Y:", Y.shape, Y),
      )
      for item in items:
        for i in item:
          print(i)
    else:
      print("X: None", "Y: None", sep='\n')
  
  def gety(self, pp):
    """
      Get Labels
      The class categories/communities start from 0
      pp from the input is used for sampling
        pp:  [p1, p2, ..., pn]
        sum(pp) == 1
      tt is a list of ramdonly assigned number to the samples, 
        from unifrom distribution over [0,1)
        tt = [tt1, tt2, ..., ttn] 
      label is given based on tt:
        if tt1 < p1 => label_1 = 0
        if p1 < tt1 < (p2+p1) => label_1 = 1
        ...... 
    """
    np.random.seed(2)
    
    n = self.n
    Y = np.zeros((n,1), dtype=int)
    
    # Create n*1 array from a uniform distribution over [0, 1).
    tt = np.random.rand(n,1) 
    # get thresholds for the labels(the indices)
    thres = [sum(pp[:i+1]) for i in range(len(pp))]
    
    # assgin labels based on the thredsholds, see explaination in docstrings
    for i in range(len(tt)):
      for j in range(len(thres)-1, -1, -1):
        if tt[i] < thres[j]:
          Y[i,0] = j
    
    return Y

  def SBM(self, d, pp, Bl):
    """
      SBM: Stochastic Block Model 
      pp is used for generating labels(classes) -- the input for gety()
      d is total label number. In total, d different labels.
      Bl is a d*d matrix. Bl[i,j] indicates the probability of a edge between 
        classes(labels) i and j

      Dis is the n*n matrix, Dis[i,j] indicates the connection between vertex i
      and vertex j, with the probability Bl[i_label,j_label]
       
    """ 
    sbm = copy.deepcopy(self)
    sbm.name = "SBM"
    n = sbm.n
    Y = self.gety(pp)  

    edg_num = 0
    np.random.seed(2)
    Dis = np.zeros((n,n), dtype=int)
    for i in range(n):
      Dis[i,i] = 0 # assign diagonal 0
      for j in range(i+1,n):
        Dis[i,j] = int(np.random.rand() < Bl[Y[i], Y[j]])
        Dis[j,i] = Dis[i,j]
        if Dis[i,j] == 1:
          edg_num += 1
    
    sbm.X = Dis
    sbm.Y = Y
    sbm.d = d
    sbm.bl = Bl
    sbm.pp = pp
    sbm.edg_num = edg_num

    return sbm

  def DC_SBM(self, d, pp, Bl):
    """
      DC-SBM: Degree corrected Stochastic Block Model  
      pp is used for generating labels(classes) -- the input for gety()
      d is total label number. In total, d different labels.
      Bl is a d*d matrix. Bl[i,j] indicates the probability between 
        classes(labels) i and j
      
      theta is the n*1 array, each element contains the degree of a vertex(node).

      Dis is the n*n matrix, Dis[i,j] indicates the connection between vertex i
      and vertex j, with the probability theta[i]*theta[j]*Bl[i_label,j_label]

    """

    dcsbm = copy.deepcopy(self)
    dcsbm.name = "DC-SBM" 

    n = dcsbm.n
    Y = self.gety(pp)

    # theta is the n*1 array, from beta distribution with a=1, b=4
    theta = np.random.beta(1,4, (n,1)) 

    Dis = np.zeros((n,n), dtype=int)
    for i in range(n):
      Dis[i,i] = 0 # assign diagonal 0
      for j in range(i+1,n):
        Dis[i,j] = int(np.random.rand() < theta[i]*theta[j]*Bl[Y[i], Y[j]])
        Dis[j,i] = Dis[i,j]   
    
    dcsbm.X = Dis
    dcsbm.Y = Y
    dcsbm.d = d
    return dcsbm

  def DC_SBM_edg_list(self, d, pp, Bl):
      """
        DC-SBM: Degree corrected Stochastic Block Model  
        pp is used for generating labels(classes) -- the input for gety()
        d is total label number. Intotal, d different labels.
        Bl is a d*d matrix. Bl[i,j] indicates the probability between 
          classes(labels) i and j
        
        theta is the n*1 array, each element contains the degree of a vertex(node).

        Dis is the n*n matrix, Dis[i,j] indicates the connection between vertex i
        and vertex j, with the probability theta[i]*theta[j]*Bl[i_label,j_label]

        return s*3 edge list. 
          each row in the edge list contains the nodes and the weight(1) of the conection.
          e.g. i,j,1 => nodei and nodej has connection with weight 1
      """

      dcsbm_edg = copy.deepcopy(self)
      dcsbm_edg.name = "DC-SBM s*3 edge list" 

      n = dcsbm_edg.n
      Y = self.gety(pp)

      # theta is the n*1 array, from beta distribution with a=1, b=4
      theta = np.random.beta(1,4, (n,1))  

      Dis = []
      for i in range(n):
        for j in range(i+1,n):
          if np.random.rand() < theta[i]*theta[j]*Bl[Y[i], Y[j]]:
            Dis.append([i,j,1])
      
      Dis = np.array(Dis)   
      
      dcsbm_edg.X = Dis
      dcsbm_edg.Y = Y
      dcsbm_edg.d = d

      return dcsbm_edg

  def nonsym_adj_to_edg(self,A):
    """
      input is the adjacency matrix: A
      other variables in this function:
      s: number of edges
      return edg_list -- matrix format with shape(edg_sum,3):
      example row in edg_list(matrix): [vertex1, vertex2, connection weight from Adj matrix]
    """

    n = A.shape[0] 
    # construct the initial edg_list matrix with the size of (edg_sum, 3)
    edg_list = []
    for i in range(n):
      for j in range(n):
        if A[i,j] > 0:
          row = [i, j, A[i,j]]
          row_reverse = [j, i, A[i,j]]
          if (row not in edg_list) and (row_reverse not in edg_list):
            edg_list.append(row)
    edg = np.array(edg_list)
    return edg
  
  def sym_adj_to_edg(self,A):
    """
      input is the symmetric adjacency matrix: A
      other variables in this function:
      s: number of edges
      return edg_list -- matrix format with shape(edg_sum,3):
      example row in edg_list(matrix): [vertex1, vertex2, connection weight from Adj matrix]
    """

    n = A.shape[0] 
    # construct the initial edg_list matrix with the size of (edg_sum, 3)
    edg_list = []
    for i in range(n):
      for j in range(i, n):
        if A[i,j] > 0:
          row = [i, j, A[i,j]]
          edg_list.append(row)
    edg = np.array(edg_list)
    return edg

  def to_edge_list(self):
    """
      change X from adjacnecy matrix to s3 edge list
    """
    DataSet = copy.deepcopy(self)
    X = DataSet.X

    DataSet.X = self.sym_adj_to_edg(X)
    return DataSet

  def add_unknown(self, unlabel_ratio):
    """
      input is the ratio for unlabeled set range is [0,1]
    """

    DataSet = copy.deepcopy(self)
    d = DataSet.d
    n = DataSet.n
    u = unlabel_ratio # unlabeled
    l = 1 - u   

    Y_ori = DataSet.Y
    Y = np.copy(Y_ori)
    
    Y_1st_dim = Y.shape[0]

    np.random.seed(0)

    # stratified unlabel with given ratio
    for i in range(d):
      i_indices  = np.argwhere(Y==i)[:,0]
      len_i = i_indices.shape[0]
      i_ran_permu = np.random.permutation(len_i)  #randomly permute the indices of the i_indices
      i_ran_permu = i_ran_permu[:math.floor(len_i*u)] #pick the indices of i_indices by ratio u
      unlabel_idx_i = i_indices[i_ran_permu]
      Y[unlabel_idx_i, 0] = -1

    DataSet.Y_ori = Y_ori
    DataSet.Y = Y
    return DataSet

  def for_cluster(self):
    """
      input is the ratio for unlabeled set range is [0,1]
    """

    DataSet = copy.deepcopy(self)
    d = DataSet.d


    Y_ori = DataSet.Y
    Y = np.array([[d]])

    DataSet.Y_ori = Y_ori
    DataSet.Y = Y
    return DataSet
  
  def split_sets(self, test_ratio):
    """
      Split Adjacency matrix for training and testing 
      input is the ratio for test set.
    """

    DataSet = copy.deepcopy(self)
    
    # X_ori = DataSet.X
    Y_ori = DataSet.Y
    # X = np.copy(X_ori) 
    Y = np.copy(Y_ori)
    t = test_ratio
    n = DataSet.n

    Y_1st_dim = Y.shape[0]

    np.random.seed(0)
    indices = np.random.permutation(Y_1st_dim)  #randomly permute the 1st indices

    # Generate indices for splits
    test_ind_split_point = math.floor(Y_1st_dim*t)
    test_idx, train_idx = indices[:test_ind_split_point], indices[test_ind_split_point:]

    # get the Y_test label
    Y_test = Y[test_idx]
    # get the Y_train label
    Y_train = Y[train_idx]
    # mark the test position as unknown: -1
    Y[test_idx, 0] = -1    


    DataSet.Y = Y
    DataSet.Y_ori = Y_ori
    DataSet.Y_test = Y_test     
    DataSet.Y_train = Y_train 
    DataSet.test_idx = test_idx
    DataSet.train_idx = train_idx  
    DataSet.test_ratio = test_ratio  
    return DataSet
  
  def output_files(self, outfolder):
    """
     Assume the edges have no special weights.
     output two files. 
     one file end with ".node_labels". 
      each line has a node and its lable and they are seperated by a comma. 
      e.g. "1,2" means node 1 has label 2
     one file end with ".edges".
      each line has two nodes that forms an edge and a universe edge weight 1.
      e.g. "1,870,1" means node 1 and node 870 forms an edge and the weight is 1 
    """
    DataSet = copy.deepcopy(self)
    if DataSet.edglist:
      edge_list = DataSet.X
    else:
      edge_list = self.sym_adj_to_edg(DataSet.X)
    
    output_folder = f"{outfolder}/{DataSet.name}_{DataSet.n}_nodes"
    if not os.path.exists(output_folder):
      os.makedirs(f"{output_folder}")

    labels = DataSet.Y_ori 
    f1 = open(f"{output_folder}/{DataSet.name}_node_labels_ori.tsv", "w")
    for i in range(labels.shape[0]):
      f1.write(f"{i}\t{labels[i,0]}\n")
    f1.close()

    labels = DataSet.Y # with the same order as Y_ori, the test nodes are labeled as -1 
    f2 = open(f"{output_folder}/{DataSet.name}_node_labels_with_unknown.tsv", "w")
    for i in range(labels.shape[0]):
      f2.write(f"{i}\t{labels[i,0]}\n")
    f2.close()
    
    train_labels = DataSet.Y_train 
    train_idx = DataSet.train_idx
    f3 = open(f"{output_folder}/{DataSet.name}_node_labels_train.tsv", "w")
    for i in range(train_labels.shape[0]):
      f3.write(f"{train_idx[i]}\t{train_labels[i,0]}\n")
    f3.close()
    
    test_labels = DataSet.Y_test
    test_idx = DataSet.test_idx
    f4 = open(f"{output_folder}/{DataSet.name}_node_labels_test.tsv", "w")
    for i in range(test_labels.shape[0]):
      f4.write(f"{test_idx[i]}\t{test_labels[i,0]}\n")
    f4.close()

    f5 = open(f"{output_folder}/{DataSet.name}_edges.tsv", "w")
    for (i,j,w) in edge_list:
      f5.write(f"{i}\t{j}\t{w}\n")
    f5.close()
    
    bl = DataSet.bl # matrix of probability
    f6 = open(f"{output_folder}/{DataSet.name}_bl.txt", "w")
    for row in bl:
      line = '\t'.join([str(float(i)) for i in row])
      f6.write(f'{line}\n')
    f6.close()
    
    pp = DataSet.pp #list
    f7 = open(f"{output_folder}/{DataSet.name}_pp.txt", "w")
    line = '\t'.join([str(item) for item in pp])
    f7.write(f'{line}')
    f7.close() 

    vars = ["node_num", "edg_num", "class_num", "test_ratio"]
    stats = [DataSet.n, DataSet.edg_num, DataSet.d, DataSet.test_ratio]
    f8 = open(f"{output_folder}/{DataSet.name}_stats.txt", "w")
    for i in range(len(vars)):
      f8.write(f'{vars[i]}: {stats[i]}\n')
    f8.close() 
