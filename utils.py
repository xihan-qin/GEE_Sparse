# -*- coding: utf-8 -*-
################################################################################
import copy
import numpy as np
################################################################################
#-----------------input_data_from_file_start-----------------------------------#
class DataSet:
  def __init__(self):
    self.X = None  # edg_list
    self.n = None  # number of nodes
    self.Y_ori = None  # original labels contains all original labels
    self.Y = None  # unknown labels marked as -1, shape(n,1)
    self.Y_train = None # labels for train set
    self.train_idx = None # index for train set
    self.Y_test = None # labels for test set
    self.test_idx = None # index for test set

  def get_initial_values(self, edg_file, label_file_ori, label_file_train, label_file_test, stat_file):
    resultSet = copy.deepcopy(self)

    Y_ori, map_new_old_keys = self.read_node_ori_file(label_file_ori)
    n = len(Y_ori)
    Y_train, train_idx = self.read_node_other_file(label_file_train, map_new_old_keys)
    Y_test, test_idx = self.read_node_other_file(label_file_test, map_new_old_keys)

    X, edg_num = self.read_edge_file_with_remap(edg_file, map_new_old_keys)
    k = self.get_k(Y_ori)

    Y_ori = np.array(Y_ori).reshape(-1,1) # (n,1)
    Y = self.mark_unknown_labels(Y_ori,test_idx)

    test_ratio = self.read_stat_file(stat_file)

    resultSet.X = X # edg_list
    resultSet.edg_num = edg_num
    resultSet.Y_ori = Y_ori
    resultSet.Y = Y
    resultSet.n = n
    resultSet.Y_train = Y_train
    resultSet.train_idx = train_idx
    resultSet.Y_test = Y_test
    resultSet.test_idx = test_idx
    resultSet.d = k
    resultSet.A = self.edge_list_to_adjacency_matrix(X, n)
    resultSet.test_ratio = test_ratio

    return resultSet


  def read_node_ori_file(self, filename):
    """
      node file with any node names

    """
    label_dict = {}
    labels = open(filename, "r")
    map_new_old_keys = {}

    for l in labels:
      (node_i, label_i) = l.strip().split("\t")
      label_dict[node_i] = int(label_i)

    keys = sorted(list(label_dict.keys()))
    new_node_idx = [i for i in range(len(keys))] # map the idx that starts with 0 with original keys (node name)
    Y_ori = [] # stores the labels only
    for i in range(len(keys)):
      map_new_old_keys[keys[i]] = new_node_idx[i]
      Y_ori.append(label_dict[keys[i]])

    return Y_ori, map_new_old_keys

  def read_node_other_file(self, filename, map_new_old_keys):
    """
      based on the input file
      if test file, returns test_idx and test_labels
      if train file, returns train_idx and train_labels
    """
    idx_part = []
    Y_part = []
    labels = open(filename, "r")

    for l in labels:
      (node_i, label_i) = l.strip().split("\t")
      new_key = map_new_old_keys[node_i]
      idx_part.append(new_key)
      Y_part.append(int(label_i))

    return Y_part, idx_part

  def read_edge_file_with_remap(self, filename, map_new_old_keys):
    """
      for the ids that are remaped from the node file,
      need to remap id for edge list as well
    """
    edg_list = []
    edges = open(filename, "r")

    edg_num = 0
    for l in edges:
      edg_num += 1
      elements = l.strip().split("\t")
      if len(elements) > 2:
        (node_i, node_j, w) = elements
        # some edge files have nodes as float numbers
        (node_i, node_j) = (str(int(float(node_i))), str(int(float(node_j))))
        new_idx_i = map_new_old_keys[node_i]
        new_idx_j = map_new_old_keys[node_j]
        edg_list.append([new_idx_i, new_idx_j, float(w)])
      else:
        (node_i, node_j) = elements
        new_idx_i = map_new_old_keys[node_i]
        new_idx_j = map_new_old_keys[node_j]
        edg_list.append([new_idx_i, new_idx_j, 1])
    edg = np.array(edg_list)
    return edg, edg_num

  def get_k(self, Y_ori):
    """
      get the number of classes: k
    """
    k = len(set(Y_ori))
    return k

  def read_stat_file(self, filename):
    stats = open(filename, "r")
    test_ratio = None
    for l in stats:
      if "test_ratio" in l:
        elements = l.strip().split(": ")
        test_ratio = elements[1]
    return test_ratio

  def edge_list_to_adjacency_matrix(self, edg_list, n):
    A = np.zeros((n,n))
    for [i, j, w] in edg_list:
      i = int(i)
      j = int(j)
      if A[i,j] != w:
        A[i,j] = w
      if A[j,i] != w:
        A[j,i] = w
    return A

  def mark_unknown_labels(self, Y_ori, test_idx):
      Y = Y_ori.copy()
      # mark the test position as unknown: -1
      Y[test_idx, 0] = -1
      return Y

#-----------------input_data_from_file_end-----------------------------------#
