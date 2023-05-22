# -*- coding: utf-8 -*-
################################################################################
import sys
import numpy as np
import copy
from tensorflow import keras
from keras.utils import to_categorical
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time

#early stop
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from scipy import sparse

from GEE_sparse import *
from simulated_cases.cases import Case
################################################################################
def main():
  clf = sys.argv[1]  # can be "LDA" or "NN"

  opts = {"Laplacian": True, "DiagA": True, "Correlation": True}
  case = Case(3000)
  test_case = case.case_10_fully_known() 

  GEE = GraphEncoderEmbed_sparse()
  Z, W, emb_time = GEE.run(test_case.X, test_case.Y, test_case.n, Laplacian = opts["Laplacian"], 
                                    DiagA = opts["DiagA"], Correlation = opts["Correlation"], EdgeList = True)        
  
  test_case = dataset_preprocess(test_case, Z)

  if clf == "NN":
    gnn = GNN(test_case)
    acc, train_time  = gnn.GNN_run("direct")

  if clf == "LDA":
    lda = LDA(test_case)
    acc, train_time  = lda.LDA_run()

  print("--- embed %s seconds ---" % emb_time)  
  print("--- train %s seconds ---" % train_time)
  print("--- accuracy: %s ---" % acc)   
################################################################################
#---------------------------------pre_prep_start-------------------------------#
def dataset_preprocess(dataset, Z):
  train_idx = dataset.train_idx
  test_idx = dataset.test_idx
  dataset.z_train= Z[train_idx]
  dataset.z_test = Z[test_idx]
  dataset.y_train = dataset.Y[train_idx].ravel()
  dataset.y_test = dataset.Y_test.ravel() 
  if sparse.issparse(dataset.z_train):
      dataset.z_train = dataset.z_train.toarray()

  if sparse.issparse(dataset.z_test):
      dataset.z_test = dataset.z_test.toarray()
    
  return dataset
#---------------------------------pre_prep_end---------------------------------#
#---------------------------------GNN_start------------------------------------#

# https://www.kaggle.com/c/talkingdata-mobile-user-demographics/discussion/22567
# https://github.com/tkipf/pygcn/blob/1600b5b748b3976413d1e307540ccc62605b4d6d/pygcn/utils.py#L73

def batch_generator(X, y, k, batch_size, shuffle):
    number_of_batches = int(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:]
        y_batch = to_categorical(y[batch_index], num_classes=k)
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

class Hyperperameters:
  """
    define perameters for GNN.
    default values are for GNN learning -- "Leaner" ==2:
      embedding via partial label, then learn unknown label via two-layer NN

  """
  def __init__(self):
    # there is no scaled conjugate gradiant in keras optimiser, use defualt instead
    # use whatever default
    self.learning_rate = 0.01  # Initial learning rate.
    self.epochs = 100 #Number of epochs to train.
    self.hidden = 20 #Number of units in hidden layer 
    self.val_split = 0.1 #Split 10% of training data for validation
    self.loss = 'categorical_crossentropy' # loss function

class GNN:
  def __init__(self, DataSets):
    GNN.DataSets = DataSets
    GNN.hyperM = Hyperperameters()
    GNN.model = self.GNN_model()  #model summary: GNN.model.summary()
      
 
  def GNN_model(self):
    """
      build GNN model
    """
    hyperM = self.hyperM
    DataSets = self.DataSets

    z_train = DataSets.z_train
    k = DataSets.d

    feature_num = z_train.shape[1]
    
    model = keras.Sequential([
    keras.layers.Flatten(input_shape = (feature_num,)),  # input layer 
    keras.layers.Dense(hyperM.hidden, activation='relu'),  # hidden layer -- no tansig activation function in Keras, use relu instead
    keras.layers.Dense(k, activation='softmax') # output layer, matlab used softmax for patternnet default ??? max(opts.neuron,K)? opts 
    ])

    optimizer = keras.optimizers.Adam(learning_rate = hyperM.learning_rate)

    model.compile(optimizer='adam',
                  loss=hyperM.loss,
                  metrics=['accuracy'])

    return model
    
  def GNN_run(self, flag):
    """
      Train and test directly.
      Do not learn from the unknown labels.
    """
    gnn = copy.deepcopy(self)
    hyperM = gnn.hyperM
    DataSets = self.DataSets
    k = DataSets.d
    z_train = DataSets.z_train
    y_train = DataSets.y_train
    y_test = DataSets.y_test
    z_test = DataSets.z_test
    model = gnn.model    

    # no batch generator
    if flag == "direct":
      y_train_one_hot = to_categorical(y_train, num_classes=k)
      train_strat = time.time() 
      history = model.fit(z_train, y_train_one_hot, 
        validation_split=hyperM.val_split,
        epochs=hyperM.epochs, 
        shuffle=True,
        verbose=0)
    else:
      early_stopping_callback = EarlyStopping(monitor='loss', patience=5, verbose=0)
      checkpoint_callback = ModelCheckpoint('GNN.h5', monitor='loss', save_best_only=True, mode='min', verbose=0)
      
      train_strat = time.time()
      history = model.fit(batch_generator(z_train, y_train, k, 32, True),
                      epochs=hyperM.epochs,
                      steps_per_epoch=z_train.shape[0],
                      callbacks=[early_stopping_callback, checkpoint_callback],
                      verbose=0)
    train_end = time.time()
    train_time = train_end - train_strat 

    y_test_one_hot = to_categorical(y_test, num_classes=k) 
    # set verbose to 0 to silent the output
    test_loss, test_acc = gnn.model.evaluate(z_test,  y_test_one_hot, verbose=0) 
    return test_acc, train_time
#---------------------------------GNN_end--------------------------------------#
#---------------------------------LDA_start------------------------------------#
class LDA:
  def __init__(self, DataSets):
    LDA.DataSets = DataSets
    LDA.model = LinearDiscriminantAnalysis()  # asssume spseudolinear is its default setting
    
  def LDA_run(self):
    lda = copy.deepcopy(self)
    DataSets = lda.DataSets
    z_train = DataSets.z_train
    y_train = DataSets.Y_train
    z_test = DataSets.z_test
    y_test = DataSets.y_test

    model = self.model
    train_strat = time.time()
    model.fit(z_train,y_train)
    train_end = time.time()
    train_time = train_end - train_strat 
    
    test_acc = model.score(z_test, y_test)

    return test_acc, train_time
#---------------------------------LDA_end------------------------------------#
################################################################################
if __name__ == "__main__":
  main()
  

  
  
