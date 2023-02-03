
# -*- coding: utf-8 -*-
"""
most used cases in the one-hot GEE paper and its related experiments
case_10, case_11, case_20, case_21
"""

################################################################################
from .simulated_models import *
################################################################################

class Case(Model):
###-------------------case_10_start------------------------------------------###
  def case_10(self):
    """
        3 X 3 sized SBM model with block probabilities
        0.13 0.1  0.1
        0.1  0.13 0.1
        0.1  0.1  0.13
    pp is used to generate y labels
    [0.2,0.3,0.5]
    """
    d = 3
    pp = [0.2,0.3,0.5]

    # posibilities between classes including the classes with themselves
    # bd is the probability within class
    bd = 0.13  #0.13 anything above 0.15 should be high, close to 0.1 should be low

    Bl = np.full((d, d), 0.1)
    np.fill_diagonal(Bl, bd)

    case = self.SBM(d, pp, Bl)
    # case = case.add_unknown(0.95)
    case = case.split_sets(0.95)

    case.bd = bd
    case.name = "case10"
    case.info = """
    SBM with 3 classes and defined probabilities with 95% unknown labels.
    """
    return case

  def case_10_1(self):
    d = 3
    pp = [0.2,0.3,0.5]

    # posibilities between classes including the classes with themselves
    # bd is the probability within class
    bd = 0.13  #0.13 anything above 0.15 should be high, close to 0.1 should be low

    Bl = np.full((d, d), 0.1)
    np.fill_diagonal(Bl, bd)

    case = self.SBM(d, pp, Bl)
    case = case.add_unknown(0.05)

    case.bd = bd
    case.name = "case10"   
    case.info = """
    SBM with 3 classes and defined probabilities with 5% unknown labels.
    """
    return case    

  def case_10_fully_known(self):
    d = 3
    pp = [0.2,0.3,0.5]

    # posibilities between classes including the classes with themselves
    # bd is the probability within class
    bd = 0.13  #0.13 anything above 0.15 should be high, close to 0.1 should be low

    Bl = np.full((d, d), 0.1)
    np.fill_diagonal(Bl, bd)

    case = self.SBM(d, pp, Bl)
    case = case.split_sets(0.2)

    case.bd = bd
    case.name = "case10"
    case.info = """
    SBM with 3 classes and defined probabilities with fully known labels
    80% for training and 20% for testing
    """
    return case

  def case_10_cluster(self):
    d = 3
    pp = [0.2,0.3,0.5]

    # posibilities between classes including the classes with themselves
    # bd is the probability within class
    bd = 0.13  #0.13 anything above 0.15 should be high, close to 0.1 should be low

    Bl = np.full((d, d), 0.1)
    np.fill_diagonal(Bl, bd)

    case = self.SBM(d, pp, Bl)
    case = case.for_cluster()

    case.bd = bd
    case.name = "case10"  
    case.info = """
    SBM with 3 classes for clustering
    """
    return case
###-------------------case_10_end--------------------------------------------###
###-------------------case_11_start------------------------------------------###
  def case_11(self):
    d = 5
    pp = [1/d]*d

    # posibilities between classes including the classes with themselves
    bd = 0.2
    Bl = np.full((d, d), 0.1)
    np.fill_diagonal(Bl, bd)

    case = self.SBM(d, pp, Bl)
    # case = case.add_unknown(0.95)
    case = case.split_sets(0.95)

    case.bd = bd
    case.name = "case11"    
    case.info = """
    SBM with 5 classes and defined probabilities with 95% unknown labels.  
    """
    return case

  def case_11_fully_known(self):
    d = 5
    pp = [1/d]*d

    # posibilities between classes including the classes with themselves
    bd = 0.2
    Bl = np.full((d, d), 0.1)
    np.fill_diagonal(Bl, bd)

    case = self.SBM(d, pp, Bl)
    case = case.split_sets(0.2)

    case.bd = bd
    case.name = "case11"   
    case.info = """
    SBM with 5 classes and defined probabilities with fully known labels
    80% for training and 20% for testing
    """
    return case

  def case_11_cluster(self):
    d = 5
    pp = [1/d]*d

    # posibilities between classes including the classes with themselves
    bd = 0.2
    Bl = np.full((d, d), 0.1)
    np.fill_diagonal(Bl, bd)

    case = self.SBM(d, pp, Bl)
    case = case.for_cluster()
    
    case.bd = bd
    case.name = "case11"
    case.info = """
    SBM with 5 classes for clustering
    """
    return case

###-------------------case_11_end--------------------------------------------###
###-------------------case_20_start------------------------------------------###
  def case_20(self):
    d = 3
    pp = [0.2,0.3,0.5]

    # posibilities between classes including the classes with themselves
    bd = [0.9,0.5,0.2]
    Bl = np.full((d, d), 0.1)
    np.fill_diagonal(Bl, bd)

    case = self.DC_SBM(d, pp, Bl)
    case = case.add_unknown(0.95)
    
    case.bd = bd
    case.name = "case20"  
    case.info = """
    DC-SBM with 3 classes and defined probabilities with 95% unknown labels.
    """
    return case

  def case_20_fully_known(self):
    d = 3
    pp = [0.2,0.3,0.5]

    # posibilities between classes including the classes with themselves
    bd = [0.9,0.5,0.2]
    Bl = np.full((d, d), 0.1)
    np.fill_diagonal(Bl, bd)

    case = self.DC_SBM(d, pp, Bl)
    case = case.split_sets(0.2)
    
    case.bd = bd
    case.name = "case20" 
    case.info = """
    DC-SBM with 3 classes and defined probabilities with fully known labels
    80% for training and 20% for testing
    """
    return case

  def case_20_cluster(self):
    d = 3
    pp = [0.2,0.3,0.5]

    # posibilities between classes including the classes with themselves
    bd = [0.9,0.5,0.2]
    Bl = np.full((d, d), 0.1)
    np.fill_diagonal(Bl, bd)

    case = self.DC_SBM(d, pp, Bl)
    case = case.for_cluster()

    case.bd = bd
    case.name = "case20"   
    case.info = """
    DC-SBM with 3 classes for clustering
    """
    return case   
###-------------------case_20_end--------------------------------------------###
###-------------------case_21_start------------------------------------------###
  def case_21(self):
    d = 10
    pp = np.full((d, 1), 1/d)

    # posibilities between classes including the classes with themselves
    bd = 0.9
    Bl = np.full((d, d), 0.1)
    np.fill_diagonal(Bl, bd)

    case = self.DC_SBM_edg_list(d, pp, Bl)
    case = case.add_unknown(0.95)
    
    case.bd = bd
    case.name = "case21"
    case.info = """
    DC-SBM with 10 classes and defined probabilities with 95% unknown labels.
    Edge list version.     
    """
    return case

  def case_21_fully_known(self):
    d = 10
    pp = np.full((d, 1), 1/d)

    # posibilities between classes including the classes with themselves
    bd = 0.9
    Bl = np.full((d, d), 0.1)
    np.fill_diagonal(Bl, bd)

    case = self.DC_SBM(d, pp, Bl)
    case = case.split_sets(0.2)
    # case = case.to_edge_list()
    
    case.bd = bd
    case.name = "case21"
    case.info = """
    DC-SBM with 10 classes and defined probabilities with fully known labels.
    Edge list version. 
    """
    return case

  def case_21_cluster(self):
    d = 10
    pp = np.full((d, 1), 1/d)

    # posibilities between classes including the classes with themselves
    bd = 0.9
    Bl = np.full((d, d), 0.1)
    np.fill_diagonal(Bl, bd)

    case = self.DC_SBM_edg_list(d, pp, Bl)
    case = case.for_cluster()

    case.bd = bd
    case.name = "case21"
    case.info = """
    DC-SBM with 10 classes for clustering.
    Edge list version. 
    """
    return case     
###-------------------case_21_end--------------------------------------------###
###-------------------case_100_start-----------------------------------------###
  def case_100(self):
    """
        label probabilities: average, 1/3
    """
    d = 2
    pp = np.full((d, 1), 1/d)

    # posibilities between classes including the classes with themselves
    Bl = np.array([[0.050, 0.013], [0.013, 0.051]])

    case = self.SBM(d, pp, Bl)
    case = case.split_sets(0.2)
    
    case.name = "case100"
    case.info = """
    Affinity SBM from two-truth paper
    """
    return case
###-------------------case_100_end-------------------------------------------###
###-------------------case_paper_SBM_start-----------------------------------###
  def case_paper_SBM(self):
    """
        The paper uses 2000 nodes
        Two classes: 0.5 and 0.5
    """
    d = 2
    pp = np.full((d, 1), 1/d)

    # posibilities between classes including the classes with themselves
    Bl = np.array([[0.13, 0.10], [0.10, 0.13]])

    case = self.SBM(d, pp, Bl)
    case = case.split_sets(0.2)
    
    case.name = "case paper SBM"
    case.info = """
    SBM from GEE paper
    """
    return case
###-------------------case_paper_SBM_end-------------------------------------###
###-------------------case_paper_SBM_start-----------------------------------###
  def case_paper_DCSBM(self):
    """
        The paper uses 2000 nodes
        Two classes: 0.5 and 0.5
    """
    d = 2
    pp = np.full((d, 1), 1/d)

    # posibilities between classes including the classes with themselves
    Bl = np.array([[0.9, 0.1], [0.1, 0.5]])

    case = self.DC_SBM(d, pp, Bl)
    case = case.split_sets(0.2)
    
    case.name = "case paper DCSBM"  
    case.info = """
    DC-SBM from GEE paper
    """
    return case

###-------------------case_paper_SBM_end-------------------------------------###