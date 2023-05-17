# -*- coding: utf-8 -*-
################################################################################
import sys
from GEE_sparse import *
from GEE_edge_list import *
from simulated_cases.cases import Case
################################################################################
def main():
  emb = sys.argv[1]  # can be "sparse" or "edglist" or "compare"
  opts = {"Laplacian": False, "DiagA": False, "Correlation": False}
  run(emb, opts)


################################################################################
def run(emb, opts):
  case = Case(1000)
  test_case = case.case_paper_SBM()   
  if emb == "sparse":  
    Z, W, emb_time = run_sparse(test_case, opts)

    print(f"Embed matrix: \n{Z}")
    print(f"Weight matrix: \n{W}")    
    print(f"GEE_sparse embed time: {emb_time}")  

  elif emb == "edglist":
    Z, W, emb_time = run_edglist(test_case, opts)

    print(f"Embed matrix: \n{Z}")
    print(f"Weight matrix: \n{W}")    
    print(f"GEE_edg_list embed time: {emb_time}")      

  elif emb == "compare":
    emb_time_s, emb_time_e = run_together(test_case, opts)
    print(f"embed using GEE_sparse: {emb_time_s}")
    print(f"embed using GEE_edg_list: {emb_time_e}")
  
  else:
    print("""
    please choose one of these below to run the program:
    1. python3 compare.py sparse
    2. python3 compare.py edglist
    3. python3 compare.py compare
    """)
# -----------------------------------------------------------------------------#        
def run_sparse(test_case, opts):
  GEE = GraphEncoderEmbed_sparse()
  Z, W, emb_time = GEE.run(test_case.X, test_case.Y, test_case.n, Laplacian = opts["Laplacian"], 
                                    DiagA = opts["DiagA"], Correlation = opts["Correlation"], EdgeList = True)        
  return Z, W, emb_time

# -----------------------------------------------------------------------------#        
def run_edglist(test_case, opts):
  GEE = GraphEncoderEmbed_Edge()
  Z, W, emb_time = GEE.run(test_case.X, test_case.Y, test_case.n, Laplacian = opts["Laplacian"], 
                                    DiagA = opts["DiagA"], Correlation = opts["Correlation"])
  return Z, W, emb_time  

# -----------------------------------------------------------------------------#        
def run_together(test_case, opts):
  GEE_sparse = GraphEncoderEmbed_sparse()
  _Zs, _Ws, emb_time_s = GEE_sparse.run(test_case.X, test_case.Y, test_case.n, Laplacian = opts["Laplacian"], 
                                    DiagA = opts["DiagA"], Correlation = opts["Correlation"], EdgeList = True)       
  GEE_edg = GraphEncoderEmbed_Edge()
  _Ze, _We, emb_time_e = GEE_edg.run(test_case.X, test_case.Y, test_case.n, Laplacian = opts["Laplacian"], 
                                    DiagA = opts["DiagA"], Correlation = opts["Correlation"]) 
  
  
  return emb_time_s, emb_time_e  
################################################################################
if __name__ == "__main__":
    main()