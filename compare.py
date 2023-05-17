# -*- coding: utf-8 -*-
################################################################################
from utils import DataSet
from GEE_sparse import *
# from GEE_edge_list import *
from other_versions.GEE_edge_list_for_sparse import *
from GNN import *
import os
import itertools
from simulated_cases.cases import Case
################################################################################
def main():

    #-----set_outfolder_start---------#
    output_folder = "results"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    #-----set_outfolder_end-----------#


    #-----set_case_and_node_start-----#
    case_name_list = ['case10']   #['case10', 'case11', 'case20', 'case21'] #['case21']
    node_num_list = [10]   #[100, 1000, 3000, 5000, 10000] #100000 is too large, will be kiiled if run

    # real casea
    # case_name_list = ['citeseer', 'cora', 'proteins-all', 'PubMed', "CL-100K-1d8-L9"]
    # node_num_list = [3264, 2708, 43471, 19717, 92482]
    # case_name_list = ['CL-10M-1d8-L5']      
    # node_num_list = [10000000]    
    #-----set_case_and_node_end-----#


    #-----set_options_start---------#
    # opt_all = generate_opts()

    # opt_all = [{"Laplacian": True, "DiagA": True, "Correlation": True, "Weight": 0},
    #            {"Laplacian": True, "DiagA": True, "Correlation": False, "Weight": 0},
    #            {"Laplacian": True, "DiagA": False, "Correlation": True, "Weight": 0},
    #            {"Laplacian": True, "DiagA": False, "Correlation": False, "Weight": 0},
    #            {"Laplacian": False, "DiagA": True, "Correlation": True, "Weight": 0},
    #            {"Laplacian": False, "DiagA": True, "Correlation": False, "Weight": 0},
    #            {"Laplacian": False, "DiagA": False, "Correlation": True, "Weight": 0},
    #            {"Laplacian": False, "DiagA": False, "Correlation": False, "Weight": 0}]

    opt_all = [{"Laplacian": True, "DiagA": True, "Correlation": True, "Weight": 0}]
    # opt_all = [{"Laplacian": False, "DiagA": False, "Correlation": False, "Weight": 0}]
    #-----set_options_end-----------#


    #----GEE_NN_LDA_choice_start----#
    # choice = "GEE+NN"
    choice = "GEE_only"
    #----GEE_NN_LDA_choice_end------#


    #----GEE_sparse_start-----------#
    gee_flag = "sparse"
    out_file_name = "model_sparse_case_10000_node"
    run(gee_flag, case_name_list, node_num_list, opt_all, output_folder, out_file_name, choice)
    #----GEE_sparse_end-------------#


    #----GEE_edgList_start----------#
    # gee_flag = "Edge_list"
    # out_file_name = "model_edge_10000_node_test"
    # run(gee_flag, case_name_list, node_num_list, opt_all, output_folder, out_file_name, choice)
     #----GEE_edgList_end-----------#


################################################################################
def run(gee_flag, case_name_list, node_num_list, opt_all, output_folder, out_file_name, choice):
    # real_case or simulated case
    case_flag = "real"
    # case_flag = "simulated"

    # supervised
    class_flag = "supervised"
    # # --- from files ----
    # run_from_files(case_flag, class_flag, gee_flag, case_name_list, node_num_list, opt_all, output_folder, out_file_name, choice)

    # ---- from models -----
    run_from_models(gee_flag, case_name_list, node_num_list,  opt_all, output_folder, out_file_name, choice)

# -----------------------------------------------------------------------------#
def run_from_files(case_flag, class_flag, gee_flag, case_name_list, node_num_list,  opt_all, output_folder, out_file_name, choice):
    if gee_flag == "sparse":
        GEE = GraphEncoderEmbed_sparse()
        f = open(f"{output_folder}/{out_file_name}.txt", "w")
    if gee_flag == "Edge_list":
        GEE = GraphEncoderEmbed_Edge()
        f = open(f"{output_folder}/{out_file_name}.txt", "w")
    GEE_with_case_from_files(GEE, case_flag, class_flag, case_name_list, node_num_list, opt_all, f, choice)
    f.close()

# -----------------------------------------------------------------------------#
def GEE_with_case_from_files(GEE, case_flag, class_flag, case_name_list, node_num_list, opt_all, f, choice):
    if case_flag == "real":
        in_main_folder = "real_sets"
        
        if class_flag == "supervised":
            in_folder = f"{in_main_folder}/supervised"

        for i in range(len(node_num_list)):
            node_num = node_num_list[i]
            case_name = case_name_list[i]
            f.write(f'{case_name}_{node_num}\n')

            test_case = test_case_from_files(in_folder, case_name, node_num)
            # GEE only
            if choice ==  "GEE_only":
                GEE_to_result(test_case, opt_all, f, GEE)

            # GEE + NN
            if choice == "GEE+NN":
                GEE_NN_to_result(test_case, opt_all, f, GEE)   
    
    
    else:
        in_main_folder = "simulated_cases"

        if class_flag == "supervised":
            in_folder = f"{in_main_folder}/supervised"

        for node_num in node_num_list:
            for case_name in case_name_list:
                f.write(f'{case_name}_{node_num}\n')

                test_case = test_case_from_files(case_flag, in_folder, case_name, node_num)
                # GEE only
                if choice ==  "GEE_only":
                    GEE_to_result(test_case, opt_all, f, GEE)

                # GEE + NN
                if choice == "GEE+NN":
                    GEE_NN_to_result(test_case, opt_all, f, GEE)

# -----------------------------------------------------------------------------#
def run_from_models(gee_flag, case_name_list, node_num_list,  opt_all, output_folder, out_file_name, choice):
    if gee_flag == "sparse":
        GEE = GraphEncoderEmbed_sparse()
        f = open(f"{output_folder}/{out_file_name}.txt", "w")
    if gee_flag == "Edge_list":
        GEE = GraphEncoderEmbed_Edge()
        f = open(f"{output_folder}/{out_file_name}.txt", "w")
    GEE_with_case_from_models(GEE, case_name_list, node_num_list,  opt_all, f, choice)
    f.close()

# -----------------------------------------------------------------------------#
def GEE_with_case_from_models(GEE, case_name_list, node_num_list,  opt_all, f, choice):

    for node_num in node_num_list:
        case = Case(node_num)
        for case_name in case_name_list:
            f.write(f'{case_name}_{node_num}\n')
            test_case = test_case_from_select_model(case, case_name)

            # GEE only
            if choice ==  "GEE_only":
                GEE_to_result(test_case, opt_all, f, GEE)

            # GEE + NN
            if choice == "GEE+NN":
                GEE_NN_to_result(test_case, opt_all, f, GEE)

# -----------------------------------------------------------------------------#
def test_case_from_select_model(case, case_name):
    if case_name == "case10":
        test_case = case.case_10_fully_known()
    if case_name == "case11":
        test_case = case.case_11_fully_known()
    if case_name == "case20":
        test_case = case.case_20_fully_known()
    if case_name == "case21":
        test_case = case.case_21_fully_known()
    if case_name == "case_paper_SBM":
        test_case = case.case_paper_SBM()
    return test_case

# -----------------------------------------------------------------------------#
def GEE_to_result(test_case, opt_all, f, GEE):
    for opt in opt_all:
        f.write(f'Laplacian = {opt["Laplacian"]}, DiagA = {opt["DiagA"]}, Correlation = {opt["Correlation"]}, Weight = {opt["Weight"]}\n')
        # print(f'Laplacian = {opt["Laplacian"]}, DiagA = {opt["DiagA"]}, Correlation = {opt["Correlation"]}, Weight = {opt["Weight"]}')
        Z, W, emb_time = GEE.run(test_case.X, test_case.Y, test_case.n, Laplacian = opt["Laplacian"], DiagA = opt["DiagA"], Correlation = opt["Correlation"], Weight = opt["Weight"], EdgeList = True)

        f.write(f'emb\t{emb_time}s\n')
        f.write(f'edg_num\t{test_case.edg_num}\n')

# -----------------------------------------------------------------------------#
def GEE_NN_to_result(test_case, opt_all, f, GEE):
    for opt in opt_all:
        f.write(f'Laplacian = {opt["Laplacian"]}, DiagA = {opt["DiagA"]}, Correlation = {opt["Correlation"]}, Weight = {opt["Weight"]}\n')
        # print(f'Laplacian = {opt["Laplacian"]}, DiagA = {opt["DiagA"]}, Correlation = {opt["Correlation"]}, Weight = {opt["Weight"]}')
        Z, W, emb_time = GEE.run(test_case.X, test_case.Y, test_case.n, Laplacian = opt["Laplacian"], DiagA = opt["DiagA"], Correlation = opt["Correlation"], Weight = opt["Weight"], EdgeList = True)
        test_case = repack_test_case(test_case, Z)

        gnn = GNN(test_case)
        acc, train_time  = gnn.GNN_run("direct")

        f.write(f'emb\t{emb_time}s\n')
        f.write(f'train\t{train_time}s\n')
        f.write(f'acc\t{acc}\n')
        f.write(f'edg_num\t{test_case.edg_num}\n')

# -----------------------------------------------------------------------------#
def generate_opts():
    options_1 = ["Laplacian", "DiagA", "Correlation"]
    choices_1 = [True, False]

    options_2 = ["Weight"]
    choices_2 = [0, 1, 2]
    opt_all = []

    options_all = options_1 + options_2
    for p in itertools.product(choices_1,repeat=len(options_1)):
        for cho in choices_2:
            p_all = p + (cho,)
            params = dict(zip(options_all,p_all))
            opt_all.append(params)

    return opt_all


# -----------------------------------------------------------------------------#
def test_case_from_files(in_folder, case_name, node_num):
    input_folder = f"{in_folder}/{case_name}_{node_num}_nodes"
    edg_file = f"{input_folder}/{case_name}_edges.tsv"
    node_ori_file = f"{input_folder}/{case_name}_node_labels_ori.tsv"
    node_train_file = f"{input_folder}/{case_name}_node_labels_train.tsv"
    node_test_file = f"{input_folder}/{case_name}_node_labels_test.tsv"
    stats_file = f"{input_folder}/{case_name}_stats.txt"

    test_case = DataSet().get_initial_values(edg_file, node_ori_file, node_train_file, node_test_file, stats_file)

    return test_case


# -----------------------------------------------------------------------------#
def repack_test_case(test_case, Z):
    train_idx = test_case.train_idx
    test_idx = test_case.test_idx
    test_case.z_train= Z[train_idx]
    test_case.z_test = Z[test_idx]
    test_case.y_train = test_case.Y_train
    test_case.y_test = test_case.Y_test

    return test_case


# -----------------------------------------------------------------------------#

################################################################################
if __name__ == "__main__":
    main()
