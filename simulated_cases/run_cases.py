################################################################################
from cases import *
from plotting import *
# # from community import community_louvain

# import networkx as nx

################################################################################
def main():
    n_list = [100, 1000, 3000, 5000, 10000]
    for n in n_list:
        run(n)
################################################################################
def generate_results(test_case, outfolder):
    test_case.output_files(outfolder)

    A = test_case.X

    y = test_case.Y_ori
    bl = test_case.bl
    pp = test_case.pp

    fig_name = f"{outfolder}/{test_case.name}_{test_case.n}_nodes/{test_case.name}_four_plots"

    plotg = plot_graph()
    plotg.four_plots(A, y, bl, pp, fig_name)

# -----------------------------------------------------------------------------#
def run(n):
    case = Case(n)
    
    outfolder = "supervised"
    if not os.path.exists(outfolder):
      os.makedirs(f"{outfolder}")

    test_case = case.case_10_fully_known()
    generate_results(test_case, outfolder)

    test_case = case.case_11_fully_known()
    generate_results(test_case, outfolder)


    outfolder = "semi_supervised"
    if not os.path.exists(outfolder):
      os.makedirs(f"{outfolder}")
    test_case = case.case_10()
    generate_results(test_case, outfolder)

    test_case = case.case_11()
    generate_results(test_case, outfolder)
################################################################################
if __name__ == "__main__":
    main()
    