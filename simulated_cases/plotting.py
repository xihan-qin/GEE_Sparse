################################################################################
import matplotlib.pyplot as plt
import numpy as np
################################################################################
class plot_graph:
    def block_prob(self, Bl, ax=None):
        plot_block_probabilities(Bl, ax=None)
        plt.show()
    
    def reord_Adj(self, A_reorder, block_idx, ax=None):
        plot_reordered_Adj(A_reorder, block_idx, ax=None)
        plt.show()

    def labels(self, y, bl):
        plot_labels(y, bl)
        plt.show()
    
    def pp(self, bl, pp):
        label_pp_pie(bl, pp)
        plt.show()

    def two_plots(self, A,y,bl):
        """
            given a adjacency matrix, its node labels, and its block probability matrix
            reorder the adjacency matrix by its node labels
            generate two plots 
            Left plot is for reordered adjacency matrix
            Right plot is for block probability matrix
        """       
        A_reorder, block_idx = re_order_by_label(A, y)

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(15, 6))
        
        plot_reordered_Adj(A_reorder, block_idx, ax = ax1)
        plot_block_probabilities(bl, ax = ax2)
        fig.tight_layout()
        plt.show()

    def four_plots(self, A,y,bl,pp, fig_name):     
        A_reorder, block_idx = re_order_by_label(A, y)

        plt.figure(figsize=(13, 12))
        ax1 = plt.subplot2grid((7, 8), (0, 0), colspan=4, rowspan=4)
        ax2 = plt.subplot2grid((7, 8), (0, 4), colspan=4, rowspan=4)
        ax3 = plt.subplot2grid((7, 8), (4, 0), colspan=4, rowspan=3)
        ax4 = plt.subplot2grid((7, 8), (4, 4), colspan=3, rowspan=3)
        
        plot_reordered_Adj(A_reorder, block_idx, ax = ax1)
        plot_block_probabilities(bl, ax = ax2)
        plot_labels(y,bl,ax = ax3)
        label_pp_pie(bl, pp, ax = ax4)
        
        plt.tight_layout()
        plt.savefig(f"{fig_name}")
        
        

# -----------------------------------------------------------------------------#
# ---------------------block_prob_start----------------------------------------#
# @auto_plot
def plot_block_probabilities(Bl, ax=None):
    """
        given a block probabilities matrix and plot a heatmap for it       
    """
    if ax is None:
        ax = plt.gca() # Get the current Axes instance on the current figure matching the given keyword args, or create one.
    im = ax.imshow(Bl, cmap="Greys", interpolation='none')
    block_num =  Bl.shape[0]
    block_labels = ["community_" + str(i) for i in range(block_num)]
    # set the limits of the plot to the limits of the data
    ax.set_xticks(np.arange(block_num), labels = block_labels, fontsize=13)
    ax.set_yticks(np.arange(block_num), labels = block_labels, fontsize=13) 

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")    

    # Loop over data dimensions and create text annotations.
    for i in range(block_num):
        for j in range(block_num):
            text = ax.text(j, i, Bl[i, j],
                        ha="center", va="center", color="gray")
    
    block_idices = [i for i in range(block_num)]
    # Draw block lines
    draw_block_lines(block_idices, ax)

    ax.set_title("Blcok probabilities",  fontsize=16)

    # create colorbar
    cbar = ax.figure.colorbar(im, cmap="Greys")
    cbar.ax.set_ylabel("block_probabilities", rotation=-90, va="bottom", fontsize=13)
    
   
# -----------------------------------------------------------------------------#
def draw_block_lines(block_idices, ax):
    n = block_idices[-1]
    for crent_bl_idx in block_idices[:-1]:
        # draw horriline, adjust 0.5 so that the line will not go through the squares
        ax.axline((0, crent_bl_idx+0.5), (n,crent_bl_idx+0.5), color="gray", linestyle = "-.")
        # draw vertiline
        ax.axline((crent_bl_idx+0.5, 0), (crent_bl_idx+0.5, n), color="gray", linestyle = "-.")

# ---------------------block_prob_end------------------------------------------#
# ---------------------reord_Adj_start-----------------------------------------#
# @auto_plot
def plot_reordered_Adj(A_reorder, block_idx, ax=None):
    """
        given reordered adjacency matrix and a dictionary that indicates the end index for each block.
        plot a heatmap based on the reordered adjacency matrix
    """
    if ax is None:
        ax = plt.gca() # Get the current Axes instance on the current figure matching the given keyword args, or create one.

    ax.imshow(A_reorder, cmap="Greys", interpolation='none')
    
    blocks = list(block_idx.keys())
    block_num = len(blocks)
    block_labels = ["community_" + str(i) for i in blocks]

    block_idices = [v for k,v in block_idx.items()]
    n = A_reorder.shape[0]

    label_idx = []
    pre_idx = 0
    for crent_idx in block_idices:
        label_idx.append(int((crent_idx + pre_idx)/2))
        pre_idx = crent_idx
        
    # set the limits of the plot to the limits of the data
    ax.set_xticks(label_idx, labels = block_labels, fontsize=13)
    ax.set_yticks(label_idx, labels = block_labels, fontsize=13)     

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")    

    # Draw block lines
    draw_block_lines(block_idices, ax)

    ax.set_title("Adjacnecy Matrix reordered by node labels",  fontsize=16)

# -----------------------------------------------------------------------------#
def re_order_by_label(A, y):
    """
        input adjacency matrix and its labels
        reorder the matrix by the labels.
    """
    n = A.shape[0]
    y_reorder = np.sort(y)
    labels = list(set(y_reorder.flatten()))
    
    label_count_dict = {k:0 for k in labels} # count the number of nodes for each label

    for l in y_reorder:
        # numpy array with shape (n,1), thus is iterating rows
        label_count_dict[l[0]] += 1
    
    
    block_idx = {labels[0]:0} # the start_idx for each block after the loop below. 
    for i in range(1, len(labels)):
        cnt = 0
        for j in range(i):
            cnt += label_count_dict[j]
        block_idx[i] = cnt 

    node_idx_map = {} # {idx_old: idx_new} map the ord idx in A to new idx in A_reorder.
    for node in range(len(y)):
        block = y[node,0]
        idx = block_idx[block]
        node_idx_map[node] = idx
        block_idx[block] += 1 
    
    # adjust the extra one idx back
    block_idx = {k:v-1 for k,v in block_idx.items()} #Now it becomes the end_idx for each block after the next loop.

    A_reorder = np.zeros((n,n), dtype=int)
    
    for node_i in range(n):            
        for node_j in range(i+1,n):
            A_reorder[node_idx_map[node_i], node_idx_map[node_j]] = A[node_i,node_j]
            A_reorder[node_idx_map[node_j], node_idx_map[node_i]] = A_reorder[node_idx_map[node_i], node_idx_map[node_j]]
    
    return A_reorder, block_idx

# ---------------------reord_Adj_end-------------------------------------------#

# ---------------------labels_his_start----------------------------------------#
def plot_labels(Y, Bl, ax= None):
    """
    size in inches 
    """
    if ax is None:
        ax = plt.gca() # Get the current Axes instance on the current figure matching the given keyword args, or create one.

    label_vs_count = count_labels(Y)
    # Get the keys of the dictionary and sorted the values by increasing order
    x_list = list(label_vs_count)
    x_list.sort()

    y_list = []
    for x in x_list:
        y_list.append(label_vs_count[x])

    block_num =  Bl.shape[0]
    block_labels = ["community_" + str(i) for i in range(block_num)]

    # plot the histogram
    x = np.arange(len(x_list))  # the label locations
    width = 0.35  # the width of the bars
    rects = ax.bar(x, y_list, width)

    ax.set_xticks(x)
    ax.set_xticklabels(x_list)
    ax.set_xticklabels(block_labels)    
    ax.set_ylabel('Count', fontsize=15)
    ax.set_xlabel('Label', fontsize=15)
    autolabel(ax, rects)

    ax.set_title("Community Label vs Node Count", fontsize=16)

    # x, y = size
    # fig.set_size_inches(x, y)
    # fig.tight_layout()
# -----------------------------------------------------------------------------#
def count_labels(Y):
    Y = Y[:,0]
    unique_y = set(Y)
    label_vs_count = {}
    for label in unique_y:
        # find a value in 1 D Numpy array
        # return a tuple of an array and dtype. 
        # i.e.(array([110, 111, 219], dtype=int64),)
        ind_tuple = np.where(Y == label)
        # get the list of indices.
        ind_list = list(ind_tuple[0])
        label_vs_count[label] = len(ind_list)
    return label_vs_count

# -----------------------------------------------------------------------------#
def autolabel(figure, rects):
    """
        Attach a text label above each bar in *rects*, displaying its height.
    """
    for rect in rects:
        height = rect.get_height()
        figure.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

# ---------------------labels_his_end------------------------------------------#
# ---------------------labels_pie_start----------------------------------------#
def label_pp_pie(Bl, pp, ax = None):
    if ax is None:
        ax = plt.gca() # Get the current Axes instance on the current figure matching the given keyword args, or create one.
    
    block_num =  Bl.shape[0]
    block_labels = ["community_" + str(i) for i in range(block_num)]

    sizes = [i*100 for i in pp]
    
    ax.pie(sizes, labels=block_labels, autopct='%1.2f%%',
        startangle=90, textprops={'fontsize': 14})
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title("Initial Community label percentage before sampling", fontsize=16)
# ---------------------labels_pie_end------------------------------------------#

###------------back_up_start-------------------------------------------------###
# -------------functions_to_explore_some_properties_start----------------------#
def try_different_interpolation_methods(grid):
    """
        given a grid and then plot with different interpolation methods
        example: input a block probability matrix
    """
    methods = [None, 'none', 'nearest', 'bilinear', 'bicubic', 'spline16',
            'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric',
            'catrom', 'gaussian', 'bessel', 'mitchell', 'sinc', 'lanczos']

    fig, axs = plt.subplots(nrows=3, ncols=6, figsize=(9, 6),
                        subplot_kw={'xticks': [], 'yticks': []})
    for ax, interp_method in zip(axs.flat, methods):
      ax.imshow(grid, interpolation=interp_method, cmap='binary') 
      ax.set_title(str(interp_method))      
    plt.tight_layout()
    plt.show()
# -------------functions_to_explore_some_properties_end------------------------#
# -------------Wrapper_function_start------------------------------------------#
def auto_plot(plot_function):
    """
        Wrapper function for showing plot.
    """
    def wrapper(*args, **kwargs):
        plot_function(*args, **kwargs)
        plt.show()
    return wrapper
# -------------Wrapper_function_end--------------------------------------------#
###------------back_up_end---------------------------------------------------###