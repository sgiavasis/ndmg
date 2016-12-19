import os
import networkx as nx
import matplotlib
matplotlib.use('Agg')  # very important above pyplot import
import matplotlib.pyplot as plt
import numpy as np

def graph2png(path_to_graph_dir):
        if (not os.path.isdir('graph_png/')):
            os.mkdir('graph_png/')
        for f in os.listdir(path_to_graph_dir):
            if (os.path.isdir(path_to_graph_dir + f)): 
                continue
            try:
                graph = nx.read_gpickle(path_to_graph_dir + f)
            except:
                graph = nx.read_graphml(path_to_graph_dir + f)                                       
            g = nx.adj_matrix(graph).todense()
            fig = plt.figure(figsize=(7,7))
            p = plt.imshow(g, interpolation='None', cmap='jet')
            save_location = 'graph_png/' + os.path.splitext(f)[0] + '.png'
            plt.savefig(save_location, format='png')
            print('done!')

if __name__ == "__main__":
    graph2png('/brainstore/MR/data/BNU1/ndmg_v0033/graphs/desikan/')
    graph2png('/brainstore/MR/data/HNU1/ndmg_v0033/graphs/desikan/')
    graph2png('/brainstore/MR/data/NKI1/ndmg_v0033/graphs/desikan/')
