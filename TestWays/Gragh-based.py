import matplotlib.pyplot as plt
import numpy as np
from skimage import data, segmentation, color
from skimage.future import graph
from skimage.measure import label
import networkx as nx
from skimage.segmentation import slic
from skimage.future import graph


def merge_nodes_weight(graph, src, dst, n):
    default = {'weight': 0.0}
    edge_data = graph[src].get(dst, default)
    return {'weight': edge_data['weight'] / n}


def plot_saliency(img, segments):
    # 根据边界权重计算显著性
    g = graph.rag_mean_color(img, segments)
    for edge in g.edges():
        n = g[edge[0]][edge[1]]['count']
        g[edge[0]][edge[1]]['weight'] = merge_nodes_weight(g, edge[0], edge[1], n)['weight']

    saliency_map = np.zeros(segments.shape)
    for region in np.unique(segments):
        node_neighbors = list(g.neighbors(region))
        edges = [(region, neighbor) for neighbor in node_neighbors]
        edge_weights = [g[u][v]['weight'] for u, v in edges]
        saliency_map[segments == region] = np.mean(edge_weights)

    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))
    plt.imshow(saliency_map, cmap='hot')
    plt.colorbar()
    plt.title("Graph-based Saliency")
    plt.show()


# Load image
img = data.coffee()
segments = segmentation.slic(img, n_segments=250, compactness=10, sigma=1)
plot_saliency(img, segments)
