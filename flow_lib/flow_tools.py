import numpy as np
import networkx as nx
import tqdm as tqdm
from collections import defaultdict
from kappa_tools import *
import sklearn as sk
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import igraph as ig
import leidenalg


def to_np(G:nx.multidigraph)->np.array:
    """Transforms a networkx graph into a numpy (adjacency) matrix

    Args:
        G (nx.multidigraph): networkx graph

    Returns:
        np.array: 
    """    
    edges = list(G.edges())#gets edges from generated network
    l = []
    [(l.append(x[0]), l.append(x[1])) for x in edges] #shape of matrix 
    m = max(l) +1
    A = np.zeros([m,m])

    for e in edges:
        A[e[0], e[1]] = 1


    return A

# Inducing flow
def flow_ij(G:nx.Graph,i:int,j:int) -> float:
    """Given a networkx graph G and nodes i,j we induce a flow in the graph using the maxflow algorithm, with source node i and sink node j, returns the flow interaction index
    ID of the matrix

    Args:
        G nx.Graph: graph og a network G 
        i (_type_): int representing node i
        j (_type_): int representing node j

    Returns:
        float: Flow between nodes I and j
    """    
    return nx.flow.maximum_flow(G, i, j)[0]

def create_id(A):
    """Creates the flow interaction index for a adjacency matrix A from a agraph G

    Args:
        A (_type_): _description_

    Returns:
        _type_: _description_
    """    
    G = nx.from_numpy_array(A, parallel_edges=True,create_using=nx.DiGraph, edge_attr="capacity")
    m = A.shape[0]
    ind_i = ind_j = np.arange(0,m)
    Id = np.zeros([m,m])

    return Id