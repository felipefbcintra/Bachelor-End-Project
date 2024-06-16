import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import pickle
import sklearn as sk
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from munkres import Munkres
import pandas as pd



def tranlsate_communities(cm:list[set]):
    community_lva = []
    node_lva = []

    for i,set in enumerate(cm):
        for n in list(set):
            node_lva.append(n)
            community_lva.append(i)

    gen_labels_lva = pd.DataFrame({"node": node_lva, "community":community_lva})
    gen_labels_lva = gen_labels_lva.sort_values(by="node")
    #gen_labels_lva = gen_labels_lva[gen_labels_lva["node"]!=0]

    return gen_labels_lva


def assign_to_mat(A, ground,pred,c):
    """ # assigns l`_ij =  AivBj - Ai^Bj if i <= c (minus 1 is for python indexing reasons) """    
    m = A.shape[0]
    for i,j in np.ndindex(m, m): 
        if i <= (c -1):
            A[i,j] = np.sum(np.logical_or(ground == i , pred == j )) - np.sum(np.logical_and(ground==i, pred==j))

    return A


def create_cost_matrix(ground: np.array, pred:np.array):
    """Creates a cost matrix for the assignment problem (source evaluating community detection algorithms)"""
    u = (len(np.unique(ground)), len(np.unique(pred))) # gets the ammount of communities

    c_star = max(u) 
    c = min(u)

    A = np.zeros((c_star, c_star))
    A = assign_to_mat(A, ground, pred, c)

    return A    


def remap_communities(ground:np.array, pred:np.array):
    """Remaps communities from the predictions to the correspondent ground communities using a cost matrix and the Munkres algorithm"""
    
    cost_mat = create_cost_matrix(ground,pred)
    
    # ------------------------ caculates munkres algorithm ----------------------- #
    mun = Munkres()
    indexes = mun.compute(cost_mat)

    # ---------------------------------- mapping --------------------------------- #
    index_dict = {i:j for i, j in indexes} #new map

    pred = pd.Series(pred) # to use series.map
    pred = pred.map(index_dict)

    return pred


def calculate_kappa(ground, pred):
    """Calculates the kappa index for a set of predictions and ground truths"""

    # accuracy = accuracy_score(ground, pred)
    # ea = expected_accuracy(ground)
    # kappa_index = (accuracy - ea)/(1-ea)
    # return kappa_index
    ground =pd.Series(ground)
    freq = ground.value_counts() /len(ground)
    ea = np.sum([freq[i]*freq[j] for i,j in zip(ground, ground)])/len(ground)
    return (accuracy_score(ground,pred)-ea)/ (1- ea)
    #return 



def total_kappa(gcl, lcl):
    """calculates the kappa for the generated partitions

    Args:
        gcl (list): ground community labels
        lcl (list): predicted community labels
        
    Returns:
        float: kappa index 
    """      
    #calculating kappa   
    u = (len(pd.Series(gcl).unique())), len(pd.Series(gcl).unique())
    c_star = max(u) 
    c = min(u)

    A = np.zeros((c_star, c_star))
    A = assign_to_mat(A, np.array(gcl), np.array(lcl), c)
    #calculating kappa    
    cost_mat = A

    mun = Munkres()
    indexes = mun.compute(cost_mat)

    if len(pd.Series(gcl).unique()) < len(pd.Series(lcl).unique()):
        index_dict = {i:j for i, j in indexes} #new map
        ground = pd.Series(gcl) # to use series.map
        ground = ground.map(index_dict)
        return calculate_kappa(ground, pd.Series(np.array(lcl)))

    else:
        index_dict = {i:j for i, j in indexes} #new map
        pred = pd.Series(lcl) # to use series.map
        pred = pred.map(index_dict)
        return  calculate_kappa(pd.Series(np.array(gcl)), pred)
    

def total_accuracy(gcl, lcl):
    """calculates the kappa for the generated partitions

    Args:
        gcl (list): ground community labels
        lcl (list): predicted community labels
        
    Returns:
        float: kappa index 
    """      
    #calculating kappa   
    u = (len(pd.Series(gcl).unique())), len(pd.Series(gcl).unique())
    c_star = max(u) 
    c = min(u)

    A = np.zeros((c_star, c_star))
    A = assign_to_mat(A, np.array(gcl), np.array(lcl), c)
    #calculating kappa    
    cost_mat = A

    mun = Munkres()
    indexes = mun.compute(cost_mat)

    if len(pd.Series(gcl).unique()) < len(pd.Series(lcl).unique()):
        index_dict = {i:j for i, j in indexes} #new map
        ground = pd.Series(gcl) # to use series.map
        ground = ground.map(index_dict)
        return sk.metrics.f1_score(ground, pd.Series(np.array(lcl)), average="macro")

    else:
        index_dict = {i:j for i, j in indexes} #new map
        pred = pd.Series(lcl) # to use series.map
        pred = pred.map(index_dict)
        return  sk.metrics.f1_score(pd.Series(np.array(gcl)), pred, average="macro")
    


# matrix to numpy
def to_np(G:nx.multidigraph):
    edges = list(G.edges())#gets edges from generated network
    l = []
    [(l.append(x[0]), l.append(x[1])) for x in edges] #shape of matrix 
    m = max(l) +1
    A = np.zeros([m,m])

    for e in edges:
        A[e[0], e[1]] = 1


    return A


# Inducing flow
def flow_ij(G,i,j):
    return nx.flow.maximum_flow(G, i, j)[0]

def create_id(A):
    G = nx.from_numpy_array(A, parallel_edges=True,create_using=nx.DiGraph, edge_attr="capacity")
    m = A.shape[0]
    ind_i = ind_j = np.arange(0,m)
    Id = np.zeros([m,m])

    return Id