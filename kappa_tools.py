import numpy as np
from munkres import Munkres
import pandas as pd
from sklearn.metrics import accuracy_score


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

np.vectorize(assign_to_mat)

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
    
    cost_mat = create_cost_matrix(ground, pred)
    
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

    accuracy = accuracy_score(ground, pred)
    expected_accuracy = 1/(len(ground.unique()))


    kappa_index = (accuracy - expected_accuracy)/(1-expected_accuracy)
    return kappa_index


def kappa(ground:np.array, pred:np.array):
    """" Calculates the kappa index for a given set of groudn truth communities and a set of labels

    Args:
        ground (np.array): ground communities
        pred (np.array): labels

    Returns:
        float: kappa index 
    """    

    accuracy = accuracy_score(ground, pred)
    expected_accuracy = 1/(len(ground.unique()))

    kappa_index = (accuracy - expected_accuracy)/(1-expected_accuracy)
    return kappa_index