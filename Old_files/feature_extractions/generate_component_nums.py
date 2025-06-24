"""
In this file, we add component_no column to the amazon_df_labels.csv dataset.
We consider the bipartite graph whose vertices consist of all Amazon products and incidents/recalls, 
and an amazon product is joined to an incident when there is a match.
Each connected component in this graph is labelled with a distict integer, 
and the component_no column records this integer for each amazon product.
"""


import numpy as np
import pandas as pd
from sknetwork.topology import get_connected_components
from scipy.sparse import csr_matrix
from ast import literal_eval

def get_components(matches_df):
    """
    Assigns a connected component number to each product in the matches DataFrame.

    This function constructs a bipartite graph where nodes are Amazon products and incidents/recalls,
    and an edge exists between a product and an incident if there is a match (as listed in the
    'incident_indices' column). It then computes the connected components of this graph and assigns
    a unique integer label ('component_no') to each product, indicating its component membership.

    Parameters:
        matches_df (pd.DataFrame): DataFrame with columns 'asin' (product ID) and 'incident_indices'
                                   (list of incident/recall indices for each product).

    Returns:
        pd.Series: A Series indexed by product with the assigned component number for each product.

    Notes:
        - The function assumes 'incident_indices' is a string representation of a list and will parse it.
        - No product (asin) should appear more than once in the input DataFrame.
        - Useful for downstream tasks that require grouping products by their connected incident clusters.
    """

    # Verify there are no duplicate asins
    assert(matches_df.asin.duplicated().sum()==0)


    # Convert incident_indices to lists
    matches_df.loc[:,"incident_indices"]=matches_df["incident_indices"].apply(literal_eval)

    # Build biadjacency matrix as a csr matrix
    # This is an m*n matrix where m is the number of products, 
    # and n is the number of matched incident reports/recalls

    matrix=dict()

    m=matches_df.shape[0]

    for i,data in matches_df.iterrows():
        indices=data['incident_indices']
        for j in indices:
            entry=matrix.get(j,[])
            entry.append(i)
            matrix[j]=entry

    column=list(matrix.keys())
    n=len(column)
    matrix=np.array([[int(i in matrix[column[j]]) for i in matches_df.index] for j in range(n)])
    matrix=csr_matrix(matrix)

    # Label connected component for each item, and add to matches_df
    components=pd.DataFrame(get_connected_components(matrix,force_bipartite=True)[len(column):],
                            index=matches_df.asin).rename(columns={0:'component_no'})
    matches_df=matches_df.join(components,on='asin')

    return matches_df['component_no']
