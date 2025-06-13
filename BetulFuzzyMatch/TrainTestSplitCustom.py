#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
We aim to split the Amazon product data into training and test sets. However, many products are very similar

but have different ASINs (product identifiers), making it difficult to ensure a clean separation between train

and test. Ideally, we would identify and remove all similar products across the dataset before splitting.

But due to the large size of the dataset, that approach is impractical. Instead, we focus on the labeled products

(those with label = 1), which are matched to incident reports. Since a single incident report can be linked to

multiple Amazon products, we must ensure that all products linked to the same incident are assigned to the same set — either training or test — to avoid information leakage.

Therefore, we split the labeled Amazon products based on their linked incident reports, ensuring that no incident

and consequently no group of matched products appears in both the training and test sets.

The Problem: Many-to-Many Relationship Between Amazon Products and Incidents

In our dataset:

Each Amazon product (row in amazon_df) can be linked to multiple incidents.

Each incident can be linked to multiple products.

This creates a many-to-many relationship:

Imagine product A is linked to incidents 1 and 2.

Product B is linked to incident 2.

If we put product A in train and product B in test → incident 2 leaks into both sets!

This violates a key rule in machine learning:

No information should be shared between training and test sets.

So what we do here:

Graph-Based Connected Component Splitting:

We treat the data as a graph:

Nodes: Amazon product IDs and incident IDs.

Edges: Between a product and each incident it's linked to.

Why this works:

If we find connected components of this graph, each one is a “self-contained group” of linked products

and incidents. So we can:

Keep entire components together

Assign each component to a train/test or fold without causing leakage.

"""

import pandas as pd
import os
import ast
import networkx as nx
from sklearn.model_selection import train_test_split

this_dir = os.path.dirname(__file__)
data_path = os.path.join(this_dir, "..", "Data", "amazon_df_labels.csv")
amazon_df = pd.read_csv(os.path.abspath(data_path))

#amazon_df = pd.read_csv("Data/amazon_df_labels.csv")
amazon_df.drop_duplicates(subset=['asin'], inplace=True)
# indices column turned to str after loading csv, turn it back to list

amazon_df['incident_indices'] = amazon_df['incident_indices'].apply(
    lambda x: ast.literal_eval(x) if isinstance(x, str) else x
)

def train_test_split_custom(amazon_df, test_size=0.2, random_state=42):
    """
    Splits the Amazon data into train/test sets:
    - Uses connected components to split label 1 (matched) data based on incident links
    - Randomly splits label 0 (non-matched) data using the same test_size
    """

    # ---------- STEP 1: Split Label 1 data using connected components ----------
    label_1_df = amazon_df[amazon_df['match'] == 1].copy()

    G = nx.Graph()

    # Build bipartite graph: product_idx <-> incident_id
    for idx, row in label_1_df.iterrows():
        for incident in row['incident_indices']:
            G.add_edge(f"product_{idx}", f"incident_{incident}")
    
    # Get number of connected components
    num_components = nx.number_connected_components(G)
    print(f"Number of connected components: {num_components}")

    # Get connected components of the graph
    components = list(nx.connected_components(G))

    # Split the connected components into train/test
    comps_train, comps_test = train_test_split(
        components, test_size=test_size, random_state=random_state
    )

    # Extract Amazon product indices from each component group
    def extract_product_ids(component_sets):
        product_indices = set()
        for comp in component_sets:
            product_indices.update({
                int(node.replace("product_", ""))
                for node in comp if node.startswith("product_")
            })
        return product_indices

    train_label1_idx = extract_product_ids(comps_train)
    test_label1_idx = extract_product_ids(comps_test)

    # Get final label 1 train/test sets
    train_label1_df = amazon_df.loc[list(train_label1_idx)]
    test_label1_df = amazon_df.loc[list(test_label1_idx)]

    # ---------- STEP 2: Split Label 0 data using same test_size ----------
    label_0_df = amazon_df[amazon_df['match'] == 0].copy()
    
    label1_test_ratio = len(test_label1_df) / (len(train_label1_df) + len(test_label1_df))

    train_label0_df, test_label0_df = train_test_split(
        label_0_df,
        test_size=label1_test_ratio,
        random_state=random_state,
        stratify=None  
    )

    # ---------- STEP 3: Combine label 1 and label 0 ----------
    train_df = pd.concat([train_label1_df, train_label0_df], axis=0)
    test_df = pd.concat([test_label1_df, test_label0_df], axis=0)
    
    # ---------- STEP 4: Shuffle ----------
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return train_df, test_df

train_df, test_df = train_test_split_custom(amazon_df, test_size=0.2, random_state=42)

