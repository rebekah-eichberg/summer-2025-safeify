import numpy as np
from collections import Counter
from sklearn.neighbors import KNeighborsRegressor

def predict_rank(entry,categories):
    """
    Predicts the 'Toys & Games' ranking of a product based on its ranking in another category,
    using K-Nearest Neighbors regression trained on overlapping entries.

    Parameters:
    ----------
    entry : pandas.Series
        A row from `meta_df`, expected to have:
        - `max_intersect`: a tuple (count, category) where `count` is the number of overlapping
          entries with 'Toys & Games' in that category.
        - `rank_dict`: a dictionary mapping category names to the product's rank in those categories.

    Returns:
    -------
    float or None
        The predicted rank in 'Toys & Games' based on nearest neighbors from a related category.
        Returns None if the overlap count is less than 10 or if required data is missing.

    Notes:
    -----
    - Uses a weighted KNN regressor (`weights='distance'`, `n_neighbors=5`).
    - Merges the category DataFrame with 'Toys & Games' on the `index` column to find common entries.
    - Assumes global access to a `categories` DataFrame with per-category sub-DataFrames under the 'df' column.
    """
    # Do not predict if too few overlapping data points exist
    if entry.max_intersect[0] < 10:
        return None

    category = entry.max_intersect[1]

    # Merge category DataFrame with 'Toys & Games' on shared product indices
    merged_df = categories.loc[category].df.merge(
        categories.loc['Toys & Games'].df, on='index'
    )

    # Train a weighted K-Nearest Neighbors regressor
    model = KNeighborsRegressor(n_neighbors=5, weights='distance')
    model.fit(merged_df[['rank_x']].values, merged_df.rank_y)

    # Predict 'Toys & Games' rank based on rank in the intersecting category
    return model.predict(np.array(entry.rank_dict[category]).reshape(1, 1))[0]

def max_intersect_category(rank_dict,categories):
    """
    Determines the category from `rank_dict` that has the highest overlap
    with 'Toys & Games' based on previously computed statistics.

    Parameters:
    ----------
    rank_dict : dict or None
        A dictionary mapping category names to ranks for a given product.

    Returns:
    -------
    tuple or None
        Returns a tuple (max_count, category_name) where:
        - max_count is the maximum number of products in that category
          also associated with 'Toys & Games' (`num_tgs`).
        - category_name is the name of the category with that maximum.

        Returns None if `rank_dict` is None or if no matching categories
        exist in the `categories` DataFrame.

    Notes:
    -----
    - Relies on a global `categories` DataFrame that contains `num_tgs` values
      for each category, which represent the number of times that category co-occurred
      with 'Toys & Games'.
    """
    if rank_dict is None:
        return None

    # Select num_tgs for all categories listed in rank_dict
    intersects = categories.loc[rank_dict.keys()].num_tgs

    if intersects.shape[0] == 0:
        return None

    # Return max intersection value and its corresponding category
    return (int(intersects.max()), intersects.idxmax())

def contains_tg(dict):
    if dict==None:
        return False
    return 'Toys & Games' in dict.keys()


def extract_rank_dict(rank_entry):
    """
    Extracts product rankings from a string or list of strings into a dictionary.

    Parameters:
    -----------
    rank_entry : str or list of str
        A single rank string or a list of such strings. Each string is expected to contain
        a rank (preceded by '#') and a category (after the word 'in').

    Returns:
    --------
    dict or None
        A dictionary mapping category names (str) to their ranks (int).
        Returns None if input is invalid or not in an expected format.

    Examples:
    ---------
    >>> extract_rank_dict("56,123 in Software (")
    {'Software': 56123}

    >>> extract_rank_dict(["#1 in Toys & Games", "#23 in Educational Toys"])
    {'Toys & Games': 1, 'Educational Toys': 23}
    """
    output=dict()
    if type(rank_entry)==list:
        if rank_entry==[]:
            return None
        for string in rank_entry:
            index_in=string.find('in')
            index_hash=string[:index_in].find('#')
            if index_in==-1 or index_hash==-1:
                return None
            index_see=string.find('(See')
            if index_see!=-1:
                category=string[index_in+3:index_see-1]
            else:
                category=string[index_in+3:]
            rank=int(string[index_hash+1:index_in].replace(',',''))
            output[category]=rank
        return output
    if type(rank_entry)!=str:
        return None
    index_in=rank_entry.find('in')
    if index_in==-1:
        return None
    string=rank_entry
    index_in=string.find('in')
    category=string[index_in+3:-2]
    rank=int(string[:index_in-1].replace(',',''))
    output[category]=rank
    return output

def combine(entry):
    output=entry.also_buy+entry.also_view
    output=list(set(output)) # Remove duplicates
    if output==[]:
        return None
    return output

def predict_rank_similar_prods(entry,lookup_rank):
    if not np.isnan(entry.item_rank):
        return entry.item_rank
    similar_prods=entry.similar
    if similar_prods==None:
        return None
    similar_ranks=[]
    for prod in similar_prods:
        rank=lookup_rank.get(prod,-1)
        if rank!=-1:
            similar_ranks.append(rank)
    if len(similar_ranks)>0:
        geometric_mean = np.exp(np.mean(np.log(similar_ranks)))
        return geometric_mean
    return None

def filter_junk(price):
    if price=='':
        return None
    if len(price)>=12:
        if price[0:12]=='.a-box-inner':
            return None
    return price

def extract_subcategory(cat):
    if len(cat)>0:
        return cat[1]
    return None

def most_frequent(lst):
    return Counter(lst).most_common(1)[0][0]

def predict_category_similar_prods(entry,lookup_cat):
    if entry.category!=None:
        return entry.category
    similar_prods=entry.similar
    if similar_prods==None:
        return None
    similar_categories=[]
    for prod in similar_prods:
        category=lookup_cat.get(prod,-1)
        if category!=-1:
            similar_categories.append(category)
    if len(similar_categories)>0:
        return most_frequent(similar_categories)
    return None
