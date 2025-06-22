
import pandas as pd
import numpy as np
from collections import defaultdict

def StratifiedGroupSplit(df, label_col, component_col, test_size=0.2, n_splits=2, random_state=None):
    """
    Splits a pandas DataFrame into train/test or multiple folds, preserving both label proportions (stratification)
    and component integrity (no component appears in more than one split).

    Parameters:
        df (pd.DataFrame): DataFrame to split.
        label_col (str): Name of the column containing class labels.
        component_col (str): Name of the column containing component IDs.
        test_size (float): Fraction of data to assign to the test set (only used if n_splits=2).
        n_splits (int): Number of splits to create (default 2).
        random_state (int, optional): Seed for reproducibility.

    Returns:
        tuple or list:
            - If n_splits == 2: (train_df, test_df)
            - If n_splits > 2: List of DataFrames, one per split.

    Notes:
        - For n_splits=2: Performs a stratified group split, assigning groups to train/test sets so that
          the test set label distribution matches the overall distribution as closely as possible.
        - For n_splits>2: Recursively applies the n=2 logic to create multiple splits, each preserving
          group integrity and approximate stratification.
        - Typical use cases include grouped cross-validation, train/test/validation splitting for grouped data,
          and any scenario where both stratification and group integrity are required.
    """
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    np.random.seed(random_state)

    # Group -> label distribution
    component_col_labels = df.groupby(component_col)[label_col].agg(lambda x: x.value_counts().to_dict())
    component_col_sizes = df.groupby(component_col).size()

    # Format: (group_id, group_size, label_dist)
    component_info = [(comp, component_col_sizes[comp], labels) for comp, labels in component_col_labels.items()]
    np.random.shuffle(component_info)

    total_label_counts = df[label_col].value_counts().to_dict()

    if n_splits == 2:
        test_label_target = {k: v * test_size for k, v in total_label_counts.items()}
        test_label_current = defaultdict(float)

        test_components = set()
        train_components = set()

        for component, size, label_dist in component_info:
            can_add = True
            for label, count in label_dist.items():
                if test_label_current[label] + count > test_label_target[label] + 1e-6:
                    can_add = False
                    break
            if can_add:
                test_components.add(component)
                for label, count in label_dist.items():
                    test_label_current[label] += count
            else:
                train_components.add(component)

        train_df = df[df[component_col].isin(train_components)]
        test_df = df[df[component_col].isin(test_components)]

        return train_df, test_df

    else:
        # For n>2, run the n=2 case recursively, adjusting the test size accordingly.
        int_split,final_split=StratifiedGroupSplit(df, label_col, component_col, test_size=1/n_splits, random_state=random_state)
        split_list=[]
        if n_splits>3:
            split_list=StratifiedGroupSplit(int_split,label_col, component_col, n_splits=n_splits-1, random_state=random_state)
        else:
            split1,split2=StratifiedGroupSplit(int_split,label_col, component_col, test_size=0.5, random_state=random_state)
            split_list=[split1,split2]
        split_list.append(final_split)
        return split_list

