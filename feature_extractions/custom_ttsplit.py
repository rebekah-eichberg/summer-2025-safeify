import pandas as pd
import numpy as np
from collections import defaultdict

def StratifiedGroupSplit(df, label_col, component_col, test_size=0.2, n_splits=2, random_state=None):
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
        # === Original Behavior ===
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
        # === n > 2: Multi-split ===
        # Assign each group to a split to match label proportions
        label_targets = [
            {k: v * test_size for k, v in total_label_counts.items()} for _ in range(n_splits)
        ]
        label_current = [defaultdict(float) for _ in range(n_splits)]
        split_components = [set() for _ in range(n_splits)]

        for component, size, label_dist in component_info:
            best_split = None
            min_imbalance = float('inf')

            for i in range(n_splits):
                imbalance = 0
                for label, count in label_dist.items():
                    projected = label_current[i][label] + count
                    imbalance += abs(projected - label_targets[i][label])
                if imbalance < min_imbalance:
                    min_imbalance = imbalance
                    best_split = i

            split_components[best_split].add(component)
            for label, count in label_dist.items():
                label_current[best_split][label] += count

        # Return list of test splits
        return [df[df[component_col].isin(split)] for split in split_components]


# import pandas as pd
# import numpy as np
# from collections import defaultdict

# def StratifiedGroupSplit(df, label_col, component_col, test_size=0.2, random_state=None):
#     np.random.seed(random_state)
    
#     component_col_labels = df.groupby(component_col)[label_col].agg(lambda x: x.value_counts().to_dict())
#     component_col_sizes = df.groupby(component_col).size()

#     component_info = []
#     for component, labels in component_col_labels.items():
#         component_info.append((component, component_col_sizes[component], labels))

#     np.random.shuffle(component_info)

#     total_label_counts = df[label_col].value_counts().to_dict()
#     test_label_target = {k: v * test_size for k, v in total_label_counts.items()}
#     test_label_current = defaultdict(float)

#     test_components = set()
#     train_components = set()

#     for component, size, label_dist in component_info:
#         should_add_to_test = True
#         for label, count in label_dist.items():
#             if test_label_current[label] + count > test_label_target[label] + 1e-6:
#                 should_add_to_test = False
#                 break
#         if should_add_to_test:
#             test_components.add(component)
#             for label, count in label_dist.items():
#                 test_label_current[label] += count
#         else:
#             train_components.add(component)

#     train_df = df[df[component_col].isin(train_components)]
#     test_df = df[df[component_col].isin(test_components)]

#     return train_df, test_df


