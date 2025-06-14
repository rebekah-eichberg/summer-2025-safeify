import pandas as pd
import numpy as np
from collections import defaultdict

def StratifiedGroupSplit(df, label_col, component_col, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    
    component_col_labels = df.groupby(component_col)[label_col].agg(lambda x: x.value_counts().to_dict())
    component_col_sizes = df.groupby(component_col).size()

    component_info = []
    for component, labels in component_col_labels.items():
        component_info.append((component, component_col_sizes[component], labels))

    np.random.shuffle(component_info)

    total_label_counts = df[label_col].value_counts().to_dict()
    test_label_target = {k: v * test_size for k, v in total_label_counts.items()}
    test_label_current = defaultdict(float)

    test_components = set()
    train_components = set()

    for component, size, label_dist in component_info:
        should_add_to_test = True
        for label, count in label_dist.items():
            if test_label_current[label] + count > test_label_target[label] + 1e-6:
                should_add_to_test = False
                break
        if should_add_to_test:
            test_components.add(component)
            for label, count in label_dist.items():
                test_label_current[label] += count
        else:
            train_components.add(component)

    train_df = df[df[component_col].isin(train_components)]
    test_df = df[df[component_col].isin(test_components)]

    return train_df, test_df
