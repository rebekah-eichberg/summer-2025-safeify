#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
from sklearn.pipeline       import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition  import PCA
from sklearn.compose       import ColumnTransformer

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["review_span"]           = (df["max_date"] - df["min_date"]).dt.days
    df["missing_price"]         = df["missing_price"].astype(int)
    df["product_lifespan_days"] = df["product_lifespan"].dt.days
    df.drop(
        ["min_date", "max_date", "product_lifespan",
         "percent_positive", "percent_negative",
         "unique_reviewer_count", "review_span"],
        axis=1, inplace=True, errors="ignore"
    )
    return df

def make_transformer(df, r, s, *, drop_first=True):
    rev  = [c for c in df if c.startswith("embedding_")]
    summ = [c for c in df if c.startswith("embed_")]
    num  = [c for c in df if c not in rev+summ+["category"]]

    rev_pipe  = ("drop" if r == 0 else
                 Pipeline([("scale",StandardScaler()), ("pca",PCA(r,random_state=42))]))
    sum_pipe  = ("drop" if s == 0 else
                 Pipeline([("scale",StandardScaler()), ("pca",PCA(s,random_state=42))]))

    return ColumnTransformer(
        [("num", StandardScaler(), num),
         ("cat", OneHotEncoder(handle_unknown="ignore",
                               drop="first" if drop_first else None,
                               sparse_output=False), ["category"]),
         ("rev", rev_pipe,  rev),
         ("sum", sum_pipe,  summ)
        ]).set_output(transform="pandas")

