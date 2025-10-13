import pandas as pd

def load_csv(path):
    return pd.read_csv(path)

def load_excel(path, **kwargs):
    return pd.read_excel(path, **kwargs)
