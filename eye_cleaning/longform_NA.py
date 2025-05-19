import pandas as pd
from numpy import nan

df = pd.read_csv("data/new_fillers.csv")
for i in range(1,78):
    df[f"fp_{i}"].replace(0,nan,inplace=True)
df.to_csv("data/new_fillers_NA.csv",na_rep="NA")