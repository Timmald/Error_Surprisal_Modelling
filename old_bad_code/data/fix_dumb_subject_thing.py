import pandas as pd 
df = pd.read_csv("data/target_items_long_freq.csv")
df["subject"] = "fp_"+df["subject"].astype(str)
df.to_csv("data/target_items_long_freq_s.csv",na_rep="NA")