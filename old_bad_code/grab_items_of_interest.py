import pandas as pd

df = pd.read_csv("data/target_items_long_freq_s.csv")
df2 = df[(df["word_pos"]==df["disambPosition_0idx"])|(df["word_pos"]==df["disambPosition_0idx"]+1)]
df2.to_csv("target_interest_items.csv",na_rep="NA")