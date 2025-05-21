import pandas as pd
import numpy as np

df = pd.read_csv("data/target_items.csv")
df = pd.DataFrame(np.repeat(df.values,77,axis=0))
fake_subj_col = np.tile(np.arange(1,78),2430)
#2429 rows?
df["subject"] = fake_subj_col
df.to_csv("data/target_items_long.csv",na_rep="NA",header=["","unnamed","item","condition","ambiguity","Sentence","token","word","word_pos","sum_surprisal","sum_surprisal_base2","mean_surprisal","mean_surprisal_base2","disambPositionAmb","disambPositionUnamb","disambPosition_0idx","spillover_1","spillover_2","subject"])