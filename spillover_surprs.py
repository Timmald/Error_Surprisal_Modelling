import pandas as pd
from numpy import nan

df = pd.read_csv("data/script_trial_items_pivot.gpt2.csv")
#spillover: where the "subject" is the same and the "Sentence" but word_pos is word_pos -1 or word_pos-2
def getspillover(row,amount):
    if row["word_pos"]-amount < 0:
        return nan
    #spilloverEntries = df[(df["subject"]==row["subject"]) & (df["Sentence"]==row["Sentence"]) & (df["word_pos"] == row["word_pos"]-amount)]
    spilloverEntries = df[(df["Sentence"]==row["Sentence"]) & (df["word_pos"] == row["word_pos"]-amount)]
    return spilloverEntries["sum_surprisal"].iloc[0]

df["spillover_1"] = df.apply(lambda row: getspillover(row,1),axis=1,result_type="reduce")
print("spillover 1!")
df["spillover_2"] = df.apply(lambda row: getspillover(row,2),axis=1,result_type="reduce")
print("spillover 2")
df.to_csv("data/target_items.csv",na_rep="NA")
print("DONE")