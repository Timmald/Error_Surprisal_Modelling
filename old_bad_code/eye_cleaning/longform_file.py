import pandas as pd
import os
from numpy import nan

df = pd.read_csv("data/filler_items.csv")

i = 0

def add_eye_data_cols(row):
    global i
    i += 1
    item = row["item"]
    wordPos1 = row["word_pos"]+1
    rodf = pd.read_csv(os.path.join("eye_cleaning/cleaned_eyetrack/ro",f"{item}.ixs"))
    rodf = rodf[pd.notna(rodf[f"R{wordPos1}"])]

    subj = rodf["subj"].astype(pd.Int16Dtype(),copy=True).apply(lambda val:f"ro_{val}")
    row = row.append(pd.Series(list(rodf[f"R{wordPos1}"]),list(subj),dtype=pd.Int16Dtype()))

    
    fpdf = pd.read_csv(os.path.join("eye_cleaning/cleaned_eyetrack/fp",f"{item}.ixs"))
    fpdf = fpdf[pd.notna(fpdf[f"R{wordPos1}"])]

    subj = fpdf["subj"].astype(pd.Int16Dtype(),copy=True).apply(lambda val:f"fp_{val}")
    row = row.append(pd.Series(list(fpdf[f"R{wordPos1}"]),list(subj),dtype=pd.Int16Dtype()))
    print(row)
    return row


df = df.apply(add_eye_data_cols,1)
#for subj in range(1,77):
    #df = df.astype({f"ro_{subj}":pd.Int32Dtype(),f"fp_{subj}":pd.Int32Dtype()})
#0-3 Sentence,Unnamed: 0, ambiguity, condition
fpidxs = list(range(4,81))
#81-85 item,len,lg_freq,mean_surp,meansurp2
roidxs = list(range(86,163))
#164-168 sumsurp,sumsurp2,token,word,pos
df = df.iloc[:,[81,3,2,0,166,165,167,82,83,84,85,163,164]+fpidxs+roidxs+[1]]
df.to_csv("data/new_fillers.csv",na_rep="NA")

    #takes a series row, gets eyetracking data, adds shit to it