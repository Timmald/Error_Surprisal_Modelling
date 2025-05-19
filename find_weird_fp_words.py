import pandas as pd
df = pd.read_csv("filler_spillover.pivot.csv")
print(df[df["fp"]>2000][["word","subject","Sentence"]].head(20))
print(df[df["word"].str.len()>=13][["word","subject","fp"]])