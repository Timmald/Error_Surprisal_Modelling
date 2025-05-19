import pandas as pd
import numpy as np
df = pd.read_csv("data/new_fillers_NA.csv")
fp_cols=df.loc[:,[f"fp_{i}" for i in range(1,78)]]
ro_cols=df.loc[:,[f"ro_{i}" for i in range(1,78)]]
fp_mean= np.mean(fp_cols,axis=1)
ro_mean = np.mean(ro_cols,axis=1)
fp_stdev = np.std(fp_cols,axis=1)
ro_stdev = np.std(ro_cols, axis=1)
df["fp_mean"] = fp_mean
df["fp_stdev"] = fp_stdev
df["ro_mean"] = ro_mean
df["ro_stdev"] = ro_stdev
print(df)
df.to_csv("filler_means.csv",na_rep="NA")