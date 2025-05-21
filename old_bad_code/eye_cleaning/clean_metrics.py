import os

import pandas as pd
from numpy import nan

foldernames = ["ff","fp","gp","ro","tt"]
itemnames = list(range(81,181))
for folder in foldernames:
    for item in itemnames:
        path = os.path.join("eye_cleaning","cleaned_eyetrack",folder,f"{item}.ixs")
        df = pd.read_csv(path)
        df = df.astype(pd.Int64Dtype(),copy=True)
        if folder != "ro" and folder != "ff":
            df.replace(0,nan)
        df.to_csv(path,na_rep="NA")
"""        with open(path) as reader:
            rows = reader.readlines()
            dataRows = rows[1:]
            for row in range(len(dataRows)):
                if folder == "ro" or folder == "ff":
                    dataRows[row] = dataRows[row].replace(",,",",NA,")
                else:
                    dataRows[row] = dataRows[row].replace(",0,",",NA,")
        with open(path,"w") as writer:
            writer.writelines([rows[0]]+dataRows)"""


