from os import mkdir
filenames = ["ff","fp","gp","ro","tt"]
for fn in filenames:
    with open("raw_eyetrack/"+fn+".ixs") as reader:
        rows = reader.readlines()
        header = rows[0]
        rows = rows[1:]
        for itemNum in range(1,101):
            toWrite = [header]
            for row in rows:
                rowList = row.split(",")
                rowItem = int(rowList[2])
                if rowItem == itemNum:
                    toWrite.append(row)
            with open(f"eye_cleaning/cleaned_eyetrack/{fn}/{itemNum+80}.ixs","w") as writer:
                writer.writelines(toWrite)
        
