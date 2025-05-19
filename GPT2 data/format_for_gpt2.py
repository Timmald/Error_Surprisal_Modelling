import csv
with open("allsentences.txt", "r") as reader:
    lineNo = 0
    conds = {1: ("PREP_DAT",True),
             2: ("PREP_DAT",False),
             3: ("COM_SBJ",True),
             4: ("COM_SBJ",False),
             5: ("FILLER", False)}
    writer = csv.DictWriter(open("data/script_items_pivot.csv","w"),fieldnames=["","item","condition","ambiguity","Sentence"])
    writer.writeheader()
    for line in reader:
        line = line.split()
        cnd = line[0]
        item = line[1]
        sentence = " ".join(line[2:])[:-2]
        condition = conds[int(cnd)][0]
        ambiguous = not conds[int(cnd)][1]
        writer.writerow({"":lineNo,"item":item,"condition":condition,"ambiguity":ambiguous,"Sentence":sentence})
        lineNo +=1
        