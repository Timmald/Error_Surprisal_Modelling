import csv

with open("data/script_items_pivot.gpt2.csv") as csv_reader:
    with open("data/script_trial_items_pivot.gpt2.csv","w") as writer:
        rows = []
        reader = csv.DictReader(csv_reader)
        csvWriter = csv.DictWriter(writer,fieldnames=list(reader.fieldnames)+["disambPositionAmb","disambPositionUnamb","disambPosition_0idx"])
        csvWriter.writeheader()
        for row in reader:
            rows.append(row)
        for i in range(1,81):
            itemRows = [row for row in rows if int(row["item"]) == i]
            condition = itemRows[0]["condition"]
            ambRows = [row for row in itemRows if row["ambiguity"]=="True"]
            unAmbRows = [row for row in itemRows if row["ambiguity"]=="False"]
            if condition == "COM_SBJ":
                idx = 0
                for uword,aword in zip(unAmbRows[0]["Sentence"].split(),ambRows[0]["Sentence"].split()):
                    if uword != aword:
                        break
                    else:
                        idx += 1
                for i in range(len(itemRows)):
                    itemRows[i]["disambPositionAmb"] = idx+1
                    itemRows[i]["disambPositionUnamb"] = idx+1
                    itemRows[i]["disambPosition_0idx"] = idx+1
            else:
                idx = 0
                for uword,aword in zip(unAmbRows[0]["Sentence"].split(),ambRows[0]["Sentence"].split()):
                    if uword != aword:
                        break
                    else:
                        idx += 1
                for i in range(len(itemRows)):
                    itemRows[i]["disambPositionAmb"] = idx+1
                    itemRows[i]["disambPositionUnamb"] = idx
                    itemRows[i]["disambPosition_0idx"] = idx-1
            csvWriter.writerows(itemRows)


