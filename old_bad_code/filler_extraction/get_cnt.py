import re
cnt = []
with open("allsentences.txt") as reader:
    for sentence in reader:
        #item, condition, num region, region boundaries
        line = sentence.strip().split()
        condition = line[0]
        if condition == "5":#then it's filler
            item = line[1]
            idx = 0
            boundaries = ["0"]
            for word in line[2:]:
                word = re.sub("\\n","",word)
                idx += len(word)
                boundaries.append(str(idx))
                idx+=1
            cnt.append([item,condition,str(len(boundaries)-1)]+boundaries)#why 
with open("fillers.cnt", "w") as writer:
    writer.writelines([" ".join(i)+"\n" for i in cnt])


