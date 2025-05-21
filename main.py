import pandas as pd
import os
import subprocess
#TODO: os.path.join
#TODO: make it more generalizable
#TODO: Deal with the phantom row columns that keep showing up
#TODO: keep outputs cleaner
#TODO: Do this in a less insane way

def print_sep():
    print("-----------------------------------------")

def parse_experiment_file(filename):
    script_file = open(filename, 'r')		# Script file, output from eyetrack_reading.py

    #the Item number of the first item, as counted in the EyeTrack .script file
    #(Item number comes after the "I" in the trial id number)

    first_item = 1    
        
    #the number of items

    num_items = 180

    #the Condition number of the first item, as counted in the EyeTrack .script file
    #(Condition number comes after the "E" in the trial id number)

    first_cond = 1

    #the total number of conditions, as counted in the EyeTrack .script file
    #(If multiple experiments with different numbers of conditions were included in the
    #script, then the number of conditions will be a multiple of the actual number.)

    num_conds = 5

    #define range of item numbers:
    items = range(first_item, first_item+num_items)
    conds = range(first_cond, first_cond+num_conds)

    #make list of lines in script_file:
    lines = script_file.readlines()
    script_file.close()

    #define empty list for sentences:
    sentences = []
    #define variable that keeps track of whether the next "inline" found is
    #a sentence and not a question:
    get = 0

    for line in lines:
        if line != "":
            l = line.split(" ")
            if 'trial' in l[0]:
                if l[1][0]=="E":
                    if l[1].split('D')[1][0]=='0':
                        cond = l[1].split('I')[0][1:]
                        item = l[1].split('I')[1].split('D')[0]
                        sentences.append(cond+' '+item)
                        get = 1
                    else:
                        get = 0
            elif 'inline' in l:
                if get == 1:
                    sent = line.split('|')[1]
                    s = len(sentences)-1
                    sentences[s] = sentences[s]+' '+sent
                    get = 0                

    # open output text file
    allsentences = open('get_word_info/trial_info.txt', 'w')

    for row in sentences:
        allsentences.write(str(row))

    allsentences.close()


#-------------------------------------------------

def format_for_gpt2():
    import csv
    with open("get_word_info/trial_info.txt", "r") as reader:
        lineNo = 0
        conds = {1: ("PREP_DAT",True),
                2: ("PREP_DAT",False),
                3: ("COM_SBJ",True),
                4: ("COM_SBJ",False),
                5: ("FILLER", False)}
        writer = csv.DictWriter(open("get_word_info/trial_info_formatted.csv","w"),fieldnames=["row","item","condition","ambiguity","Sentence"])
        writer.writeheader()
        for line in reader: # find the sentence, item, condition, ambiguity
            line = line.split()
            cnd = line[0]
            item = line[1]
            sentence = " ".join(line[2:])[:-2]
            condition = conds[int(cnd)][0]
            ambiguous = not conds[int(cnd)][1]
            writer.writerow({"row":lineNo,"item":item,"condition":condition,"ambiguity":ambiguous,"Sentence":sentence})
            lineNo +=1


def get_gpt2_surps():
    #magic code to get GPT-2 surprisals
    #source: Huang et al. 2023

    import re

    def clean(token):
        return re.sub("[^a-zA-Z0-9*.,!?\-]", "", token) # filter out non-alphanumeric or punctuation characters

    def align(words, wordpieces, debug=False):
        # Remove the "not beginning of sentence" character from the wordpieces
        wordpieces = [clean(piece) for piece in wordpieces]

        aligned = [] # list containing lists of wordpieces that make up each word
        idx_word = 0 # idx of the next word
        current_pieces = [] # wordpieces that don't align with the next word

        for idx_piece, piece in enumerate(wordpieces):
            if idx_word < len(words):
                if debug: print("not EOS")
                word = words[idx_word]
            else:
                current_pieces += wordpieces[idx_piece:]
                break

            if debug: print(piece, word, piece == word[:len(piece)])

            if piece == word[:len(piece)]:
                # if the new wordpiece is aligned to the next word

                # all current pieces belong to the current word
                aligned.append(current_pieces)

                # and the new piece belongs to the next word
                idx_word += 1
                current_pieces = [piece]
            else:
                # otherwise, the new piece belongs to the current word too
                current_pieces.append(piece)

        # at EOS, all remaining wordpieces belong to the last word
        aligned.append(current_pieces)
        if debug: print("EOS, merging the rest in: " + ",".join(current_pieces))

        # First entry in aligned is always empty (first wordpiece should always match the first word)
        aligned = aligned[1:]

        # get the indices of the wordpiece that correspond to word boundaries (breaks)
        breaks = [len(pieces) for pieces in aligned]
        breaks = [0] + [sum(breaks[:i+1]) for i in range(len(breaks))]

        return aligned, breaks

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    import torch.nn.functional as F

    import csv
    import re
    import numpy as np

    from tqdm import tqdm

    # TODO option for selecting subword merges to compute

    # how can we combine subwords/punctuation to get one surprisal per word?
    merge_fs = {"sum_":sum}

    # Load models from HF transformers
    tokenizer = AutoTokenizer.from_pretrained("gpt2", add_prefix_space=True)
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    #if args.cuda:
        #device = torch.device("cuda")
        #model.to(device)
    #I commented that out for cross-platformness. SUFFER!


    in_f = open("get_word_info/trial_info_formatted.csv", "r")
    stims = csv.DictReader(in_f)

    out = []
    for stim_row in tqdm(stims):
        inputs = tokenizer("<|endoftext|> " + stim_row["Sentence"], return_tensors="pt")
        ids = inputs["input_ids"]
        tokens = tokenizer.tokenize(stim_row["Sentence"])

        # run the model
        outputs = model(**inputs, labels=ids)
        logprobs = F.log_softmax(outputs.logits[0], 1)
        
        words = stim_row["Sentence"].split()
        piecess, breaks = align(words, tokens)
        # get surp for each word (avgd over pieces) and write it to a new row
        for i, (word, pieces) in enumerate(zip(words, piecess)):
            row = stim_row.copy() # new object, not a reference
            row["token"] = ".".join(pieces)
            row["word"] = word
            row["word_pos"] = i
            # correct for alignment difference due to initial EOS in the model input. see get_lstm.py for details
            surps = [-logprobs[j][ids[0][j+1]].item() for j in range(breaks[i], breaks[i+1])]
            surps_base2 = [surp/np.log(2.0) for surp in surps]
            for merge_fn, merge_f in merge_fs.items():
                row[merge_fn + "surprisal"] = merge_f(surps)
                #row[merge_fn + "surprisal_base2"] = merge_f(surps_base2)

            out.append(row)

    with open("get_word_info/gpt2_surps.csv", "w") as out_f:
        writer = csv.DictWriter(out_f, fieldnames = out[0].keys())
        writer.writeheader()
        writer.writerows(out)


def add_disamb():#add idxs of interest
    import csv
    with open("get_word_info/gpt2_surps.csv") as csv_reader:
        with open("get_word_info/idx_of_interest.csv","w") as writer:
            rows = []
            reader = csv.DictReader(csv_reader)
            csvWriter = csv.DictWriter(writer,fieldnames=list(reader.fieldnames)+["zdisambPositionAmb","zdisambPositionUnamb","zdisambPosition_0idx"])
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
                        if uword=="are" or uword=="were":
                            break
                        else:
                            idx += 1
                    for i in range(len(itemRows)):
                        itemRows[i]["zdisambPositionAmb"] = idx+1
                        itemRows[i]["zdisambPositionUnamb"] = idx+1
                        itemRows[i]["zdisambPosition_0idx"] = idx
                else:
                    idx = 0
                    for uword,aword in zip(unAmbRows[0]["Sentence"].split(),ambRows[0]["Sentence"].split()):
                        if uword != aword:
                            break
                        else:
                            idx += 1
                    for i in range(len(itemRows)):
                        itemRows[i]["zdisambPositionAmb"] = idx+2
                        itemRows[i]["zdisambPositionUnamb"] = idx+3
                        itemRows[i]["zdisambPosition_0idx"] = idx+1
                csvWriter.writerows(itemRows)
            for i in range(81,181):
                filler_rows = [row for row in rows if int(row["item"]) == i]
                csvWriter.writerows(filler_rows)

def add_freqs():
    import csv
    import math
    import re

    coca_freqs = csv.DictReader(open("raw_data/freqs_coca.csv"))#tells it where the freqs at

    freqs = {}
    for row in coca_freqs:
        freqs[row["word"]] = int(row["count"])

    def get_uni_freq(word):
        return freqs.get(re.sub("[.,?!;:']", "", word.lower()), 0)

    fillers = {}
    with open("get_word_info/idx_of_interest.csv", "r") as rd:
        with open("get_word_info/freq.csv","w") as wr:
            reader = csv.DictReader(rd)
            writer = csv.DictWriter(wr,fieldnames=list(reader.fieldnames)+["lg_freq","len"])
            writer.writeheader()
            for row in reader:
                fq = math.log2(get_uni_freq(row["word"])) if (re.sub("[.,?!;:]", "", row["word"].lower()) in freqs and freqs[re.sub("[.,?!;:]", "", row["word"].lower())] > 0) else "NA"
                row["lg_freq"] = fq
                row["len"] = len(row["word"])
                writer.writerow(row)


def add_spillover():
    import pandas as pd
    from numpy import nan

    df = pd.read_csv("get_word_info/freq.csv")
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
    df.to_csv("get_word_info/word_info.csv",na_rep="NA")
    print("DONE")


def clean_eyetracking_files():
    import os

    import pandas as pd
    from numpy import nan



    #replace 0 FP times with NA
    df = pd.read_csv("raw_data/fp.ixs")
    df = df.astype(pd.Int64Dtype(),copy=True)
    df = df.replace(0,nan)
    df.to_csv("eyetrack_data/fp_clean.ixs",na_rep="NA")

    #pandas should autointerepret the ,, columns of RO as NA

    ro_df = pd.read_csv("raw_data/ro.ixs")
    ro_df = ro_df.astype(pd.Int64Dtype(),copy=True)
    ro_df.to_csv("eyetrack_data/ro_clean.ixs",na_rep="NA")


def split_eyetrack_items():#an artifact of me being a fool
    filenames = ["fp","ro"]
    for fn in filenames:
        with open("eyetrack_data/"+fn+"_clean.ixs") as reader:
            rows = reader.readlines()
            header = rows[0]
            rows = rows[1:]
            for itemNum in range(1,101):
                toWrite = [header]
                for row in rows:
                    rowList = row.split(",")
                    rowItem = int(rowList[3])
                    if rowItem == itemNum:
                        toWrite.append(row)
                with open(f"eyetrack_data/{fn+"_by_item"}/{itemNum+80}.ixs","w") as writer:
                    writer.writelines(toWrite)


def wideform_data():
    import pandas as pd
    import os
    from numpy import nan
    from tqdm import tqdm



    df = pd.read_csv("get_word_info/word_info.csv")
    df = df[df["condition"]=="FILLER"]

    #if you are wondering why I did it this way, just know my old code was even more foolhardy
    def add_eye_data_cols(row):
        item = row["item"]
        wordPos1 = row["word_pos"]+1
        rodf = pd.read_csv(os.path.join("eyetrack_data/ro_by_item",f"{item}.ixs"))
        rodf = rodf[pd.notna(rodf[f"R{wordPos1}"])]

        subj = rodf["subj"].astype(pd.Int16Dtype(),copy=True).apply(lambda val:f"ro_{val}")
        row = pd.concat([row,pd.Series(list(rodf[f"R{wordPos1}"]),list(subj),dtype=pd.Int16Dtype())])

        
        fpdf = pd.read_csv(os.path.join("eyetrack_data/fp_by_item",f"{item}.ixs"))
        fpdf = fpdf[pd.notna(fpdf[f"R{wordPos1}"])]

        subj = fpdf["subj"].astype(pd.Int16Dtype(),copy=True).apply(lambda val:f"fp_{val}")
        row = pd.concat([row,pd.Series(list(fpdf[f"R{wordPos1}"]),list(subj),dtype=pd.Int16Dtype())])
        return row


    df = df.apply(add_eye_data_cols,1)
    #for subj in range(1,77):
        #df = df.astype({f"ro_{subj}":pd.Int32Dtype(),f"fp_{subj}":pd.Int32Dtype()})
    print(list(enumerate(list(df.columns.values))))
    #the above line is how I got the indices of colnames
    #0-4 fakerow,Sentence,Unnamed: 0, ambiguity, condition
    fpidxs = list(range(4,81))
    #82-86 item,len,lg_freq,mean_surp,meansurp2
    roidxs = list(range(84,161))
    #164-171 row,spillover1,spillover2,sumsurp,sumsurp2,token,word,pos
    df = df.iloc[:,[161,3,81,2,0,165,167,170,164,162,163,82,83]+fpidxs+roidxs]


    df.to_csv("filler_wideform_data.csv",na_rep="NA")
    print("DONE")


def fake_target_longform():#repeat each nonfiller item 77 times to match the shape of the longform data
    import pandas as pd
    import numpy as np

    df = pd.read_csv("get_word_info/word_info.csv")
    df = df[df["condition"] != "FILLER"]#only grab target items
    df = pd.DataFrame(np.repeat(df.values,77,axis=0))
    
    fake_subj_col = np.tile(np.arange(1,78).astype(str),2430)
    #2429 rows?
    df["subject"] = fake_subj_col
    df["subject"] = "fp_"+df["subject"].astype(str)
    df.to_csv("longform_targets.csv",na_rep="NA",header=["","row","item","condition","ambiguity","Sentence","token","word","word_pos","sum_surprisal","irrelevant","irrelevant2","disamb_0idx","lg_freq","len","spillover_1","spillover_2","subject"])


def grab_WOIs():
    import pandas as pd

    df = pd.read_csv("longform_targets.csv")
    #if it's a prep_dat and ambiguious, -1. Else, just use the index?

    df2= df[((df["condition"]=="PREP_DAT") & (df["ambiguity"]==True) & (df["word_pos"]==df["disamb_0idx"]-1))|
    (~((df["condition"]=="PREP_DAT") & (df["ambiguity"]==True)) & (df["word_pos"]==df["disamb_0idx"]))]

    #df2 = df[((df['ambiguity'] == False) & (df['word_pos'] == df['disamb_0idx'])) |
    #((df['ambiguity'] == True) & (df['word_pos'] == df['disamb_0idx']-1))]
    
    df2.to_csv("target_interest_items.csv",na_rep="NA")

#------------

if __name__=="__main__":
    parse_experiment_file("raw_data/ad.script")
    with open("get_word_info/trial_info.txt") as trial_info:
        print("Parsed Experimental Info:")
        print("\n".join(trial_info.readlines(4)))

    print_sep()

    format_for_gpt2()
    print("Formatted for GPT2:")
    print_df = pd.read_csv("get_word_info/trial_info_formatted.csv")
    print(print_df)
        
    print_sep()

    if not os.path.isfile("get_word_info/gpt2_surps.csv"):#getting surprisals takes a long time
        get_gpt2_surps()
        print("Got GPT2 Surps:")
        print_df = pd.read_csv("get_word_info/gpt2_surps.csv")
        print(print_df)

        print_sep()

    add_disamb()
    print("Added idxs of interest:")
    print_df = pd.read_csv("get_word_info/idx_of_interest.csv")
    print(print_df)

    print_sep()

    add_freqs()
    print("Added freq column:")
    print_df = pd.read_csv("get_word_info/freq.csv")
    print(print_df)

    print_sep()

    add_spillover()
    print("Added spillover columns:")
    print_df = pd.read_csv("get_word_info/word_info.csv")
    print(print_df)

    #NOTE: Here and below only works on the filler data because that's the eyetracking data I had. My bad!
    clean_eyetracking_files()
    print("Cleaned eyetracking files:")
    print_df = pd.read_csv("eyetrack_data/fp_clean.ixs")
    print(print_df)
    print_df = pd.read_csv("eyetrack_data/ro_clean.ixs")
    print(print_df)

    print_sep()

    split_eyetrack_items()
    print("Split eyetracking files into individual items")
    print_sep()

    if not os.path.isfile("filler_wideform_data.csv"):#this code takes forever
        wideform_data()
        print("Made data wideform:")
        print_df = pd.read_csv("filler_wideform_data.csv")
        print(print_df)
        print_sep()

    subprocess.run(["Rscript","pivot.r"])#in theory, we now have beautiful longform data
    print_df = pd.read_csv("filler_longform_data.csv")
    print(print_df)
    print_sep()

    fake_target_longform()
    print("Data is long-form:")
    print_df = pd.read_csv("longform_targets.csv")
    print(print_df)

    print_sep()

    grab_WOIs()
    print("target WOIs grabbed:")
    print_df = pd.read_csv("target_interest_items.csv")
    print(print_df)

    print_sep()

    subprocess.run(["Rscript","analysis.R"])#in theory this should print out our analysis



