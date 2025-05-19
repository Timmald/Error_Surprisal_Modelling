import csv
import math
import re

coca_freqs = csv.DictReader(open("filler_extraction/freqs_coca.csv"))

freqs = {}
for row in coca_freqs:
    freqs[row["word"]] = int(row["count"])

def get_uni_freq(word):
    return freqs.get(re.sub("[.,?!;:']", "", word.lower()), 0)

fillers = {}
with open("data/target_items_long.csv", "r") as rd:
    with open("data/target_items_long_freq.csv","w") as wr:
        reader = csv.DictReader(rd)
        writer = csv.DictWriter(wr,fieldnames=list(reader.fieldnames)+["lg_freq","len"])
        writer.writeheader()
        for row in reader:
            fq = math.log2(get_uni_freq(row["word"])) if (re.sub("[.,?!;:]", "", row["word"].lower()) in freqs and freqs[re.sub("[.,?!;:]", "", row["word"].lower())] > 0) else "NA"
            row["lg_freq"] = fq
            row["len"] = len(row["word"])
            writer.writerow(row)