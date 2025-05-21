#by: Shevaun Lewis
#created: 10/22/10

#This script extracts the sentences for a particular set of items and
#conditions from an EyeTrack .script file.

#input: python script-get_sentences.py 'input file'
    #input file must be a .script file (output from eyetrack_reading.py)
#output: a list of all the sentences (allsentences.txt), and a list of just the sentences
# in the experiment (expsentences.txt) in same directory

#########################################################################################

import sys

try:
    script_file = open(sys.argv[1], 'r')		# Script file, output from eyetrack_reading.py
except NameError:
    print('Error: Specify input file (open script for usage info)')

#User enters the Item number of the first item, as counted in the EyeTrack .script file
#(Item number comes after the "I" in the trial id number)

try:
    first_item = int(input('Number of first item:'))    
except NameError:
    print('Error: must be an integer')
    exit()
except ValueError:
    print('Error: must be an integer')
    exit()
    
#User enters the number of items

try:
    num_items = int(input('Total number of items:'))
except NameError:
    print('Error: must be an integer')
    exit()
except ValueError:
    print('Error: must be an integer')
    exit()

#User enters the Condition number of the first item, as counted in the EyeTrack .script file
#(Condition number comes after the "E" in the trial id number)

try:
    first_cond = int(input('Number of first condition:'))
except NameError:
    print('Error: must be an integer')
    exit()
except ValueError:
    print('Error: must be an integer')
    exit()

#User enters the total number of conditions, as counted in the EyeTrack .script file
#(If multiple experiments with different numbers of conditions were included in the
#script, then the number of conditions will be a multiple of the actual number.)

try:
    num_conds = int(input('Total number of conditions:'))
except NameError:
    print('Error: must be an integer')
    exit()
except ValueError:
    print('Error: must be an integer')
    exit()

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
allsentences = open('allsentences.txt', 'w')

for row in sentences:
    allsentences.write(str(row))

allsentences.close()

expsent = []

for row in sentences:
    r = row.split(" ")
    if int(r[0]) in conds:
        if int(r[1]) in items:
            expsent.append(row)

expsentences = open('expsentences.txt','w')

for row in expsent:
    expsentences.write(str(row))

expsentences.close()
