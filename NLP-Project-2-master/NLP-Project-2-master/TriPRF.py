import numpy
import regex
import os

train = "train1.txt"
test = "test1.txt"
output = "output.csv"
new = "newtrain.txt"
fresh = "newtest.txt"

k=0.2
tagfor = {}
i=0
for line in open(train):
    line.rstrip()
    tokens = line.split()
    for t in tokens:
        if i%3==0:
            if t in tagfor:
                tagfor[t] += 1
            else:
                tagfor[t]=1
    i=i+1
i=0
f1u = open(new,"w")
for line in open(train):
    line.rstrip()
    tokens = line.split()
    newL = ""
    for t in tokens:
        if i%3==0:
            if tagfor[t]!=1:
                newL += t + " "
            else:
                t="<unk>"
                newL += t + " "
        else:
            newL += t + " "
    i=i+1
    newL += "\n"
    f1u.write(newL)
f1u.close()
f2u = open(fresh,"w")
i=0
for line in open(test):
    line.rstrip()
    tokens = line.split()
    newL = ""
    for t in tokens:
        if i%3==0:
            if t in tagfor:
                newL += t + " "
            else:
                t="<unk>"
                newL += t + " "
        else:
            newL += t + " "
    i=i+1
    newL += "\n"
    f2u.write(newL)
f2u.close()
WORD=[] #words
POS=[] #part of speech
IOB=[] #Named entities
entities = ["B-ORG","I-ORG","B-PER","I-PER","B-LOC","I-LOC","B-MISC","I-MISC", "O"]

BORG = 0
IORG = 0
BPER = 0
IPER = 0
BLOC = 0
ILOC = 0
BMISC = 0
IMISC = 0
O = 0
wordcount = 0
i=0
for line in open(train):
    line.rstrip()
    tokens = line.split()
    if i%3==0:
        for t in tokens:
            WORD.append(t)
    elif i%3==1:
        for t in tokens:
            POS.append(t)
    else:
        for t in tokens:
            IOB.append(t)
    i=i+1

#print(WORD)
#print(POS)
#print(IOB)

#State Transition
count = {} #key is a NE bigram, values are ([following word's NE], [corresponding occurences of this NE trigram pair])
uni_counts = {} #keys are NE in vocab, value is their number of occurences
bigram_counts = {} #keys are NE bigrams, value is # of occurences
prob = []
for i,t in enumerate(IOB[:-2]):
    if t in uni_counts:
        uni_counts[t] += 1
    else:
        uni_counts[t] = 1
    #count entities
    if t=="B-ORG":
        BORG+=1
    elif t=="B-PER":
        BPER+=1
    elif t=="B-LOC":
        BLOC+=1
    elif t=="B-MISC":
        BMISC+=1
    elif t=="I-ORG":
        IORG+=1
    elif t=="I-PER":
        IPER+=1
    elif t=="I-LOC":
        ILOC+=1
    elif t=="I-MISC":
        IMISC+=1
    else:
        O+=1

    wordcount+=1
    t=t+IOB[i+1]
    if t in bigram_counts:
        bigram_counts[t] += 1
    else:
        bigram_counts[t] = 1
    if t in count:
        if IOB[i+2] in count[t][0]:
            index = count[t][0].index(IOB[i+2])
            count[t][1][index] += 1
        else:
            count[t][0].append(IOB[i+2])
            count[t][1].append(1)
    else:
        count[t] = ([IOB[i+2]], [1])

#count last 2 entities
wordcount+=2
t= IOB[-2]
if t=="B-ORG":
    BORG+=1
elif t=="B-PER":
    BPER+=1
elif t=="B-LOC":
    BLOC+=1
elif t=="B-MISC":
    BMISC+=1
elif t=="I-ORG":
    IORG+=1
elif t=="I-PER":
    IPER+=1
elif t=="I-LOC":
    ILOC+=1
elif t=="I-MISC":
    IMISC+=1
else:
    O+=1
t = IOB[-1]
if t=="B-ORG":
    BORG+=1
elif t=="B-PER":
    BPER+=1
elif t=="B-LOC":
    BLOC+=1
elif t=="B-MISC":
    BMISC+=1
elif t=="I-ORG":
    IORG+=1
elif t=="I-PER":
    IPER+=1
elif t=="I-LOC":
    ILOC+=1
elif t=="I-MISC":
    IMISC+=1
else:
    O+=1
if IOB[-2]+IOB[-1] in bigram_counts:
    bigram_counts[IOB[-2]+IOB[-1]]+=1
else:
    bigram_counts[IOB[-2]+IOB[-1]]=1
if IOB[-1] in uni_counts:
    uni_counts[IOB[-1]] += 1
else:
    uni_counts[IOB[-1]] = 1
if IOB[-2] in uni_counts:
    uni_counts[IOB[-2]] += 1
else:
    uni_counts[IOB[-2]] = 1

tran = {}  # keys are NE bigrams that we have seen, values are ([following NE's], [probability of corresponding NE appearing afterwards])
for c in count:
    tran[c] = ([], [])
    for i,x in enumerate(count[c][1]):
        word = count[c][0][i]
        tran[c][0].append(word)
        tran[c][1].append((x+k) / (bigram_counts[c]+k*9))
    sumP = sum(tran[c][1])
    for i,x in enumerate(tran[c][1]):
        tran[c][1][i] = x / sumP
#State transition matrix finished, and the probabilities of any that not in matrix 'tran' are zero
#print(tran)


#Observation using entities as keys and words as their values
count1 = {} #key is entity, values are {word assigned to entity: corresponding # occurences}
uni_counts1 = {} #key is an entity, value is number of occurences of this entity
for i,t in enumerate(WORD):
    if IOB[i] in count1:
        if t in count1[IOB[i]]:
            count1[IOB[i]][t]+=1
        else:
            count1[IOB[i]][t] = 1
    else:
        count1[IOB[i]] = {t: 1}
    if IOB[i] in uni_counts1:
        uni_counts1[IOB[i]] += 1
    else:
        uni_counts1[IOB[i]] = 1
obsv = {} #key is a entity, value is {word : Prob. of word|entity}
for c in count1:
    obsv[c] = {}
    for i,word in enumerate(count1[c]):
        x = count1[c][word]
        obsv[c][word] = (x+k) / (uni_counts1[c]+k*9)
#Observation using words as keys and entities  as their values
count2 = {} #key is word, values are ([NE of these words] : [# occurences of the corresponding NE])
uni_counts2 = {} #key is a word, value is number of occurences of this word
for i,t in enumerate(WORD,start=0):
    if t in count2:
        if IOB[i] in count2[t][0]:
            index = count2[t][0].index(IOB[i])
            count2[t][1][index] += 1
        else:
            count2[t][0].append(IOB[i])
            count2[t][1].append(1)
    else:
        count2[t] = ([IOB[i]], [1])
    if t in uni_counts2:
        uni_counts2[t] += 1
    else:
        uni_counts2[t] = 1
obsv2 = {} #key is a word, value is {NE : Prob. of corresponding NE}
for c in count2:
    obsv2[c] = {}
    for i,x in enumerate(count2[c][1]):
        tag = count2[c][0][i]
        obsv2[c][tag] = (x+k) / (uni_counts2[c]+k*9)
    sumP = sum(obsv2[c].values())
    for i,e in enumerate(obsv2[c]):
        obsv2[c][e] = obsv2[c][e] / sumP


#Observation using words as keys and entities  as their values
count3 = {} #key is bigram, values are ([NE of these words] : [# occurences of the corresponding NE])
uni_counts3 = {} #key is a bigram, value is number of occurences of this bigram
for i,t in enumerate(WORD[:-1],start=0):
    t = t+" " +WORD[i+1]
    if t in count3:
        if IOB[i] in count3[t][0]:
            index = count3[t][0].index(IOB[i]+IOB[i+1])
            count3[t][1][index] += 1
        else:
            count3[t][0].append(IOB[i]+IOB[i+1])
            count3[t][1].append(1)
    else:
        count3[t] = ([IOB[i]+IOB[i+1]], [1])
    if t in uni_counts3:
        uni_counts3[t] += 1
    else:
        uni_counts3[t] = 1
obsv3 = {} #key is a bigram, value is {NE : Prob. of corresponding NE}
for c in count3:
    obsv3[c] = {}
    for i,x in enumerate(count3[c][1]):
        tag = count3[c][0][i]
        obsv3[c][tag] = (x+k) / (uni_counts3[c]+k*9)
    sumP = sum(obsv3[c].values())
    for i,e in enumerate(obsv3[c]):
        obsv3[c][e] = obsv3[c][e] / sumP

#Observation matrix finished, and the probabilities of any that not in matrix 'obsv' are zero
#print(obsv)

#Transition prob is NER-tag-based P = #sequences from Ti to Tj / # Ti occurences
#Emission prob is observation prob is P(word | NER tag)

def viterbi(words):
    n = len(words)
    if n<2:
        return [0 for _ in range(n)]

    score = [[] for _ in range(len(bigram_counts))]
    bptr = [[] for _ in range(len(bigram_counts))]
    ents = ["B-ORG", "I-ORG", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-MISC", "I-MISC", "O"]
    for i,e in enumerate(bigram_counts):
            if words[0]+" " + words[1] not in obsv3:
                p_b = 0.0001
            elif e in obsv3[words[0]+" " +words[1]]:
                p_b = obsv3[words[0]+" "+words[1]][e]
            else:
                p_b = 0.0001
            if e in bigram_counts:
                score[i].append( (bigram_counts[e]/(wordcount-1)) * p_b )
            else:
                score[i].append(0)
            bptr[i].append(0)
    #iteration
    for t in range(1, n):
        for i,e in enumerate(bigram_counts):
            maxscore = 0
            maxi = 8
            for j,e2 in enumerate(entities):
                if e2 in tran[e][0]:
                    index = tran[e][0].index(e2)
                    p = tran[e][1][index]
                else:
                    p=0
                newscore = score[j][t-1] * p
                # print(score[j][t-1])
                if newscore > maxscore:
                    maxscore=newscore
                    maxi = j
            if words[t] not in obsv2:
                #WE NEVER SEEN THIS WORD USE A LOW PROB.
                p_w = 0.00001
            elif e in obsv2[words[t]]:
                p_w = obsv2[words[t]][e]
            else:
                p_w = 0.00001
            score[i].append(maxscore * p_w)
            bptr[i].append(maxi)
    #identify sequence
    tr = [0 for _ in range(n)]
    maxscore = 0
    for i,e in enumerate(entities):
        if score[i][n-1] > maxscore:
            maxscore = i
    tr[n-1] = maxscore
    for i in range(n-2, -1, -1):
        #for i=n-1 to 1 do
        tr[i] = bptr[tr[i+1]][i+1]
    return tr

test_pos = []
test_iob = []
tn = []
word_count = 0
#test line by line
i=0
for line in open(test):
    test_words = []
    line.rstrip()
    tokens = line.split()
    if i%3==0:
        for t in tokens:
            test_words.append(t)
            word_count+=1
        tn = tn + viterbi(test_words)
    elif i%3==1:
        for t in tokens:
            test_pos.append(t)
    else:
        for t in tokens:
            test_iob.append(t)
    i=i+1

#BUILD OUTPUT
results = {"PER": [], "LOC": [], "ORG": [], "MISC": []}
i=0
n=word_count
while i<n:
    j=i
    if entities[tn[i]] == "B-ORG":
        while j < n-1 and (entities[tn[j+1]]=="I-ORG" or entities[tn[j+1]]=="B-ORG"):
            print("continuation found")
            j+=1
        results["ORG"].append(str(i)+"-"+str(j))
        i=j
    elif entities[tn[i]] == "B-LOC":
        while j < n-1 and (entities[tn[j+1]]=="I-LOC" or entities[tn[j+1]]=="B-LOC"):
            print("continuation found")
            j+=1
        results["LOC"].append(str(i)+"-"+str(j))
        i=j
    elif entities[tn[i]] == "B-PER":
        while j < n-1 and (entities[tn[j+1]]=="I-PER" or entities[tn[j+1]]=="B-PER"):
            print("continuation found")
            j+=1
        results["PER"].append(str(i)+"-"+str(j))
        i=j
    elif entities[tn[i]] == "B-MISC":
        while j < n-1 and (entities[tn[j+1]]=="I-MISC" or entities[tn[j+1]]=="B-MISC"):
            print("continuation found")
            j+=1
        results["MISC"].append(str(i)+"-"+str(j))
        i=j
    i+=1


results1 = {"PER": [], "LOC": [], "ORG": [], "MISC": []}
i=0
n=len(test_iob)
while i<n:
    j=i
    if test_iob[i] == "B-ORG":
        while j < n-1 and (test_iob[j+1]=="I-ORG" or test_iob[j+1]=="B-ORG"):
            #print("continuation found")
            j+=1
        results1["ORG"].append(str(i)+"-"+str(j))
        i=j
    elif test_iob[i] == "B-LOC":
        while j < n-1 and (test_iob[j+1]=="I-LOC" or test_iob[j+1]=="B-LOC"):
            #print("continuation found")
            j+=1
        results1["LOC"].append(str(i)+"-"+str(j))
        i=j
    elif test_iob[i] == "B-PER":
        while j < n-1 and (test_iob[j+1]=="I-PER" or test_iob[j+1]=="B-PER"):
            #print("continuation found")
            j+=1
        results1["PER"].append(str(i)+"-"+str(j))
        i=j
    elif test_iob[i] == "B-MISC":
        while j < n-1 and (test_iob[j+1]=="I-MISC" or test_iob[j+1]=="B-MISC"):
            #print("continuation found")
            j+=1
        results1["MISC"].append(str(i)+"-"+str(j))
        i=j
    i+=1

numofpre=len(results["PER"])+len(results["LOC"])+len(results["ORG"])+len(results["MISC"])
numofentitiy=len(results1["PER"])+len(results1["LOC"])+len(results1["ORG"])+len(results1["MISC"])
numofcorrect=0
for e in ["PER","LOC","ORG","MISC"]:
    for r in results[e]:
        if r in results1[e]:
            numofcorrect += 1
#print(numofpre)
#print(numofentitiy)
#print(numofcorrect)
Precision=numofcorrect/numofpre
Recall=numofcorrect/numofentitiy
F1=2*Precision*Recall/(Precision+Recall)
print("Precision: ",Precision)
print("Recall: ",Recall)
print("F1: ",F1)