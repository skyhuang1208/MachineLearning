'''
This code reads in emails from 2 people
and distinguish the test emails from them

by Sky
'''

import sys
import re
import numpy as np
import random

if len(sys.argv) != 5:
    print("Email classfication btw author p1 and p2:")
    exit("Usage: %s [emails_p1(file)] [emails_p2(file)] Name_p1 Name_p2" % sys.argv[0])

### PARS ###
ratio_TrainTest= 5
### PARS ###

### VARS ###
Ne1= 0  # N of emails frm p1
Ne2= 0  # N of emails frm p2
Ne1t= 0 # N of emails frm p1 used in training
Ne2t= 0 # N of emails frm p2 used in training
words_dict= {}
words_list= []
words_training= [] # a vectorized sparse matrix 
email_test= []
### VARS ###


######### Reading p1 #########
one= ""
with open(sys.argv[1], "r") as IFILE1:
    for line in IFILE1:
        if "=====" in line:
            Ne1 +=1
            IStrain= random.randint(0,ratio_TrainTest) # ran select training data
            if IStrain:
                Ne1t +=1
                txts= one.strip().split()
                words= [ re.sub('[^A-Za-z0-9]+', '', t).lower() for t in txts] # rm sp char; to lower
                for w in words:
                    if words_dict.get(w, None)==None: words_dict[w]= {} # new word for all
                    if words_dict[w].get(Ne1t-1, None)==None: words_dict[w][Ne1t-1]= 0 # new word for the email
                    words_dict[w][Ne1t-1] +=1
            else: email_test.append(one)
            one= ""
        else: one += line

######### Reading p2 #########
one= ""
with open(sys.argv[2], "r") as IFILE2:
    for line in IFILE2:
        if "=====" in line:
            Ne2 +=1
            IStrain= random.randint(0,ratio_TrainTest)
            if IStrain:
                Ne2t +=1
                txts= one.strip().split()
                words= [ re.sub('[^A-Za-z0-9]+', '', t).lower() for t in txts] # rm sp char; to lower
                for w in words:
                    if words_dict.get(w, None)==None: words_dict[w]= {} 
                    if words_dict[w].get(Ne1t+Ne2t-1, None)==None: words_dict[w][Ne1t+Ne2t-1]= 0
                    words_dict[w][Ne1t+Ne2t-1] +=1
            else: email_test.append(one)
            one= ""
        else: one += line

######### Print #########
print("Reading completed.")
print("N of emails from %s: %d. Training/testing: %d/%d." % (sys.argv[3], Ne1, Ne1t, Ne1-Ne1t))
print("N of emails from %s: %d. Training/testing: %d/%d." % (sys.argv[4], Ne2, Ne2t, Ne2-Ne2t))
print("Starting fitting and predicting...")

######### Making sparse matrix #########
n= -1
words_training= np.zeros( (Ne1t+Ne2t, len(words_dict)) )
for word, counts in words_dict.items():
    n +=1
    words_list.append(word)
    for iSample, count in counts.items():
        words_training[iSample, n]= count
labels_training= [ 0 if i<Ne1t else 1 for i in range(Ne1t+Ne2t) ]

#print(words_dict)
#print(words_list)
#print(words_training)

######### Do fitting #########
from sklearn.naive_bayes import MultinomialNB
clf= MultinomialNB()
clf.fit(words_training, labels_training)

######### Do predict, cal accuracy #########
Ntesting= Ne1+Ne2-Ne1t-Ne2t
words_testing= np.zeros( (Ntesting, len(words_dict)) )
for isample, email in enumerate(email_test):
    txts= email.strip().split()
    words= [ re.sub('[^A-Za-z0-9]+', '', t).lower() for t in txts] # rm sp char; to lower
    for w in words:
        try: ifeature= words_list.index(w)
        except: continue
        words_testing[isample, ifeature] +=1
labels_testing= [0 for i in range(Ne1-Ne1t)] + [1 for i in range(Ne2-Ne2t)]
predict_testing= clf.predict(words_testing)
#print(labels_testing)
#print(predict_testing)
accuracy= ( Ntesting - sum(np.absolute(np.subtract(labels_testing, predict_testing)))) / Ntesting
print("Accuracy: ", accuracy)
