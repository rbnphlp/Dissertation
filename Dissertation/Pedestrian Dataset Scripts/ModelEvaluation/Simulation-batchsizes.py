
'''Simulation file to run the keras model a 100 times , see what the average accuracy , precision and TPR are : used for obtaining graphsin Fig24 and Fig 25 in Report'''




'''Code to run the model a 100 times , change batch sizes in kerasmdel.py and run the below code '''

import subprocess
import sys
from tqdm import *
for scriptInstance in tqdm(range(100)):
    sys.stdout = open('resultbatch32%s.txt' % scriptInstance, 'w')
    subprocess.check_call(['python', 'kerasmodel.py'],
                          stdout=sys.stdout, stderr=subprocess.STDOUT)

for scriptInstance in tqdm(range(100)):
    sys.stdout = open('resultbatch200%s.txt' % scriptInstance, 'w')
    subprocess.check_call(['python', 'kerasmodel200.py'],
                          stdout=sys.stdout, stderr=subprocess.STDOUT)

for scriptInstance in tqdm(range(100)):
    sys.stdout = open('resultbatch300%s.txt' % scriptInstance, 'w')
    subprocess.check_call(['python', 'kerasmodel300.py'],
                          stdout=sys.stdout, stderr=subprocess.STDOUT)

## the results are printed to a txt file 

#################################################################################################################################################################




#############################################################################################################################################################################

### The below code , combines all the text file , into one and then takes the accuracy , roc , recall etc and prints to the screen #######

import glob
read_files = glob.glob("*.txt")

with open("result200.txt", "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            outfile.write(infile.read())




Accuracy=[]
Precision=[]
roc_auc_score=[]
Recall=[]

flag=False
flag1=False
with open('result200.txt','r') as f:
    for line in f:
        if line.startswith('Accuracy '):
            flag=True
        if flag:
            Accuracy.append(line)
        if line.strip().startswith('Recall:'):
           flag=False

print( ''.join(Accuracy))


        if line.startswith('precision:'):
            flag=True
         if flag:
             Precision.append(line)
        if line.startswith('roc_auc_score:'):
            flag=True
         if flag:
             roc_auc_score.append(line)
        if line.startswith('Recall:'):
            flag=True
         if flag:
             Recall.append(line)


print (Accuracy,Precision,roc_auc_score,Recall)



###################################################################################################################