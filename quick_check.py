from time import time
start_time = time()
t0 = time()
print "Importing magical pythons..."
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn import ensemble
from datetime import datetime, date
print "Imported magical pythons:",round(time()-t0,3),"s"

t0 = time()
print "Importing training and testing sets..."
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print "Set imported:",round(time()-t0,3),"s"

t0 = time()
print "Checking amounts..."
for i in np.arange(0,10,.1):
    for r in np.arange(0,10,.1):
        train_reduced = train[(train.x >= i) & (train.x <= (i + .1)) & (train.y >= r) & (train.y <= (r + .1))]
        if i == 9.8 and r == 9.8:
            test_reduced = test[(test.x >= i) & (test.x <= (i + .1)) & (test.y >= r) & (test.y <= (r + .1))]
        elif i == 9.8:
            test_reduced = test[(test.x >= i) & (test.x <= (i + .1)) & (test.y >= r) & (test.y < (r + .1))]
        elif r == 9.8:
            test_reduced = test[(test.x >= i) & (test.x < (i + .1)) & (test.y >= r) & (test.y <= (r + .1))]
        else:
            test_reduced = test[(test.x >= i) & (test.x < (i + .1)) & (test.y >= r) & (test.y < (r + .1))]
        if len(test_reduced) == 0 or len(train_reduced) == 0:
            print "WON'T WORK!!!!!!!!!!!!!!!", i, r, len(train_reduced), len(test_reduced) 
        else:
            print "All Good", i, r, len(train_reduced), len(test_reduced)
print "Checking loop complete:",round((time()-start_time)/60,2),"m"

print "Script End:",round((time()-start_time)/60,2),"m"