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
print "Create ids and predictions lists..."
ids = []
predictions = []
print "Lists created:",round(time()-t0,3),"s"

t0 = time()
print "Importing training and testing sets..."
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
print "Set imported:",round(time()-t0,3),"s"

t0 = time()
print "Running training loop..."
for i in np.arange(8,10,.1):
    for r in np.arange(0,10,.1):
        train_reduced = train[(train.x >= i) & (train.x <= (i + .1)) & (train.y >= r) & (train.y <= (r + .1))]
        if i == 9.9 and r == 9.9:
            test_reduced = test[(test.x >= i) & (test.x <= (i + .1)) & (test.y >= r) & (test.y <= (r + .1))]
        elif i == 9.9:
            test_reduced = test[(test.x >= i) & (test.x <= (i + .1)) & (test.y >= r) & (test.y < (r + .1))]
        elif r == 9.9:
            test_reduced = test[(test.x >= i) & (test.x < (i + .1)) & (test.y >= r) & (test.y <= (r + .1))]
        else:
            test_reduced = test[(test.x >= i) & (test.x < (i + .1)) & (test.y >= r) & (test.y < (r + .1))]


        print 'Test: ',i, (i + .1), r, (r + .1), len(train_reduced), len(test_reduced)
        t1 = time()
        train_reduced['seconds'] = (train_reduced['time'] * 60)
        train_reduced['date_time'] = pd.to_datetime(train_reduced['seconds'],unit='s')
        train_reduced['hour'] = train_reduced['date_time'].dt.hour
        train_reduced['day'] = train_reduced['date_time'].dt.day
        train_reduced['dow'] = train_reduced['date_time'].dt.dayofweek
        train_reduced.x.replace(0, .0001, inplace=True)
        train_reduced.y.replace(0, .0001, inplace=True)
        train_reduced['div']= (train_reduced.x / train_reduced.y)
        train_reduced['multi']= (train_reduced.x * train_reduced.y)
        train_reduced['squadd']= (train_reduced.x**2 + train_reduced.y**2)
        train_reduced['acc_squ'] = (train_reduced.accuracy**2 / (train_reduced.x / train_reduced.y))
        train_reduced['acc_x'] = (train_reduced.accuracy * train_reduced.x)
        train_reduced['acc_y'] = (train_reduced.accuracy * train_reduced.y)

        test_reduced['seconds'] = (test_reduced['time'] * 60)
        test_reduced['date_time'] = pd.to_datetime(test_reduced['seconds'],unit='s')
        test_reduced['hour'] = test_reduced['date_time'].dt.hour
        test_reduced['day'] = test_reduced['date_time'].dt.day
        test_reduced['dow'] = test_reduced['date_time'].dt.dayofweek
        test_reduced.x.replace(0, .0001, inplace=True)
        test_reduced.y.replace(0, .0001, inplace=True)
        test_reduced['div'] = (test_reduced.x / test_reduced.y)
        test_reduced['multi'] = (test_reduced.x * test_reduced.y)
        test_reduced['squadd'] = (test_reduced.x**2 + test_reduced.y**2)
        test_reduced['acc_squ'] = (test_reduced.accuracy**2 / (test_reduced.x / test_reduced.y))
        test_reduced['acc_x'] = (test_reduced.accuracy * test_reduced.x)
        test_reduced['acc_y'] = (test_reduced.accuracy * test_reduced.y)

        features = [c for c in train_reduced.columns if c in ['x', 'y', 'accuracy', 'hour', 'day', 'dow', 'div', 'multi', 'squadd', 'acc_squ', 'acc_x', 'acc_y']]
        
        forest = ensemble.RandomForestClassifier(n_estimators=60, min_samples_leaf=5, n_jobs=-1).fit(train_reduced[features], train_reduced['place_id'])
        
        probs = pd.DataFrame(forest.predict_proba(test_reduced[features]))
        probs.columns = np.unique(train_reduced['place_id'].sort_values().values)
        preds = pd.DataFrame([list([p.sort_values(ascending=False)[:3].index.values]) for x,p in probs.iterrows()])
        
        print 'All Good: ',i, (i + .1), r, (r + .1), len(test_reduced['row_id']), len(preds)
        print "Analysis time:",round(time()-t1,3),"s"
        
        ids.append(list(test_reduced['row_id'].values))
        predictions.append(preds[0])
print "Training loop completed:",round((time()-start_time)/60,2),"m"

t0 = time()
print "Id and predictions lengths..."
print len(ids), len(predictions)
print "Printed lengths:",round(time()-t0,3),"s"

t0 = time()
print "Flattening ids and predictions lists..."
ids = [val for sublist in ids for val in sublist]
predictions = [val for sublist in predictions for val in sublist]
print "Lists flattened:",round(time()-t0,3),"s"

t0 = time()
print "Checking list lengths again..."
print len(ids), len(predictions)
print "List lengths checked:",round(time()-t0,3),"s"

t0 = time()
print "Creating submission file..."
submission = pd.DataFrame()
submission['row_id'] = ids
submission['place_id'] = [' '.join(str(x) for x in y) for y in predictions]
print "Submission file created:",round(time()-t0,3),"s"

t0 = time()
print "Exporting submission file..."
submission.to_csv('submissions/submission-8-10.csv', index=False)
print "Submission file exported:",round(time()-t0,3),"s"

print "Script End:",round((time()-start_time)/60,2),"m"
