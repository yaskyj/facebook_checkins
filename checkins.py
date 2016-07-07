from time import time
start_time = time()
t0 = time()
print "Importing magical pythons..."
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn import ensemble, neighbors, preprocessing
from datetime import datetime, date
import itertools as itertools
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

def horizontally_bin_data(data, NX, NY):
    """Add columns to data indicating X and Y bins.

    Divides the grid into `NX` bins in X and `NY` bins in Y, and adds columns 
    to `data` containing the bin number in X and Y. 
    """

    NX = int(NX)
    NY = int(NY)

    assert((NX >= 5) and (NX <= 1000))
    assert((NY >= 5) and (NY <= 1000))

    x_bounds = (0., 10.)
    y_bounds = (0., 10.)

    delta_X = (x_bounds[1] - x_bounds[0]) / float(NX)
    delta_Y = (y_bounds[1] - y_bounds[0]) / float(NY)

    # very fast binning algorithm, just divide by delta and round down
    xbins = np.floor((data.x.values - x_bounds[0])
                     / delta_X).astype(np.int32)
    ybins = np.floor((data.y.values - y_bounds[0])
                     / delta_Y).astype(np.int32)

    # some points fall on the upper/right edge of the domain
    # tweak their index to bring them back in the box
    xbins[xbins == NX] = NX-1
    ybins[ybins == NY] = NY-1

    xlabel = 'x_bin_{0:03d}'.format(NX)
    ylabel = 'y_bin_{0:03d}'.format(NY)

    data[xlabel] = xbins
    data[ylabel] = ybins
    return

t0 = time()
print "Binning data..."
horizontally_bin_data(train, 100, 100)
horizontally_bin_data(test, 100, 100)
print "Data binned:",round(time()-t0,3),"s"

t0 = time()
print "Running training loop..."
estimator = 60
#Choose this line for the whole dataset.
for i_bin_x, i_bin_y in itertools.product(xrange(100), xrange(100)):
# for i_bin_x, i_bin_y in bin_numbers:
    t1 = time()
    print("Bin {},{}".format(i_bin_x, i_bin_y))

    training_set = train[(train.x_bin_100 == i_bin_x) & (train.y_bin_100 == i_bin_y)]
    testing_set = test[(test.x_bin_100 == i_bin_x) & (test.y_bin_100 == i_bin_y)]

    print 'Place IDs:', len(np.unique(training_set['place_id']))
    print 'Train length:', len(training_set), 'Test length:', len(testing_set)

    minute = 2*np.pi*((training_set["time"]//5)%288)/288
    training_set['minute'] = minute
    training_set['minute_sin'] = (np.sin(minute)+1).round(4)
    training_set['minute_cos'] = (np.cos(minute)+1).round(4)
    del minute
    day = 2*np.pi*((training_set['time']//1440)%365)/365
    training_set['day'] = day
    training_set['day_of_year_sin'] = (np.sin(day)+1).round(4)
    training_set['day_of_year_cos'] = (np.cos(day)+1).round(4)
    del day
    weekday = 2*np.pi*((training_set['time']//1440)%7)/7
    training_set['weekday'] = weekday
    training_set['weekday_sin'] = (np.sin(weekday)+1).round(4)
    training_set['weekday_cos'] = (np.cos(weekday)+1).round(4)
    del weekday
    training_set['year'] = (((training_set['time'])//525600))
    training_set.drop(['time'], axis=1, inplace=True)
    training_set['month'] = ((training_set['weekday']//30)%12+1)*2.73
    training_set['accuracy'] = np.log10(training_set['accuracy'])*14.4

    training_set.loc[:,'x'] *= 465.0
    training_set.loc[:,'y'] *= 975.0
    training_set['squadd']= (training_set.x**2 + training_set.y**2)

    
    
    minute = 2*np.pi*((testing_set["time"]//5)%288)/288
    testing_set['minute'] = minute
    testing_set['minute_sin'] = (np.sin(minute)+1).round(4)
    testing_set['minute_cos'] = (np.cos(minute)+1).round(4)
    del minute
    day = 2*np.pi*((testing_set['time']//1440)%365)/365
    testing_set['day'] = day
    testing_set['day_of_year_sin'] = (np.sin(day)+1).round(4)
    testing_set['day_of_year_cos'] = (np.cos(day)+1).round(4)
    del day
    weekday = 2*np.pi*((testing_set['time']//1440)%7)/7
    testing_set['weekday'] = weekday
    testing_set['weekday_sin'] = (np.sin(weekday)+1).round(4)
    testing_set['weekday_cos'] = (np.cos(weekday)+1).round(4)
    del weekday
    testing_set['year'] = (((testing_set['time'])//525600))
    testing_set.drop(['time'], axis=1, inplace=True)
    testing_set['month'] = ((testing_set['weekday']//30)%12+1)*2.73
    testing_set['accuracy'] = np.log10(testing_set['accuracy'])*14.4

    testing_set.loc[:,'x'] *= 465.0
    testing_set.loc[:,'y'] *= 975.0
    testing_set['squadd']= (testing_set.x**2 + testing_set.y**2)

    features = [c for c in training_set.columns if c in ['year','month','dow','squadd','hour','time_kde','x_x', 'y_x','acc_norm']]

    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(training_set.place_id.values)

    forest = ensemble.RandomForestClassifier(n_estimators=estimator, min_samples_leaf=5, n_jobs=-1).fit(training_set.drop(['row_id', 'place_id', 'x_bin_100', 'y_bin_100',], axis=1).values, labels)
    probs = forest.predict_proba(testing_set.drop(['row_id', 'x_bin_100', 'y_bin_100'], axis=1).values)

    # probs.columns = np.unique(training_set['place_id'].sort_values().values)
    preds = pd.DataFrame(le.inverse_transform(np.argsort(probs, axis=1)[:,::-1][:,:3]))

    
    ids.append(list(testing_set['row_id'].values))
    predictions.append(preds.values)
    print "Analysis time:",round(time()-t1,3),"s"
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
submission.sort_values('row_id', inplace=True)
print "Submission file created:",round(time()-t0,3),"s"

t0 = time()
print "Exporting submission file..."
submission.to_csv('final_submission.csv', index=False)
print "Submission file exported:",round(time()-t0,3),"s"

print "Script End:",round((time()-start_time)/60,2),"m"