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

    # choose the correct bin, sort values in time to better simulate
    # the real train/test split for k-fold validation
    training_set = train[(train.x_bin_100 == i_bin_x) & (train.y_bin_100 == i_bin_y)]
    testing_set = test[(test.x_bin_100 == i_bin_x) & (test.y_bin_100 == i_bin_y)]

    print 'Place IDs:', len(np.unique(training_set['place_id']))
    print 'Train length:', len(training_set), 'Test length:', len(testing_set)

    training_set['day_number'] = ((training_set['time']/60)//24).astype(int)
    training_set['hour'] = (training_set['time']//60)%24+1 # 1 to 24
    training_set['dow'] = (training_set['time']//1440)%7+1
    training_set['month'] = (training_set['time']//43200)%12+1 # rough estimate, month = 30 days
    training_set['year'] = (training_set['time']//525600)+1
    # training_set['seconds'] = (training_set['time'] * 60)
    # training_set['date_time'] = pd.to_datetime(training_set['seconds'],unit='s')
    # training_set['hour'] = training_set['date_time'].dt.hour
    # training_set['dow'] = training_set['date_time'].dt.dayofweek
    # training_set['week_of_year'] = training_set['date_time'].dt.weekofyear


    accuracy_means = training_set.groupby(['place_id'], as_index=False)[["x", "y", "accuracy"]].mean()
    time_mean = training_set.groupby(['place_id'], as_index=False)[["hour", "dow", "month"]].mean()

    accuracy_kde = neighbors.KernelDensity(kernel='gaussian', bandwidth=0.2).fit(accuracy_means[["x", "y", "accuracy"]].values)
    time_kde = neighbors.KernelDensity(kernel='gaussian', bandwidth=0.2).fit(time_mean[["hour", "dow", "month"]].values)


    training_set['acc_kde'] = accuracy_kde.score_samples(training_set[["x", "y", "accuracy"]].values)
    training_set['time_kde'] = time_kde.score_samples(training_set[["hour", "dow", "month"]].values)
    mean_group = training_set.groupby(['hour'], as_index=False)[["x", "y"]].mean()
    training_set = pd.merge(training_set, mean_group, on='hour')
    training_set['acc_norm'] = preprocessing.MinMaxScaler(feature_range=(0,25)).fit_transform(np.array(training_set.accuracy.astype(np.float)).reshape((len(training_set.accuracy), 1)))
    #     training_set['time_proportion'] = ((training_set.time - training_set.time.min()) / (training_set.time.max() - training_set.time.min()))

    #     training_set['r'] = np.sqrt((training_set.x_x-training_set.x_y)**2+(training_set.y_x-training_set.y_y)**2)
    #     training_set['minute'] = training_set['date_time'].dt.minute

    #     training_set['day'] = training_set['date_time'].dt.day
    training_set.x_x.replace(0, .0001, inplace=True)
    training_set.y_x.replace(0, .0001, inplace=True)
    training_set['squadd']= (training_set.x_x**2 + training_set.y_x**2)
    #     training_set['acc_squ'] = (training_set.accuracy**2 / (training_set.x / training_set.y))
    #     training_set['acc_x'] = (training_set.accuracy * training_set.x)
    #     training_set['acc_y'] = (training_set.accuracy * training_set.y)
    #     training_set['time_change'] = ((training_set.time - np.mean(training_set.time))/np.std(training_set.time))

    testing_set['hour'] = (testing_set['time']//60)%24+1 # 1 to 24
    testing_set['dow'] = (testing_set['time']//1440)%7+1
    testing_set['month'] = (testing_set['time']//43200)%12+1 # rough estimate, month = 30 days
    testing_set['year'] = (testing_set['time']//525600)+1

    # testing_set['seconds'] = (testing_set['time'] * 60)
    # testing_set['date_time'] = pd.to_datetime(testing_set['seconds'],unit='s')
    # testing_set['hour'] = testing_set['date_time'].dt.hour
    # testing_set['dow'] = testing_set['date_time'].dt.dayofweek
    # testing_set['week_of_year'] = testing_set['date_time'].dt.weekofyear

    testing_set['acc_kde'] = accuracy_kde.score_samples(testing_set[["x", "y", "accuracy"]].values)
    testing_set['time_kde'] = time_kde.score_samples(testing_set[["hour", "dow", "month"]].values)

    mean_group = testing_set.groupby(['hour'], as_index=False)[["x", "y"]].mean()
    testing_set = pd.merge(testing_set, mean_group, on='hour')
    testing_set['acc_norm'] = preprocessing.MinMaxScaler(feature_range=(0,25)).fit_transform(np.array(testing_set.accuracy.astype(np.float)).reshape((len(testing_set.accuracy), 1)))
    #     testing_set['time_proportion'] = (abs((testing_set.time - testing_set.time.min()) - testing_set.time.max()) / (testing_set.time.max() - testing_set.time.min()))

    #     testing_set['r'] = np.sqrt((testing_set.x_x-testing_set.x_y)**2+(testing_set.y_x-testing_set.y_y)**2)
    #     testing_set['minute'] = testing_set['date_time'].dt.minute

    testing_set.x_x.replace(0, .0001, inplace=True)
    testing_set.y_x.replace(0, .0001, inplace=True)
    testing_set['squadd']= (testing_set.x_x**2 + testing_set.y_x**2)
    #     testing_set['acc_squ'] = (testing_set.accuracy**2 / (testing_set.x / testing_set.y))
    #     testing_set['acc_x'] = (testing_set.accuracy * testing_set.x)
    #     testing_set['acc_y'] = (testing_set.accuracy * testing_set.y)
    #     testing_set['time_change'] = ((testing_set.time - np.mean(testing_set.time))/np.std(testing_set.time))

    features = [c for c in training_set.columns if c in ['year','month','dow','squadd','hour','time_kde','x_x', 'y_x','acc_norm']]


    #     features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(train_in_bin[features], train_in_bin['place_id'], test_size=0.70)
    forest = ensemble.RandomForestClassifier(n_estimators=estimator, min_samples_leaf=5, n_jobs=-1).fit(training_set[features], training_set['place_id'])
    #     boost = xgb.XGBClassifier(objective='multi:softprob', n_estimators=estimator, nthread=4).fit(training_set[features], training_set['place_id'])
    probs = pd.DataFrame(forest.predict_proba(testing_set[features]))
    probs.columns = np.unique(training_set['place_id'].sort_values().values)
    preds = pd.DataFrame([list([r.sort_values(ascending=False)[:3].index.values]) for i,r in probs.iterrows()])

    
    ids.append(list(testing_set['row_id'].values))
    predictions.append(preds[0])
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