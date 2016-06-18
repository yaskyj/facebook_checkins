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
import itertools as itertools
print "Imported magical pythons:",round(time()-t0,3),"s"

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

    assert((NX >= 5) and (NX <= 2000))
    assert((NY >= 5) and (NY <= 2000))

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
horizontally_bin_data(train, 2000, 2000)
horizontally_bin_data(test, 2000, 2000)
print "Data binned:",round(time()-t0,3),"s"


t0 = time()
print "Running training loop..."

#Choose this line for the whole dataset.
for i_bin_y in xrange(200):
    t1 = time()
    print("Bin {}".format(i_bin_y))

    # choose the correct bin, sort values in time to better simulate
    training_set = train[(train.y_bin_2000 == i_bin_y)]
    testing_set = test[(test.y_bin_2000 == i_bin_y)]

    print "Train length:", len(training_set), "Test length:", len(testing_set)
    print "Analysis time:",round(time()-t1,3),"s"
print "Training loop completed:",round((time()-start_time)/60,2),"m"

print "Script End:",round((time()-start_time)/60,2),"m"