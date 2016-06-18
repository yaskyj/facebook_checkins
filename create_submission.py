from time import time
start_time = time()
t0 = time()
print "Importing magical pythons..."
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
print "Imported magical pythons:",round(time()-t0,3),"s"

t0 = time()
print "Importing submission files..."
submission_1 = pd.read_csv('submissions/submission-0-2.csv')
submission_2 = pd.read_csv('submissions/submission-2-4.csv')
submission_3 = pd.read_csv('submissions/submission-4-6.csv')
submission_4 = pd.read_csv('submissions/submission-6-8.csv')
submission_5 = pd.read_csv('submissions/submission-8-10.csv')
print "Files imported:",round(time()-t0,3),"s"

t0 = time()
print "Concating submission files..."
submission_concat = pd.concat([submission_1,submission_2,submission_3,submission_4, submission_5], ignore_index=True)
print "Files concated:",round(time()-t0,3),"s"

t0 = time()
print "Removing any duplicates..."
submission_concat.drop_duplicates('row_id', inplace=True)
print "Duplicates removed:",round(time()-t0,3),"s"

t0 = time()
print "Sorting concated files by row_ids..."
submission_concat.sort_values('row_id', inplace=True)
print "Concated files sorted:",round(time()-t0,3),"s"

t0 = time()
print "Exporting complete submission file..."
submission_concat.to_csv('final_submission.csv', index=False)
print "Submission file exported:",round(time()-t0,3),"s"

print "Script End:",round((time()-start_time)/60,2),"m"