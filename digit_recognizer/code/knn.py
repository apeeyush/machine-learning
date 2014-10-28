import pandas as pd
import numpy as np
import csv as csv
from sklearn.neighbors import KNeighborsClassifier
from time import gmtime, strftime

# Load the train file into a dataframe
train_df = pd.read_csv('../data/train.csv', header=0)

# Load the test file into a dataframe
test_df = pd.read_csv('../data/test.csv', header=0)

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values
ids = list(range(1,len(test_data)+1))

print 'Fitting...'
knn = KNeighborsClassifier(n_neighbors=5, algorithm="kd_tree")
print knn.fit(train_data[0::,1::], train_data[0::,0])

print 'Predicting... Started at', strftime("%Y-%m-%d %H:%M:%S", gmtime())
predictions = knn.predict(test_data).astype(int)

predictions_file = open("../data/knn.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ImageId","Label"])
open_file_object.writerows(zip(ids,predictions))
predictions_file.close()
print 'Done. Time', strftime("%Y-%m-%d %H:%M:%S", gmtime())
