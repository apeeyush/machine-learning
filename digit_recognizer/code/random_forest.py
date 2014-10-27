import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

# Load the train file into a dataframe
train_df = pd.read_csv('../data/train.csv', header=0)

# TEST DATA
test_df = pd.read_csv('../data/test.csv', header=0)        # Load the test file into a dataframe

# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values
ids = list(range(1,len(test_data)+1))

print 'Training...'
forest = RandomForestClassifier(n_estimators=1000)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting...'
output = forest.predict(test_data).astype(int)


predictions_file = open("../data/random_forest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["ImageId","Label"])
open_file_object.writerows(zip(ids,output))
predictions_file.close()
print 'Done.'
