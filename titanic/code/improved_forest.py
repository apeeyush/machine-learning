import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier

# Load the train file into a dataframe
train_df = pd.read_csv('../data/train.csv', header=0)

# Add Gender coumn in DF such that female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

train_df["embarkedPresent"] = 1
train_df.loc[ (train_df.Embarked.isnull()), 'embarkedPresent'] = 0

# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

train_df["agePresent"] = 1
train_df.loc[ (train_df.Age.isnull()), 'agePresent'] = 0

# All the ages with no data -> make the median of all Ages
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df_copy = train_df
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 


# TEST DATA
test_df = pd.read_csv('../data/test.csv', header=0)        # Load the test file into a dataframe

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

test_df["embarkedPresent"] = 1
test_df.loc[ (test_df.Embarked.isnull()), 'embarkedPresent'] = 0

# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)

test_df["agePresent"] = 1
test_df.loc[ (test_df.Age.isnull()), 'agePresent'] = 0


# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
age_present_ids = test_df.PassengerId[test_df.agePresent == 1].values
age_absent_ids = test_df.PassengerId[test_df.agePresent == 0].values

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df_copy = test_df
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 


# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df[ train_df.agePresent == 1 ].values
test_data = test_df[ test_df.agePresent == 1 ].values

print 'Training...'
forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

print 'Predicting...'
output = forest.predict(test_data).astype(int)

predictions_file = open("../data/improved_random_forest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(age_present_ids, output))


train_df_copy = train_df_copy.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId','Age'], axis=1)
test_df_copy = test_df_copy.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId','Age'], axis=1) 
train_data = train_df_copy.values
test_data = test_df_copy[ test_df.agePresent == 0 ].values

forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit( train_data[0::,1::], train_data[0::,0] )

output = forest.predict(test_data).astype(int)

open_file_object.writerows(zip(age_absent_ids, output))
predictions_file.close()
print 'Done.'
