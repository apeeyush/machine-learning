import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv as csv
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from math import *
from datetime import datetime
import sklearn.cluster as cluster
from sklearn.decomposition import PCA


def split_list(alist, wanted_parts=1):
    length = len(alist)
    return [ alist[i*length // wanted_parts: (i+1)*length // wanted_parts] 
            for i in range(wanted_parts) ]

def rmsle(predicted, actual):
    error = 0
    for i in range(len(actual)):
        error += pow(log(actual[i]+1)-log(predicted[i]+1), 2)
    return sqrt(error/len(actual))

def remove_negative(items):
    newlist = []
    for item in items:
        if item>0:
            newlist.append(item)
        else:
            newlist.append(0)
    return newlist

def getTimeData(df):
    datetime_values = df['datetime'].values
    hour_values = []
    for datetime_value in datetime_values:
        datetime_object = datetime.strptime(datetime_value, '%Y-%m-%d %H:%M:%S')
        hour_values.append(datetime_object.hour)
    df['hour'] = hour_values
    return df

def getMonthData(df):
    datetime_values = df['datetime'].values
    month_values = []
    for datetime_value in datetime_values:
        datetime_object = datetime.strptime(datetime_value, '%Y-%m-%d %H:%M:%S')
        month_values.append(datetime_object.month)
    df['month'] = month_values
    return df

def transform_data(df):
    epoch = datetime.utcfromtimestamp(0)
    datetime_values = df['datetime'].values
    time_values = []
    date_values = []
    month_values = []
    year_values =[]
    weekday_values = []
    isSunday_values = []
    time_since_epoch_values = []
    hour_cluster_values = []
    month_cluster_values = []
    for datetime_value in datetime_values:
        datetime_object = datetime.strptime(datetime_value, '%Y-%m-%d %H:%M:%S')
        time_values.append(datetime_object.hour)
        date_values.append(datetime_object.day)
        month_values.append(datetime_object.month)
        year_values.append(datetime_object.year-2011)
        weekday_values.append(datetime_object.weekday())
        isSunday_values.append(1 if datetime_object.weekday() == 6 else 0)
        time_since_epoch_values.append(int((datetime_object-epoch).total_seconds()/3600))
        hour_cluster_values.append(hour_clusters[datetime_object.hour])
        month_cluster_values.append(month_clusters[datetime_object.month-1])
    df['time'] = time_values
    df['date'] = date_values
    df['month'] = month_values
    df['year'] = year_values
    df['weekday'] = weekday_values
    df['isSunday'] = isSunday_values
    df['time_since_epoch'] = time_since_epoch_values
    df['hourCluster'] = hour_cluster_values
    df['monthCluster'] = month_cluster_values
    return df

def train_test_split(X, y, date):
    indexes = list(np.where(date == 19)[0])
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(len(X)):
        if date[i] == 19:
            X_test.append(X[i,:])
            y_test.append(y[i,:])
        else:
            X_train.append(X[i,:])
            y_train.append(y[i,:])
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test

def validate():
    # Validation
    X_train, X_test, y_train, y_test = train_test_split(X_train_data[0::,0::], y_train_data[0::,0::], X_train_date)

    ## GBM ##
    #  Parameter Tuning #
    if algo == 'gbm':
        #  n_estimators : 100 better than 1000
        #  max_depth    : 6 better than 7 (5 slightly better than 6)
        ## Casual GBM
        gbm_casual = GradientBoostingRegressor(n_estimators=gbm_estimators, max_depth = gbm_depth, random_state = 0)
        gbm_casual.fit(X_train, y_train[0::,1])
        output_gbm_casual = gbm_casual.predict(X_test)
        output_gbm_casual = [int(exp(x)-1) for x in output_gbm_casual]
        ## Resistered GBM
        gbm_registered = GradientBoostingRegressor(n_estimators=gbm_estimators, max_depth = gbm_depth, random_state = 0)
        gbm_registered.fit(X_train, y_train[0::,2])
        output_gbm_registered = gbm_registered.predict(X_test)
        output_gbm_registered = [int(exp(x)-1) for x in output_gbm_registered]
        ## Combining GBM output
        output = [x + y for x, y in zip(output_gbm_casual, output_gbm_registered)]
    elif algo == 'rf':
        ## Random Forest ##
        #  Parameter Tuning #
        #  min_sample_split  :  11 better than 10
        ## Casual Random Forest
        rf_casual = RandomForestRegressor(n_estimators=rf_estimators, min_samples_split = rf_split, random_state = 0, n_jobs = -1)
        rf_casual.fit(X_train, y_train[0::,1])
        output_rf_casual = rf_casual.predict(X_test)
        output_rf_casual = [int(exp(x)-1) for x in output_rf_casual]
        ## Resistered Random Forest
        rf_registered = RandomForestRegressor(n_estimators=rf_estimators, min_samples_split = rf_split, random_state = 0, n_jobs = -1)
        rf_registered.fit(X_train, y_train[0::,2])
        output_rf_registered = rf_registered.predict(X_test)
        output_rf_registered = [int(exp(x)-1) for x in output_rf_registered]
        ## Combine rf output
        output = [x + y for x, y in zip(output_rf_casual, output_rf_registered)]

    output = [log(1+x) for x in output]
    error = rmsle(output, y_test[0::,0])
    print error
    return error


if __name__ == '__main__':
    validation = True
    toCategorical = False
    algo = 'rf'

    df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')

    hour_df = getTimeData(df)
    hour_cluster_data = hour_df.groupby(['hour']).agg(lambda x: x.mean())[['count']]
    hour_clust = cluster.KMeans(n_clusters=6)
    hour_clusters = np.array(hour_clust.fit_predict(split_list(hour_cluster_data.iloc[:,0].values,24)))

    month_df = getMonthData(df)
    month_cluster_data = month_df.groupby(['month']).agg(lambda x: x.mean())[['count']]
    month_clust = cluster.KMeans(n_clusters=4)
    month_clusters = np.array(month_clust.fit_predict(split_list(month_cluster_data.iloc[:,0].values,12)))

    df = transform_data(df)
    test_df = transform_data(test_df)

    df['count'] = [log(1+x) for x in df['count']]
    df['casual'] = [log(1+x) for x in df['casual']]
    df['registered'] = [log(1+x) for x in df['registered']]

    # Convert data to categorical
    if toCategorical:
        print 'Converting variables to categorical..'
        df['weather'] = df['weather'].astype('category')
        df['holiday'] = df['holiday'].astype('category')
        df['workingday'] = df['workingday'].astype('category')
        df['season'] = df['season'].astype('category')
        df['time'] = df['time'].astype('category')

    # Adding 'time_since_epoch' overfits (on kaggle as well on validation)
    # Adding 'month' overfits (on kaggle as well as on validation)
    # Removing 'holiday' reduces the accuracy (on kaggle)
    # Adding 'isSunday' overfits (on kaggle as well as on validation)
    # Adding 'hourCluster' overfits (on validation)
    # Adding 'monthCluster' performs similar
    features = ['season','holiday','workingday','weather','temp','atemp','humidity','windspeed','time','weekday','year']
    X_train_date = df[['date']].values
    X_train_data = df[features].values
    y_train_data = df[['count', 'casual', 'registered']].values

    if algo == 'gbm':
        # Estimating depth
        gbm_estimators = 120
        x_vals = [2,3,4,5,6,7,8,9,10,11,12,13,14]
        error_vals = []
        for gbm_depth in x_vals:
            error = validate()
            error_vals.append(error)
        plt.plot(x_vals, error_vals)
        plt.ylabel('Error')
        plt.xlabel('Depth of tree')
        plt.savefig('gbm_depth_tuning.png')

        # Estimating number of estimators
        gbm_depth = 6
        x_vals = [20,50,80,100,120,200,400,500,1000,2000,3000]
        error_vals = []
        for gbm_estimators in x_vals:
            error = validate()
            error_vals.append(error)
        plt.plot(x_vals, error_vals)
        plt.ylabel('Error')
        plt.xlabel('Number of estimators')
        plt.savefig('gbm_estimator_tuning.png')

    elif algo == 'rf':
        # Estimating depth
        rf_estimators = 1000
        x_vals = [5,6,7,8,9,10,11,12,13,14,15]
        error_vals = []
        for rf_split in x_vals:
            error = validate()
            error_vals.append(error)
        plt.plot(x_vals, error_vals)
        plt.ylabel('Error')
        plt.xlabel('rf split')
        plt.savefig('rf_split_tuning.png')
        plt.show()

        # Estimating number of estimators
        rf_split = 11
        x_vals = [20,50,80,100,200,500,1000,2000,3000,4000]
        error_vals = []
        for rf_estimators in x_vals:
            error = validate()
            error_vals.append(error)
        plt.plot(x_vals, error_vals)
        plt.ylabel('Error')
        plt.xlabel('Number of estimators')
        plt.savefig('rf_estimator_tuning.png')
