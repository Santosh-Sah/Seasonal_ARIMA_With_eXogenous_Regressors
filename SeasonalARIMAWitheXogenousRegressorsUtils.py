# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:50:57 2020

@author: Santosh Sah
"""
import pandas as pd
import pickle
from statsmodels.tsa.stattools import adfuller
"""
Import dataset and read specific column.
"""
def importSeasonalARIMAWitheXogenousRegressorsDataset(seasonalARIMAWitheXogenousRegressorsDatasetFileName):
    
    seasonalARIMAWitheXogenousRegressorsDataset = pd.read_csv(seasonalARIMAWitheXogenousRegressorsDatasetFileName, index_col='date',parse_dates=True)
    
    #the dataset is daily dataset. Hence setting its frequency as daily.
    seasonalARIMAWitheXogenousRegressorsDataset.index.freq = "D"
    
    #dropping null values
    seasonalARIMAWitheXogenousRegressorsDataset = seasonalARIMAWitheXogenousRegressorsDataset.dropna()
    
    #change the data types of the columns
    columns = ['rest1','rest2','rest3','rest4','total']
    
    for col in columns:
        seasonalARIMAWitheXogenousRegressorsDataset[col] = seasonalARIMAWitheXogenousRegressorsDataset[col].astype(int)
    
    return seasonalARIMAWitheXogenousRegressorsDataset

def importSeasonalARIMAWitheXogenousRegressorsDatasetWithMissingDataForHoliday(seasonalARIMAWitheXogenousRegressorsDatasetFileName):
    
    seasonalARIMAWitheXogenousRegressorsDataset = pd.read_csv(seasonalARIMAWitheXogenousRegressorsDatasetFileName, index_col='date',parse_dates=True)
    
    #the dataset is daily dataset. Hence setting its frequency as daily.
    seasonalARIMAWitheXogenousRegressorsDataset.index.freq = "D"
    
# =============================================================================
#     #change the data types of the columns
#     columns = ['rest1','rest2','rest3','rest4','total']
#     
#     for col in columns:
#         seasonalARIMAWitheXogenousRegressorsDataset[col] = seasonalARIMAWitheXogenousRegressorsDataset[col].astype(int)
# =============================================================================
    
    return seasonalARIMAWitheXogenousRegressorsDataset

#splitting dataset into training and testing set
def splitSeasonalARIMAWitheXogenousRegressorsDataset(seasonalARIMAWitheXogenousRegressorsDataset):
    
    #splitting the dataset into training and testing set.
    seasonalARIMAWitheXogenousRegressorsTrainingSet = seasonalARIMAWitheXogenousRegressorsDataset.iloc[:436]
    seasonalARIMAWitheXogenousRegressorsTestingSet = seasonalARIMAWitheXogenousRegressorsDataset.iloc[436:]
    
    return seasonalARIMAWitheXogenousRegressorsTrainingSet, seasonalARIMAWitheXogenousRegressorsTestingSet

#test dataset is stationary or non stationary
def agumentedDickeyFullerTest(series,title=''):
    
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")
        
"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)

"""
read X_train from pickle file
"""
def readSeasonalARIMAWitheXogenousRegressorsXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readSeasonalARIMAWitheXogenousRegressorsXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
Save SeasonalARIMAWitheXogenousRegressors as a pickle file.
"""
def saveSeasonalARIMAWitheXogenousRegressorsModel(seasonalARIMAWitheXogenousRegressorsModel):
    
    #Write SeasonalARIMAWitheXogenousRegressorsModel as a picke file
    with open("SeasonalARIMAWitheXogenousRegressorsModel.pkl",'wb') as seasonalARIMAWitheXogenousRegressorsModel_Pickle:
        pickle.dump(seasonalARIMAWitheXogenousRegressorsModel, seasonalARIMAWitheXogenousRegressorsModel_Pickle, protocol = 2)

"""
read SeasonalARIMAWitheXogenousRegressors from pickle file
"""
def readSeasonalARIMAWitheXogenousRegressorsModel():
    
    #load SeasonalARIMAWitheXogenousRegressorsModel model
    with open("SeasonalARIMAWitheXogenousRegressorsModel.pkl","rb") as seasonalARIMAWitheXogenousRegressorsModel:
        seasonalARIMAWitheXogenousRegressorsModel = pickle.load(seasonalARIMAWitheXogenousRegressorsModel)
    
    return seasonalARIMAWitheXogenousRegressorsModel

"""
Save SeasonalARIMAWitheXogenousRegressors as a pickle file.
"""
def saveSeasonalARIMAWitheXogenousRegressorsModelForFullDataset(seasonalARIMAWitheXogenousRegressorsModelForFullDataset):
    
    #Write SeasonalARIMAWitheXogenousRegressorsModelForFullDataset as a picke file
    with open("SeasonalARIMAWitheXogenousRegressorsModelForFullDataset.pkl",'wb') as seasonalARIMAWitheXogenousRegressorsModelForFullDataset_Pickle:
        pickle.dump(seasonalARIMAWitheXogenousRegressorsModelForFullDataset, seasonalARIMAWitheXogenousRegressorsModelForFullDataset_Pickle, protocol = 2)

"""
read SeasonalARIMAWitheXogenousRegressors from pickle file
"""
def readSeasonalARIMAWitheXogenousRegressorsModelForFullDataset():
    
    #load SeasonalARIMAWitheXogenousRegressorsModelForFullDataset model
    with open("SeasonalARIMAWitheXogenousRegressorsModelForFullDataset.pkl","rb") as seasonalARIMAWitheXogenousRegressorsModelForFullDataset:
        seasonalARIMAWitheXogenousRegressorsModelForFullDataset = pickle.load(seasonalARIMAWitheXogenousRegressorsModelForFullDataset)
    
    return seasonalARIMAWitheXogenousRegressorsModelForFullDataset

"""
save SeasonalARIMAWitheXogenousRegressors PredictedValues as a pickle file
"""

def saveSeasonalARIMAWitheXogenousRegressorsPredictedValues(seasonalARIMAWitheXogenousRegressorsPredictedValues):
    
    #Write SeasonalARIMAWitheXogenousRegressorsPredictedValues in a picke file
    with open("SeasonalARIMAWitheXogenousRegressorsPredictedValues.pkl",'wb') as seasonalARIMAWitheXogenousRegressorsPredictedValues_Pickle:
        pickle.dump(seasonalARIMAWitheXogenousRegressorsPredictedValues, seasonalARIMAWitheXogenousRegressorsPredictedValues_Pickle, protocol = 2)

"""
read SeasonalARIMAWitheXogenousRegressors PredictedValues from pickle file
"""
def readSeasonalARIMAWitheXogenousRegressorsPredictedValues():
    
    #load SeasonalARIMAWitheXogenousRegressorsPredictedValues
    with open("SeasonalARIMAWitheXogenousRegressorsPredictedValues.pkl","rb") as seasonalARIMAWitheXogenousRegressorsPredictedValues_pickle:
        seasonalARIMAWitheXogenousRegressorsPredictedValues = pickle.load(seasonalARIMAWitheXogenousRegressorsPredictedValues_pickle)
    
    return seasonalARIMAWitheXogenousRegressorsPredictedValues

"""
save SeasonalARIMAWitheXogenousRegressors ForecastedValues as a pickle file
"""

def saveSeasonalARIMAWitheXogenousRegressorsForecastedValues(seasonalARIMAWitheXogenousRegressorsForecastedValues):
    
    #Write SeasonalARIMAWitheXogenousRegressorsForecastedValues in a picke file
    with open("SeasonalARIMAWitheXogenousRegressorsForecastedValues.pkl",'wb') as seasonalARIMAWitheXogenousRegressorsForecastedValues_Pickle:
        pickle.dump(seasonalARIMAWitheXogenousRegressorsForecastedValues, seasonalARIMAWitheXogenousRegressorsForecastedValues_Pickle, protocol = 2)

"""
read SeasonalARIMAWitheXogenousRegressorsForecastedValues from pickle file
"""
def readSeasonalARIMAWitheXogenousRegressorsForecastedValues():
    
    #load SeasonalARIMAWitheXogenousRegressorsForecastedValues
    with open("SeasonalARIMAWitheXogenousRegressorsForecastedValues.pkl","rb") as seasonalARIMAWitheXogenousRegressorsForecastedValues_pickle:
        seasonalARIMAWitheXogenousRegressorsForecastedValues = pickle.load(seasonalARIMAWitheXogenousRegressorsForecastedValues_pickle)
    
    return seasonalARIMAWitheXogenousRegressorsForecastedValues


