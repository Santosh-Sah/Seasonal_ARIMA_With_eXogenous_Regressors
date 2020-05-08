# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:52:22 2020

@author: Santosh Sah
"""

from SeasonalARIMAWitheXogenousRegressorsUtils import (readSeasonalARIMAWitheXogenousRegressorsXTest, 
                                                       readSeasonalARIMAWitheXogenousRegressorsModel, 
                                                       saveSeasonalARIMAWitheXogenousRegressorsPredictedValues, 
                                                       readSeasonalARIMAWitheXogenousRegressorsXTrain,
                                                       readSeasonalARIMAWitheXogenousRegressorsPredictedValues)

from SeasonalARIMAWitheXogenousRegressorsVisualization import (visualizeSeasonalARIMAWitheXogenousRegressorsPredictedValues,
                                                               visualizeSeasonalARIMAWitheXogenousRegressorsPredictedValuesWithHolidays)


"""
test the model on testing dataset
"""
def testSeasonalARIMAWitheXogenousRegressorsModel():
    
    #reading the training dataset
    X_train = readSeasonalARIMAWitheXogenousRegressorsXTrain()
    
    #reading testing set
    X_test = readSeasonalARIMAWitheXogenousRegressorsXTest()
    
    start = len(X_train)
    
    end = len(X_train) + len(X_test) - 1
    
    exog_forecast = X_test[['holiday']]
    
    #reading model from pickle file
    seasonalARIMAWitheXogenousRegressorsModel = readSeasonalARIMAWitheXogenousRegressorsModel()
    
    #forecasting
    #Passing dynamic=False means that forecasts at each point are generated using the full history up to that point (all lagged values).
    #Passing typ='levels' predicts the levels of the original endogenous variables. 
    #If we'd used the default typ='linear' we would have seen linear predictions in terms of the differenced endogenous variables.
    predictedValues =seasonalARIMAWitheXogenousRegressorsModel.predict(start = start, end = end, exog=exog_forecast).rename("SARIMAX(0, 0, 0)x(1, 0, 1, 7) Prediction")
    
    #saving the foreasted values
    saveSeasonalARIMAWitheXogenousRegressorsPredictedValues(predictedValues)

def plotSeasonalARIMAWitheXogenousRegressorsPredictedValues():
    
    #reading testing set
    X_test = readSeasonalARIMAWitheXogenousRegressorsXTest()
    
    #reading predicted value
    predictedValues = readSeasonalARIMAWitheXogenousRegressorsPredictedValues()
    
    #visualizing the predicted values with training set and the testing set
    visualizeSeasonalARIMAWitheXogenousRegressorsPredictedValues(X_test, predictedValues)

def plotSeasonalARIMAWitheXogenousRegressorsPredictedValuesWithHolidays():
    
    #reading testing set
    X_test = readSeasonalARIMAWitheXogenousRegressorsXTest()
    
    #reading predicted value
    predictedValues = readSeasonalARIMAWitheXogenousRegressorsPredictedValues()
    
    #visualizing the predicted values with training set and the testing set
    visualizeSeasonalARIMAWitheXogenousRegressorsPredictedValuesWithHolidays(X_test, predictedValues)
    
    
if __name__ == "__main__":
    #testSeasonalARIMAWitheXogenousRegressorsModel()
    #plotSeasonalARIMAWitheXogenousRegressorsPredictedValues()
    plotSeasonalARIMAWitheXogenousRegressorsPredictedValuesWithHolidays()