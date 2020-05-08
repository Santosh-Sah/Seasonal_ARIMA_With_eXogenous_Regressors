# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:52:57 2020

@author: Santosh Sah
"""

from SeasonalARIMAWitheXogenousRegressorsUtils import (importSeasonalARIMAWitheXogenousRegressorsDataset, 
                                                       saveSeasonalARIMAWitheXogenousRegressorsForecastedValues,
                                                       readSeasonalARIMAWitheXogenousRegressorsForecastedValues, 
                                                       readSeasonalARIMAWitheXogenousRegressorsModelForFullDataset,
                                                       importSeasonalARIMAWitheXogenousRegressorsDatasetWithMissingDataForHoliday)

from SeasonalARIMAWitheXogenousRegressorsVisualization import (visualizeSeasonalARIMAWitheXogenousRegressorsForecastedValues,
                                                               visualizeSeasonalARIMAWitheXogenousRegressorsForecastedValuesWithHolidays)

def forecastSeasonalARIMAWitheXogenousRegressorsModel():
    
    #reading the model whichis trained on the whole dataset
    seasonalARIMAWitheXogenousRegressorsModel = readSeasonalARIMAWitheXogenousRegressorsModelForFullDataset()
    
    #reading the dataset
    seasonalARIMAWitheXogenousRegressorsDataset = importSeasonalARIMAWitheXogenousRegressorsDataset("RestaurantVisitors.csv")
    
    #reading the dataset with holidays
    seasonalARIMAWitheXogenousRegressorsDatasetWithHoliday = importSeasonalARIMAWitheXogenousRegressorsDatasetWithMissingDataForHoliday("RestaurantVisitors.csv")
    
    exog_forecast = seasonalARIMAWitheXogenousRegressorsDatasetWithHoliday[478:][['holiday']]
    
    #forecasting for 38 months
    seasonalARIMAWitheXogenousRegressorsForecastedValues = seasonalARIMAWitheXogenousRegressorsModel.predict(len(seasonalARIMAWitheXogenousRegressorsDataset),
                                                                                                             len(seasonalARIMAWitheXogenousRegressorsDataset)+38,
                                                                                                             exog=exog_forecast).rename("SARIMAX(0, 0, 0)x(1, 0, 1, 7) Prediction")
    #saving the forecasted values
    saveSeasonalARIMAWitheXogenousRegressorsForecastedValues(seasonalARIMAWitheXogenousRegressorsForecastedValues)

def plotSeasonalARIMAWitheXogenousRegressorsForecastedValues():
    
    #reading the dataset
    seasonalARIMAWitheXogenousRegressorsDataset = importSeasonalARIMAWitheXogenousRegressorsDataset("RestaurantVisitors.csv")
    
    #reading the forecated values
    seasonalARIMAWitheXogenousRegressorsForecastedValues = readSeasonalARIMAWitheXogenousRegressorsForecastedValues()
    
    #visualizing the forecated values
    visualizeSeasonalARIMAWitheXogenousRegressorsForecastedValues(seasonalARIMAWitheXogenousRegressorsDataset, 
                                                                  seasonalARIMAWitheXogenousRegressorsForecastedValues)

def plotSeasonalARIMAWitheXogenousRegressorsForecastedValuesWithHolidays():
    
    #reading the dataset with holidays
    seasonalARIMAWitheXogenousRegressorsDatasetWithHoliday = importSeasonalARIMAWitheXogenousRegressorsDatasetWithMissingDataForHoliday("RestaurantVisitors.csv")
    
    #reading the dataset
    seasonalARIMAWitheXogenousRegressorsDataset = importSeasonalARIMAWitheXogenousRegressorsDataset("RestaurantVisitors.csv")
    
    #reading the forecated values
    seasonalARIMAWitheXogenousRegressorsForecastedValues = readSeasonalARIMAWitheXogenousRegressorsForecastedValues()
    
    #visualizing the forecated values
    visualizeSeasonalARIMAWitheXogenousRegressorsForecastedValuesWithHolidays(seasonalARIMAWitheXogenousRegressorsDataset, 
                                                                  seasonalARIMAWitheXogenousRegressorsForecastedValues, 
                                                                  seasonalARIMAWitheXogenousRegressorsDatasetWithHoliday)
    


if __name__ == "__main__":
    #forecastSeasonalARIMAWitheXogenousRegressorsModel()
    #plotSeasonalARIMAWitheXogenousRegressorsForecastedValues()
    plotSeasonalARIMAWitheXogenousRegressorsForecastedValuesWithHolidays()
    