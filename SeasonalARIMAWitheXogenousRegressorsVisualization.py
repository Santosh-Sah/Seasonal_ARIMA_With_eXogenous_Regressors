# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:53:28 2020

@author: Santosh Sah
"""
import pylab
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

def visualizeSeasonalARIMAWitheXogenousRegressorsPredictedValues(seasonalARIMAWitheXogenousRegressorsXTest, 
                                                                 seasonalARIMAWitheXogenousRegressorsPredictedValues):
    
    #plotting the predicted values, and testing set
    title = 'Restaurant Visitors'
    
    ylabel='Visitors per day'
    
    xlabel='' 

    ax = seasonalARIMAWitheXogenousRegressorsXTest['total'].plot(legend=True,figsize=(12,6),title=title)
    
    seasonalARIMAWitheXogenousRegressorsPredictedValues.plot(legend=True)
    
    ax.autoscale(axis='x',tight=True)
    
    ax.set(xlabel=xlabel, ylabel=ylabel)
    
    pylab.savefig('PredeictedValues.png')

def visualizeSeasonalARIMAWitheXogenousRegressorsForecastedValues(seasonalARIMAWitheXogenousRegressorsDataset, 
                                                                  seasonalARIMAWitheXogenousRegressorsForecastedValues):
    
    #plotting the predicted values, and testing set
    title = 'Restaurant Visitors'
    
    ylabel='Visitors per day'
    
    xlabel='' 

    ax = seasonalARIMAWitheXogenousRegressorsDataset['total'].plot(legend=True,figsize=(12,6),title=title)
    
    seasonalARIMAWitheXogenousRegressorsForecastedValues.plot(legend=True)
    
    ax.autoscale(axis='x',tight=True)
    
    ax.set(xlabel=xlabel, ylabel=ylabel)
    
    pylab.savefig('ForecastedValues.png')

def visualizeACFPlot(seasonalARIMAWitheXogenousRegressorsDataset):
    
    title = 'Restaurant Visitors'
    lags = 40
    plot_acf(seasonalARIMAWitheXogenousRegressorsDataset['total'],title=title,lags=lags)
    pylab.savefig('acf_plot.png')

def visualizePACFPlot(seasonalARIMAWitheXogenousRegressorsDataset):
    
    title = 'Restaurant Visitors'
    lags = 40
    plot_pacf(seasonalARIMAWitheXogenousRegressorsDataset['total'],title=title,lags=lags)
    pylab.savefig('pacf_plot.png')

def visualizeSourceDataPlot(seasonalARIMAWitheXogenousRegressorsDataset):
    
    #plotting the source dataset
    title = 'Restaurant Visitors'
    
    ylabel='Visitors per day'
    
    xlabel='' 

    ax = seasonalARIMAWitheXogenousRegressorsDataset['total'].plot(figsize=(16,5),title=title)
    
    ax.autoscale(axis='x',tight=True)
    
    ax.set(xlabel=xlabel, ylabel=ylabel)
    
    pylab.savefig('SourceDatasetPlot.png')
    
def visualizeSourceDataPlotWithHolidays(seasonalARIMAWitheXogenousRegressorsDataset):

    #plotting the source dataset with holidays
    title = 'Restaurant Visitors'
    
    ylabel='Visitors per day'
    
    xlabel='' 

    ax = seasonalARIMAWitheXogenousRegressorsDataset['total'].plot(figsize=(16,5),title=title)
    
    ax.autoscale(axis='x',tight=True)
    
    ax.set(xlabel=xlabel, ylabel=ylabel)
    
    for x in seasonalARIMAWitheXogenousRegressorsDataset.query('holiday==1').index:  # for days where holiday == 1
        
        ax.axvline(x=x, color='k', alpha = 0.3);  # add a semi-transparent grey line
    
    pylab.savefig('SourceDatasetPlotWithHolidays.png')

def visualizeEtsDecomposition(seasonalARIMAWitheXogenousRegressorsDataset):
    
    result = seasonal_decompose(seasonalARIMAWitheXogenousRegressorsDataset["total"])
    
    result.plot()
    
    pylab.savefig('SeasonalDecompose.png')

def visualizeSeasonalARIMAWitheXogenousRegressorsPredictedValuesWithHolidays(seasonalARIMAWitheXogenousRegressorsXTest,
                                                                         seasonalARIMAWitheXogenousRegressorsPredictedValues):
    
    
    #plotting the predicted values, and testing set
    title = 'Restaurant Visitors'
    
    ylabel='Visitors per day'
    
    xlabel='' 

    ax = seasonalARIMAWitheXogenousRegressorsXTest['total'].plot(legend=True,figsize=(12,6),title=title)
    
    seasonalARIMAWitheXogenousRegressorsPredictedValues.plot(legend=True)
    
    ax.autoscale(axis='x',tight=True)
    
    ax.set(xlabel=xlabel, ylabel=ylabel)
    
    for x in seasonalARIMAWitheXogenousRegressorsXTest.query('holiday==1').index: 
        
        ax.axvline(x=x, color='k', alpha = 0.3);
    
    pylab.savefig('PredeictedValuesWithHolidays.png')

def visualizeSeasonalARIMAWitheXogenousRegressorsForecastedValuesWithHolidays(seasonalARIMAWitheXogenousRegressorsDataset, 
                                                                  seasonalARIMAWitheXogenousRegressorsForecastedValues,
                                                                  seasonalARIMAWitheXogenousRegressorsDatasetWithHoliday):
    
    #plotting the predicted values, and testing set
    title = 'Restaurant Visitors'
    
    ylabel='Visitors per day'
    
    xlabel='' 

    ax = seasonalARIMAWitheXogenousRegressorsDataset['total'].plot(legend=True,figsize=(12,6),title=title)
    
    seasonalARIMAWitheXogenousRegressorsForecastedValues.plot(legend=True)
    
    ax.autoscale(axis='x',tight=True)
    
    ax.set(xlabel=xlabel, ylabel=ylabel)
    
    for x in seasonalARIMAWitheXogenousRegressorsDatasetWithHoliday.query('holiday==1').index: 
        
        ax.axvline(x=x, color='k', alpha = 0.3);
    
    pylab.savefig('ForecastedValuesWithHolidays.png')