# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:51:54 2020

@author: Santosh Sah
"""
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from statsmodels.tsa.statespace.tools import diff

from SeasonalARIMAWitheXogenousRegressorsUtils import (saveSeasonalARIMAWitheXogenousRegressorsModel, 
                                                       readSeasonalARIMAWitheXogenousRegressorsXTrain, 
                                                       importSeasonalARIMAWitheXogenousRegressorsDataset, 
                                                       saveSeasonalARIMAWitheXogenousRegressorsModelForFullDataset,
                                                       agumentedDickeyFullerTest)

from SeasonalARIMAWitheXogenousRegressorsVisualization import (visualizeACFPlot, visualizePACFPlot, visualizeSourceDataPlot,
                                                               visualizeSourceDataPlotWithHolidays, visualizeEtsDecomposition)


"""
Train SeasonalARIMAWitheXogenousRegressors model on training set
"""
def trainSeasonalARIMAWitheXogenousRegressorsModel():
    
    X_train = readSeasonalARIMAWitheXogenousRegressorsXTrain()
    
    X_train["total"] = X_train["total"].astype('float64')
    
    #training model on the training set
    seasonalARIMAWitheXogenousRegressorsModel = SARIMAX(X_train['total'],
                                                        exog=X_train['holiday'], 
                                                        order=(0,0,0),seasonal_order=(1,0,1,7), 
                                                        enforce_invertibility=False)
    
    seasonalARIMAWitheXogenousRegressorsModelFitResult = seasonalARIMAWitheXogenousRegressorsModel.fit()
    
    saveSeasonalARIMAWitheXogenousRegressorsModel(seasonalARIMAWitheXogenousRegressorsModelFitResult)
    
    print(seasonalARIMAWitheXogenousRegressorsModelFitResult.summary())
    
# =============================================================================
#                                      SARIMAX Results
#     =================================================================================
#     Dep. Variable:                     total   No. Observations:                  436
#     Model:             SARIMAX(1, 0, [1], 7)   Log Likelihood               -2098.576
#     Date:                   Fri, 08 May 2020   AIC                           4205.152
#     Time:                           15:27:00   BIC                           4221.462
#     Sample:                       01-01-2016   HQIC                          4211.589
#                                 - 03-11-2017
#     Covariance Type:                     opg
#     ==============================================================================
#                      coef    std err          z      P>|z|      [0.025      0.975]
#     ------------------------------------------------------------------------------
#     holiday       70.0714      3.972     17.640      0.000      62.286      77.857
#     ar.S.L7        1.0000   4.74e-05   2.11e+04      0.000       1.000       1.000
#     ma.S.L7       -0.9556      0.022    -42.738      0.000      -0.999      -0.912
#     sigma2       809.3932     47.196     17.150      0.000     716.890     901.896
#     ===================================================================================
#     Ljung-Box (Q):                       56.41   Jarque-Bera (JB):                21.14
#     Prob(Q):                              0.04   Prob(JB):                         0.00
#     Heteroskedasticity (H):               1.01   Skew:                             0.23
#     Prob(H) (two-sided):                  0.96   Kurtosis:                         3.97
#     ===================================================================================
# =============================================================================
        
"""
Train SeasonalARIMAWitheXogenousRegressors model on full dataset
"""
def trainSeasonalARIMAWitheXogenousRegressorsModelOnFullDataset():
    
    seasonalARIMAWitheXogenousRegressorsDataset = importSeasonalARIMAWitheXogenousRegressorsDataset("RestaurantVisitors.csv")
    
    seasonalARIMAWitheXogenousRegressorsDataset["total"] = seasonalARIMAWitheXogenousRegressorsDataset["total"].astype('float64')
    
    #training model on the whole dataset
    seasonalARIMAWitheXogenousRegressorsModel = SARIMAX(seasonalARIMAWitheXogenousRegressorsDataset['total'],
                                                        exog=seasonalARIMAWitheXogenousRegressorsDataset['holiday'],
                                                        order=(0,0,0),seasonal_order=(1,0,1,7), 
                                                        enforce_invertibility=False)
    
    seasonalARIMAWitheXogenousRegressorsModelFitResult = seasonalARIMAWitheXogenousRegressorsModel.fit()
    
    saveSeasonalARIMAWitheXogenousRegressorsModelForFullDataset(seasonalARIMAWitheXogenousRegressorsModelFitResult)
    
    print(seasonalARIMAWitheXogenousRegressorsModelFitResult.summary())
    
# =============================================================================
#                                      SARIMAX Results
#     =================================================================================
#     Dep. Variable:                     total   No. Observations:                  478
#     Model:             SARIMAX(1, 0, [1], 7)   Log Likelihood               -2290.989
#     Date:                   Fri, 08 May 2020   AIC                           4589.978
#     Time:                           15:27:59   BIC                           4606.656
#     Sample:                       01-01-2016   HQIC                          4596.535
#                                 - 04-22-2017
#     Covariance Type:                     opg
#     ==============================================================================
#                      coef    std err          z      P>|z|      [0.025      0.975]
#     ------------------------------------------------------------------------------
#     holiday       70.1667      3.837     18.286      0.000      62.646      77.688
#     ar.S.L7        1.0000   2.76e-05   3.62e+04      0.000       1.000       1.000
#     ma.S.L7       -1.0300      0.024    -43.225      0.000      -1.077      -0.983
#     sigma2       734.4792     47.088     15.598      0.000     642.188     826.770
#     ===================================================================================
#     Ljung-Box (Q):                       54.07   Jarque-Bera (JB):                22.58
#     Prob(Q):                              0.07   Prob(JB):                         0.00
#     Heteroskedasticity (H):               0.91   Skew:                             0.22
#     Prob(H) (two-sided):                  0.55   Kurtosis:                         3.97
#     ===================================================================================
# =============================================================================

def testIsDatasetStationary():
    
    seasonalARIMAWitheXogenousRegressorsDataset = importSeasonalARIMAWitheXogenousRegressorsDataset("RestaurantVisitors.csv")
    
    #order of p,d,q and P, D, Q is SARIMAX(0,0,0)x(1,0,1,7)
    #hence we do not have take diff to check stationarity.    
    agumentedDickeyFullerTest(seasonalARIMAWitheXogenousRegressorsDataset["total"])
    
# =============================================================================
#     Augmented Dickey-Fuller Test:
#     ADF test statistic       -5.592497
#     p-value                   0.000001
#     # lags used              18.000000
#     # observations          459.000000
#     critical value (1%)      -3.444677
#     critical value (5%)      -2.867857
#     critical value (10%)     -2.570135
#     Strong evidence against the null hypothesis
#     Reject the null hypothesis
#     Data has no unit root and is stationary
# =============================================================================
    
def determineSARIMAXOrderOfPAndQ():
    
    seasonalARIMAWitheXogenousRegressorsDataset = importSeasonalARIMAWitheXogenousRegressorsDataset("RestaurantVisitors.csv")
    
    # For SARIMA Orders we set seasonal=True and pass in an m value
    autoArimaResult = auto_arima(seasonalARIMAWitheXogenousRegressorsDataset["total"], seasonal = True, m = 7, trace = True)
    
    print(autoArimaResult.summary()) #SARIMAX(0,0,0)x(1,0,1,7)
    
# =============================================================================
#      
#     
#     Fit ARIMA(0,0,0)x(1,0,1,7) [intercept=True]; AIC=4782.211, BIC=4798.889, Time=0.682 seconds
#     
#                                      SARIMAX Results
#     =================================================================================
#     Dep. Variable:                         y   No. Observations:                  478
#     Model:             SARIMAX(1, 0, [1], 7)   Log Likelihood               -2387.105
#     Date:                   Fri, 08 May 2020   AIC                           4782.211
#     Time:                           14:41:05   BIC                           4798.889
#     Sample:                                0   HQIC                          4788.768
#                                        - 478
#     Covariance Type:                     opg
#     ==============================================================================
#                      coef    std err          z      P>|z|      [0.025      0.975]
#     ------------------------------------------------------------------------------
#     intercept      5.9554      2.023      2.943      0.003       1.990       9.921
#     ar.S.L7        0.9543      0.015     62.493      0.000       0.924       0.984
#     ma.S.L7       -0.7330      0.054    -13.532      0.000      -0.839      -0.627
#     sigma2      1318.0895     84.030     15.686      0.000    1153.394    1482.785
#     ===================================================================================
#     Ljung-Box (Q):                       72.85   Jarque-Bera (JB):                58.97
#     Prob(Q):                              0.00   Prob(JB):                         0.00
#     Heteroskedasticity (H):               0.86   Skew:                             0.73
#     Prob(H) (two-sided):                  0.33   Kurtosis:                         3.91
#     ===================================================================================
# =============================================================================
    
def plotACFPlot():
    
    seasonalARIMAWitheXogenousRegressorsDataset = importSeasonalARIMAWitheXogenousRegressorsDataset("RestaurantVisitors.csv")
    visualizeACFPlot(seasonalARIMAWitheXogenousRegressorsDataset)

def plotPACFPlot():
    
    seasonalARIMAWitheXogenousRegressorsDataset = importSeasonalARIMAWitheXogenousRegressorsDataset("RestaurantVisitors.csv")
    visualizePACFPlot(seasonalARIMAWitheXogenousRegressorsDataset)

def plotTheSourceData():
    
    seasonalARIMAWitheXogenousRegressorsDataset = importSeasonalARIMAWitheXogenousRegressorsDataset("RestaurantVisitors.csv")
    visualizeSourceDataPlot(seasonalARIMAWitheXogenousRegressorsDataset)

def plotTheSourceDataWithHoliday():
    
    seasonalARIMAWitheXogenousRegressorsDataset = importSeasonalARIMAWitheXogenousRegressorsDataset("RestaurantVisitors.csv")
    visualizeSourceDataPlotWithHolidays(seasonalARIMAWitheXogenousRegressorsDataset)

def etsDecomposition():
    
    seasonalARIMAWitheXogenousRegressorsDataset = importSeasonalARIMAWitheXogenousRegressorsDataset("RestaurantVisitors.csv")
    visualizeEtsDecomposition(seasonalARIMAWitheXogenousRegressorsDataset)
        
if __name__ == "__main__":
    #plotTheSourceData()
    #plotTheSourceDataWithHoliday()
    #etsDecomposition()
    #determineSARIMAXOrderOfPAndQ()
    #testIsDatasetStationary()   
    #plotACFPlot()
    #plotPACFPlot()
    #trainSeasonalARIMAWitheXogenousRegressorsModel()
    trainSeasonalARIMAWitheXogenousRegressorsModelOnFullDataset()
