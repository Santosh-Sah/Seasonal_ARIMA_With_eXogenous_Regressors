# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:41:20 2020

@author: Santosh Sah
"""
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

from SeasonalARIMAWitheXogenousRegressorsUtils import (readSeasonalARIMAWitheXogenousRegressorsXTest, 
                                                       readSeasonalARIMAWitheXogenousRegressorsPredictedValues)

"""

calculating SeasonalARIMAWitheXogenousRegressors metrics

"""
def testSeasonalARIMAWitheXogenousRegressorsMetrics():
    
    #reading testing set
    X_test = readSeasonalARIMAWitheXogenousRegressorsXTest()
    
    X_test = X_test[["total"]]
    
    #reading predicted value
    predictedValues = readSeasonalARIMAWitheXogenousRegressorsPredictedValues()
    
    meanSquredError = mean_squared_error(X_test, predictedValues)
    
    meanAbsoluteError = mean_absolute_error(X_test, predictedValues)
    
    rootMeanSquaredError = np.sqrt(mean_squared_error(X_test, predictedValues))
    
    print(meanSquredError) #520.5870393342598
    
    print(meanAbsoluteError) #18.442404329401118
    
    print(rootMeanSquaredError) #22.816376560143368
    
    
    
if __name__ == "__main__":
    testSeasonalARIMAWitheXogenousRegressorsMetrics()