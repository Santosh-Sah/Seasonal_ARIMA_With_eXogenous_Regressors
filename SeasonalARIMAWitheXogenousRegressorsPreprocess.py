# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:51:38 2020

@author: Santosh Sah
"""

from SeasonalARIMAWitheXogenousRegressorsUtils import (importSeasonalARIMAWitheXogenousRegressorsDataset, saveTrainingAndTestingDataset, 
                                splitSeasonalARIMAWitheXogenousRegressorsDataset)

def preprocess():
    
    seasonalARIMAWitheXogenousRegressorsDataset = importSeasonalARIMAWitheXogenousRegressorsDataset("RestaurantVisitors.csv")
    
    X_train, X_test = splitSeasonalARIMAWitheXogenousRegressorsDataset(seasonalARIMAWitheXogenousRegressorsDataset)
    
    saveTrainingAndTestingDataset(X_train, X_test)
    

if __name__ == "__main__":
    preprocess()