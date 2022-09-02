# Mod14_challenge
# UW-finctech-2022
This is  a public repo for the Module 13 Challenge of the UW Fintech Bootcamp in 2022.


## Technologies and Libraries

Jupyter lab
pandas. 1.3.5
scikit-learn 1.0.2


## Installation Guide

Install jupyter lab by running the command jupyter lab in your terminal

Install the following dependencies an dmocdules from the libraries above

```
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import hvplot.pandas
    import matplotlib.pyplot as plt
    from sklearn import svm
    from sklearn.preprocessing import StandardScaler
    from pandas.tseries.offsets import DateOffset
    from sklearn.metrics import classification_report

```


## Overview of the analysis

* Purpose of the analysis

 Assuming the role of a financial advisor at one of the top five financial advisory firms in the world, combine your algorithmic trading skills with your existing skills in financial Python programming and machine learning to create an algorithmic trading bot that learns and adapts to new data and evolving markets.



* Description of the stages of creating thr Algorithmic Trading Bot

     * Establish a Baseline Performance

1. Import the OHLCV dataset into a Pandas DataFrame.
2. Generate trading signals using short- and long-window SMA values.
3. Split the data into training and testing datasets.
4. Use the SVC classifier model from SKLearn's support vector machine (SVM) learning method to fit the training data and make predictions based on the testing data. Review the predictions.
5. Review the classification report associated with the SVC model predictions.
6. Create a predictions DataFrame that contains columns for “Predicted” values, “Actual Returns”, and “Strategy Returns”.
7. Create a cumulative return plot that shows the actual returns vs. the strategy returns. 
8. Write your conclusions about the performance of the baseline trading algorithm.

    * Tune the Baseline Trading Algorithm
In this section,  adjust the model’s input features to find the parameters that result in the best trading outcomes. (Choose the best by comparing the cumulative products of the strategy returns.) 

1. Tune the training algorithm by adjusting the size of the training dataset. To do so, slice your data into different periods. Rerun the notebook with the updated parameters, and record the results.
2. Tune the trading algorithm by adjusting the SMA input features. Adjust one or both of the windows for the algorithm. Rerun the notebook with the updated parameters, and record the results.

    * Evaluate a New Machine Learning Classifier

1. Import a new classifier (LogisticRegression). 
2. Using the original training data as the baseline model, fit another model with the new classifier.
3. Backtest the new model to evaluate its performance. 

   

## Results
SVM MODEL
<img src="https://github.com/Virginia440/Mod14_challenge/blob/main/Images/SVM%20model%20-Actual%20vs%20Strategy%20cumulative%20returns.PNG" width=500 height=300>

    *  Original model's results(SMA 4 and 100 and Dataset 3 months)
<img src="https://github.com/Virginia440/Mod14_challenge/blob/main/Images/Original%20model%20SMA%204%20and%20100%20report%2C%203mnths.PNG" width=500 height=300>
   

    *   Alternative Model 1 results(SMA 4 and 100 and Dataset 6 months)
<img src="https://github.com/Virginia440/Mod14_challenge/blob/main/Images/Diff%20time%20period-%20%206months.PNG" width=500 height=300>

What impact resulted from increasing or decreasing the training window?
   

    *   Alternative Model 2 Results(SMA 50 and 100 and Dataset of 3 months)
<img src="https://github.com/Virginia440/Mod14_challenge/blob/main/Images/Diff%20SMAs%2C%20SMA%2050%20and%20SMA%20100%20report.PNG" width=500 height=300>

What impact resulted from increasing or decreasing either or both of the SMA windows?

LogisticsRegression MODEL
 <img src="https://github.com/Virginia440/Mod14_challenge/blob/main/Images/Lr%20Model%20Actual%20vs%20Strategy%20cumulative%20returns.PNG" width=500 height=300>

 Did this new model perform better or worse than the provided baseline model? Did this new model perform better or worse than your tuned trading algorithm?
---

## Summary
Optimization of the model was tested by using different number of hidden layers for two of the models, and different epochs for two of the models.

    


## License
 The code is made without a license, however, the materials used for research are licensed.
---


