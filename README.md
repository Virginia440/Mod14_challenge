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
**SVM MODEL (Plot of Cumulated Actual returns vs Strategy Returns)**

<img src="https://github.com/Virginia440/Mod14_challenge/blob/main/Images/SVM%20model%20-Actual%20vs%20Strategy%20cumulative%20returns.PNG" width=500 height=300>


    *  SVM Model- Classification Report(SMA-4 and 100 and Dataset-3 months)
<img src="https://github.com/Virginia440/Mod14_challenge/blob/main/Images/SVM%20Original%20Model%20-Classification%20Report.PNG" width=500 height=300>
   

    *   Alternative SVM Model 1 results (SMA-4 and 100 and Dataset 6 months)
<img src="https://github.com/Virginia440/Mod14_challenge/blob/main/Images/Diff%20time%20period-%20%206months.PNG" width=500 height=300>

What impact resulted from increasing or decreasing the training window?
   An increased training window led to a lower accuracy in prediction. This is reflected by the recall (1.00 vs 0.69).

    *   Alternative SVM Model 2 results (SMA-50 and 100 and Dataset-3 months)
<img src="https://github.com/Virginia440/Mod14_challenge/blob/main/Images/Diff%20SMAs%2C%20SMA%2050%20and%20SMA%20100%20report.PNG" width=500 height=300>

What impact resulted from increasing or decreasing either or both of the SMA windows?
An increase in one of the SMAs, from 4 to 50, led to a decline in accuracy of predictions by almost 30% (1.00 vs 0.71)

**LogisticsRegression MODEL ((Plot of Cumulated Actual returns vs Strategy Returns)**

 <img src="https://github.com/Virginia440/Mod14_challenge/blob/main/Images/Lr%20Model%20Actual%20vs%20Strategy%20cumulative%20returns.PNG" width=500 height=300>

    * LR Model- Classification Report(SMA-4 and 100 and Dataset-3 months)
<img src="https://github.com/Virginia440/Mod14_challenge/blob/main/Images/LR%20Model%20-Classification%20Report.PNG" width=500 height=300>

Did this new model perform better or worse than the provided baseline model? Looking at the cumulated graphs of Actual returns vs Strategy returns of both the SVM model and LogisticsRegeression model, little to no difference is visible. However, classification reports indicate that the SVM model made 100% accurate predicions, compared to the LR model whose that was only 66% accurate as reflected in the recall values (1.00 vs 0.66). Hence the conclusion, SVM model performed better that the LR model.
---

## Summary
-The SVM model performs a bit better than the logistic regression model since it has a higher accuracy score.
-An increased training window led to a lower accuracy in prediction
-An increase in one of the SMAs, from 4 to 50, led to a decline in accuracy of predictions  


## License
 The code is made without a license, however, the materials used for research are licensed.
---


