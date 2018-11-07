# Predicting Stock Market Crashes

With this project I am introduciong the design of machine learning models that predict stock market crashes based on past 
price information. I used public availble market data from seven major stock market indices. This is a classification problem to forecast whether or not a crash will occur within the next 1, 3 or 6 months at any point in time. I trained and tested the linear regression and logistic regression models, support vector machines, decision trees and recurrent neural networks with long short term memory (RNN LSTM). For a detailed description of the problem and findings please refer to the article [blog post link].

Following files and directories are available:

**exploration.ipynb:** Data exploration and crash definition and identification for all seven data sets.

**results.ipynb:** A summary of the results for training, validation and test results of all tested models. For more detail
refer to the jupyter notebooks in the models directory.

**models:** Folder containing jupyter notebooks of all tested models. The notebooks illustrate how the models were trained,
tuned and tested and present the results on the test set

**data:** .csv files of the daily price information of all datasets used. This data is available on yahoo finance.

**final predictor:** pretrained logistic regression model that shows current forecasting probabilities for a crash occuring
within the next 1, 3 and 6 months. Clone this repository, download most recent historical price information of any stock market from yahoo finance (at least 3 years of data), specify the filename in **inputs.json** and run **main.py** to get prediction results.
