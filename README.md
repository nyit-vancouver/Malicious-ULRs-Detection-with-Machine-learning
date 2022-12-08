# Detect malicious URLs using machine learning models
## Environment tips:
This project runs under python3.11. When you install **lightgbm** on macOS, there will be a problem as you need **gcc** to complie the package. [Here](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#macos) is the instruction to install **lightgbm** on your macOS.
## Description:
1. **features_extraction.py** is used to extract 31 features, including general features, length features, count features, ratio features and domain features as shown in the features table.
2. **model training.py** is used to train different ML models and draw the heapmap.
## The datasets we collected from: 
1. [Kaggle](https://www.kaggle.com/code/siddharthkumar25/detect-malicious-url-using-ml/data?select=urldata.csv) 
2. [UNB](https://www.unb.ca/cic/datasets/url-2016.html)
3. [URLhaus](https://urlhaus.abuse.ch/browse/)
4. [Mendeley](https://data.mendeley.com/datasets/gdx3pkwp47/2)
## The applied models are:
Logistic, KNN, SVM, Decision Trees, Random Forest, Bagging, and AdaBoosting
## Contributors:
[xinyanzhang27](https://github.com/XinyanZhang027)
