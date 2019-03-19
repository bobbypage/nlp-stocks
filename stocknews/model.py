import pandas as pd
import numpy as np
import itertools
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn import linear_model

# input_file = "averageSentimentPerDate.txt"
# df = pd.read_csv(input_file, index_col=0)
# print(df.columns)
# print(df['textBlobSentimentAverage'])
# # print(df.head(5))
# df["textBlobSentimentAverage_cv"] = df["textBlobSentimentAverage"].rolling(window=5).sum() / 5
# df["vaderSentimentAverage_cv"] = df["vaderSentimentAverage"].rolling(window=5).sum() / 5
# stock_df = pd.read_csv("DJIA_table.csv", index_col=1)
# stock_df = stock_df.iloc[::-1]

# df1 = pd.merge(df, stock_df)

# df2 = df1.drop('Volume', axis=1)
# df2 = df2.drop('High', axis=1)
# df2 = df2.drop('Low', axis=1)
# df2 = df2.drop('Close', axis=1)

# df2["trend"] = 0

# # Assigning class to each day stock trend
# for i in range(1,len(df2["Adj Close"])):
#     if(df2["Adj Close"].iloc[i]>=df2["Adj Close"].iloc[i-1]):
#         df2["trend"].iloc[i]=1
#     else:
#         df2["trend"].iloc[i]=0
# df2 = df2.drop(df2.index[[0]])
# df2 = df2.drop(df2.index[[0]])
# df2 = df2.drop(df2.index[[0]])
# df2 = df2.drop(df2.index[[0]])
# df2.to_csv("clean_date.cvs", sep=',')

# train-test split
input_file = "clean_date.cvs"
df2 = pd.read_csv(input_file, index_col=0)

train = df2.loc[0:1392,:]
test = df2.drop(train.index)

date_train = train["Date"]
date_test = test["Date"]

train = train.drop('Date', axis=1)
test = test.drop('Date',axis=1)

X = train.loc[:, train.columns!="trend"]
y = train.iloc[:,-1]

test_X = test.loc[:, test.columns!="trend"]
test_y = test.iloc[:,-1]



# clf = RandomForestClassifier(max_depth=2, random_state=0)
# clf = GradientBoostingRegressor(n_estimators=1000, max_depth=10)
clf = linear_model.LinearRegression()
clf.fit(X,y)
test["predict"] = clf.predict(test_X)
test["change"] = 0
test["change_predict"] = 0
predict = test["predict"]

for i in range(1,len(test["Adj Close"])):
    if(test["Adj Close"].iloc[i]>=test["Adj Close"].iloc[i-1]):
        test["change"].iloc[i]=1
    else:
        test["change"].iloc[i]=0

for i in range(1,len(test["predict"])):
    if(test["predict"].iloc[i]>=test["predict"].iloc[i-1]):
        test["change_predict"].iloc[i]=1
    else:
        test["change_predict"].iloc[i]=0
print(test.head(5))
print(accuracy_score(test["change"], test["change_predict"]))

