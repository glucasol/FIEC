import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

#plt.style.use('fivethirtyeight')

def create_features(df):
    """
    Create time series features based on time series index.
    """
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

def add_lags(df):
    target_map = df['T (degC)'].to_dict()
    df['lag1'] = (df.index - pd.Timedelta('2 hours')).map(target_map)
    return df

df = pd.read_csv('jena_climate_2009_2016.csv')
df = df[5::6]
df = df[['Date Time', 'T (degC)']]
df = df.set_index('Date Time')
df.index = pd.to_datetime(df.index)

train = df.loc[df.index < '01-01-2016']
test = df.loc[df.index >= '01-01-2016']

tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
df = df.sort_index()

df = create_features(df)

df = add_lags(df)

tss = TimeSeriesSplit(n_splits=5, test_size=24*365*1, gap=24)
df = df.sort_index()


fold = 0
preds = []
scores = []
for train_idx, val_idx in tss.split(df):
    train = df.iloc[train_idx]
    test = df.iloc[val_idx]

    train = create_features(train)
    test = create_features(test)

    FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month','year',
                'lag1']
    TARGET = 'T (degC)'

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    reg = xgb.XGBRegressor(base_score=0.5, booster='gbtree',    
                           n_estimators=1000,
                           early_stopping_rounds=50,
                           objective='reg:linear',
                           max_depth=3,
                           learning_rate=0.01)
    reg.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=100)

    y_pred = reg.predict(X_test)
    preds.append(y_pred)
    score = np.sqrt(mean_squared_error(y_test, y_pred))
    scores.append(score)

print(f'Score across folds {np.mean(scores):0.4f}')
print(f'Fold scores:{scores}')

test['Predicted T (degC)'] = reg.predict(X_test)

#ax = test[['T (degC)']].plot(figsize=(15, 5))
#test['Predicted T (degC)'].plot(ax=ax, style='.')
#plt.legend(['Truth Data', 'Predictions'])
#ax.set_title('Raw Dat and Prediction')
#plt.show()

score = np.sqrt(mean_squared_error(test['T (degC)'], test['Predicted T (degC)']))
print(f'RMSE Score on Test set: {score:0.2f}')
