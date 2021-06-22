from sklearn.ensemble import RandomForestRegressor
from math import floor

from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit, KFold
from tqdm import tqdm

import pickle
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

# BACKTEST FUNCTION
def backtest_equity(data, starting_equity=int(1000), leverage=1, commission=0.00075, slippage=0.02, mode="Long/Short", output=True):
    current_equity = starting_equity
    buyandhold_equity = starting_equity
    current_position = 0

    equity_curve = []
    buyandhold_curve = []

    first = True
    for period in tqdm(data.values, total=len(data)):
        if first == False:
            buyandhold_equity = buyandhold_equity * (1 + leverage * period[0])
            if current_position == 1:
                current_equity = current_equity * (1 + leverage * period[0])
            elif current_position == -1:
                current_equity = current_equity * (1 - leverage * period[0])

        # Long signal
        if period[1] == 1.0:
            if mode in ["Long/Short", "Long"]:
                if current_position != 1:
                    current_equity = current_equity * (1 - commission) * (1 - slippage)
                current_position = 1
            else:
                if current_position != 0:
                    current_equity = current_equity * (1 - commission) * (1 - slippage)
                current_position = 0
        # Short signal
        else:
            if mode in ["Long/Short", "Short"]:
                if current_position != -1:
                    current_equity = current_equity * (1 - commission) * (1 - slippage)
                current_position = -1
            elif mode == "Long":
                if current_position != 0:
                    current_equity = current_equity * (1 - commission) * (1 - slippage)
                current_position = 0

        equity_curve.append(current_equity)
        buyandhold_curve.append(buyandhold_equity)

        first = False

    equity_curves = pd.DataFrame({
        'Buy and Hold Equity': buyandhold_curve,
        'Strategy Equity': equity_curve})

    if output == True:
        print('ending equity: ' + str(equity_curves['Strategy Equity'].values[-1]))
        print('buy and hold equity: ' + str(equity_curves['Buy and Hold Equity'].values[-1]))
        print('ending return: ' + str(((equity_curves['Strategy Equity'].values[-1] -\
                                       equity_curves['Strategy Equity'].values[0])\
                                      /equity_curves['Strategy Equity'].values[0])*100) + '%')
        print('buy and hold return: ' + str(((equity_curves['Buy and Hold Equity'].values[-1] - \
                                       equity_curves['Buy and Hold Equity'].values[0]) \
                                      / equity_curves['Buy and Hold Equity'].values[0])*100) + '%')

    return equity_curves

# ROLLING PREDICTION AND RETRAIN FUNCTION
def leaveoneout_predict(data, model, mode='Timeseries', modelDir="models/", output=True, ret=["model", "predictions"]):
    if mode == 'Timeseries':
        split = TimeSeriesSplit(n_splits=len(data)-1, max_train_size=None)
    else:
        split = KFold(n_folds=len(data)-1, shuffle=True, test_size=1)

    predictions = []
    scores = {'validation': []}
    for i, j in tqdm(split.split(data), total=len(data)):
        train, validation = {}, {}
        train_data = data.iloc[i]
        validation_data = data.iloc[j]

        train['features'] = train_data.drop('target', axis=1)
        validation['features'] = validation_data.drop('target', axis=1)
        train['targets'] = train_data['target']
        validation['targets'] = validation_data['target']

        model.fit(train['features'], train['targets'])

        validation_predictions = pd.DataFrame(model.predict(np.array(validation['features'])))[:][0]
        # validation_score = model.score(validation['features'], validation['targets'])

        # scores['validation'].append(validation_score)
        predictions.append(validation_predictions)

    # scores = pd.DataFrame(scores)
    # predictions = pd.DataFrame({'train': train_predictions, 'validation': validation_predictions})
    predictions = pd.DataFrame(predictions)
    predictions.index = data.index[1:]

    # if output == True:
    #     print('validation_score: ' + str(scores['validation'].mean()))

    if ("model" in ret) and ("predictions" in ret):
        return [model, predictions]
    elif "model" in ret:
        return model
    elif "predictions" in ret:
        return predictions

def test_starting_point(starting_point, plots=["zoomed", "volatility", "residuals", "inventories"], timeframe="D"):
    # LOAD DATA
    df = pd.read_csv('BITMEX_XBTUSD, 1D.csv', index_col=['time'])

    # CHANGE STARTING POINT
    df = df.iloc[floor(len(df)*starting_point):]

    # CALCULATE REALIZED VOLATILITY (NOT-ANNUALIZED)
    df['log(close)'] = np.log(df['close'])
    df['logdiff(close)'] = df['log(close)'].diff()
    df['rvar'] = df['logdiff(close)'] ** 2
    df['rvol'] = np.sqrt(df['rvar'])

    # CLEAN UP DATAFRAME AND STORE PRICE CLOSES FOR LATER CHARTING
    close = df[['close']].iloc[1:]
    df = df[['rvol']]

    if timeframe == "D":
        # FEATURES FOR DAILY VOLATILITY
        lag_periods = 7
        for i in range(1,lag_periods+1):
            exec('df[\'rvol_t-1_{}\'] = df[\'rvol\'].shift({})'.format(i, i))
            exec('df[\'rvol_t-7_{}\'] = df[\'rvol\'].rolling(window=7).mean().shift({})'.format(i, i))
            exec('df[\'rvol_t-30_{}\'] = df[\'rvol\'].rolling(window=30).mean().shift({})'.format(i, i))
            exec('df[\'rvol_t-60_{}\'] = df[\'rvol\'].rolling(window=60).mean().shift({})'.format(i, i))
    elif timeframe == "W":
    # FEATURES FOR WEEKLY VOLATILITY
        lag_periods = 4
        for i in range(1,lag_periods+1):
            exec('df[\'rvol_t-1_{}\'] = df[\'rvol\'].shift({})'.format(i, i))
            exec('df[\'rvol_t-4_{}\'] = df[\'rvol\'].rolling(window=4).mean().shift({})'.format(i, i))
            exec('df[\'rvol_t-8_{}\'] = df[\'rvol\'].rolling(window=8).mean().shift({})'.format(i, i))
            exec('df[\'rvol_t-16_{}\'] = df[\'rvol\'].rolling(window=16).mean().shift({})'.format(i, i))

    # DROP RECORDS WITH AN NA DUE TO NUMBER OF LAG WINDOWS
    df.dropna(inplace=True)
    df.columns = ['target'] + list(df.columns)[1:]
    data = df

    # TEST TRAIN LOGIC FOR SELECTING A MODEL
    # train_size = floor(len(data)*0.8)
    # train_data = data[:train_size]
    # test_data = data[train_size:]

    # test = {}
    # test['features'] = pd.DataFrame(test_data.drop(['target'], axis=1), index=test_data.index)
    # test['target'] = pd.DataFrame(test_data[['target']])
    #
    # train = {}
    # train['features'] = pd.DataFrame(train_data.drop(['target'], axis=1), index=train_data.index)
    # train['target'] = pd.DataFrame(np.array(train_data[['target']]))

    full = {}
    full['features'] = pd.DataFrame(data.drop(['target'], axis=1), index=data.index)
    full['target'] = pd.DataFrame(np.array(data[['target']]))

    # IF PREDICTIONS FILE EXISTS, LOAD INTO
    if not os.path.exists('ret.pkl'):
        model = RandomForestRegressor(n_estimators=1000, max_depth=7, n_jobs=12)
        ret = leaveoneout_predict(pd.DataFrame(data, columns=data.columns), model)
        model = ret[0]
        predictions = ret[1]
        test_r2 = r2_score(full['target'].iloc[:-1], predictions)

        print('test r2: ' + str(test_r2))
        with open('ret.pkl', 'wb') as f:
            pickle.dump(ret, f)
    else:
        with open('ret.pkl', 'rb') as f:
            ret = pickle.load(f)
            model = ret[0]
            predictions = ret[1]

    if "volatility" in plots:
        full['target'].plot()
        plt.plot(full['target'].index[1:], predictions.values, label='test_predicted')
        plt.legend()
        plt.title("Out of Sample RVol")
        plt.show()

    if "residuals" in plots:
        residuals = (np.array(predictions.values) - np.array(full['target'].values)[1:])/np.array(full['target'].values)[1:]
        plt.hist(residuals, range=(-1,1), bins=20)
        plt.title("Out of Sample Residuals")
        plt.show()

    close['predicted_vol'] = predictions
    close['close_t-1'] = close['close'].shift(1)
    close.dropna(inplace=True)
    close['upper_band'] = close['close_t-1'] + close['close_t-1'] * close['predicted_vol']
    close['lower_band'] = close['close_t-1'] - close['close_t-1'] * close['predicted_vol']
    close['midpoint'] = (close['upper_band'] - close['lower_band'])/2

    bands = close[['upper_band', 'lower_band']]

    df6 = pd.read_csv("BITMEX_XBTUSD, 360.csv", index_col=['time'])
    df6 = df6[['close']]

    merged = df6.join(bands,sort=True)

    merged.index = pd.to_datetime(merged.index)
    merged_filled = merged.ffill()
    merged_filled['midpoint'] = merged_filled['upper_band'] - (merged_filled['upper_band'] - merged_filled['lower_band'])/2
    merged_filled['lower_quart'] = (merged_filled['midpoint'] - merged_filled['lower_band'])/2
    merged_filled['upper_quart'] = (merged_filled['upper_band'] - merged_filled['midpoint'])/2
    merged_filled.dropna(inplace=True)
    # merged_filled = merged_filled.iloc[floor(len(merged_filled)*0.98):]

    if "zoomed" in plots:
        fig, ax = plt.subplots(1)

        merged_filled.plot(y='close', ax=ax, logy=True, figsize=(19.2*2, 9.6))
        merged_filled.plot(y='upper_band', color="r", ax=ax)
        merged_filled.plot(y='lower_band', color="g", ax=ax)
        merged_filled.plot(y='midpoint', color="k", ax=ax)
        # merged_filled.plot(y='upper_quart', color="r", ax=ax, logy=True)
        # merged_filled.plot(y='lower_quart', color="g", ax=ax, logy=True)

        day_starts = merged_filled[merged_filled.index.hour == 0].index
        for day in day_starts:
            plt.vlines(day, ymin=0, ymax=merged_filled['upper_band'].max())

        plt.show()

    # GENERATE SIGNALS
    signals = []

    for index, period in tqdm(merged_filled.iterrows()):
        if period['close'] <= period['lower_band']:
            signals.append(0.5)
        elif period['close'] <= period['midpoint']:
            signals.append(0.33)
        elif period['close'] > period['midpoint']:
            signals.append(0.25)
        elif period['close'] >= period['upper_band']:
            signals.append(0.1)



    # PERFORM BACKTEST
    closes = merged_filled[['close']].dropna()
    signals = pd.DataFrame(signals, index=merged_filled.index)
    closes = closes.join(signals, sort=True)

    daily_allocation = 1000
    remaining_allocation = 1000

    inventory = 0
    dca_inventory = 0
    inventories = []
    dca_inventories = []

    for index, period in tqdm(closes.iterrows()):
        allocation = (daily_allocation * period[0])
        if allocation <= remaining_allocation:
            inventory += allocation/period['close']
            remaining_allocation -= allocation
        else:
            inventory += remaining_allocation/period['close']
            remaining_allocation = 0

        # DAILY CLEANUP
        if pd.to_datetime(index).hour == 0:
            if remaining_allocation > 0:
                inventory += remaining_allocation/period['close']
            remaining_allocation = daily_allocation
            dca_inventory += daily_allocation/period['close']

        # STORE RESULTS
        inventories.append(inventory)
        dca_inventories.append(dca_inventory)

    inventories = pd.DataFrame({'inventory': inventories, 'dca_inventory': dca_inventories})
    inventories['ratio'] = inventories['inventory'] / inventories['dca_inventory']
    inventories['ratio_avg'] = inventories['ratio']
    if "inventories" in plots:
        inventories[['ratio']].iloc[:200].plot()
        plt.show()

    return inventories

# min = 0
# max = 0.75
# ns = random.sample(min, max, 1000)
#
# for i in range(0,1000):
#     n = ns[i]

test_starting_point(0, plots=["inventories"])
