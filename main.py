from tqdm import tqdm

import time
import numpy as np
import pandas as pd

from mxnet import nd, autograd, gluon
from mxnet.gluon import nn, rnn
import mxnet as mx


import math
import talib as ta
from collections import deque
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.varmax import VARMAX
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from vae import VAE

model_ctx = mx.gpu()


def load_dataframe():
    """

    :return:
    """
    df = pd.read_csv('data/AIA.NZ.csv', parse_dates=[0], index_col='Date')
    df.dropna(how='any', axis=0, inplace=True)

    dataset_AIR_df = pd.read_csv('data/AIR.NZ.csv', parse_dates=[0], index_col='Date')
    dataset_THL_df = pd.read_csv('data/THL.NZ.csv', parse_dates=[0], index_col='Date')
    dataset_MFT_df = pd.read_csv('data/MFT.NZ.csv', parse_dates=[0], index_col='Date')

    dataset_NZDAUD_df = pd.read_csv('data/NZDAUD.csv', parse_dates=[0], index_col='Date')
    dataset_NZDCNY_df = pd.read_csv('data/NZDCNY.csv', parse_dates=[0], index_col='Date')
    dataset_NZDUSD_df = pd.read_csv('data/NZDUSD.csv', parse_dates=[0], index_col='Date')
    dataset_NZDJPY_df = pd.read_csv('data/NZDJPY.csv', parse_dates=[0], index_col='Date')

    dataset_NZ50_df = pd.read_csv('data/NZ50.csv', parse_dates=[0], index_col='Date')

    # Merge datasets
    stock = pd.Series(dataset_AIR_df['Close'], name='AIR')
    df = df.join(stock)
    stock = pd.Series(dataset_THL_df['Close'], name='THL')
    df = df.join(stock)
    stock = pd.Series(dataset_MFT_df['Close'], name='MFT')
    df = df.join(stock)

    forex = pd.Series(dataset_NZDAUD_df['Close'], name='NZD/AUD')
    df = df.join(forex)
    forex = pd.Series(dataset_NZDCNY_df['Close'], name='NZD/CNY')
    df = df.join(forex)
    forex = pd.Series(dataset_NZDUSD_df['Close'], name='NZD/USD')
    df = df.join(forex)
    forex = pd.Series(dataset_NZDJPY_df['Close'], name='NZD/JPY')
    df = df.join(forex)

    comp = pd.Series(dataset_NZ50_df['Close'], name='NZ50')
    df = df.join(comp)
    return df


def generate_ti(df):
    """

    :param df:
    :return:
    """
    df['MA7'] = ta.MA(df['Close'], timeperiod=7, matype=0)
    df['MA21'] = ta.MA(df['Close'], timeperiod=21, matype=0)

    df['Upperband'], df['Middleband'], df['Lowerband'] = ta.BBANDS(df['Close'], timeperiod=20, nbdevup=2, nbdevdn=2,
                                                                   matype=0)
    df['MACD'], _, _ = ta.MACDFIX(df['Close'], signalperiod=9)
    df['EMA'] = df['Close'].ewm(com=0.5).mean()
    df['MOM'] = ta.MOM(df['Close'], timeperiod=10)
    return df


def fft_trend(df):
    """

    :param df:
    :return:
    """
    fft = np.fft.fft(np.asarray(df['Close'].tolist()))
    fft_df = pd.DataFrame({'fft': fft})
    fft_df['magnitude'] = fft_df['fft'].apply(lambda x: np.abs(x))
    fft_df['angle'] = fft_df['fft'].apply(lambda x: np.angle(x))
    fft_list = np.asarray(fft_df['fft'].tolist())

    for num_ in [3, 6, 9]:
        fft_list_m10 = np.copy(fft_list)
        fft_list_m10[num_:-num_] = 0

        dd = pd.DataFrame()
        dd['Date'] = df.index
        dd = dd.join(pd.Series(np.fft.ifft(fft_list_m10), name="FFT_" + str(num_)))
        dd = dd.set_index('Date')
        df = df.join(dd["FFT_" + str(num_)])
    return df


def lag_difference(df: pd.DataFrame, factor=1):
    """

    :param df:
    :param factor:
    :return:
    """
    df['Lag'] = df['Close'].shift(factor)
    df['Diff'] = df['Close'].diff(factor)
    return df


def arima_model(df):
    """

    :param df:
    :return:
    """
    model = ARIMA(df['Close'], order=(2, 1, 0))
    model_fit = model.fit()
    model_ft = pd.Series(model_fit.fittedvalues, name='ARIMA')
    df = df.join(model_ft)
    return df


def scale_inputs(df):
    scale = MinMaxScaler()
    df[df.columns] = scale.fit_transform(df[df.columns])
    return df


def vae_feature_extraction(x, y):
    n_hidden = 40
    n_latent = 12
    n_layers = 3
    n_output = x.shape[1]
    model_prefix = 'vae_gluon_{}d{}l{}h.params'.format(n_latent, n_layers, n_hidden)

    batch_size = 127
    train_iter = mx.io.NDArrayIter(data={'data': x}, label={'label': y['Close']}, batch_size=batch_size)

    net = VAE(hidden=n_hidden, latent=n_latent, layers=n_layers, output=n_output, batch_size=batch_size)
    net.collect_params().initialize(mx.init.Xavier(), ctx=model_ctx)
    net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': .001})

    n_epoch = 1

    training_loss = []
    for _ in tqdm(range(n_epoch)):
        epoch_loss = 0

        train_iter.reset()

        n_batch_train = 0
        for batch in train_iter:
            n_batch_train += 1
            data = batch.data[0].as_in_context(model_ctx)
            with autograd.record():
                loss = net(data)
            loss.backward()
            trainer.step(data.shape[0])
            epoch_loss += nd.mean(loss).asscalar()

        epoch_loss /= n_batch_train
        training_loss.append(epoch_loss)

    net.save_parameters('models/' + model_prefix)
    net2 = VAE(hidden=n_hidden, latent=n_latent, layers=n_layers, output=n_output, batch_size=batch_size)
    net2.load_parameters('models/' + model_prefix, ctx=model_ctx)

    results = np.empty((0, n_latent), dtype=float)
    train_iter.reset()
    for batch in train_iter:
        net2(batch.data[0].as_in_context(model_ctx))
        results = np.append(results, net2.mu.asnumpy(), axis=0)
    return pd.DataFrame(results)


def main():
    """

    """
    df = load_dataframe()
    df = generate_ti(df)
    df.dropna(axis=0, how='any', inplace=True)
    df = arima_model(df)
    df = lag_difference(df)
    df.dropna(axis=0, how='any', inplace=True)
    df_y = df[['Close']].copy()
    df_X = df.drop(['Close', 'Adj Close'], axis=1)
    df_X_sc = scale_inputs(df_X)
    df_y_sc = scale_inputs(df_y)

    df_X_sc = vae_feature_extraction(df_X_sc, df_y_sc)


if __name__ == '__main__':
    main()
