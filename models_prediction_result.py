import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import signal
from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import GRU, Dropout, Dense, LSTM
from keras.optimizers import Adam
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor as XGBR
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_percentage_error as MAPE, \
    mean_absolute_error as MAE


class ModelsPrediction:
    def __init__(self):
        self.v = "data_filename"
        self._data_path = "../../data/" + self.v + "/train_data_water_density.csv"
        self._data = pd.read_csv(self._data_path, index_col=0)
        self.actual_heat_load = self._data.iloc[:, -1].values
        self.split_datasets()

    def split_datasets(self):

        min_heat_load_ = self._data.loc[self._data.index[0]:,
                         ["实际需热量sd1", "实际需热量sd2", "实际需热量prediction"]].min(axis=1)  # 最小热负荷
        actual_heat_load_ = self._data.loc[self._data.index[0]:, "实际需热量prediction"]  # 实际热负荷
        y = 0.5 * min_heat_load_ + 0.5 * actual_heat_load_  # 均值

        # y = self._data.loc[self._data.index[0]:,
        #     ["实际需热量sd1", "实际需热量sd2", "实际需热量prediction"]].mean(axis=1)

        columns_to_drop = ["实际需热量sd1", "实际需热量sd2", "实际需热量prediction"
            , "二次网供水温度sd1", "二次网回水温度sd1"
            , "二次网供水温度sd2", "二次网回水温度sd2"
            , "二次网供水温度prediction", "二次网回水温度prediction"]
        x = self._data.drop(columns_to_drop, axis=1)
        self.scaled_features_y = {}
        min_, max_ = y.min(), y.max()
        self.scaled_features_y[0] = [min_, max_]
        y = (y - min_) / (max_ - min_)

        self.scaled_features_x = {}
        for columns_index in range(x.shape[1]):
            # 求出每一列的最大值和最小值
            cur_columns_data = x.iloc[:, columns_index]
            min_, max_ = cur_columns_data.min(), cur_columns_data.max()
            self.scaled_features_x[columns_index] = [min_, max_]
            x.iloc[:, columns_index] = (cur_columns_data - min_) / (max_ - min_)

        # 打乱数据集
        np.random.seed = 420
        idx_lst = [*range(x.shape[0])]
        np.random.shuffle(idx_lst)
        np.savetxt('../../data/' + self.v + '/random_lst.csv'
                   , idx_lst
                   , encoding='utf-8'
                   , delimiter=',')
        idx_lst = np.loadtxt('../../data/' + self.v + '/random_lst.csv'
                             , encoding='utf-8'
                             , delimiter=',')
        self.idx_lst = idx_lst.astype('int32')
        x = np.array(x)
        y = np.array(y)
        x = x[self.idx_lst]
        y = y[self.idx_lst]

        self.actual_heat_load = self.actual_heat_load[self.idx_lst]

        self.train_targets = y[:-168]
        self.train_features = x[:-168, :]

        # # ------------------------------------------------- 24小时预测 ---------------------------------------------
        self.prediction_targets_24h = y[-168:-144]
        self.prediction_features_24h = x[-168:-144, :]

        # # ------------------------------------------------- 72小时预测 ---------------------------------------------
        self.prediction_targets_72h = y[-168:-96]
        self.prediction_features_72h = x[-168:-96, :]

        # ------------------------------------------------- 168小时预测 ---------------------------------------------
        self.prediction_targets_168h = y[-168:]
        self.prediction_features_168h = x[-168:, :]

        print("24h ... Train shape {0} ... Prediction shape {1}".format(self.train_features.shape,
                                                                        self.prediction_features_24h.shape))
        print("72h ... Train shape {0} ... Prediction shape {1}".format(self.train_features.shape,
                                                                        self.prediction_features_72h.shape))
        print("168h ... Train shape {0} ... Prediction shape {1}".format(self.train_features.shape,
                                                                         self.prediction_features_168h.shape))

    def cal_ESR_CCI(self, y, y_hat):
        """计算预测曲线和实际曲线的节能率和互相关系数
        """
        y_hat_1, y_hat_N = y_hat[0], y_hat[-1]  # 获取到第一个预测值和最后一个预测值
        y_hat_middle = y_hat[1: -1]  # 获取到 (1-N) 之间的预测值
        y_1, y_N = y[0], y[-1]  # 获取到第一个真实值和最后一个真实值
        y_middle = y[1: -1]  # 获取到 (1-N) 之间的真实值
        E_hat = 0.5 * (y_hat_1 + y_hat_N) + np.sum(y_hat_middle)
        E = 0.5 * (y_1 + y_N) + np.sum(y_middle)
        ESR = 1 - (E_hat / E)
        ESR = np.round(ESR, 3)
        n = len(y)
        sum_x_y, sum_x, sum_y, sum_x_2, sum_y_2 = 0, 0, 0, 0, 0
        for x_i, y_i in zip(y_hat, y):
            sum_x_y += x_i * y_i
            sum_x += x_i
            sum_y += y_i
            sum_x_2 += x_i ** 2
            sum_y_2 += y_i ** 2
        fm1 = (n * sum_x_2) - sum_x ** 2
        fm2 = (n * sum_y_2) - sum_y ** 2
        CCI = (n * sum_x_y - sum_x * sum_y) / np.sqrt(fm1 * fm2)
        CCI = np.round(CCI, 3)
        return ESR, CCI

    def sd_mlr(self):
        '''多元线性回归预测模型
        '''
        MLR_model = LinearRegression(fit_intercept=False, normalize=False, copy_X=True)
        MLR_model.fit(self.train_features, self.train_targets)
        # 24小时
        predictions_24h_mlr = MLR_model.predict(self.prediction_features_24h)
        min_, max_ = self.scaled_features_y[0]
        self.predictions_24h_mlr_ = predictions_24h_mlr * (max_ - min_) + min_
        self.predictions_24h_target_ = self.prediction_targets_24h * (max_ - min_) + min_
        # 72小时
        predictions_72h_mlr = MLR_model.predict(self.prediction_features_72h)
        min_, max_ = self.scaled_features_y[0]
        self.predictions_72h_mlr_ = predictions_72h_mlr * (max_ - min_) + min_
        self.predictions_72h_target_ = self.prediction_targets_72h * (max_ - min_) + min_
        # 168小时
        predictions_168h_mlr = MLR_model.predict(self.prediction_features_168h)
        min_, max_ = self.scaled_features_y[0]
        self.predictions_168h_mlr_ = predictions_168h_mlr * (max_ - min_) + min_
        self.predictions_168h_target_ = self.prediction_targets_168h * (max_ - min_) + min_

        fc = 100
        fs = 700
        w_n = 2 * fc / fs
        b, a = signal.butter(6, w_n, "lowpass")
        # 24小时
        self.predictions_24h_mlr_filter = signal.filtfilt(b, a, self.predictions_24h_mlr_, axis=-1,
                                                          padtype='odd', padlen=None, method='pad',
                                                          irlen=None)
        self.predictions_24h_target_filter = signal.filtfilt(b, a, self.predictions_24h_target_, axis=-1,
                                                             padtype='odd', padlen=None, method='pad',
                                                             irlen=None)
        print('RMSE:', np.sqrt(MSE(self.predictions_24h_target_, self.predictions_24h_mlr_)))
        print('MAPE', MAPE(self.predictions_24h_target_, self.predictions_24h_mlr_))
        print('MAE', MAE(self.predictions_24h_target_, self.predictions_24h_mlr_))
        # 72小时
        self.predictions_72h_mlr_filter = signal.filtfilt(b, a, self.predictions_72h_mlr_, axis=-1,
                                                          padtype='odd', padlen=None, method='pad',
                                                          irlen=None)
        self.predictions_72h_target_filter = signal.filtfilt(b, a, self.predictions_72h_target_, axis=-1,
                                                             padtype='odd', padlen=None, method='pad',
                                                             irlen=None)
        print('RMSE', np.sqrt(MSE(self.predictions_72h_target_, self.predictions_72h_mlr_)))
        print('MAPE', MAPE(self.predictions_72h_target_, self.predictions_72h_mlr_))
        print('MAE', MAE(self.predictions_72h_target_, self.predictions_72h_mlr_))

        # 168小时
        self.predictions_168h_mlr_filter = signal.filtfilt(b, a, self.predictions_168h_mlr_, axis=-1,
                                                           padtype='odd', padlen=None, method='pad',
                                                           irlen=None)
        self.predictions_168h_target_filter = signal.filtfilt(b, a, self.predictions_168h_target_, axis=-1,
                                                              padtype='odd', padlen=None, method='pad',
                                                              irlen=None)
        print('RMSE', np.sqrt(MSE(self.predictions_168h_target_, self.predictions_168h_mlr_)))
        print('MAPE', MAPE(self.predictions_168h_target_, self.predictions_168h_mlr_))
        print('MAE', MAE(self.predictions_168h_target_, self.predictions_168h_mlr_))

    def sd_xgboost(self):
        xgb_model = XGBR(n_estimators=45).fit(self.train_features, self.train_targets)
        # 24小时
        predictions_24h_xgb = xgb_model.predict(self.prediction_features_24h)
        min_, max_ = self.scaled_features_y[0]
        self.predictions_24h_xgb_ = predictions_24h_xgb * (max_ - min_) + min_
        self.predictions_24h_target_ = self.prediction_targets_24h * (max_ - min_) + min_

        # 72小时
        predictions_72h_xgb = xgb_model.predict(self.prediction_features_72h)
        min_, max_ = self.scaled_features_y[0]
        self.predictions_72h_xgb_ = predictions_72h_xgb * (max_ - min_) + min_
        self.predictions_72h_target_ = self.prediction_targets_72h * (max_ - min_) + min_

        # 168小时
        predictions_168h_xgb = xgb_model.predict(self.prediction_features_168h)
        min_, max_ = self.scaled_features_y[0]
        self.predictions_168h_xgb_ = predictions_168h_xgb * (max_ - min_) + min_
        self.predictions_168h_target_ = self.prediction_targets_168h * (max_ - min_) + min_

        fc = 100
        fs = 700
        w_n = 2 * fc / fs
        b, a = signal.butter(6, w_n, "lowpass")
        # 对实际需热量进行滤波处理
        self.predictions_24h_xgb_filter = signal.filtfilt(b, a, self.predictions_24h_xgb_, axis=-1,
                                                          padtype='odd', padlen=None, method='pad',
                                                          irlen=None)
        self.predictions_24h_target_filter = signal.filtfilt(b, a, self.predictions_24h_target_, axis=-1,
                                                             padtype='odd', padlen=None, method='pad',
                                                             irlen=None)
        print('RMSE', np.sqrt(MSE(self.predictions_24h_target_, self.predictions_24h_xgb_)))
        print('MAPE', MAPE(self.predictions_24h_target_, self.predictions_24h_xgb_))
        print('MAE', MAE(self.predictions_24h_target_, self.predictions_24h_xgb_))

        # 72小时
        self.predictions_72h_xgb_filter = signal.filtfilt(b, a, self.predictions_72h_xgb_, axis=-1,
                                                          padtype='odd', padlen=None, method='pad',
                                                          irlen=None)
        self.predictions_72h_target_filter = signal.filtfilt(b, a, self.predictions_72h_target_, axis=-1,
                                                             padtype='odd', padlen=None, method='pad',
                                                             irlen=None)
        print('RMSE', np.sqrt(MSE(self.predictions_72h_target_, self.predictions_72h_xgb_)))
        print('MAPE', MAPE(self.predictions_72h_target_, self.predictions_72h_xgb_))
        print('MAE', MAE(self.predictions_72h_target_, self.predictions_72h_xgb_))
        # 168小时
        self.predictions_168h_xgb_filter = signal.filtfilt(b, a, self.predictions_168h_xgb_, axis=-1,
                                                           padtype='odd', padlen=None, method='pad',
                                                           irlen=None)
        self.predictions_168h_target_filter = signal.filtfilt(b, a, self.predictions_168h_target_, axis=-1,
                                                              padtype='odd', padlen=None, method='pad',
                                                              irlen=None)
        print('RMSE', np.sqrt(MSE(self.predictions_168h_target_, self.predictions_168h_xgb_)))
        print('MAPE', MAPE(self.predictions_168h_target_, self.predictions_168h_xgb_))
        print('MAE', MAE(self.predictions_168h_target_, self.predictions_168h_xgb_))

    def sd_rf(self):
        '''随机森林回归预测模型
        '''
        # RF -- 随机森林
        RF_model = RandomForestRegressor(n_estimators=60,
                                         random_state=60,
                                         max_depth=5)
        RF_model.fit(self.train_features, self.train_targets)

        # 24小时
        predictions_24h_RF = RF_model.predict(self.prediction_features_24h)
        min_, max_ = self.scaled_features_y[0]
        self.predictions_24h_RF_ = predictions_24h_RF * (max_ - min_) + min_
        self.predictions_24h_target_ = self.prediction_targets_24h * (max_ - min_) + min_

        # 72小时
        predictions_72h_RF = RF_model.predict(self.prediction_features_72h)
        min_, max_ = self.scaled_features_y[0]
        self.predictions_72h_RF_ = predictions_72h_RF * (max_ - min_) + min_
        self.predictions_72h_target_ = self.prediction_targets_72h * (max_ - min_) + min_

        # 168小时
        predictions_168h_RF = RF_model.predict(self.prediction_features_168h)
        min_, max_ = self.scaled_features_y[0]
        self.predictions_168h_RF_ = predictions_168h_RF * (max_ - min_) + min_
        self.predictions_168h_target_ = self.prediction_targets_168h * (max_ - min_) + min_

        fc = 100
        fs = 700
        w_n = 2 * fc / fs
        b, a = signal.butter(6, w_n, "lowpass")
        # 对实际需热量进行滤波处理
        self.predictions_24h_RF_filter = signal.filtfilt(b, a, self.predictions_24h_RF_, axis=-1,
                                                         padtype='odd', padlen=None, method='pad',
                                                         irlen=None)
        self.predictions_24h_target_filter = signal.filtfilt(b, a, self.predictions_24h_target_, axis=-1,
                                                             padtype='odd', padlen=None, method='pad',
                                                             irlen=None)
        print('RMSE', np.sqrt(MSE(self.predictions_24h_target_, self.predictions_24h_RF_)))
        print('MAPE', MAPE(self.predictions_24h_target_, self.predictions_24h_RF_))
        print('MAE', MAE(self.predictions_24h_target_, self.predictions_24h_RF_))

        # 72小时
        self.predictions_72h_RF_filter = signal.filtfilt(b, a, self.predictions_72h_RF_, axis=-1,
                                                         padtype='odd', padlen=None, method='pad',
                                                         irlen=None)
        self.predictions_72h_target_filter = signal.filtfilt(b, a, self.predictions_72h_target_, axis=-1,
                                                             padtype='odd', padlen=None, method='pad',
                                                             irlen=None)
        print('RMSE', np.sqrt(MSE(self.predictions_72h_target_, self.predictions_72h_RF_)))
        print('MAPE', MAPE(self.predictions_72h_target_, self.predictions_72h_RF_))
        print('MAE', MAE(self.predictions_72h_target_, self.predictions_72h_RF_))

        # 168小时
        self.predictions_168h_RF_filter = signal.filtfilt(b, a, self.predictions_168h_RF_, axis=-1,
                                                          padtype='odd', padlen=None, method='pad',
                                                          irlen=None)
        self.predictions_168h_target_filter = signal.filtfilt(b, a, self.predictions_168h_target_, axis=-1,
                                                              padtype='odd', padlen=None, method='pad',
                                                              irlen=None)
        print('RMSE', np.sqrt(MSE(self.predictions_168h_target_, self.predictions_168h_RF_)))
        print('MAPE', MAPE(self.predictions_168h_target_, self.predictions_168h_RF_))
        print('MAE', MAE(self.predictions_168h_target_, self.predictions_168h_RF_))

    def sd_svm(self):
        '''支持向量机预测模型
        '''
        SVM_model = svm.SVR(kernel='rbf', C=1, epsilon=0.01)
        SVM_model.fit(self.train_features, self.train_targets)
        # 24小时
        predictions_24h_SVM = SVM_model.predict(self.prediction_features_24h)
        min_, max_ = self.scaled_features_y[0]
        self.predictions_24h_SVM_ = predictions_24h_SVM * (max_ - min_) + min_
        self.predictions_24h_target_ = self.prediction_targets_24h * (max_ - min_) + min_
        # 72小时
        predictions_72h_SVM = SVM_model.predict(self.prediction_features_72h)
        min_, max_ = self.scaled_features_y[0]
        self.predictions_72h_SVM_ = predictions_72h_SVM * (max_ - min_) + min_
        self.predictions_72h_target_ = self.prediction_targets_72h * (max_ - min_) + min_
        # 168小时
        predictions_168h_SVM = SVM_model.predict(self.prediction_features_168h)
        min_, max_ = self.scaled_features_y[0]
        self.predictions_168h_SVM_ = predictions_168h_SVM * (max_ - min_) + min_
        self.predictions_168h_target_ = self.prediction_targets_168h * (max_ - min_) + min_

        fc = 100
        fs = 700
        w_n = 2 * fc / fs
        b, a = signal.butter(6, w_n, "lowpass")
        # 24小时
        self.predictions_24h_SVM_filter = signal.filtfilt(b, a, self.predictions_24h_SVM_, axis=-1,
                                                          padtype='odd', padlen=None, method='pad',
                                                          irlen=None)
        self.predictions_24h_target_filter = signal.filtfilt(b, a, self.predictions_24h_target_, axis=-1,
                                                             padtype='odd', padlen=None, method='pad',
                                                             irlen=None)
        print('RMSE', np.sqrt(MSE(self.predictions_24h_target_, self.predictions_24h_SVM_)))
        print('MAPE', MAPE(self.predictions_24h_target_, self.predictions_24h_SVM_))
        print('MAE', MAE(self.predictions_24h_target_, self.predictions_24h_SVM_))

        # 72小时
        self.predictions_72h_SVM_filter = signal.filtfilt(b, a, self.predictions_72h_SVM_, axis=-1,
                                                          padtype='odd', padlen=None, method='pad',
                                                          irlen=None)
        self.predictions_72h_target_filter = signal.filtfilt(b, a, self.predictions_72h_target_, axis=-1,
                                                             padtype='odd', padlen=None, method='pad',
                                                             irlen=None)
        print('RMSE', np.sqrt(MSE(self.predictions_72h_target_, self.predictions_72h_SVM_)))
        print('MAPE', MAPE(self.predictions_72h_target_, self.predictions_72h_SVM_))
        print('MAE', MAE(self.predictions_72h_target_, self.predictions_72h_SVM_))

        # 168小时
        self.predictions_168h_SVM_filter = signal.filtfilt(b, a, self.predictions_168h_SVM_, axis=-1,
                                                           padtype='odd', padlen=None, method='pad',
                                                           irlen=None)
        self.predictions_168h_target_filter = signal.filtfilt(b, a, self.predictions_168h_target_, axis=-1,
                                                              padtype='odd', padlen=None, method='pad',
                                                              irlen=None)
        print('RMSE', np.sqrt(MSE(self.predictions_168h_target_, self.predictions_168h_SVM_)))
        print('MAPE', MAPE(self.predictions_168h_target_, self.predictions_168h_SVM_))
        print('MAE', MAE(self.predictions_168h_target_, self.predictions_168h_SVM_))

    def sd_lstm(self):
        '''长短时记忆网络预测模型
        '''
        lstm_size = 300  # 神经元个数
        batch_size = 64  # 训练批次
        epochs = 200  # 迭代次数
        dropout = 0.3
        train_x = np.array(self.train_features)
        train_y = np.array(self.train_targets)
        test_x_168 = np.array(self.prediction_features_168h)
        test_x_72 = np.array(self.prediction_features_72h)
        test_x_24 = np.array(self.prediction_features_24h)
        # 改变数据尺寸
        train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
        train_y = np.reshape(train_y, (-1, 1))
        test_x_24 = np.reshape(test_x_24, (test_x_24.shape[0], 1, test_x_24.shape[1]))
        test_x_72 = np.reshape(test_x_72, (test_x_72.shape[0], 1, test_x_72.shape[1]))
        test_x_168 = np.reshape(test_x_168, (test_x_168.shape[0], 1, test_x_168.shape[1]))
        # 搭建模型
        lstm_model = Sequential()
        lstm_model.add(
            LSTM(lstm_size, activation='relu', return_sequences=True))
        lstm_model.add(Dropout(dropout))
        lstm_model.add(
            LSTM(lstm_size, activation='relu', return_sequences=False))
        lstm_model.add(Dropout(dropout))
        lstm_model.add(Dense(1, activation='linear'))
        lstm_model.compile(loss="mse", optimizer=Adam(1e-4))
        lstm_model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=0)
        lstm_model.save('../../models/' + self.v + '/lstm_model/lstm.h5')

        lstm_model = load_model('../../models/' + self.v + '/lstm_model/lstm.h5')
        predictions_lstm_24h = lstm_model.predict(test_x_24)
        predictions_lstm_72h = lstm_model.predict(test_x_72)
        predictions_lstm_168h = lstm_model.predict(test_x_168)

        min_, max_ = self.scaled_features_y[0]
        self.predictions_24h_lstm_ = predictions_lstm_24h * (max_ - min_) + min_
        self.predictions_24h_target_ = self.prediction_targets_24h * (max_ - min_) + min_

        min_, max_ = self.scaled_features_y[0]
        self.predictions_72h_lstm_ = predictions_lstm_72h * (max_ - min_) + min_
        self.predictions_72h_target_ = self.prediction_targets_72h * (max_ - min_) + min_

        min_, max_ = self.scaled_features_y[0]
        self.predictions_168h_lstm_ = predictions_lstm_168h * (max_ - min_) + min_
        self.predictions_168h_target_ = self.prediction_targets_168h * (max_ - min_) + min_

        fc = 100
        fs = 700
        w_n = 2 * fc / fs
        b, a = signal.butter(6, w_n, "lowpass")
        # 24小时
        self.predictions_24h_lstm_filter = signal.filtfilt(b, a, self.predictions_24h_lstm_.squeeze(), axis=-1,
                                                           padtype='odd', padlen=None, method='pad',
                                                           irlen=None)
        self.predictions_24h_target_filter = signal.filtfilt(b, a, self.predictions_24h_target_, axis=-1,
                                                             padtype='odd', padlen=None, method='pad',
                                                             irlen=None)
        print('RMSE', np.sqrt(MSE(self.predictions_24h_target_, self.predictions_24h_lstm_)))
        print('MAPE', MAPE(self.predictions_24h_target_, self.predictions_24h_lstm_))
        print('MAE', MAE(self.predictions_24h_target_, self.predictions_24h_lstm_))

        # 72小时
        self.predictions_72h_lstm_filter = signal.filtfilt(b, a, self.predictions_72h_lstm_.squeeze(), axis=-1,
                                                           padtype='odd', padlen=None, method='pad',
                                                           irlen=None)
        self.predictions_72h_target_filter = signal.filtfilt(b, a, self.predictions_72h_target_, axis=-1,
                                                             padtype='odd', padlen=None, method='pad',
                                                             irlen=None)
        print('RMSE', np.sqrt(MSE(self.predictions_72h_target_, self.predictions_72h_lstm_)))
        print('MAPE', MAPE(self.predictions_72h_target_, self.predictions_72h_lstm_))
        print('MAE', MAE(self.predictions_72h_target_, self.predictions_72h_lstm_))

        # 168小时
        self.predictions_168h_lstm_filter = signal.filtfilt(b, a, self.predictions_168h_lstm_.squeeze(), axis=-1,
                                                            padtype='odd', padlen=None, method='pad',
                                                            irlen=None)
        self.predictions_168h_target_filter = signal.filtfilt(b, a, self.predictions_168h_target_, axis=-1,
                                                              padtype='odd', padlen=None, method='pad',
                                                              irlen=None)
        print('RMSE', np.sqrt(MSE(self.predictions_168h_target_, self.predictions_168h_lstm_)))
        print('MAPE', MAPE(self.predictions_168h_target_, self.predictions_168h_lstm_))
        print('MAE', MAE(self.predictions_168h_target_, self.predictions_168h_lstm_))

    def sd_gru(self):
        '''门控神经网络预测模型
        '''
        gru_size = 150  # 神经元个数
        batch_size = 32  # 训练批次
        epochs = 150  # 迭代次数
        train_x = np.array(self.train_features)
        train_y = np.array(self.train_targets)
        test_x_168 = np.array(self.prediction_features_168h)
        test_x_72 = np.array(self.prediction_features_72h)
        test_x_24 = np.array(self.prediction_features_24h)
        # 改变数据尺寸
        train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
        train_y = np.reshape(train_y, (-1, 1))
        test_x_24 = np.reshape(test_x_24, (test_x_24.shape[0], 1, test_x_24.shape[1]))
        test_x_72 = np.reshape(test_x_72, (test_x_72.shape[0], 1, test_x_72.shape[1]))
        test_x_168 = np.reshape(test_x_168, (test_x_168.shape[0], 1, test_x_168.shape[1]))
        # 搭建模型
        gru_model = Sequential()
        gru_model.add(GRU(gru_size, activation='relu', return_sequences=True))
        gru_model.add(Dropout(0.2))
        gru_model.add(GRU(gru_size, activation='relu', return_sequences=False))
        gru_model.add(Dropout(0.2))
        gru_model.add(Dense(1, activation='linear'))
        gru_model.compile(loss="mse", optimizer=Adam(1e-4))
        gru_model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=0)
        gru_model.save('../../models/' + self.v + '/gru_model/gru.h5')

        lstm_model = load_model('../../models/' + self.v + '/gru_model/gru.h5')
        predictions_gru_24h = lstm_model.predict(test_x_24)
        predictions_gru_72h = lstm_model.predict(test_x_72)
        predictions_gru_168h = lstm_model.predict(test_x_168)

        min_, max_ = self.scaled_features_y[0]
        self.predictions_24h_gru_ = predictions_gru_24h * (max_ - min_) + min_
        self.predictions_24h_target_ = self.prediction_targets_24h * (max_ - min_) + min_

        min_, max_ = self.scaled_features_y[0]
        self.predictions_72h_gru_ = predictions_gru_72h * (max_ - min_) + min_
        self.predictions_72h_target_ = self.prediction_targets_72h * (max_ - min_) + min_

        min_, max_ = self.scaled_features_y[0]
        self.predictions_168h_gru_ = predictions_gru_168h * (max_ - min_) + min_
        self.predictions_168h_target_ = self.prediction_targets_168h * (max_ - min_) + min_

        fc = 100
        fs = 700
        w_n = 2 * fc / fs
        b, a = signal.butter(6, w_n, "lowpass")
        # 24小时
        self.predictions_24h_gru_filter = signal.filtfilt(b, a, self.predictions_24h_gru_.squeeze(), axis=-1,
                                                          padtype='odd', padlen=None, method='pad',
                                                          irlen=None)
        self.predictions_24h_target_filter = signal.filtfilt(b, a, self.predictions_24h_target_, axis=-1,
                                                             padtype='odd', padlen=None, method='pad',
                                                             irlen=None)
        print('RMSE', np.sqrt(MSE(self.predictions_24h_target_, self.predictions_24h_gru_)))
        print('MAPE', MAPE(self.predictions_24h_target_, self.predictions_24h_gru_))
        print('MAE', MAE(self.predictions_24h_target_, self.predictions_24h_gru_))

        # 72小时
        self.predictions_72h_gru_filter = signal.filtfilt(b, a, self.predictions_72h_gru_.squeeze(), axis=-1,
                                                          padtype='odd', padlen=None, method='pad',
                                                          irlen=None)
        self.predictions_72h_target_filter = signal.filtfilt(b, a, self.predictions_72h_target_, axis=-1,
                                                             padtype='odd', padlen=None, method='pad',
                                                             irlen=None)
        print('RMSE', np.sqrt(MSE(self.predictions_72h_target_, self.predictions_72h_gru_)))
        print('MAPE', MAPE(self.predictions_72h_target_, self.predictions_72h_gru_))
        print('MAE', MAE(self.predictions_72h_target_, self.predictions_72h_gru_))

        # 168小时
        self.predictions_168h_gru_filter = signal.filtfilt(b, a, self.predictions_168h_gru_.squeeze(), axis=-1,
                                                           padtype='odd', padlen=None, method='pad',
                                                           irlen=None)
        self.predictions_168h_target_filter = signal.filtfilt(b, a, self.predictions_168h_target_, axis=-1,
                                                              padtype='odd', padlen=None, method='pad',
                                                              irlen=None)
        print('RMSE', np.sqrt(MSE(self.predictions_168h_target_, self.predictions_168h_gru_)))
        print('MAPE', MAPE(self.predictions_168h_target_, self.predictions_168h_gru_))
        print('MAE', MAE(self.predictions_168h_target_, self.predictions_168h_gru_))

    def sd_dnn(self):
        '''深度神经网络预测模型
        '''
        input_nodes = self.train_features.shape[1]
        # 设置超参数
        epochs = 1000
        learning_rate = 0.001
        hidden1_nodes = 50
        hidden2_nodes = 40
        output_nodes = 1
        batch_size = 32
        rate = 0.1  # (0.1 - 0.7) 丢弃率
        # 重置图
        tf.reset_default_graph()
        # 设置数据接口
        x_train = tf.placeholder(dtype=tf.float32, shape=[None, input_nodes])
        y_train = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        is_training = tf.placeholder(tf.bool)  # hold 模型状态这样一个属性
        # 搭建网络
        fc1 = tf.layers.dense(x_train, hidden1_nodes, activation=tf.nn.relu,
                              kernel_initializer=tf.random_normal_initializer(stddev=0.1), name="layer1")
        fc1 = tf.layers.dropout(fc1, rate, training=is_training)
        fc2 = tf.layers.dense(fc1, hidden2_nodes, activation=tf.nn.relu,
                              kernel_initializer=tf.random_normal_initializer(stddev=0.1), name="layer2")
        fc2 = tf.layers.dropout(fc2, rate, training=is_training)
        logits = tf.layers.dense(fc2, output_nodes, activation=None,
                                 kernel_initializer=tf.random_normal_initializer(stddev=0.1), name="outputs")
        # 定义损失函数和优化器
        loss = tf.reduce_mean(tf.square(y_train - logits))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        init = tf.global_variables_initializer()
        # 配置config_proto
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        # 开始训练 -- 训练好保存模型之后直接测试即可
        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            sess.run(init)
            losses = {"train": []}
            for epoch in range(epochs):
                batch = np.random.choice([*range(self.train_features.shape[0])], size=batch_size)
                sess.run(optimizer, feed_dict={
                    x_train: self.train_features[batch, :],
                    y_train: self.train_targets[batch][:, None],
                    is_training: True
                })
                train_loss = sess.run(loss, feed_dict={
                    x_train: self.train_features,
                    y_train: self.train_targets[:, None],
                    is_training: True
                })
                sys.stdout.write(
                    "\rProgress: {:2.1f}".format(100 * epoch / float(epochs)) + "% ... Training loss: " + str(
                        train_loss)[:5])
                sys.stdout.flush()
                losses["train"].append(train_loss)
            saver.save(sess, "../../models/" + self.v + "/dnn_model/dnn.ckpt")

        saver = tf.train.Saver()
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            # 会将已经保存的变量值restore到变量中
            saver.restore(sess, "../../models/" + self.v + "/dnn_model/dnn.ckpt")
            predictions_dnn_24h = sess.run(logits, feed_dict={x_train: self.prediction_features_24h,
                                                              is_training: False})
            predictions_dnn_72h = sess.run(logits, feed_dict={x_train: self.prediction_features_72h,
                                                              is_training: False})
            predictions_dnn_168h = sess.run(logits, feed_dict={x_train: self.prediction_features_168h,
                                                               is_training: False})
            # 预测数据反归一化
            min_, max_ = self.scaled_features_y[0]
            self.predictions_24h_dnn_ = predictions_dnn_24h * (max_ - min_) + min_
            self.predictions_24h_target_ = self.prediction_targets_24h * (max_ - min_) + min_
            min_, max_ = self.scaled_features_y[0]
            self.predictions_72h_dnn_ = predictions_dnn_72h * (max_ - min_) + min_
            self.predictions_72h_target_ = self.prediction_targets_72h * (max_ - min_) + min_
            min_, max_ = self.scaled_features_y[0]
            self.predictions_168h_dnn_ = predictions_dnn_168h * (max_ - min_) + min_
            self.predictions_168h_target_ = self.prediction_targets_168h * (max_ - min_) + min_

            fc = 100
            fs = 700
            w_n = 2 * fc / fs
            b, a = signal.butter(6, w_n, "lowpass")
            # 24小时
            self.predictions_24h_dnn_filter = signal.filtfilt(b, a, self.predictions_24h_dnn_.squeeze(), axis=-1,
                                                              padtype='odd', padlen=None, method='pad',
                                                              irlen=None)
            self.predictions_24h_target_filter = signal.filtfilt(b, a, self.predictions_24h_target_, axis=-1,
                                                                 padtype='odd', padlen=None, method='pad',
                                                                 irlen=None)
            print('RMSE', np.sqrt(MSE(self.predictions_24h_target_, self.predictions_24h_dnn_)))
            print('MAPE', MAPE(self.predictions_24h_target_, self.predictions_24h_dnn_))
            print('MAE', MAE(self.predictions_24h_target_, self.predictions_24h_dnn_))

            # 72小时
            self.predictions_72h_dnn_filter = signal.filtfilt(b, a, self.predictions_72h_dnn_.squeeze(), axis=-1,
                                                              padtype='odd', padlen=None, method='pad',
                                                              irlen=None)
            self.predictions_72h_target_filter = signal.filtfilt(b, a, self.predictions_72h_target_, axis=-1,
                                                                 padtype='odd', padlen=None, method='pad',
                                                                 irlen=None)
            print('RMSE', np.sqrt(MSE(self.predictions_72h_target_, self.predictions_72h_dnn_)))
            print('MAPE', MAPE(self.predictions_72h_target_, self.predictions_72h_dnn_))
            print('MAE', MAE(self.predictions_72h_target_, self.predictions_72h_dnn_))

            # 168小时
            self.predictions_168h_dnn_filter = signal.filtfilt(b, a, self.predictions_168h_dnn_.squeeze(), axis=-1,
                                                               padtype='odd', padlen=None, method='pad',
                                                               irlen=None)
            self.predictions_168h_target_filter = signal.filtfilt(b, a, self.predictions_168h_target_, axis=-1,
                                                                  padtype='odd', padlen=None, method='pad',
                                                                  irlen=None)
            print('RMSE', np.sqrt(MSE(self.predictions_168h_target_, self.predictions_168h_dnn_)))
            print('MAPE', MAPE(self.predictions_168h_target_, self.predictions_168h_dnn_))
            print('MAE', MAE(self.predictions_168h_target_, self.predictions_168h_dnn_))

    def save_prediction_data(self):
        """保存未滤波之前的网络预测结果
        """
        models_prediction_data_24 = np.concatenate([self.predictions_24h_RF_[:, None],
                                                    self.predictions_24h_SVM_[:, None],
                                                    self.predictions_24h_xgb_[:, None],
                                                    self.predictions_24h_dnn_,
                                                    self.predictions_24h_lstm_,
                                                    self.predictions_24h_gru_,
                                                    self.predictions_24h_target_[:, None],
                                                    self.actual_heat_load[-168:-144][:, None]], axis=1)
        np.savetxt('../../data/' + self.v + '/models_prediction_data_24.csv', models_prediction_data_24, delimiter=','
                   , encoding='utf-8', fmt='%.5f')

        models_prediction_data_72 = np.concatenate([self.predictions_72h_RF_[:, None],
                                                    self.predictions_72h_SVM_[:, None],
                                                    self.predictions_72h_xgb_[:, None],
                                                    self.predictions_72h_dnn_,
                                                    self.predictions_72h_lstm_,
                                                    self.predictions_72h_gru_,
                                                    self.predictions_72h_target_[:, None],
                                                    self.actual_heat_load[-168:-96][:, None]], axis=1)
        np.savetxt('../../data/' + self.v + '/models_prediction_data_72.csv', models_prediction_data_72, delimiter=','
                   , encoding='utf-8', fmt='%.5f')

        models_prediction_data_168 = np.concatenate([self.predictions_168h_RF_[:, None],
                                                     self.predictions_168h_SVM_[:, None],
                                                     self.predictions_168h_xgb_[:, None],
                                                     self.predictions_168h_dnn_,
                                                     self.predictions_168h_lstm_,
                                                     self.predictions_168h_gru_,
                                                     self.predictions_168h_target_[:, None],
                                                     self.actual_heat_load[-168:][:, None]], axis=1)
        np.savetxt('../../data/' + self.v + '/models_prediction_data_168.csv', models_prediction_data_168,
                   delimiter=','
                   , encoding='utf-8', fmt='%.5f')

    def run(self):
        self.sd_rf()
        print('------------------------------')
        self.sd_svm()
        print('------------------------------')
        self.sd_xgboost()
        print('------------------------------')
        self.sd_dnn()
        print('------------------------------')
        self.sd_lstm()
        print('------------------------------')
        self.sd_gru()
        print('------------------------------')
        self.save_prediction_data()


if __name__ == '__main__':
    ModelsPrediction().run()
