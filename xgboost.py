# Xgboost
import xgboost
# Firest Xgboost model for Pima Indains dataset
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load data
def load_data(file_name, sequence_length=10, split=0.89):
    # load_data(file_name, sequence_length=10, split=0.8)
    df = pd.read_csv(file_name, sep=',', usecols=[1])
    data_all = np.array(df).astype(float)
    scaler = MinMaxScaler()
    data_all = scaler.fit_transform(data_all)
    data = []
    for i in range(len(data_all) - sequence_length - 1):
        data.append(data_all[i: i + sequence_length + 1])
    reshaped_data = np.array(data).astype('float64')
    np.random.shuffle(reshaped_data)
    # 对x进行统一归一化，而y则不归一化
    x = reshaped_data[:, :-1]
    y = reshaped_data[:, -1]
    split_boundary = int(reshaped_data.shape[0] * split)
    train_x = x[: split_boundary]
    test_x = x[split_boundary:]
    train_y = y[: split_boundary]
    test_y = y[split_boundary:]
    return train_x, train_y, test_x, test_y, scaler

if __name__ == '__main__':
    train_x, train_y, test_x, test_y, scaler = load_data('DE150_time.csv')
    # Fit model no training data
    model = XGBClassifier()
    model.fit(train_x,train_y)
    # Make predictions for test data
    y_pred = model.predict(test_x)
    predictions = [round(value) for value in y_pred]  # ????
    # Evaluate predictions
    accuracy = accuracy_score(test_y, predictions)
    print('Accuracy:%.2f%%' % (accuracy * 100.0))
