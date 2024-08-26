import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def preprocess_data(data, target_col):
    data['Target'] = data[target_col].shift(-1)
    data = data.dropna()
    
    scaler = MinMaxScaler()
    data.loc[:, ['Close', 'Google_Trends']] = scaler.fit_transform(data[['Close', 'Google_Trends']])
    
    return data

def split_data(data, test_size):
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=False)
    return train_data, test_data

mydata = pd.read_csv("mydata.csv")
amazon_trends = pd.read_csv("amazon_trends.csv")

mydata = mydata.merge(amazon_trends, on="Date", how='left')

data = preprocess_data(mydata, target_col='Close')

train_data, test_data = split_data(data, test_size=0.2)

X_train = train_data[['Close', 'Google_Trends']].values
y_train = train_data['Target'].values
X_test = test_data[['Close', 'Google_Trends']].values
y_test = test_data['Target'].values

svm_param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma': [0.001, 0.01, 0.1]
}
svm_model = GridSearchCV(SVR(), svm_param_grid, cv=3)
svm_model.fit(X_train, y_train)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
mlp_model = Sequential()
mlp_model.add(Dense(units=50, activation='relu', input_shape=(X_train.shape[1],)))
mlp_model.add(Dropout(0.2))
mlp_model.add(Dense(units=50, activation='relu'))
mlp_model.add(Dense(1))
mlp_model.compile(loss='mean_squared_error', optimizer='adam')
mlp_model.fit(X_train, y_train, epochs=120, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

y_pred_svm = svm_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)
y_pred_mlp = mlp_model.predict(X_test).flatten()

models = {
    'SVM': (svm_model, y_pred_svm),
    'Linear Regression': (lr_model, y_pred_lr),
    'MLP': (mlp_model, y_pred_mlp)
}

def plot_actual_vs_predicted(model_name, y_test, y_pred):
    plt.figure(figsize=(12, 8))
    plt.plot(y_test, label=f'Actual - {model_name}', linestyle='--')
    plt.plot(y_pred, label=f'Predicted - {model_name}')
    plt.title(f'Actual vs. Predicted for {model_name}')
    plt.xlabel('Data Points')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

for model_name, (model, y_pred) in models.items():
    plot_actual_vs_predicted(model_name, y_test, y_pred)

mse_scores = {}
mae_scores = {}
r2_scores = {}

for model_name, (model, y_pred) in models.items():
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    mse_scores[model_name] = mse
    mae_scores[model_name] = mae
    r2_scores[model_name] = r2

metrics_table = pd.DataFrame({
    'Model': list(models.keys()),
    'MSE': list(mse_scores.values()),
    'MAE': list(mae_scores.values()),
    'R2': list(r2_scores.values())
})

print(metrics_table)
