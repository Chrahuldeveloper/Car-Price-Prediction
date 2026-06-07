import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import kagglehub

path = kagglehub.dataset_download("nehalbirla/vehicle-dataset-from-cardekho")

os.listdir(path)

df = pd.read_csv(path + "/car data.csv")

df = df.drop(["Car_Name", "Seller_Type", "Owner","Fuel_Type"], axis=1)
df["Transmission"] = df["Transmission"].map({"Manual": 1, "Automatic": 0})

X = df.drop("Selling_Price", axis=1)
Y = df["Selling_Price"]


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

x_mean = X_train.mean()
y_mean =  Y_train.mean()
y_std =  Y.std()
x_std = X.std()


X_train = (X_train- x_mean) / x_std
X_test = (X_test-x_mean) / x_std

Y_train = (Y_train - y_mean) / y_std
Y_test = (Y_test - y_mean) / y_std

w = np.random.randn(X_train.shape[1])
b = 0
lr = 0.001
epochs = 1000
losses = []

for i in range(epochs):
    y_pred = np.dot(X_train,w)+b

    loss = np.mean((Y_train-y_pred)**2)
    losses.append(loss)

    dw = (-2/len(X_train)) * np.dot(X_train.T, (Y_train - y_pred))
    db = (-2/len(X_train)) * np.sum(Y_train - y_pred)

    w -= lr * dw
    b -= lr * db

    y_train_pred = np.dot(X_train, w) + b
    y_test_pred = np.dot(X_test, w) + b

    train_mse = np.mean((Y_train - y_train_pred)**2)
    print("Train MSE:", train_mse)

    test_mse = np.mean((Y_test - y_test_pred)**2)
    print("Test MSE:", test_mse)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot)

print("Train R²:", r2_score(Y_train, y_train_pred))
print("Test R²:", r2_score(Y_test, y_test_pred))

plt.plot(range(epochs), losses, color="black", linewidth=2, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()


def predict(new_data):

    data_normalization = (new_data - x_mean) / x_std

    y_pred_scaled = np.dot(data_normalization,w) + b

    y_pred_original = (y_pred_scaled * y_std) + y_mean

    return y_pred_original


sample = np.array([[2020, 1, 50000, 1]])
print("Predicted Price:", predict(sample))