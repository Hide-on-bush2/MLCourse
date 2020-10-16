import numpy as np
import pandas as pd

def read_data_numpy(filename):
    anscombe = np.loadtxt(filename, dtype=np.str, delimiter=",")
    # print(anscombe)
    return anscombe

def data_processing_numpy(data):
    x = np.array(data[1:, 1].astype(np.float))
    y = np.array(data[1:, 2].astype(np.float))

    print("x:\n", x)
    print("x_sum: ", x.sum())
    print("x_min: ", x.min())
    print("x_max: ", x.max())
    print("x_mean: ", x.mean())
    print("x_var: ", x.var())
    print("x_std: ", x.std())
    print("x_argmax: ", x.argmin())
    print("x_argmin: ", x.argmax())
    print()
    print("y:\n", y)
    print("y_sum: ", y.sum())
    print("y_min: ", y.min())
    print("y_max: ", y.max())
    print("y_mean: ", y.mean())
    print("y_var: ", y.var())
    print("y_std: ", y.std())
    print("y_argmax: ", y.argmin())
    print("y_argmin: ", y.argmax())

def read_data_pandas(filename):
    df = pd.read_csv(filename)
    return df 

def data_processing_pandas(data):
    print(data.describe())

    print(data.groupby('dataset').describe())

if __name__ == "__main__":
    data = read_data_numpy("./data/anscombe.csv")
    data_processing_numpy(data)

    data = read_data_pandas("./data/anscombe.csv")
    data_processing_pandas(data)