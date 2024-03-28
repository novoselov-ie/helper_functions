import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import pywt
from sklearn.preprocessing import MinMaxScaler


def create_dataloaders(path_file, num_values, batch_size=512, test_size=0.3, valid_size=0.5, random_state=42):
    print("Загрузка файла...")
    data_file = pd.read_csv(path_file)

    # Предобработка файла
    print("Предобработка файла...")
    ## Удаление значений с нулем
    zero_counts_P = (data_file['P'] == 0).sum()
    zero_counts_N = (data_file['N'] == 0).sum()
    zero_counts_K = (data_file['K'] == 0).sum()
    print(f"Количество нулей в колонке 'P': {zero_counts_P}")
    print(f"Количество нулей в колонке 'N': {zero_counts_N}")
    print(f"Количество нулей в колонке 'K': {zero_counts_K}")
    print('_' * 100)

    if zero_counts_P > 0 or zero_counts_N > 0 or zero_counts_K > 0:
        print("Удаляются строки с нулевыми значениями в колонках 'P', 'N' и 'K'...")
        data_file = data_file.drop(data_file[data_file['N'] == 0].index)
        data_file = data_file.drop(data_file[data_file['P'] == 0].index)
        data_file = data_file.drop(data_file[data_file['K'] == 0].index)
        zero_counts_P = (data_file['P'] == 0).sum()
        zero_counts_N = (data_file['N'] == 0).sum()
        zero_counts_K = (data_file['K'] == 0).sum()
        print(f"Количество нулей в колонке 'P': {zero_counts_P}")
        print(f"Количество нулей в колонке 'N': {zero_counts_N}")
        print(f"Количество нулей в колонке 'K': {zero_counts_K}")
        print('_' * 100)

    X = data_file.drop(columns=['pH(H2O)', 'P', 'N', 'K', 'OC', 'CaCO3'], axis=1).values
    wl = data_file.drop(columns=['pH(H2O)', 'P', 'N', 'K', 'OC', 'CaCO3'], axis=1).columns.values

    print("Уменьшение размерной сетки...")
    if num_values < 4200:
        print(f"Уменьшение размерной сетки до {num_values}...")
        new_size = num_values
    else:
        print("Размерная сетка уже меньше или равна 4200.")
        new_size = 4200
    size_arr = (new_size * 2, int(2100 / new_size))
    X_filtered = np.zeros((X.shape[0], size_arr[0]))

    for i in range(X.shape[0]):
        X_reshape = X[i,].reshape(size_arr)
        for j in range(X_reshape.shape[0]):
            X_filtered[i, j] = np.mean(X_reshape[j,])

    wl_old = np.arange(400, 2500, 0.5)  # Существующая сетка длин волн
    wl_new = np.linspace(400, wl_old.max(), new_size * 2)  # Новая сетка длин волн
    X = X_filtered
    wl = wl_new
    ##Применение разных фильтров к параметрам X
    print("Применение разных фильтров к параметрам X...")
    X1 = savgol_filter(X, 11, polyorder=2, deriv=1)
    X_dwt = pywt.dwt(X, 'db1')

    ##Нормализация данных
    print("Нормализация данных...")
    scaler_X = MinMaxScaler()
    scaler_X1 = MinMaxScaler()
    scaler_X_dwt0 = MinMaxScaler()
    scaler_X_dwt1 = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X[:, ::2])
    X1_scaled = scaler_X1.fit_transform(X1[:, ::2])
    X_dwt0_scaled = scaler_X_dwt0.fit_transform(X_dwt[0])
    X_dwt1_scaled = scaler_X_dwt1.fit_transform(X_dwt[1])

    print("Размеры преобразованных данных:")
    print(X_scaled.shape, X1_scaled.shape, X_dwt0_scaled.shape, X_dwt1_scaled.shape)

    y = data_file[['N', 'P', 'K']].values
    print("Размеры меток:")
    print(y.shape)

    # Загрузка данных
    print("Преобразование в тензоры и разделение на обучающий, валидационный и тестовый наборы данных...")
    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y)
    y_tensor = torch.Tensor(y_scaled)

    n = min(num_values, X_scaled.shape[1])  # Обрабатываем случай, если num_values больше, чем размер данных

    X_combined_scaled = torch.stack([torch.Tensor(X_scaled[:, :n]), torch.Tensor(X1_scaled[:, :n]),
                                     torch.Tensor(X_dwt0_scaled[:, :n]), torch.Tensor(X_dwt1_scaled[:, :n])], dim=2)

    # Разделение на обучающий, валидационный и тестовый наборы данных
    X_train_tensor, X_temp_tensor, y_train_tensor, y_temp_tensor = train_test_split(X_combined_scaled, y_tensor,
                                                                                    test_size=test_size,
                                                                                    random_state=random_state)
    X_valid_tensor, X_test_tensor, y_valid_tensor, y_test_tensor = train_test_split(X_temp_tensor, y_temp_tensor,
                                                                                    test_size=valid_size,
                                                                                    random_state=random_state)

    # Перестановка размерностей для DataLoader
    X_train_tensor = X_train_tensor.permute(0, 2, 1)
    X_valid_tensor = X_valid_tensor.permute(0, 2, 1)
    X_test_tensor = X_test_tensor.permute(0, 2, 1)

    # Создание наборов данных
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # Создание DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Готово!")

    return train_loader, valid_loader, test_loader, target_scaler

