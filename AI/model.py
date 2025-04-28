# %%
# Анализ и прогнозирование рынка акций из Excel-файла с помощью Linear Regression
# В этом скрипте проведем EDA и построим модель линейной регрессии для прогнозирования цены закрытия акции на следующий день.
# Данные: Date, Open, High, Low, Close, Volume (в формате .xlsx)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# %%
# 1. Загрузка и предобработка данных из Excel
# Убедитесь, что установлены: pip install openpyxl
excel_path = 'data.xlsx'
df = pd.read_excel(excel_path, sheet_name=0)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.set_index('Date', inplace=True)

# Проверка
print(df.head())
print(df.info())
print('Пропущенные значения:', df.isnull().sum().to_dict())
print('Дубликаты:', df.duplicated().sum())

# %%
# 1\.1 Фильтрация выбросов (отдельный модуль)
# Считаем цены закрытия больше 100 выбросами и исключаем их из анализа
df = df[df['Close'] <= 100]
print('После фильтрации выбросов (Close>100):', df.shape)

# %%
# 2. Исследовательский анализ (EDA)
print(df.describe())
corr = df.corr()
print('Корреляция:\n', corr)

# %%
# 3. Визуализация цен и объема
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Close'], label='Close')
plt.title('Цена закрытия')
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.legend()
plt.show()

plt.figure(figsize=(12,4))
plt.bar(df.index, df['Volume'])
plt.title('Объем торгов')
plt.xlabel('Дата')
plt.ylabel('Volume')
plt.show()

# %%
# 4. Технические индикаторы: SMA7, SMA21
df['SMA_7'] = df['Close'].rolling(7).mean()
df['SMA_21'] = df['Close'].rolling(21).mean()

plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Close')
plt.plot(df['SMA_7'], label='SMA 7')
plt.plot(df['SMA_21'], label='SMA 21')
plt.title('Close и SMA')
plt.legend()
plt.show()

# %%
# 5. Формирование признаков и целевой переменной
df_feat = df.copy()
df_feat['Target'] = df_feat['Close'].shift(-1)
df_feat.dropna(inplace=True)
features = ['Open','High','Low','Close','Volume','SMA_7','SMA_21']
X = df_feat[features]
y = df_feat['Target']

# %%
# 6. Разбиение на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print('Train size:', X_train.shape)
print('Test size:', X_test.shape)

# %%
# 7. Обучение модели линейной регрессии
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# %%
# 8. Оценка качества линейной регрессии
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"LinearRegression: RMSE={rmse:.4f}, R2={r2:.4f}")

evaluate(y_test, y_pred)

# %%
# 9. Визуализация прогноза vs фактических значений для Linear Regression
plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted LR')
plt.title('Actual vs Predicted (Linear Regression)')
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.legend()
plt.show()

# %%
# 10. График зависимости предсказанных значений от фактических с прямой линейной регрессии
plt.figure(figsize=(8,8))
plt.scatter(y_test, y_pred, alpha=0.6)
# линия y = x для идеального прогноза
lims = [min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))]
plt.plot(lims, lims, '--', linewidth=2)
plt.title('Predicted vs Actual with Regression Line')
plt.xlabel('Actual Close Price')
plt.ylabel('Predicted Close Price')
plt.show()

# %%
# Выводы
# Модель линейной регрессии позволяет получить базовый прогноз цены закрытия.
# Для повышения точности можно добавить новые признаки или регуляризовать модель.
# Модель линейной регрессии позволяет получить базовый прогноз цены закрытия.
# Для повышения точности можно добавить новые признаки или регуляризовать модель.
