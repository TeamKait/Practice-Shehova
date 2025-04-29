# %%
# Анализ и прогнозирование рынка акций из Excel-файла с линейной регрессией
# Добавлены модули предварительной обработки: удаление выбросов, дубликатов, обработка пропусков, масштабирование и кодирование.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# %%
# 1. Загрузка данных из Excel
# Установите openpyxl для работы с .xlsx
excel_path = 'data.xlsx'
df = pd.read_excel(excel_path, sheet_name=0)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
df.set_index('Date', inplace=True)

# 1.1 Удаление дубликатов
df = df.drop_duplicates()
print('После удаления дубликатов:', df.shape)

# 1.2 Обработка пропущенных значений
# Для числовых признаков заполним медианой
num_cols = ['Open','High','Low','Close','Volume']
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
print('Пропуски после заполнения медианой:', df[num_cols].isnull().sum().to_dict())

# 1.3 Преобразование типов (уже числовые, но убеждаемся)
df[num_cols] = df[num_cols].astype(float)

# 1.4 Фильтрация выбросов: считаем цены закрытия >100 выбросами
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

# Определяем признаки
features = ['Open','High','Low','Close','Volume','SMA_7','SMA_21']
X = df_feat[features]
y = df_feat['Target']

# %%
# 6. Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print('Train size:', X_train.shape)
print('Test size:', X_test.shape)

# %%
# 7. Масштабирование признаков (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
# 8. Обучение модели линейной регрессии
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)

# %%
# 9. Оценка качества линейной регрессии
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"LinearRegression: RMSE={rmse:.4f}, R2={r2:.4f}")

evaluate(y_test, y_pred)

# %%
# 10. Визуализация: прогноз vs фактические значения
plt.figure(figsize=(12,6))
plt.plot(y_test.index, y_test, label='Actual')
plt.plot(y_test.index, y_pred, label='Predicted LR')
plt.title('Actual vs Predicted (Linear Regression)')
plt.xlabel('Дата')
plt.ylabel('Цена')
plt.legend()
plt.show()

# %%
# 11. График Predicted vs Actual с линией y=x
plt.figure(figsize=(8,8))
plt.scatter(y_test, y_pred, alpha=0.6)
lims = [min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))]
plt.plot(lims, lims, '--', linewidth=2)
plt.title('Predicted vs Actual with Regression Line')
plt.xlabel('Actual Close Price')
plt.ylabel('Predicted Close Price')
plt.show()

# %%
# Выводы
# Добавлены модули предварительной обработки: удаление дубликатов, заполнение пропусков, масштабирование.
# Модель линейной регрессии позволяет получить базовый прогноз цены. Для улучшения — добавить новые признаки или регуляризовать модель.