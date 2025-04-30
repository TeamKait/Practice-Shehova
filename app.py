import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Путь к файлу Excel
EXCEL_PATH = "AI/data.xlsx"

@st.cache_data
def load_data():
    df = pd.read_excel(EXCEL_PATH, sheet_name=0)
    # если нет столбца Date, пробуем первый столбец
    if 'Date' not in df.columns:
        df.columns = ['Date'] + list(df.columns[1:])
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df.set_index('Date', inplace=True)
    return df

@st.cache_data
def preprocess(df):
    df = df.drop_duplicates()
    num_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df = df[df['Close'] <= 100]
    df['SMA_7'] = df['Close'].rolling(7).mean()
    df['SMA_21'] = df['Close'].rolling(21).mean()
    df = df.dropna()
    return df

@st.cache_data
def train_model(df):
    features = ['Open','High','Low','Close','Volume','SMA_7','SMA_21']
    X = df[features]
    y = df['Close'].shift(-1).dropna()
    X = X.iloc[:-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    return model, scaler

# Функция добавления записи (без кэша)
def add_record(df, new_record):
    df = pd.concat([df, new_record])
    df.to_excel(EXCEL_PATH)
    return df

# Инициализация состояния для модели и переобучения
if 'needs_retrain' not in st.session_state:
    st.session_state.needs_retrain = False
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.scaler = None

# --- UI Приложения ---
st.title("Анализ и прогнозирование рынка акций")

# Краткое описание
st.markdown(
    """
    Это приложение просматривает историю цен акций, позволяет добавлять новые данные и прогнозировать цену закрытия на следующий день.
    """
)

# 1. Исходные данные
st.subheader("1. Исходные данные")
df = load_data()
min_date = df.index.min().date()
max_date = df.index.max().date()
start_date, end_date = st.slider("Период отображения данных", min_date, max_date, (min_date, max_date))
df_view = df.loc[pd.to_datetime(start_date):pd.to_datetime(end_date)]
st.dataframe(df_view)

# 2. Добавление записи
st.subheader("2. Добавление новой записи")
st.markdown("Заполните все поля (цены ≥0, объём ≥0) и нажмите **Добавить запись**.")
with st.form("form_add"):
    date = st.date_input("Дата записи")
    open_p = st.number_input("Open: цена открытия", min_value=0.0, step=0.01)
    high_p = st.number_input("High: макс. цена", min_value=0.0, step=0.01)
    low_p = st.number_input("Low: мин. цена", min_value=0.0, step=0.01)
    close_p = st.number_input("Close: цена закрытия", min_value=0.0, step=0.01)
    volume = st.number_input("Volume: объём торгов", min_value=0, step=1)
    add_btn = st.form_submit_button("Добавить запись")
    if add_btn:
        new = pd.DataFrame({
            'Open': [open_p], 'High': [high_p], 'Low': [low_p], 'Close': [close_p], 'Volume': [volume]
        }, index=[pd.to_datetime(date)])
        df = add_record(df, new)
        st.success("Новая запись добавлена!")
        st.session_state.needs_retrain = True

# 3. Обучение модели
st.subheader("3. Обучение модели")
# Описание блока
st.markdown("Нажмите кнопку ниже, чтобы переобучить модель на актуальных данных.")
# Кнопка переобучения под описанием
retrain_btn = st.button("Переобучить модель")
if retrain_btn:
    with st.spinner("Переобучение модели..."):
        df_proc = preprocess(df)
        model, scaler = train_model(df_proc)
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.needs_retrain = False
    st.success(f"Модель переобучена на {len(df_proc)} записях.")

# После первичной тренировки показываем сколько записей
if st.session_state.model is None:
    with st.spinner("Обучение модели..."):
        df_proc = preprocess(df)
        model, scaler = train_model(df_proc)
        st.session_state.model = model
        st.session_state.scaler = scaler
    st.success(f"Модель обучена на {len(df_proc)} записях.")

# 4. Сделать прогноз Сделать прогноз
st.subheader("4. Сделать прогноз")
st.markdown(
    """Для прогноза введите параметры текущего дня и рассчитанные SMA, затем нажмите **Сделать прогноз**.  
    **SMA_7**: средняя цена закрытия за 7 дней.  
    **SMA_21**: средняя цена закрытия за 21 день."""
)
with st.form("form_pred"):
    o = st.number_input("Open: цена открытия", min_value=0.0, step=0.01, key="o2")
    h = st.number_input("High: макс. цена", min_value=0.0, step=0.01, key="h2")
    l = st.number_input("Low: мин. цена", min_value=0.0, step=0.01, key="l2")
    c = st.number_input("Close: цена закрытия", min_value=0.0, step=0.01, key="c2")
    v = st.number_input("Volume: объём торгов", min_value=0, step=1, key="v2")
    s7 = st.number_input("SMA_7: скользящая средняя за 7 дней", min_value=0.0, step=0.01, key="s71")
    s21 = st.number_input("SMA_21: скользящая средняя за 21 день", min_value=0.0, step=0.01, key="s212")
    pred_btn = st.form_submit_button("Сделать прогноз")
    if pred_btn:
        X_new = np.array([[o, h, l, c, v, s7, s21]])
        Xs = st.session_state.scaler.transform(X_new)
        p = st.session_state.model.predict(Xs)[0]
        st.success(f"Прогноз цены закрытия на следующий день: **{p:.2f}**")

# Конец приложения
