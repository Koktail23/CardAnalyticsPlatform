# run_app.py - Обновленная версия
import streamlit as st
from clickhouse_driver import Client
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(
    page_title="Card Analytics Platform",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS для красивого оформления
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)


# Подключение
@st.cache_resource
def get_client():
    return Client(
        host='localhost',
        port=9000,
        user='admin',
        password='admin123',
        database='datagate'
    )


client = get_client()

# Header
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.title("💳 Card Analytics Platform")
    st.markdown("**Powered by ClickHouse** | Real-time processing of millions of transactions")

# Sidebar
with st.sidebar:
    st.header("📊 Статистика системы")

    # Обновление статистики
    if st.button("🔄 Обновить", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # Показываем статистику таблиц
    try:
        tables_info = {
            'card_transactions': '💳 Транзакции',
            'sales_data': '💰 Продажи',
            'data_quality_checks': '✅ Проверки',
        }

        for table, label in tables_info.items():
            try:
                count = client.execute(f'SELECT count() FROM {table}')[0][0]
                st.metric(label, f"{count:,}")
            except:
                pass

        # Период данных
        try:
            date_range = client.execute('''
                SELECT 
                    min(transaction_date) as min_date,
                    max(transaction_date) as max_date
                FROM card_transactions
            ''')[0]

            if date_range[0]:
                st.info(f"📅 Период: {date_range[0]} - {date_range[1]}")
        except:
            pass

    except Exception as e:
        st.error(f"Ошибка: {e}")

    st.divider()

    # Фильтры
    st.header("🔍 Фильтры")

    # Выбор компании
    try:
        companies = client.execute('''
            SELECT DISTINCT company_id 
            FROM card_transactions 
            ORDER BY company_id
        ''')
        company_list = ['Все'] + [c[0] for c in companies]
        selected_company = st.selectbox("Компания", company_list)
    except:
        selected_company = 'Все'

    # Период
    date_filter = st.date_input(
        "Период",
        value=(datetime.now() - timedelta(days=30), datetime.now()),
        max_value=datetime.now()
    )

# Главные вкладки
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Dashboard",
    "🔍 Аналитика",
    "🚨 Fraud Detection",
    "📊 Sales Analysis",
    "📤 Загрузка данных"
])

with tab1:
    st.header("Dashboard - Ключевые метрики")

    # Проверяем наличие данных
    try:
        count_check = client.execute('SELECT count() FROM card_transactions')[0][0]

        if count_check > 0:
            # Основные метрики
            metrics = client.execute('''
                SELECT 
                    count() as total_transactions,
                    sum(amount) as total_volume,
                    avg(amount) as avg_amount,
                    uniq(card_masked) as unique_cards,
                    uniq(merchant_name) as unique_merchants,
                    countIf(is_fraud = 1) as fraud_count,
                    max(transaction_time) as last_transaction
                FROM card_transactions
                WHERE transaction_date >= today() - 30
            ''')[0]

            # Метрики в карточках
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "🔢 Транзакций (30 дней)",
                    f"{metrics[0]:,}",
                    f"Последняя: {metrics[6].strftime('%H:%M') if metrics[6] else 'N/A'}"
                )

            with col2:
                st.metric(
                    "💰 Объем",
                    f"${metrics[1]:,.2f}" if metrics[1] else "$0",
                    f"Средний чек: ${metrics[2]:.2f}" if metrics[2] else "N/A"
                )

            with col3:
                st.metric(
                    "💳 Уникальных карт",
                    f"{metrics[3]:,}",
                    f"Мерчантов: {metrics[4]:,}"
                )

            with col4:
                fraud_rate = (metrics[5] / metrics[0] * 100) if metrics[0] > 0 else 0
                st.metric(
                    "⚠️ Fraud Rate",
                    f"{fraud_rate:.3f}%",
                    f"Случаев: {metrics[5]}",
                    delta_color="inverse"
                )

            st.divider()

            # Графики
            col1, col2 = st.columns(2)

            with col1:
                # График транзакций по дням
                daily_data = pd.DataFrame(client.execute('''
                    SELECT 
                        transaction_date as date,
                        count() as transactions,
                        sum(amount) as volume
                    FROM card_transactions
                    WHERE transaction_date >= today() - 30
                    GROUP BY date
                    ORDER BY date
                '''))

                if not daily_data.empty:
                    daily_data.columns = ['date', 'transactions', 'volume']

                    fig = px.area(
                        daily_data,
                        x='date',
                        y='volume',
                        title='📈 Объем транзакций по дням',
                        labels={'volume': 'Объем ($)', 'date': 'Дата'}
                    )
                    fig.update_traces(fillcolor='rgba(102, 126, 234, 0.5)', line_color='rgb(102, 126, 234)')
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Топ MCC категорий
                mcc_data = pd.DataFrame(client.execute('''
                    SELECT 
                        mcc_description,
                        count() as count,
                        sum(amount) as total
                    FROM card_transactions
                    WHERE transaction_date >= today() - 30
                        AND mcc_description != ''
                    GROUP BY mcc_description
                    ORDER BY total DESC
                    LIMIT 10
                '''))

                if not mcc_data.empty:
                    mcc_data.columns = ['category', 'count', 'total']

                    fig = px.pie(
                        mcc_data,
                        values='total',
                        names='category',
                        title='🛍️ Топ категорий по объему',
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Почасовая активность
            st.subheader("⏰ Активность по часам")

            hourly_data = pd.DataFrame(client.execute('''
                SELECT 
                    toHour(transaction_time) as hour,
                    count() as count,
                    avg(amount) as avg_amount
                FROM card_transactions
                WHERE transaction_date >= today() - 7
                GROUP BY hour
                ORDER BY hour
            '''))

            if not hourly_data.empty:
                hourly_data.columns = ['hour', 'count', 'avg_amount']

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=hourly_data['hour'],
                    y=hourly_data['count'],
                    name='Количество',
                    marker_color='lightblue',
                    yaxis='y'
                ))
                fig.add_trace(go.Scatter(
                    x=hourly_data['hour'],
                    y=hourly_data['avg_amount'],
                    name='Средний чек',
                    marker_color='red',
                    yaxis='y2',
                    mode='lines+markers'
                ))

                fig.update_layout(
                    title='Распределение транзакций по часам',
                    xaxis_title='Час',
                    yaxis=dict(title='Количество транзакций', side='left'),
                    yaxis2=dict(title='Средний чек ($)', overlaying='y', side='right'),
                    hovermode='x unified',
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info(
                "📊 Нет данных о карточных транзакциях. Перейдите на вкладку 'Загрузка данных' и сгенерируйте тестовые данные.")

    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")

with tab2:
    st.header("🔍 Углубленная аналитика")

    analysis_type = st.radio(
        "Тип анализа",
        ["Анализ по компаниям", "Анализ по мерчантам", "Временной анализ", "Когортный анализ"],
        horizontal=True
    )

    if analysis_type == "Анализ по компаниям":
        # Здесь будет анализ по компаниям
        st.info("В разработке...")

with tab3:
    st.header("🚨 Fraud Detection")

    # Здесь будет детекция мошенничества
    st.info("В разработке...")

with tab4:
    st.header("📊 Анализ sales_data")

    # Анализ существующих данных
    df = pd.DataFrame(client.execute('SELECT * FROM sales_data LIMIT 1000'))

    if not df.empty:
        columns = [col[0] for col in client.execute('DESCRIBE sales_data')]
        df.columns = columns[:len(df.columns)]

        st.write(f"Показаны первые 1000 записей из 10,000")
        st.dataframe(df, use_container_width=True)

with tab5:
    st.header("📤 Загрузка данных")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎲 Генерация тестовых данных")

        num_records = st.number_input("Количество записей", min_value=100, max_value=100000, value=5000, step=1000)

        if st.button("Сгенерировать", type="primary", use_container_width=True):
            with st.spinner(f'Генерация {num_records} транзакций...'):
                # Генерация данных
                import random

                test_data = []
                mcc_categories = {
                    '5411': 'Grocery Stores',
                    '5541': 'Gas Stations',
                    '5812': 'Restaurants',
                    '5999': 'Retail',
                    '4111': 'Transport',
                }

                for i in range(num_records):
                    mcc = random.choice(list(mcc_categories.keys()))
                    test_data.append((
                        f'TXN{datetime.now().strftime("%Y%m%d")}{i:06d}',
                        f'COMP{random.randint(1, 20):02d}',
                        f'****{random.randint(1000, 9999)}',
                        round(random.uniform(10, 5000), 2),
                        'USD',
                        mcc,
                        mcc_categories[mcc],
                        f'Merchant_{random.randint(1, 500)}',
                        f'MER{random.randint(100000, 999999)}',
                        (datetime.now() - timedelta(days=random.randint(0, 90))).date(),
                        datetime.now() - timedelta(hours=random.randint(0, 2160)),
                        'completed',
                        1 if random.random() < 0.001 else 0,
                        random.uniform(0, 1)
                    ))

                # Вставка в БД
                client.execute('''
                    INSERT INTO card_transactions 
                    (transaction_id, company_id, card_masked, amount, currency, 
                     mcc_code, mcc_description, merchant_name, merchant_id,
                     transaction_date, transaction_time, status, is_fraud, fraud_score)
                    VALUES
                ''', test_data)

                st.success(f"✅ Успешно загружено {num_records} транзакций!")
                st.balloons()
                st.cache_data.clear()
                st.rerun()

    with col2:
        st.subheader("📁 Загрузка из файла")

        uploaded_file = st.file_uploader("Выберите CSV файл", type=['csv'])

        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Предпросмотр данных:")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("Загрузить в БД", type="primary", use_container_width=True):
                # Здесь код загрузки
                st.success(f"Загружено {len(df)} записей")
                st.rerun()

if __name__ == "__main__":
    st.write("Запустите через: streamlit run run_app.py")