# run_app.py - Исправленная версия
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
        user='analyst',
        password='admin123',
        database='card_analytics'  # Правильная БД
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

    # ВЫБОР ТАБЛИЦЫ
    st.subheader("🗂️ Выбор источника данных")

    # Получаем список доступных таблиц
    available_tables = []
    table_descriptions = {
        'transactions_simple': '📝 Простая (все String)',
        'transactions_optimized': '⚡ Оптимизированная',
        'transactions_main': '🏢 Основная'
    }

    for table_name in table_descriptions.keys():
        try:
            count = client.execute(f'SELECT count() FROM {table_name}')[0][0]
            if count > 0:
                available_tables.append((table_name, count))
        except:
            pass

    if available_tables:
        # Создаем список для selectbox
        table_options = []
        for table_name, count in available_tables:
            desc = table_descriptions.get(table_name, table_name)
            table_options.append(f"{desc} ({count:,} записей)")

        # Селектор таблицы
        selected_table_option = st.selectbox(
            "Выберите таблицу для анализа:",
            table_options,
            index=0
        )

        # Извлекаем имя таблицы
        selected_table = available_tables[table_options.index(selected_table_option)][0]

        # Сохраняем в session state
        st.session_state['selected_table'] = selected_table

        # Показываем информацию о выбранной таблице
        st.success(f"✅ Используется: **{selected_table}**")
    else:
        st.error("❌ Нет доступных таблиц с данными")
        selected_table = 'transactions_simple'  # fallback

    st.divider()

    # Обновление статистики
    if st.button("🔄 Обновить", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    # Показываем статистику таблиц
    st.subheader("📈 Статистика таблиц")
    for table_name, count in available_tables:
        desc = table_descriptions.get(table_name, table_name)
        st.metric(desc, f"{count:,}")

    # Период данных
    try:
        # Проверяем, есть ли поле transaction_date в оптимизированной таблице
        if selected_table == 'transactions_optimized':
            date_range = client.execute(f'''
                SELECT 
                    min(transaction_date) as min_date,
                    max(transaction_date) as max_date
                FROM {selected_table}
            ''')[0]

            if date_range[0]:
                st.info(f"📅 Период: {date_range[0]} - {date_range[1]}")
        else:
            # Для простой таблицы используем rday
            date_range = client.execute(f'''
                SELECT 
                    min(toUInt32OrNull(rday)) as min_rday,
                    max(toUInt32OrNull(rday)) as max_rday
                FROM {selected_table}
            ''')[0]

            if date_range[0]:
                from datetime import datetime, timedelta

                # База 1900-01-01 для правильных дат
                base_date = datetime(1900, 1, 1)
                min_date = base_date + timedelta(days=date_range[0])
                max_date = base_date + timedelta(days=date_range[1])
                st.info(f"📅 Период: {min_date.date()} - {max_date.date()}")
    except Exception as e:
        st.warning(f"Не удалось получить период данных: {e}")

    st.divider()

    # Фильтры
    st.header("🔍 Фильтры")

    # Выбор банка
    try:
        banks = client.execute(f'''
            SELECT DISTINCT emitent_bank 
            FROM {selected_table}
            WHERE emitent_bank != ''
            ORDER BY emitent_bank
            LIMIT 20
        ''')
        bank_list = ['Все'] + [b[0] for b in banks]
        selected_bank = st.selectbox("Банк-эмитент", bank_list)
    except:
        selected_bank = 'Все'

    # Период (для будущих фильтров)
    date_filter = st.date_input(
        "Период",
        value=(datetime(2025, 1, 1), datetime(2025, 4, 30)),
        max_value=datetime.now()
    )

# Главные вкладки
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Dashboard",
    "🔍 Аналитика",
    "🚨 Fraud Detection",
    "📊 Анализ транзакций",
    "📤 Статус данных"
])

with tab1:
    st.header("Dashboard - Ключевые метрики")

    # Получаем выбранную таблицу
    selected_table = st.session_state.get('selected_table', 'transactions_simple')
    st.caption(f"📊 Источник данных: **{selected_table}**")

    # Проверяем наличие данных
    try:
        count_check = client.execute(f'SELECT count() FROM {selected_table}')[0][0]

        if count_check > 0:
            # Основные метрики - разные запросы для разных таблиц
            if selected_table == 'transactions_optimized':
                # Для оптимизированной таблицы - прямые запросы без конвертации
                metrics = client.execute(f'''
                    SELECT 
                        count() as total_transactions,
                        sum(amount_uzs) as total_volume,
                        avg(amount_uzs) as avg_amount,
                        uniq(hpan) as unique_cards,
                        uniq(merchant_name) as unique_merchants,
                        countIf(p2p_flag = 1) as p2p_count,
                        max(rday) as last_transaction
                    FROM {selected_table}
                ''')[0]
            else:
                # Для простой таблицы - с конвертацией типов
                metrics = client.execute(f'''
                    SELECT 
                        count() as total_transactions,
                        sum(toFloat64OrNull(amount_uzs)) as total_volume,
                        avg(toFloat64OrNull(amount_uzs)) as avg_amount,
                        uniq(hpan) as unique_cards,
                        uniq(merchant_name) as unique_merchants,
                        countIf(p2p_flag IN ('True', '1')) as p2p_count,
                        max(toUInt32OrNull(rday)) as last_transaction
                    FROM {selected_table}
                ''')[0]

            # Метрики в карточках
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "📢 Транзакций",
                    f"{metrics[0]:,}",
                    f"Всего в базе"
                )

            with col2:
                st.metric(
                    "💰 Объем",
                    f"{metrics[1]:,.0f} UZS" if metrics[1] else "0 UZS",
                    f"Средний чек: {metrics[2]:,.0f} UZS" if metrics[2] else "N/A"
                )

            with col3:
                st.metric(
                    "💳 Уникальных карт",
                    f"{metrics[3]:,}",
                    f"Мерчантов: {metrics[4]:,}"
                )

            with col4:
                p2p_rate = (metrics[5] / metrics[0] * 100) if metrics[0] > 0 else 0
                st.metric(
                    "💸 P2P переводы",
                    f"{p2p_rate:.1f}%",
                    f"Количество: {metrics[5]:,}"
                )

            st.divider()

            # Графики
            col1, col2 = st.columns(2)

            with col1:
                # График по датам (конвертируем rday в даты)
                if selected_table == 'transactions_optimized':
                    # Для оптимизированной таблицы используем готовое поле transaction_date
                    daily_data = pd.DataFrame(client.execute(f'''
                        SELECT 
                            transaction_date as date,
                            count() as transactions,
                            sum(amount_uzs) as volume
                        FROM {selected_table}
                        GROUP BY date
                        ORDER BY date
                    '''))
                else:
                    # Для простой таблицы конвертируем rday
                    daily_data = pd.DataFrame(client.execute(f'''
                        SELECT 
                            toDate('1900-01-01') + toUInt32OrNull(rday) as date,
                            count() as transactions,
                            sum(toFloat64OrNull(amount_uzs)) as volume
                        FROM {selected_table}
                        WHERE toUInt32OrNull(rday) IS NOT NULL
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
                        labels={'volume': 'Объем (UZS)', 'date': 'Дата'}
                    )
                    fig.update_traces(fillcolor='rgba(102, 126, 234, 0.5)', line_color='rgb(102, 126, 234)')
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Топ MCC категорий
                if selected_table == 'transactions_optimized':
                    # Для оптимизированной таблицы MCC уже числовой
                    mcc_data = pd.DataFrame(client.execute(f'''
                        SELECT 
                            CAST(mcc AS String) as mcc_code,
                            count() as count,
                            sum(amount_uzs) as total
                        FROM {selected_table}
                        WHERE mcc > 0
                        GROUP BY mcc_code
                        ORDER BY count DESC
                        LIMIT 10
                    '''))
                else:
                    # Для простой таблицы конвертируем
                    mcc_data = pd.DataFrame(client.execute(f'''
                        SELECT 
                            CAST(toUInt16OrNull(mcc) AS String) as mcc_code,
                            count() as count,
                            sum(toFloat64OrNull(amount_uzs)) as total
                        FROM {selected_table}
                        WHERE toUInt16OrNull(mcc) > 0
                        GROUP BY mcc_code
                        ORDER BY count DESC
                        LIMIT 10
                    '''))

                if not mcc_data.empty:
                    mcc_data.columns = ['category', 'count', 'total']
                    # Добавляем описания MCC кодов
                    mcc_descriptions = {
                        '5999': 'Розничные магазины',
                        '6011': 'ATM снятие',
                        '7372': 'Комп. услуги',
                        '5411': 'Продукты',
                        '6012': 'Фин. институты',
                        '4814': 'Телеком',
                        '5541': 'АЗС',
                        '5812': 'Рестораны'
                    }
                    mcc_data['category'] = mcc_data['category'].apply(
                        lambda x: f"{mcc_descriptions.get(x, f'MCC {x}')}"
                    )

                    fig = px.pie(
                        mcc_data,
                        values='count',
                        names='category',
                        title='🛍️ Топ категорий по количеству',
                        hole=0.4
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Почасовая активность
            st.subheader("⏰ Активность по часам")

            if selected_table == 'transactions_optimized':
                # Для оптимизированной таблицы - прямой запрос
                hourly_data = pd.DataFrame(client.execute(f'''
                    SELECT 
                        hour_num as hour,
                        count() as count,
                        avg(amount_uzs) as avg_amount
                    FROM {selected_table}
                    WHERE hour_num < 24
                    GROUP BY hour
                    ORDER BY hour
                '''))
            else:
                # Для простой таблицы - с конвертацией
                hourly_data = pd.DataFrame(client.execute(f'''
                    SELECT 
                        toUInt8OrNull(hour_num) as hour,
                        count() as count,
                        avg(toFloat64OrNull(amount_uzs)) as avg_amount
                    FROM {selected_table}
                    WHERE toUInt8OrNull(hour_num) IS NOT NULL AND toUInt8OrNull(hour_num) < 24
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
                    yaxis2=dict(title='Средний чек (UZS)', overlaying='y', side='right'),
                    hovermode='x unified',
                    height=400
                )

                st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("📊 Нет данных о карточных транзакциях. Загрузите данные через скрипты Python.")

    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")

with tab2:
    st.header("🔍 Углубленная аналитика")

    # Получаем выбранную таблицу
    selected_table = st.session_state.get('selected_table', 'transactions_simple')
    st.caption(f"📊 Источник данных: **{selected_table}**")

    analysis_type = st.radio(
        "Тип анализа",
        ["Анализ по банкам", "Анализ по мерчантам", "Временной анализ", "Когортный анализ"],
        horizontal=True
    )

    if analysis_type == "Анализ по банкам":
        # Топ банков
        if selected_table == 'transactions_optimized':
            bank_data = pd.DataFrame(client.execute(f'''
                SELECT 
                    emitent_bank,
                    count() as cnt,
                    sum(amount_uzs) as volume,
                    avg(amount_uzs) as avg_amount
                FROM {selected_table}
                WHERE emitent_bank != ''
                GROUP BY emitent_bank
                ORDER BY cnt DESC
                LIMIT 15
            '''))
        else:
            bank_data = pd.DataFrame(client.execute(f'''
                SELECT 
                    emitent_bank,
                    count() as cnt,
                    sum(toFloat64OrNull(amount_uzs)) as volume,
                    avg(toFloat64OrNull(amount_uzs)) as avg_amount
                FROM {selected_table}
                WHERE emitent_bank != ''
                GROUP BY emitent_bank
                ORDER BY cnt DESC
                LIMIT 15
            '''))

        if not bank_data.empty:
            bank_data.columns = ['bank', 'count', 'volume', 'avg_amount']

            fig = px.bar(
                bank_data,
                x='bank',
                y='count',
                title='Топ банков по количеству транзакций',
                labels={'count': 'Количество транзакций', 'bank': 'Банк'}
            )
            st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("🚨 Fraud Detection")

    # Получаем выбранную таблицу
    selected_table = st.session_state.get('selected_table', 'transactions_simple')
    st.caption(f"📊 Источник данных: **{selected_table}**")

    st.info("Модуль детекции мошенничества находится в разработке...")

    # Показываем базовую статистику
    st.subheader("Статистика по кодам ответа (respcode)")

    resp_stats = pd.DataFrame(client.execute(f'''
        SELECT 
            respcode,
            count() as cnt
        FROM {selected_table}
        WHERE respcode != ''
        GROUP BY respcode
        ORDER BY cnt DESC
        LIMIT 10
    '''))

    if not resp_stats.empty:
        resp_stats.columns = ['code', 'count']
        st.dataframe(resp_stats, use_container_width=True)

with tab4:
    st.header("📊 Анализ транзакций")

    # Получаем выбранную таблицу
    selected_table = st.session_state.get('selected_table', 'transactions_simple')
    st.caption(f"📊 Источник данных: **{selected_table}**")

    # Анализ существующих данных
    df = pd.DataFrame(client.execute(f'SELECT * FROM {selected_table} LIMIT 1000'))

    if not df.empty:
        columns = [col[0] for col in client.execute(f'DESCRIBE {selected_table}')]
        df.columns = columns[:len(df.columns)]

        # Получаем общее количество
        total_count = client.execute(f'SELECT count() FROM {selected_table}')[0][0]
        st.write(f"Показаны первые 1000 записей из {total_count:,}")

        # Добавим фильтры
        col1, col2, col3 = st.columns(3)
        with col1:
            bank_filter = st.selectbox(
                "Фильтр по банку",
                ['Все'] + df['emitent_bank'].unique().tolist()[:20]
            )
        with col2:
            mcc_filter = st.selectbox(
                "Фильтр по MCC",
                ['Все'] + df['mcc'].unique().tolist()[:20]
            )
        with col3:
            p2p_filter = st.selectbox(
                "Тип транзакции",
                ['Все', 'P2P', 'Покупки']
            )

        # Применяем фильтры
        filtered_df = df.copy()
        if bank_filter != 'Все':
            filtered_df = filtered_df[filtered_df['emitent_bank'] == bank_filter]
        if mcc_filter != 'Все':
            filtered_df = filtered_df[filtered_df['mcc'] == mcc_filter]
        if p2p_filter == 'P2P':
            filtered_df = filtered_df[filtered_df['p2p_flag'].isin(['True', '1'])]
        elif p2p_filter == 'Покупки':
            filtered_df = filtered_df[~filtered_df['p2p_flag'].isin(['True', '1'])]

        st.dataframe(filtered_df, use_container_width=True)

with tab5:
    st.header("📤 Статус данных")

    # Получаем выбранную таблицу
    selected_table = st.session_state.get('selected_table', 'transactions_simple')
    st.caption(f"📊 Анализ таблицы: **{selected_table}**")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Загруженные данные")

        # Показываем статистику загруженных данных
        try:
            if selected_table == 'transactions_optimized':
                stats = client.execute(f'''
                    SELECT 
                        count() as total,
                        uniq(hpan) as unique_cards,
                        uniq(pinfl) as unique_clients,
                        uniq(merchant_name) as unique_merchants,
                        uniq(emitent_bank) as unique_banks,
                        sum(amount_uzs) as total_volume
                    FROM {selected_table}
                ''')[0]
            else:
                stats = client.execute(f'''
                    SELECT 
                        count() as total,
                        uniq(hpan) as unique_cards,
                        uniq(pinfl) as unique_clients,
                        uniq(merchant_name) as unique_merchants,
                        uniq(emitent_bank) as unique_banks,
                        sum(toFloat64OrNull(amount_uzs)) as total_volume
                    FROM {selected_table}
                ''')[0]

            st.metric("Всего транзакций", f"{stats[0]:,}")
            st.metric("Уникальных карт", f"{stats[1]:,}")
            st.metric("Уникальных клиентов", f"{stats[2]:,}")
            st.metric("Уникальных мерчантов", f"{stats[3]:,}")
            st.metric("Банков-эмитентов", f"{stats[4]:,}")
            if stats[5]:
                st.metric("Общий объем", f"{stats[5]:,.0f} UZS")

        except Exception as e:
            st.error(f"Ошибка: {e}")

    with col2:
        st.subheader("📈 Качество данных")

        try:
            if selected_table == 'transactions_optimized':
                quality = client.execute(f'''
                    SELECT 
                        countIf(amount_uzs = 0) as empty_amounts,
                        countIf(mcc = 0) as empty_mcc,
                        countIf(merchant_name = '') as empty_merchant,
                        countIf(p2p_flag = 1) as p2p_count,
                        countIf(gender IN ('М', 'Ж')) as with_gender,
                        count() as total
                    FROM {selected_table}
                ''')[0]
            else:
                quality = client.execute(f'''
                    SELECT 
                        countIf(amount_uzs = '' OR amount_uzs = '0') as empty_amounts,
                        countIf(mcc = '' OR mcc = '0') as empty_mcc,
                        countIf(merchant_name = '') as empty_merchant,
                        countIf(p2p_flag IN ('True', '1')) as p2p_count,
                        countIf(gender IN ('М', 'Ж')) as with_gender,
                        count() as total
                    FROM {selected_table}
                ''')[0]

            total = quality[5]

            # Показываем процент заполненности
            st.progress(1 - quality[0] / total, text=f"Суммы заполнены: {(1 - quality[0] / total) * 100:.1f}%")
            st.progress(1 - quality[1] / total, text=f"MCC заполнены: {(1 - quality[1] / total) * 100:.1f}%")
            st.progress(1 - quality[2] / total, text=f"Мерчанты заполнены: {(1 - quality[2] / total) * 100:.1f}%")

            st.info(f"💸 P2P транзакций: {quality[3]:,} ({quality[3] / total * 100:.1f}%)")
            st.info(f"👥 С указанием пола: {quality[4]:,} ({quality[4] / total * 100:.1f}%)")

        except Exception as e:
            st.error(f"Ошибка: {e}")

        st.divider()

        # Топ регионов
        st.subheader("📍 Топ регионов")
        regions = pd.DataFrame(client.execute(f'''
            SELECT 
                emitent_region,
                count() as cnt
            FROM {selected_table}
            WHERE emitent_region != ''
            GROUP BY emitent_region
            ORDER BY cnt DESC
            LIMIT 5
        '''))

        if not regions.empty:
            regions.columns = ['region', 'count']
            for _, row in regions.iterrows():
                st.write(f"• {row['region']}: {row['count']:,}")

if __name__ == "__main__":
    st.write("Запустите через: streamlit run run_app.py")