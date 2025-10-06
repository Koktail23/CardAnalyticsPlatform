#!/usr/bin/env python3
"""
ML Dashboard - Интегрированный дашборд с результатами всех ML моделей
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from clickhouse_driver import Client
import joblib
import numpy as np
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="ML Analytics Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        margin: 10px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .success-metric {
        color: #00cc88;
        font-weight: bold;
    }
    .warning-metric {
        color: #ffcc00;
        font-weight: bold;
    }
    .danger-metric {
        color: #ff4444;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_client():
    """Подключение к ClickHouse"""
    return Client(
        host='localhost',
        port=9000,
        user='analyst',
        password='admin123',
        database='card_analytics'
    )


@st.cache_resource
def load_models():
    """Загрузка обученных моделей"""
    models = {}

    try:
        models['fraud_xgb'] = joblib.load('fraud_xgboost.pkl')
        models['fraud_scaler'] = joblib.load('fraud_scaler.pkl')
        st.sidebar.success("✅ Fraud Detection модель загружена")
    except:
        st.sidebar.warning("⚠️ Fraud Detection модель не найдена")

    try:
        models['prophet'] = joblib.load('prophet_main.pkl')
        st.sidebar.success("✅ Prophet модель загружена")
    except:
        st.sidebar.warning("⚠️ Prophet модель не найдена")

    try:
        models['kmeans'] = joblib.load('kmeans_model.pkl')
        models['segment_scaler'] = joblib.load('segment_scaler.pkl')
        st.sidebar.success("✅ Segmentation модель загружена")
    except:
        st.sidebar.warning("⚠️ Segmentation модель не найдена")

    return models


def fraud_detection_tab(client, models):
    """Вкладка Fraud Detection"""

    st.header("🚨 Fraud Detection Analytics")

    # Проверяем наличие модели
    if 'fraud_xgb' not in models:
        st.error("Модель Fraud Detection не загружена. Запустите: `python ml/fraud_detection.py`")
        return

    # Метрики модели
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "ROC-AUC Score",
            "0.876",
            "✅ Превышает целевой 0.85",
            delta_color="normal"
        )

    with col2:
        st.metric(
            "Precision (Fraud)",
            "67%",
            "Точность предсказаний"
        )

    with col3:
        st.metric(
            "Recall (Fraud)",
            "82%",
            "Полнота обнаружения"
        )

    with col4:
        st.metric(
            "F1-Score",
            "0.73",
            "Гармоническое среднее"
        )

    st.divider()

    # Анализ признаков
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🎯 Важность признаков")

        # Топ признаки из нашей модели
        feature_importance = pd.DataFrame({
            'feature': ['p2p_flag', 'amount_uzs', 'is_capital', 'txn_count_30d',
                        'hour_num', 'amount_deviation', 'amount_change_ratio'],
            'importance': [0.7267, 0.1564, 0.0557, 0.0156, 0.0110, 0.0088, 0.0042]
        })

        fig = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title='Топ-7 важных признаков для детекции фрода',
            labels={'importance': 'Важность', 'feature': 'Признак'},
            color='importance',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Инсайты
        st.info("""
        **Ключевые инсайты:**
        - 🔴 P2P транзакции - главный индикатор фрода (72.67%)
        - 💰 Сумма транзакции - второй по важности (15.64%)
        - 📍 Локация (столица) влияет на риск (5.57%)
        """)

    with col2:
        st.subheader("📊 Confusion Matrix")

        # Данные из результатов модели
        cm_data = pd.DataFrame({
            'Predicted Normal': [8950, 1423],
            'Predicted Fraud': [3221, 6406]
        }, index=['Actual Normal', 'Actual Fraud'])

        fig = px.imshow(
            cm_data.values,
            labels=dict(x="Предсказано", y="Фактически", color="Количество"),
            x=['Normal', 'Fraud'],
            y=['Normal', 'Fraud'],
            text_auto=True,
            color_continuous_scale='RdYlGn_r',
            title='Матрица ошибок (20,000 транзакций)'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Метрики по матрице
        tn, fp, fn, tp = 8950, 3221, 1423, 6406

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("True Positive Rate", f"{tp / (tp + fn) * 100:.1f}%")
            st.metric("False Positive Rate", f"{fp / (fp + tn) * 100:.1f}%")
        with col_b:
            st.metric("True Negative Rate", f"{tn / (tn + fp) * 100:.1f}%")
            st.metric("False Negative Rate", f"{fn / (fn + tp) * 100:.1f}%")

    # Распределение фрода
    st.divider()
    st.subheader("📈 Анализ фрода по категориям")

    # Загружаем данные из БД
    fraud_by_hour = client.execute("""
        SELECT 
            hour_num,
            countIf(respcode NOT IN ('', '00', '0')) as fraud_count,
            count() as total
        FROM transactions_optimized
        GROUP BY hour_num
        ORDER BY hour_num
    """)

    if fraud_by_hour:
        df_hour = pd.DataFrame(fraud_by_hour, columns=['hour', 'fraud', 'total'])
        df_hour['fraud_rate'] = (df_hour['fraud'] / df_hour['total'] * 100).round(2)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_hour['hour'],
            y=df_hour['total'],
            name='Всего транзакций',
            marker_color='lightblue',
            yaxis='y'
        ))
        fig.add_trace(go.Scatter(
            x=df_hour['hour'],
            y=df_hour['fraud_rate'],
            name='% Фрода',
            marker_color='red',
            yaxis='y2',
            mode='lines+markers',
            line=dict(width=3)
        ))

        fig.update_layout(
            title='Распределение фрода по часам',
            xaxis_title='Час',
            yaxis=dict(title='Количество транзакций', side='left'),
            yaxis2=dict(title='% Фрода', overlaying='y', side='right'),
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)


def segmentation_tab(client, models):
    """Вкладка Customer Segmentation"""

    st.header("👥 Customer Segmentation Analytics")

    # Проверяем наличие модели
    if 'kmeans' not in models:
        st.warning("Модель сегментации не загружена. Запустите: `python ml/customer_segmentation.py`")

    # Загружаем данные сегментов
    segments_data = client.execute("""
        SELECT 
            rfm_segment,
            count() as cnt,
            avg(txn_amount_30d) as avg_amount,
            avg(txn_count_30d) as avg_txn
        FROM customer_segments cs
        JOIN card_features cf ON cs.hpan = cf.hpan
        GROUP BY rfm_segment
        ORDER BY cnt DESC
    """)

    if segments_data:
        df_segments = pd.DataFrame(segments_data,
                                   columns=['segment', 'count', 'avg_amount', 'avg_txn'])

        # Метрики по сегментам
        col1, col2, col3, col4 = st.columns(4)

        total = df_segments['count'].sum()
        champions = df_segments[df_segments['segment'] == 'Champions']['count'].values
        champions_pct = (champions[0] / total * 100) if len(champions) > 0 else 0

        lost = df_segments[df_segments['segment'] == 'Lost']['count'].values
        lost_pct = (lost[0] / total * 100) if len(lost) > 0 else 0

        with col1:
            st.metric("Всего клиентов", f"{total:,}")

        with col2:
            st.metric("Champions", f"{champions_pct:.1f}%", "Лучшие клиенты")

        with col3:
            st.metric("At Risk", "11.2%", "Требуют внимания", delta_color="inverse")

        with col4:
            st.metric("Lost", f"{lost_pct:.1f}%", "Потерянные", delta_color="inverse")

        st.divider()

        # Визуализация сегментов
        col1, col2 = st.columns(2)

        with col1:
            # Pie chart сегментов
            fig = px.pie(
                df_segments,
                values='count',
                names='segment',
                title='Распределение клиентов по RFM сегментам',
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Bar chart по объемам
            fig = px.bar(
                df_segments,
                x='segment',
                y='avg_amount',
                title='Средний месячный объем по сегментам',
                labels={'avg_amount': 'Объем (UZS)', 'segment': 'Сегмент'},
                color='avg_amount',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

    # K-means кластеры
    st.divider()
    st.subheader("🔬 K-means Кластеризация")

    kmeans_data = client.execute("""
        SELECT 
            kmeans_cluster,
            count() as cnt,
            avg(p2p_ratio_30d) as p2p_ratio,
            avg(txn_amount_30d) as avg_amount
        FROM customer_segments cs
        JOIN card_features cf ON cs.hpan = cf.hpan
        GROUP BY kmeans_cluster
    """)

    if kmeans_data:
        df_kmeans = pd.DataFrame(kmeans_data,
                                 columns=['cluster', 'count', 'p2p_ratio', 'avg_amount'])

        # Даем названия кластерам
        cluster_names = {
            0: 'Стандарт',
            1: 'Премиум',
            2: 'P2P Активные'
        }
        df_kmeans['cluster_name'] = df_kmeans['cluster'].map(cluster_names)

        col1, col2, col3 = st.columns(3)

        for idx, row in df_kmeans.iterrows():
            with [col1, col2, col3][idx % 3]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{row['cluster_name']}</h3>
                    <p>Клиентов: {row['count']:,}</p>
                    <p>P2P доля: {row['p2p_ratio'] * 100:.1f}%</p>
                    <p>Ср. объем: {row['avg_amount']:,.0f} UZS</p>
                </div>
                """, unsafe_allow_html=True)

    # Стратегии для сегментов
    st.divider()
    st.subheader("💡 Рекомендуемые стратегии")

    strategies = {
        'Champions': '🏆 VIP программа, персональные предложения, удержание',
        'Loyal Customers': '💎 Программа лояльности, кросс-продажи',
        'Potential Loyalists': '🌱 Развитие отношений, увеличение частоты',
        'At Risk': '⚠️ Реактивация, специальные предложения, персональный контакт',
        'Hibernating': '😴 Win-back кампании, скидки на возвращение',
        'Lost': '❌ Агрессивные win-back или исключение из активных кампаний'
    }

    for segment, strategy in strategies.items():
        st.write(f"**{segment}**: {strategy}")


def forecasting_tab(client, models):
    """Вкладка Volume Forecasting"""

    st.header("📈 Volume Forecasting Analytics")

    # Проверяем наличие модели
    if 'prophet' not in models:
        st.error("Prophet модель не загружена. Запустите: `python ml/volume_forecasting.py`")
        return

    # Метрики прогноза
    st.warning("""
    ⚠️ **Внимание**: Модель показывает высокую погрешность (MAPE: 3022%) из-за недостатка исторических данных.
    Рекомендуется накопить минимум 365 дней данных для точного прогнозирования.
    """)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Дней данных", "101", "Минимум нужно 365", delta_color="inverse")

    with col2:
        st.metric("MAPE", "3022%", "Требует улучшения", delta_color="inverse")

    with col3:
        st.metric("Недельная сезонность", "✅", "Обнаружена")

    with col4:
        st.metric("Годовая сезонность", "⚠️", "Недостаточно данных")

    st.divider()

    # Исторические данные
    st.subheader("📊 Исторические объемы транзакций")

    historical_data = client.execute("""
        SELECT 
            transaction_date,
            sum(amount_uzs) as volume,
            count() as txn_count
        FROM transactions_optimized
        WHERE transaction_date IS NOT NULL
        GROUP BY transaction_date
        ORDER BY transaction_date
    """)

    if historical_data:
        df_hist = pd.DataFrame(historical_data,
                               columns=['date', 'volume', 'txn_count'])

        # График объемов
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_hist['date'],
            y=df_hist['volume'],
            mode='lines',
            name='Объем транзакций',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 100, 200, 0.2)'
        ))

        # Добавляем скользящее среднее
        df_hist['ma7'] = df_hist['volume'].rolling(window=7, min_periods=1).mean()

        fig.add_trace(go.Scatter(
            x=df_hist['date'],
            y=df_hist['ma7'],
            mode='lines',
            name='MA(7)',
            line=dict(color='red', width=2, dash='dash')
        ))

        fig.update_layout(
            title='Динамика объемов транзакций',
            xaxis_title='Дата',
            yaxis_title='Объем (UZS)',
            hovermode='x unified',
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        # Статистика
        col1, col2 = st.columns(2)

        with col1:
            st.info(f"""
            **Статистика объемов:**
            - Средний дневной объем: {df_hist['volume'].mean():,.0f} UZS
            - Максимум: {df_hist['volume'].max():,.0f} UZS
            - Минимум: {df_hist['volume'].min():,.0f} UZS
            - Волатильность: {df_hist['volume'].std() / df_hist['volume'].mean() * 100:.1f}%
            """)

        with col2:
            st.info(f"""
            **Статистика транзакций:**
            - Среднее в день: {df_hist['txn_count'].mean():.0f}
            - Максимум: {df_hist['txn_count'].max():,}
            - Минимум: {df_hist['txn_count'].min():,}
            - Всего дней: {len(df_hist)}
            """)

    # Рекомендации
    st.divider()
    st.subheader("💡 Рекомендации по улучшению прогнозирования")

    recommendations = [
        "📅 Накопить минимум 1 год исторических данных",
        "🎯 Добавить внешние факторы (праздники, события, промо-акции)",
        "📊 Использовать дополнительные регрессоры (курс валют, инфляция)",
        "🔄 Обновлять модель еженедельно с новыми данными",
        "📈 Разделить прогноз по типам транзакций (P2P, покупки)",
        "🏪 Строить отдельные модели для топ MCC категорий"
    ]

    for rec in recommendations:
        st.write(rec)


def main():
    """Главная функция дашборда"""

    # Заголовок
    st.title("🤖 Machine Learning Analytics Platform")
    st.markdown("**Комплексная аналитика на основе ML моделей**")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Настройки")

        # Загрузка моделей
        models = load_models()

        st.divider()

        # Статистика
        client = get_client()

        try:
            total_txn = client.execute("SELECT count() FROM transactions_optimized")[0][0]
            total_features = client.execute("SELECT count() FROM card_features")[0][0]
            total_segments = client.execute("SELECT count() FROM customer_segments")[0][0]

            st.metric("Транзакций", f"{total_txn:,}")
            st.metric("Признаков создано", f"{total_features:,}")
            st.metric("Клиентов сегментировано", f"{total_segments:,}")
        except:
            st.warning("БД недоступна")

        st.divider()

        # Информация
        st.info("""
        **ML Модели:**
        - Fraud Detection (XGBoost)
        - Customer Segmentation (K-means + RFM)
        - Volume Forecasting (Prophet)

        **Метрики:**
        - ROC-AUC: 0.876
        - 615 клиентов сегментировано
        - 101 день для прогнозов
        """)

    # Основные вкладки
    tab1, tab2, tab3, tab4 = st.tabs([
        "🚨 Fraud Detection",
        "👥 Customer Segmentation",
        "📈 Volume Forecasting",
        "📊 Общая статистика"
    ])

    with tab1:
        fraud_detection_tab(client, models)

    with tab2:
        segmentation_tab(client, models)

    with tab3:
        forecasting_tab(client, models)

    with tab4:
        st.header("📊 Общая статистика ML моделей")

        # Сводка по всем моделям
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🚨 Fraud Detection</h3>
                <p class="success-metric">ROC-AUC: 0.876 ✅</p>
                <p>Обработано: 100,000 транзакций</p>
                <p>Найдено фродов: 39,143</p>
                <p>Точность: 77%</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>👥 Segmentation</h3>
                <p class="success-metric">615 клиентов ✅</p>
                <p>RFM сегментов: 6</p>
                <p>K-means кластеров: 3</p>
                <p>Champions: 19.8%</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📈 Forecasting</h3>
                <p class="warning-metric">MAPE: 3022% ⚠️</p>
                <p>Дней данных: 101</p>
                <p>Прогноз: 30 дней</p>
                <p>Требует больше данных</p>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Выводы и рекомендации
        st.subheader("🎯 Ключевые выводы")

        st.success("""
        **Успешно реализовано:**
        1. ✅ Fraud Detection модель с ROC-AUC 0.876 (превышает целевой 0.85)
        2. ✅ Сегментация 615 клиентов на RFM и K-means группы
        3. ✅ Выявлен ключевой индикатор фрода - P2P транзакции (72.67% важности)
        4. ✅ Определены стратегии работы с каждым сегментом клиентов
        """)

        st.warning("""
        **Требует улучшения:**
        1. ⚠️ Volume Forecasting - накопить минимум 365 дней данных
        2. ⚠️ 24.7% клиентов в сегменте Lost - нужна программа возврата
        3. ⚠️ Высокий False Positive Rate (26.5%) в fraud detection
        """)

        st.info("""
        **Следующие шаги:**
        1. 📊 Внедрить real-time scoring для новых транзакций
        2. 🎯 Запустить таргетированные кампании по сегментам
        3. 📈 Обновлять модели еженедельно с новыми данными
        4. 🔄 A/B тестирование стратегий для разных сегментов
        """)


if __name__ == "__main__":
    main()