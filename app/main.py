"""
Главное приложение Streamlit с ClickHouse
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Добавляем путь к проекту
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from database.clickhouse_client import clickhouse
from analytics.queries import analytics
from etl.loader import DataLoader
from ai.claude_analyst import ClaudeAnalyst

# Конфигурация страницы
st.set_page_config(
    page_title="Card Analytics Platform - ClickHouse",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS стили
st.markdown("""
<style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .medium-font {
        font-size:20px !important;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("💳 Card Analytics Platform")
    st.markdown("**Powered by ClickHouse** - Processing millions of transactions in milliseconds")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")

        # Проверка соединения с ClickHouse
        try:
            version = clickhouse.execute("SELECT version()")[0][0]
            st.success(f"✅ ClickHouse {version}")

            # Статистика БД
            stats = clickhouse.execute("""
                SELECT 
                    count() as total_rows,
                    uniq(company_id) as companies,
                    uniq(card_masked) as cards
                FROM transactions
            """)[0]

            st.metric("Total Transactions", f"{stats[0]:,}")
            st.metric("Companies", stats[1])
            st.metric("Unique Cards", f"{stats[2]:,}")

        except Exception as e:
            st.error(f"❌ ClickHouse not connected: {e}")
            st.stop()

        st.divider()

        # Фильтры
        date_range = st.date_input(
            "📅 Date Range",
            value=(datetime.now() - timedelta(days=7), datetime.now()),
            max_value=datetime.now()
        )

        # Выбор компании
        companies = clickhouse.query_df("SELECT DISTINCT company_id FROM transactions ORDER BY company_id")
        selected_company = st.selectbox(
            "🏢 Company",
            options=['All'] + companies['company_id'].tolist()
        )

        # Refresh button
        if st.button("🔄 Refresh Data", type="primary", use_container_width=True):
            st.rerun()

    # Main content
    tabs = st.tabs(["📊 Dashboard", "🔍 Analytics", "🤖 AI Assistant", "📤 Data Upload"])

    with tabs[0]:
        show_dashboard(date_range, selected_company)

    with tabs[1]:
        show_analytics(date_range, selected_company)

    with tabs[2]:
        show_ai_assistant()

    with tabs[3]:
        show_data_upload()


def show_dashboard(date_range, company):
    """Dashboard с основными метриками"""

    # Получаем метрики
    company_filter = None if company == 'All' else company
    metrics_df = analytics.get_daily_metrics(
        company_id=company_filter,
        date_from=date_range[0],
        date_to=date_range[1]
    )

    if metrics_df.empty:
        st.warning("No data for selected period")
        return

    # Aggregate metrics
    total_metrics = metrics_df.agg({
        'transaction_count': 'sum',
        'total_volume': 'sum',
        'unique_cards': 'sum',
        'fraud_count': 'sum'
    })

    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📈 Transactions",
            f"{int(total_metrics['transaction_count']):,}",
            f"{metrics_df['transaction_count'].iloc[-1]:,} today"
        )

    with col2:
        st.metric(
            "💰 Total Volume",
            f"${total_metrics['total_volume']:,.2f}",
            f"${metrics_df['total_volume'].iloc[-1]:,.2f} today"
        )

    with col3:
        st.metric(
            "💳 Active Cards",
            f"{int(total_metrics['unique_cards']):,}",
            f"{metrics_df['unique_cards'].iloc[-1]:,} today"
        )

    with col4:
        fraud_rate = (total_metrics['fraud_count'] / total_metrics['transaction_count'] * 100) if total_metrics[
                                                                                                      'transaction_count'] > 0 else 0
        st.metric(
            "⚠️ Fraud Rate",
            f"{fraud_rate:.3f}%",
            f"{int(total_metrics['fraud_count'])} cases",
            delta_color="inverse"
        )

    st.divider()

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        # Transaction volume over time
        fig = px.area(
            metrics_df,
            x='transaction_date',
            y='total_volume',
            title='Daily Transaction Volume',
            labels={'total_volume': 'Volume ($)', 'transaction_date': 'Date'},
            color_discrete_sequence=['#00CC88']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Transaction count over time
        fig = px.line(
            metrics_df,
            x='transaction_date',
            y='transaction_count',
            title='Daily Transaction Count',
            labels={'transaction_count': 'Count', 'transaction_date': 'Date'},
            color_discrete_sequence=['#FF4B4B']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Real-time metrics
    st.subheader("⚡ Real-time Activity (Last 5 minutes)")

    realtime_df = analytics.get_realtime_metrics(window_minutes=5, company_id=company_filter)

    if not realtime_df.empty:
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=realtime_df['minute'],
            y=realtime_df['volume_per_minute'],
            name='Volume',
            yaxis='y',
            marker_color='lightblue'
        ))

        fig.add_trace(go.Scatter(
            x=realtime_df['minute'],
            y=realtime_df['transactions_per_minute'],
            name='Count',
            yaxis='y2',
            mode='lines+markers',
            marker_color='red'
        ))

        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Volume ($)", secondary_y=False)
        fig.update_yaxes(title_text="Transaction Count", secondary_y=True)

        fig.update_layout(
            title="Real-time Transaction Flow",
            hovermode='x unified',
            height=350
        )

        st.plotly_chart(fig, use_container_width=True)


def show_analytics(date_range, company):
    """Углубленная аналитика"""

    st.header("🔍 Deep Analytics")

    # Sub-tabs for different analyses
    analysis_type = st.selectbox(
        "Select Analysis Type",
        ["Anomaly Detection", "Fraud Analysis", "Merchant Analysis", "Company Comparison"]
    )

    if analysis_type == "Anomaly Detection":
        st.subheader("🚨 Anomaly Detection")

        company_filter = None if company == 'All' else company
        anomalies_df = analytics.detect_anomalies(company_id=company_filter)

        if not anomalies_df.empty:
            st.warning(f"Found {len(anomalies_df)} anomalies")

            # Show anomaly chart
            fig = px.scatter(
                anomalies_df,
                x='transaction_time',
                y='amount',
                color='anomaly_type',
                size='amount',
                hover_data=['merchant_name', 'card_masked'],
                title='Detected Anomalies'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Show anomaly table
            st.dataframe(
                anomalies_df[['transaction_id', 'amount', 'merchant_name', 'anomaly_type', 'z_score']],
                use_container_width=True
            )
        else:
            st.success("No anomalies detected")

    elif analysis_type == "Fraud Analysis":
        st.subheader("🔒 Fraud Analysis")

        fraud_data = analytics.get_fraud_analysis(date_from=date_range[0], date_to=date_range[1])

        # Fraud statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Fraud Rate", f"{fraud_data['stats']['fraud_rate']:.3f}%")
        with col2:
            st.metric("Fraud Volume", f"${fraud_data['stats']['fraud_volume']:,.2f}")
        with col3:
            st.metric("Avg Fraud Amount", f"${fraud_data['stats']['avg_fraud_amount']:.2f}")

        # Fraud by MCC
        if not fraud_data['top_fraud_mcc'].empty:
            fig = px.bar(
                fraud_data['top_fraud_mcc'],
                x='mcc_description',
                y='fraud_rate',
                title='Fraud Rate by Merchant Category',
                labels={'fraud_rate': 'Fraud Rate (%)', 'mcc_description': 'Category'}
            )
            st.plotly_chart(fig, use_container_width=True)

    elif analysis_type == "Merchant Analysis":
        st.subheader("🏪 Top Merchants Analysis")

        company_filter = None if company == 'All' else company
        merchants_df = analytics.get_merchant_analysis(top_n=20, company_id=company_filter)

        if not merchants_df.empty:
            fig = px.treemap(
                merchants_df,
                path=['category', 'merchant_name'],
                values='total_volume',
                title='Merchant Volume Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                merchants_df[['merchant_name', 'total_volume', 'transaction_count', 'unique_customers']],
                use_container_width=True
            )

    elif analysis_type == "Company Comparison":
        st.subheader("🏢 Company Comparison")

        comparison_df = analytics.get_company_comparison()

        if not comparison_df.empty:
            # Radar chart for comparison
            fig = go.Figure()

            for _, row in comparison_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[
                        row['total_transactions'] / comparison_df['total_transactions'].max() * 100,
                        row['total_volume'] / comparison_df['total_volume'].max() * 100,
                        row['unique_cards'] / comparison_df['unique_cards'].max() * 100,
                        row['unique_merchants'] / comparison_df['unique_merchants'].max() * 100,
                        100 - row['fraud_rate']  # Inverted for better visualization
                    ],
                    theta=['Transactions', 'Volume', 'Cards', 'Merchants', 'Security'],
                    fill='toself',
                    name=row['company_id']
                ))

            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="Company Performance Comparison"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(comparison_df, use_container_width=True)


def show_ai_assistant():
    """AI Assistant с Claude"""
    st.header("🤖 AI Analytics Assistant")

    st.info("Ask questions about your transaction data in natural language")

    # Chat interface
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Input
    if prompt := st.chat_input("Ask about your transaction data..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing..."):
                # Here you would integrate with Claude
                response = "This is where Claude's analysis would appear. Please configure Claude API."
                st.write(response)

        st.session_state.messages.append({"role": "assistant", "content": response})


def show_data_upload():
    """Интерфейс загрузки данных"""
    st.header("📤 Data Upload")

    loader = DataLoader()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload Files")

        company_id = st.text_input("Company ID", value="COMP01")

        uploaded_files = st.file_uploader(
            "Choose CSV or Parquet files",
            type=['csv', 'parquet', 'pq'],
            accept_multiple_files=True
        )

        if uploaded_files and st.button("Upload Data", type="primary"):
            progress = st.progress(0)

            for i, file in enumerate(uploaded_files):
                # Save temporary file
                temp_path = Path(f"/tmp/{file.name}")
                temp_path.write_bytes(file.read())

                # Load data
                if file.name.endswith('.csv'):
                    rows = loader.load_csv(temp_path, company_id)
                else:
                    rows = loader.load_parquet(temp_path, company_id)

                st.success(f"Loaded {rows:,} rows from {file.name}")
                progress.progress((i + 1) / len(uploaded_files))

            st.balloons()

    with col2:
        st.subheader("Batch Upload from Directory")

        directory = st.text_input("Directory Path", value=str(ROOT_DIR / "data" / "raw"))
        pattern = st.text_input("File Pattern", value="*.csv")

        if st.button("Load from Directory"):
            results = loader.load_directory(Path(directory), company_id, pattern)

            for filename, rows in results.items():
                if rows > 0:
                    st.success(f"✅ {filename}: {rows:,} rows")
                else:
                    st.error(f"❌ {filename}: Failed")


if __name__ == "__main__":
    main()