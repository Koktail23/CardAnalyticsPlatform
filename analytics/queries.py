"""
Оптимизированные аналитические запросы для ClickHouse
"""
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger

from database.clickhouse_client import clickhouse


class AnalyticsQueries:
    """Библиотека аналитических запросов"""

    def __init__(self):
        self.ch = clickhouse

    def get_daily_metrics(self,
                          company_id: Optional[str] = None,
                          date_from: Optional[datetime] = None,
                          date_to: Optional[datetime] = None) -> pd.DataFrame:
        """Получение дневных метрик"""

        conditions = []
        if company_id:
            conditions.append(f"company_id = '{company_id}'")
        if date_from:
            conditions.append(f"transaction_date >= '{date_from.date()}'")
        if date_to:
            conditions.append(f"transaction_date <= '{date_to.date()}'")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        query = f"""
        SELECT 
            transaction_date,
            company_id,
            count() as transaction_count,
            sum(amount) as total_volume,
            avg(amount) as avg_amount,
            median(amount) as median_amount,
            uniq(card_masked) as unique_cards,
            uniq(merchant_id) as unique_merchants,
            sumIf(1, is_fraud = 1) as fraud_count,
            sumIf(amount, is_fraud = 1) as fraud_volume,
            countIf(is_fraud = 1) / count() * 100 as fraud_rate,
            topK(5)(mcc_code) as top_mcc_codes,
            max(amount) as max_transaction,
            min(amount) as min_transaction
        FROM transactions
        {where_clause}
        GROUP BY transaction_date, company_id
        ORDER BY transaction_date DESC, company_id
        """

        return self.ch.query_df(query)

    def get_realtime_metrics(self,
                             window_minutes: int = 5,
                             company_id: Optional[str] = None) -> pd.DataFrame:
        """Получение метрик в реальном времени"""

        company_filter = f"AND company_id = '{company_id}'" if company_id else ""

        query = f"""
        WITH now() - INTERVAL {window_minutes} MINUTE as window_start
        SELECT 
            toStartOfMinute(transaction_time) as minute,
            count() as transactions_per_minute,
            sum(amount) as volume_per_minute,
            avg(amount) as avg_amount,
            uniq(card_masked) as unique_cards,
            sumIf(1, is_fraud = 1) as fraud_count,
            max(amount) as max_amount
        FROM transactions
        WHERE transaction_time >= window_start
        {company_filter}
        GROUP BY minute
        ORDER BY minute DESC
        """

        return self.ch.query_df(query)

    def detect_anomalies(self,
                         company_id: Optional[str] = None,
                         sensitivity: float = 3.0) -> pd.DataFrame:
        """Детекция аномалий используя статистические методы"""

        company_filter = f"AND company_id = '{company_id}'" if company_id else ""

        query = f"""
        WITH 
            (SELECT avg(amount) FROM transactions WHERE transaction_date >= today() - 30 {company_filter}) as avg_amount,
            (SELECT stddevPop(amount) FROM transactions WHERE transaction_date >= today() - 30 {company_filter}) as std_amount
        SELECT 
            transaction_id,
            company_id,
            card_masked,
            amount,
            merchant_name,
            mcc_code,
            transaction_time,
            'amount_anomaly' as anomaly_type,
            (amount - avg_amount) / std_amount as z_score
        FROM transactions
        WHERE 
            transaction_date = today()
            AND abs((amount - avg_amount) / std_amount) > {sensitivity}
            {company_filter}

        UNION ALL

        -- Velocity anomalies (слишком много транзакций с одной карты)
        SELECT 
            transaction_id,
            company_id,
            card_masked,
            amount,
            merchant_name,
            mcc_code,
            transaction_time,
            'velocity_anomaly' as anomaly_type,
            count_in_hour as z_score
        FROM (
            SELECT 
                *,
                countIf(transaction_time >= transaction_time - INTERVAL 1 HOUR) 
                    OVER (PARTITION BY card_masked ORDER BY transaction_time) as count_in_hour
            FROM transactions
            WHERE transaction_date = today()
            {company_filter}
        )
        WHERE count_in_hour > 10

        ORDER BY z_score DESC
        LIMIT 100
        """

        return self.ch.query_df(query)

    def get_fraud_analysis(self,
                           date_from: Optional[datetime] = None,
                           date_to: Optional[datetime] = None) -> Dict[str, Any]:
        """Комплексный анализ мошенничества"""

        if not date_from:
            date_from = datetime.now() - timedelta(days=30)
        if not date_to:
            date_to = datetime.now()

        # Общая статистика
        stats_query = f"""
        SELECT 
            count() as total_transactions,
            countIf(is_fraud = 1) as fraud_transactions,
            countIf(is_fraud = 1) / count() * 100 as fraud_rate,
            sumIf(amount, is_fraud = 1) as fraud_volume,
            avgIf(amount, is_fraud = 1) as avg_fraud_amount,
            avgIf(amount, is_fraud = 0) as avg_normal_amount
        FROM transactions
        WHERE transaction_date BETWEEN '{date_from.date()}' AND '{date_to.date()}'
        """

        stats = self.ch.query_df(stats_query).iloc[0].to_dict()

        # Топ MCC с мошенничеством
        mcc_query = f"""
        SELECT 
            mcc_code,
            mcc_description,
            count() as total_count,
            countIf(is_fraud = 1) as fraud_count,
            countIf(is_fraud = 1) / count() * 100 as fraud_rate
        FROM transactions
        WHERE transaction_date BETWEEN '{date_from.date()}' AND '{date_to.date()}'
        GROUP BY mcc_code, mcc_description
        HAVING fraud_count > 0
        ORDER BY fraud_rate DESC
        LIMIT 10
        """

        top_fraud_mcc = self.ch.query_df(mcc_query)

        # Временное распределение
        time_query = f"""
        SELECT 
            toHour(transaction_time) as hour,
            countIf(is_fraud = 1) as fraud_count,
            count() as total_count,
            countIf(is_fraud = 1) / count() * 100 as fraud_rate
        FROM transactions
        WHERE transaction_date BETWEEN '{date_from.date()}' AND '{date_to.date()}'
        GROUP BY hour
        ORDER BY hour
        """

        time_distribution = self.ch.query_df(time_query)

        return {
            'stats': stats,
            'top_fraud_mcc': top_fraud_mcc,
            'time_distribution': time_distribution
        }

    def get_company_comparison(self) -> pd.DataFrame:
        """Сравнение компаний холдинга"""

        query = """
        SELECT 
            company_id,
            count() as total_transactions,
            sum(amount) as total_volume,
            avg(amount) as avg_transaction,
            median(amount) as median_transaction,
            uniq(card_masked) as unique_cards,
            uniq(merchant_id) as unique_merchants,
            countIf(is_fraud = 1) / count() * 100 as fraud_rate,
            topK(1)(mcc_code)[1] as top_mcc,
            max(amount) as max_transaction
        FROM transactions
        WHERE transaction_date >= today() - 30
        GROUP BY company_id
        ORDER BY total_volume DESC
        """

        return self.ch.query_df(query)

    def get_merchant_analysis(self,
                              top_n: int = 20,
                              company_id: Optional[str] = None) -> pd.DataFrame:
        """Анализ топ мерчантов"""

        company_filter = f"WHERE company_id = '{company_id}'" if company_id else ""

        query = f"""
        SELECT 
            merchant_name,
            merchant_id,
            count() as transaction_count,
            sum(amount) as total_volume,
            avg(amount) as avg_amount,
            uniq(card_masked) as unique_customers,
            uniq(transaction_date) as active_days,
            countIf(is_fraud = 1) as fraud_count,
            any(mcc_code) as mcc_code,
            any(mcc_description) as category
        FROM transactions
        {company_filter}
        GROUP BY merchant_name, merchant_id
        ORDER BY total_volume DESC
        LIMIT {top_n}
        """

        return self.ch.query_df(query)

    def get_cohort_analysis(self,
                            cohort_field: str = 'card_masked',
                            metric: str = 'amount') -> pd.DataFrame:
        """Когортный анализ"""

        query = f"""
        WITH 
            (SELECT min(transaction_date) FROM transactions) as first_date
        SELECT 
            toStartOfMonth(min_date) as cohort_month,
            toStartOfMonth(transaction_date) as transaction_month,
            count(distinct {cohort_field}) as cohort_size,
            sum({metric}) as total_{metric},
            avg({metric}) as avg_{metric}
        FROM (
            SELECT 
                {cohort_field},
                transaction_date,
                {metric},
                min(transaction_date) OVER (PARTITION BY {cohort_field}) as min_date
            FROM transactions
        )
        GROUP BY cohort_month, transaction_month
        ORDER BY cohort_month, transaction_month
        """

        return self.ch.query_df(query)


# Экземпляр для использования
analytics = AnalyticsQueries()