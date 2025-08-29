#!/usr/bin/env python3
"""
Feature Engineering для карточных транзакций
Создание признаков для ML моделей
"""

from clickhouse_driver import Client
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Класс для создания признаков из транзакционных данных"""

    def __init__(self):
        self.client = Client(
            host='localhost',
            port=9000,
            user='analyst',
            password='admin123',
            database='card_analytics'
        )

    def create_feature_table(self):
        """Создает таблицу для хранения признаков"""

        logger.info("Создание таблицы признаков...")

        # Удаляем старую таблицу если есть
        self.client.execute("DROP TABLE IF EXISTS card_features")

        # Создаем новую таблицу признаков
        create_sql = """
        CREATE TABLE card_features
        (
            -- Идентификаторы
            hpan String,
            calculation_date Date,

            -- Транзакционные признаки (7, 30, 90 дней)
            txn_count_7d UInt32,
            txn_count_30d UInt32,
            txn_count_90d UInt32,

            txn_amount_7d Float64,
            txn_amount_30d Float64,
            txn_amount_90d Float64,

            avg_txn_amount_7d Float64,
            avg_txn_amount_30d Float64,
            avg_txn_amount_90d Float64,

            -- P2P признаки
            p2p_count_30d UInt32,
            p2p_amount_30d Float64,
            p2p_ratio_30d Float32,

            -- MCC разнообразие
            unique_mcc_7d UInt16,
            unique_mcc_30d UInt16,
            top_mcc UInt16,
            top_mcc_ratio Float32,

            -- Merchant признаки
            unique_merchants_30d UInt32,
            top_merchant String,
            top_merchant_ratio Float32,

            -- Временные паттерны
            avg_hour UInt8,
            weekend_ratio Float32,
            night_txn_ratio Float32,

            -- Velocity признаки
            max_daily_txn_count UInt32,
            max_daily_amount Float64,
            days_since_last_txn UInt16,

            -- Риск признаки
            failed_txn_count UInt32,
            reversal_count UInt32,
            international_txn_count UInt32,

            -- Сегментация
            customer_segment String,
            activity_level String,

            created_at DateTime DEFAULT now()
        )
        ENGINE = MergeTree()
        ORDER BY (hpan, calculation_date)
        PARTITION BY toYYYYMM(calculation_date)
        """

        self.client.execute(create_sql)
        logger.info("✅ Таблица card_features создана")

    def calculate_transaction_features(self) -> pd.DataFrame:
        """Рассчитывает транзакционные признаки"""

        logger.info("Расчет транзакционных признаков...")

        # Сначала проверим, какие даты есть в таблице
        date_check = self.client.execute("""
            SELECT 
                min(transaction_date) as min_date,
                max(transaction_date) as max_date,
                count() as total
            FROM transactions_optimized
        """)[0]

        logger.info(f"Период данных: {date_check[0]} - {date_check[1]}, всего записей: {date_check[2]}")

        # Используем максимальную дату из данных
        max_date = date_check[1]

        # Если дата некорректная или слишком старая/новая
        if max_date is None or max_date.year < 2020 or max_date.year > 2030:
            # Используем фиксированную дату на основе известных данных
            max_date = datetime(2025, 4, 13).date()
            logger.warning(f"Используется фиксированная дата: {max_date}")

        logger.info(f"Базовая дата для расчетов: {max_date}")

        # Рассчитываем признаки для каждой карты
        features_query = f"""
        SELECT
            hpan,
            toDate('{max_date}') as calculation_date,

            -- Количество транзакций за периоды
            countIf(transaction_date >= toDate('{max_date}') - INTERVAL 7 DAY) as txn_count_7d,
            countIf(transaction_date >= toDate('{max_date}') - INTERVAL 30 DAY) as txn_count_30d,
            countIf(transaction_date >= toDate('{max_date}') - INTERVAL 90 DAY) as txn_count_90d,

            -- Суммы транзакций
            sumIf(amount_uzs, transaction_date >= toDate('{max_date}') - INTERVAL 7 DAY) as txn_amount_7d,
            sumIf(amount_uzs, transaction_date >= toDate('{max_date}') - INTERVAL 30 DAY) as txn_amount_30d,
            sumIf(amount_uzs, transaction_date >= toDate('{max_date}') - INTERVAL 90 DAY) as txn_amount_90d,

            -- Средние суммы
            avgIf(amount_uzs, transaction_date >= toDate('{max_date}') - INTERVAL 7 DAY) as avg_txn_amount_7d,
            avgIf(amount_uzs, transaction_date >= toDate('{max_date}') - INTERVAL 30 DAY) as avg_txn_amount_30d,
            avgIf(amount_uzs, transaction_date >= toDate('{max_date}') - INTERVAL 90 DAY) as avg_txn_amount_90d,

            -- P2P метрики
            countIf(p2p_flag = 1 AND transaction_date >= toDate('{max_date}') - INTERVAL 30 DAY) as p2p_count_30d,
            sumIf(amount_uzs, p2p_flag = 1 AND transaction_date >= toDate('{max_date}') - INTERVAL 30 DAY) as p2p_amount_30d,

            -- MCC разнообразие
            uniqIf(mcc, transaction_date >= toDate('{max_date}') - INTERVAL 7 DAY) as unique_mcc_7d,
            uniqIf(mcc, transaction_date >= toDate('{max_date}') - INTERVAL 30 DAY) as unique_mcc_30d,

            -- Уникальные мерчанты
            uniqIf(merchant_name, transaction_date >= toDate('{max_date}') - INTERVAL 30 DAY) as unique_merchants_30d,

            -- Временные паттерны
            avg(hour_num) as avg_hour,
            countIf(toDayOfWeek(transaction_date) IN (6, 7)) / greatest(count(), 1) as weekend_ratio,
            countIf(hour_num >= 22 OR hour_num <= 6) / greatest(count(), 1) as night_txn_ratio,

            -- Velocity (упрощенный расчет)
            count() / greatest(uniq(transaction_date), 1) as avg_daily_txn_count,
            dateDiff('day', max(transaction_date), toDate('{max_date}')) as days_since_last_txn,

            -- Количество транзакций для сегментации
            count() as total_txn_count

        FROM transactions_optimized
        WHERE hpan != ''
            AND transaction_date <= toDate('{max_date}')
            AND transaction_date >= toDate('{max_date}') - INTERVAL 90 DAY
        GROUP BY hpan
        HAVING total_txn_count >= 5
        """

        result = self.client.execute(features_query)

        if not result:
            logger.error("Запрос не вернул данных! Проверьте даты в таблице transactions_optimized")
            logger.info("Пробуем альтернативный запрос без фильтрации по датам...")

            # Альтернативный запрос без фильтрации дат
            features_query = """
            SELECT
                hpan,
                today() as calculation_date,
                count() as txn_count_7d,
                count() as txn_count_30d,
                count() as txn_count_90d,
                sum(amount_uzs) as txn_amount_7d,
                sum(amount_uzs) as txn_amount_30d,
                sum(amount_uzs) as txn_amount_90d,
                avg(amount_uzs) as avg_txn_amount_7d,
                avg(amount_uzs) as avg_txn_amount_30d,
                avg(amount_uzs) as avg_txn_amount_90d,
                countIf(p2p_flag = 1) as p2p_count_30d,
                sumIf(amount_uzs, p2p_flag = 1) as p2p_amount_30d,
                uniq(mcc) as unique_mcc_7d,
                uniq(mcc) as unique_mcc_30d,
                uniq(merchant_name) as unique_merchants_30d,
                avg(hour_num) as avg_hour,
                countIf(toDayOfWeek(transaction_date) IN (6, 7)) / greatest(count(), 1) as weekend_ratio,
                countIf(hour_num >= 22 OR hour_num <= 6) / greatest(count(), 1) as night_txn_ratio,
                count() / 30 as avg_daily_txn_count,
                0 as days_since_last_txn,
                count() as total_txn_count
            FROM transactions_optimized
            WHERE hpan != ''
            GROUP BY hpan
            HAVING total_txn_count >= 5
            """

            result = self.client.execute(features_query)

            if not result:
                raise ValueError("Не удалось получить данные из таблицы transactions_optimized")

        df = pd.DataFrame(result)

        # Названия колонок
        columns = [
            'hpan', 'calculation_date',
            'txn_count_7d', 'txn_count_30d', 'txn_count_90d',
            'txn_amount_7d', 'txn_amount_30d', 'txn_amount_90d',
            'avg_txn_amount_7d', 'avg_txn_amount_30d', 'avg_txn_amount_90d',
            'p2p_count_30d', 'p2p_amount_30d',
            'unique_mcc_7d', 'unique_mcc_30d',
            'unique_merchants_30d',
            'avg_hour', 'weekend_ratio', 'night_txn_ratio',
            'avg_daily_txn_count', 'days_since_last_txn',
            'total_txn_count'
        ]
        df.columns = columns

        # Рассчитываем производные признаки
        df['p2p_ratio_30d'] = df['p2p_count_30d'] / df['txn_count_30d'].replace(0, 1)
        df['max_daily_txn_count'] = df['avg_daily_txn_count'].round().astype(int)  # Упрощенный расчет

        logger.info(f"✅ Рассчитано признаков для {len(df)} карт")

        return df

    def calculate_mcc_preferences(self, df: pd.DataFrame) -> pd.DataFrame:
        """Определяет предпочтения по MCC категориям"""

        logger.info("Анализ предпочтений MCC...")

        # Топ MCC для каждой карты
        top_mcc_query = """
        WITH mcc_stats AS (
            SELECT 
                hpan,
                mcc,
                count() as mcc_count,
                row_number() OVER (PARTITION BY hpan ORDER BY mcc_count DESC) as rn
            FROM transactions_optimized
            WHERE mcc > 0
            GROUP BY hpan, mcc
        )
        SELECT 
            hpan,
            mcc as top_mcc,
            mcc_count,
            mcc_count / sum(mcc_count) OVER (PARTITION BY hpan) as mcc_ratio
        FROM mcc_stats
        WHERE rn = 1
        """

        mcc_df = pd.DataFrame(self.client.execute(top_mcc_query))
        if not mcc_df.empty:
            mcc_df.columns = ['hpan', 'top_mcc', 'mcc_count', 'top_mcc_ratio']
            df = df.merge(mcc_df[['hpan', 'top_mcc', 'top_mcc_ratio']], on='hpan', how='left')

        return df

    def segment_customers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Сегментация клиентов на основе поведения"""

        logger.info("Сегментация клиентов...")

        # RFM-подобная сегментация
        df['recency_score'] = pd.cut(df['days_since_last_txn'],
                                     bins=[0, 7, 30, 90, float('inf')],
                                     labels=['Active', 'Regular', 'Sleeping', 'Lost'])

        df['frequency_score'] = pd.cut(df['txn_count_30d'],
                                       bins=[0, 5, 15, 50, float('inf')],
                                       labels=['Low', 'Medium', 'High', 'VeryHigh'])

        df['monetary_score'] = pd.qcut(df['txn_amount_30d'],
                                       q=4,
                                       labels=['Low', 'Medium', 'High', 'Premium'])

        # Комбинированный сегмент
        def get_segment(row):
            if row['recency_score'] == 'Active' and row['monetary_score'] in ['High', 'Premium']:
                return 'Champions'
            elif row['recency_score'] == 'Active' and row['frequency_score'] in ['High', 'VeryHigh']:
                return 'Loyal'
            elif row['recency_score'] == 'Regular':
                return 'Potential'
            elif row['recency_score'] == 'Sleeping':
                return 'At Risk'
            else:
                return 'Lost'

        df['customer_segment'] = df.apply(get_segment, axis=1)

        # Уровень активности
        df['activity_level'] = df['frequency_score']

        logger.info(f"✅ Сегментация выполнена")

        return df

    def save_features_to_db(self, df: pd.DataFrame):
        """Сохраняет признаки в ClickHouse"""

        logger.info("Сохранение признаков в БД...")

        # Конвертируем категориальные колонки в строки
        for col in df.columns:
            if df[col].dtype.name == 'category':
                df[col] = df[col].astype(str)

        # Заполняем пропущенные значения
        numeric_columns = df.select_dtypes(include=['float64', 'int64', 'float32', 'int32']).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)

        string_columns = df.select_dtypes(include=['object', 'string']).columns
        df[string_columns] = df[string_columns].fillna('')

        # Приводим типы данных к нужным для ClickHouse
        df['avg_hour'] = df['avg_hour'].fillna(0).round().astype(int)
        df['max_daily_txn_count'] = df['max_daily_txn_count'].fillna(0).round().astype(int)
        df['days_since_last_txn'] = df['days_since_last_txn'].fillna(0).round().astype(int)
        df['unique_mcc_7d'] = df['unique_mcc_7d'].fillna(0).astype(int)
        df['unique_mcc_30d'] = df['unique_mcc_30d'].fillna(0).astype(int)
        df['unique_merchants_30d'] = df['unique_merchants_30d'].fillna(0).astype(int)

        # Для UInt32 полей
        for col in ['txn_count_7d', 'txn_count_30d', 'txn_count_90d', 'p2p_count_30d']:
            if col in df.columns:
                df[col] = df[col].fillna(0).astype(int)

        # Для Float полей
        float_cols = ['txn_amount_7d', 'txn_amount_30d', 'txn_amount_90d',
                      'avg_txn_amount_7d', 'avg_txn_amount_30d', 'avg_txn_amount_90d',
                      'p2p_amount_30d', 'weekend_ratio', 'night_txn_ratio', 'p2p_ratio_30d']
        for col in float_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0).astype(float)

        # Убеждаемся что customer_segment и activity_level - строки
        df['customer_segment'] = df['customer_segment'].astype(str)
        df['activity_level'] = df['activity_level'].astype(str)

        # Выбираем нужные колонки
        columns_to_save = [
            'hpan', 'calculation_date',
            'txn_count_7d', 'txn_count_30d', 'txn_count_90d',
            'txn_amount_7d', 'txn_amount_30d', 'txn_amount_90d',
            'avg_txn_amount_7d', 'avg_txn_amount_30d', 'avg_txn_amount_90d',
            'p2p_count_30d', 'p2p_amount_30d', 'p2p_ratio_30d',
            'unique_mcc_7d', 'unique_mcc_30d',
            'unique_merchants_30d',
            'avg_hour', 'weekend_ratio', 'night_txn_ratio',
            'max_daily_txn_count', 'days_since_last_txn',
            'customer_segment', 'activity_level'
        ]

        # Добавляем недостающие колонки со значениями по умолчанию
        for col in columns_to_save:
            if col not in df.columns:
                if col in ['top_mcc', 'unique_mcc_7d', 'unique_mcc_30d']:
                    df[col] = 0
                elif col in ['top_mcc_ratio', 'weekend_ratio', 'night_txn_ratio', 'p2p_ratio_30d']:
                    df[col] = 0.0
                elif col in ['top_merchant', 'customer_segment', 'activity_level']:
                    df[col] = ''
                else:
                    df[col] = 0

        # Проверяем типы данных перед вставкой
        logger.info("Проверка типов данных перед вставкой:")
        for col in columns_to_save:
            dtype = df[col].dtype
            null_count = df[col].isna().sum()
            logger.debug(f"  {col}: {dtype}, nulls: {null_count}")

        # Вставка батчами
        batch_size = 1000
        total_rows = len(df)

        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i:i + batch_size]
            data = batch[columns_to_save].values.tolist()

            insert_query = f"""
            INSERT INTO card_features (
                hpan, calculation_date,
                txn_count_7d, txn_count_30d, txn_count_90d,
                txn_amount_7d, txn_amount_30d, txn_amount_90d,
                avg_txn_amount_7d, avg_txn_amount_30d, avg_txn_amount_90d,
                p2p_count_30d, p2p_amount_30d, p2p_ratio_30d,
                unique_mcc_7d, unique_mcc_30d,
                unique_merchants_30d,
                avg_hour, weekend_ratio, night_txn_ratio,
                max_daily_txn_count, days_since_last_txn,
                customer_segment, activity_level
            ) VALUES
            """

            try:
                self.client.execute(insert_query, data)

                if (i + batch_size) % 5000 == 0:
                    logger.info(f"  Загружено {min(i + batch_size, total_rows)}/{total_rows} записей")
            except Exception as e:
                logger.error(f"Ошибка при вставке батча {i // batch_size + 1}: {e}")
                logger.error(f"Первая строка батча для отладки:")
                for j, col in enumerate(columns_to_save):
                    if len(data) > 0:
                        logger.error(f"  {col}: {data[0][j]} (тип: {type(data[0][j]).__name__})")
                raise

        logger.info(f"✅ Сохранено {total_rows} записей с признаками")

    def generate_feature_report(self):
        """Генерирует отчет по созданным признакам"""

        logger.info("Генерация отчета...")

        # Статистика по признакам
        stats = self.client.execute("""
            SELECT
                count() as total_cards,
                avg(txn_count_30d) as avg_monthly_txn,
                avg(txn_amount_30d) as avg_monthly_amount,
                avg(p2p_ratio_30d) as avg_p2p_ratio,
                avg(unique_mcc_30d) as avg_unique_mcc,
                avg(unique_merchants_30d) as avg_unique_merchants
            FROM card_features
        """)[0]

        # Распределение по сегментам
        segments = self.client.execute("""
            SELECT
                customer_segment,
                count() as cnt,
                avg(txn_amount_30d) as avg_amount
            FROM card_features
            GROUP BY customer_segment
            ORDER BY cnt DESC
        """)

        print("\n" + "=" * 60)
        print("📊 ОТЧЕТ ПО FEATURE ENGINEERING")
        print("=" * 60)

        print(f"\n📈 Общая статистика:")
        print(f"  • Всего карт с признаками: {stats[0]:,}")
        print(f"  • Среднее транзакций в месяц: {stats[1]:.1f}")
        print(f"  • Средний месячный оборот: {stats[2]:,.0f} UZS")
        print(f"  • Средняя доля P2P: {stats[3]:.1%}")
        print(f"  • Среднее уникальных MCC: {stats[4]:.1f}")
        print(f"  • Среднее уникальных мерчантов: {stats[5]:.1f}")

        print(f"\n🎯 Сегментация клиентов:")
        for segment, cnt, avg_amount in segments:
            print(f"  • {segment}: {cnt:,} карт ({cnt / stats[0] * 100:.1f}%) - {avg_amount:,.0f} UZS/мес")

        print("\n" + "=" * 60)
        print("✅ Feature Engineering завершен!")
        print("=" * 60)

    def run_full_pipeline(self):
        """Запускает полный пайплайн создания признаков"""

        print("\n🚀 Запуск Feature Engineering Pipeline...")

        # 1. Создаем таблицу
        self.create_feature_table()

        # 2. Рассчитываем базовые признаки
        df = self.calculate_transaction_features()

        # 3. Добавляем MCC предпочтения
        df = self.calculate_mcc_preferences(df)

        # 4. Сегментируем клиентов
        df = self.segment_customers(df)

        # 5. Сохраняем в БД
        self.save_features_to_db(df)

        # 6. Генерируем отчет
        self.generate_feature_report()

        return df


def main():
    """Основная функция"""

    engineer = FeatureEngineer()

    # Запускаем полный процесс
    features_df = engineer.run_full_pipeline()

    # Сохраняем sample в CSV для анализа
    features_df.head(1000).to_csv('card_features_sample.csv', index=False)
    print("\n💾 Пример признаков сохранен в card_features_sample.csv")

    print("\n🎯 Следующие шаги:")
    print("  1. Используйте признаки для Fraud Detection модели")
    print("  2. Примените для прогнозирования объемов")
    print("  3. Создайте модель кластеризации клиентов")


if __name__ == "__main__":
    main()