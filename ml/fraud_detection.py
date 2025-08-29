#!/usr/bin/env python3
"""
Fraud Detection Model для карточных транзакций
Использует признаки из card_features и строит модель детекции аномалий
"""

import pandas as pd
import numpy as np
from clickhouse_driver import Client
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FraudDetector:
    """Класс для построения модели детекции мошенничества"""

    def __init__(self):
        self.client = Client(
            host='localhost',
            port=9000,
            user='analyst',
            password='admin123',
            database='card_analytics'
        )
        self.scaler = StandardScaler()
        self.isolation_forest = None
        self.xgb_model = None

    def prepare_transaction_data(self) -> pd.DataFrame:
        """Подготовка данных транзакций с метками потенциального фрода"""

        logger.info("Загрузка транзакционных данных...")

        # Загружаем транзакции с признаками потенциального фрода
        query = """
        SELECT 
            hpan,
            transaction_code,
            transaction_date,
            hour_num,
            amount_uzs,
            mcc,
            p2p_flag,
            merchant_name,
            emitent_bank,
            emitent_region,
            respcode,
            reversal_flag,
            -- Создаем метки потенциального фрода на основе бизнес-правил
            CASE 
                WHEN respcode != '' AND respcode != '00' THEN 1  -- Неуспешные транзакции
                WHEN reversal_flag = '1' THEN 1  -- Отмененные транзакции
                WHEN amount_uzs > 50000000 THEN 1  -- Очень большие суммы
                WHEN hour_num BETWEEN 2 AND 5 THEN 1  -- Ночные транзакции
                ELSE 0
            END as potential_fraud
        FROM transactions_optimized
        WHERE amount_uzs > 0
        """

        df = pd.DataFrame(self.client.execute(query))
        columns = ['hpan', 'transaction_code', 'transaction_date', 'hour_num',
                   'amount_uzs', 'mcc', 'p2p_flag', 'merchant_name',
                   'emitent_bank', 'emitent_region', 'respcode',
                   'reversal_flag', 'potential_fraud']
        df.columns = columns

        logger.info(f"Загружено {len(df)} транзакций")
        logger.info(f"Потенциальный фрод: {df['potential_fraud'].sum()} ({df['potential_fraud'].mean() * 100:.2f}%)")

        return df

    def create_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание признаков для каждой транзакции"""

        logger.info("Создание признаков для транзакций...")

        # Velocity features - скорость транзакций
        df = df.sort_values(['hpan', 'transaction_date'])

        # Время с предыдущей транзакцией
        df['prev_txn_date'] = df.groupby('hpan')['transaction_date'].shift(1)
        df['days_since_prev_txn'] = (df['transaction_date'] - df['prev_txn_date']).dt.days.fillna(999)

        # Сумма предыдущей транзакции
        df['prev_amount'] = df.groupby('hpan')['amount_uzs'].shift(1)
        df['amount_change_ratio'] = (df['amount_uzs'] / df['prev_amount'].replace(0, 1)).fillna(1)

        # Статистика по карте за последние транзакции
        df['card_avg_amount'] = df.groupby('hpan')['amount_uzs'].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )
        df['amount_deviation'] = df['amount_uzs'] / df['card_avg_amount'].replace(0, 1)

        # Частота транзакций в час
        df['txn_hour_count'] = df.groupby(['hpan', 'transaction_date', 'hour_num']).cumcount() + 1

        # Признаки MCC
        df['is_risky_mcc'] = df['mcc'].isin([6010, 6011, 6012, 7995]).astype(int)  # Финансовые и игровые MCC

        # Временные признаки
        df['is_weekend'] = df['transaction_date'].dt.dayofweek.isin([5, 6]).astype(int)
        df['is_night'] = df['hour_num'].between(22, 6).astype(int)

        # Региональные признаки
        df['is_capital'] = (df['emitent_region'] == 'Tashkent City').astype(int)

        return df

    def load_card_features(self) -> pd.DataFrame:
        """Загрузка признаков карт из card_features"""

        logger.info("Загрузка признаков карт...")

        query = """
        SELECT 
            hpan,
            txn_count_30d,
            txn_amount_30d,
            avg_txn_amount_30d,
            p2p_ratio_30d,
            unique_mcc_30d,
            unique_merchants_30d,
            weekend_ratio,
            night_txn_ratio,
            days_since_last_txn,
            customer_segment
        FROM card_features
        """

        df = pd.DataFrame(self.client.execute(query))
        columns = ['hpan', 'txn_count_30d', 'txn_amount_30d', 'avg_txn_amount_30d',
                   'p2p_ratio_30d', 'unique_mcc_30d', 'unique_merchants_30d',
                   'weekend_ratio', 'night_txn_ratio', 'days_since_last_txn',
                   'customer_segment']
        df.columns = columns

        logger.info(f"Загружено признаков для {len(df)} карт")

        return df

    def prepare_final_dataset(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Подготовка финального датасета для обучения"""

        # Загружаем транзакции
        txn_df = self.prepare_transaction_data()
        txn_df = self.create_transaction_features(txn_df)

        # Загружаем признаки карт
        card_features = self.load_card_features()

        # Объединяем
        df = txn_df.merge(card_features, on='hpan', how='left')

        # Выбираем признаки для модели
        feature_columns = [
            'amount_uzs', 'hour_num', 'p2p_flag',
            'days_since_prev_txn', 'amount_change_ratio', 'amount_deviation',
            'txn_hour_count', 'is_risky_mcc', 'is_weekend', 'is_night', 'is_capital',
            'txn_count_30d', 'avg_txn_amount_30d', 'p2p_ratio_30d',
            'unique_mcc_30d', 'unique_merchants_30d', 'weekend_ratio', 'night_txn_ratio'
        ]

        # Заполняем пропуски
        df[feature_columns] = df[feature_columns].fillna(0)

        # Ограничиваем выбросы
        df['amount_uzs'] = np.clip(df['amount_uzs'], 0, df['amount_uzs'].quantile(0.99))
        df['amount_deviation'] = np.clip(df['amount_deviation'], 0, 10)

        X = df[feature_columns]
        y = df['potential_fraud']

        logger.info(f"Финальный датасет: {X.shape[0]} транзакций, {X.shape[1]} признаков")
        logger.info(f"Распределение классов: {y.value_counts().to_dict()}")

        return X, y

    def train_isolation_forest(self, X: pd.DataFrame) -> IsolationForest:
        """Обучение Isolation Forest для детекции аномалий"""

        logger.info("Обучение Isolation Forest...")

        # Нормализация данных
        X_scaled = self.scaler.fit_transform(X)

        # Обучение модели
        self.isolation_forest = IsolationForest(
            contamination=0.01,  # Ожидаемая доля аномалий
            random_state=42,
            n_estimators=100
        )

        self.isolation_forest.fit(X_scaled)

        # Предсказания
        predictions = self.isolation_forest.predict(X_scaled)
        anomalies = (predictions == -1).sum()

        logger.info(f"Isolation Forest обучен. Найдено аномалий: {anomalies} ({anomalies / len(X) * 100:.2f}%)")

        return self.isolation_forest

    def train_xgboost(self, X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
        """Обучение XGBoost для классификации"""

        logger.info("Обучение XGBoost...")

        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Балансировка классов через веса
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

        # Обучение модели
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=False
        )

        # Оценка модели
        y_pred = self.xgb_model.predict(X_test)
        y_pred_proba = self.xgb_model.predict_proba(X_test)[:, 1]

        # Метрики
        print("\n" + "=" * 60)
        print("МЕТРИКИ XGBOOST")
        print("=" * 60)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # ROC-AUC
        if len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"\nROC-AUC Score: {roc_auc:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"TN: {cm[0, 0]:,}  FP: {cm[0, 1]:,}")
        print(f"FN: {cm[1, 0]:,}  TP: {cm[1, 1]:,}")

        return self.xgb_model

    def plot_feature_importance(self, X: pd.DataFrame):
        """Визуализация важности признаков"""

        if self.xgb_model is None:
            logger.error("Модель не обучена")
            return

        # Важность признаков
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        plt.barh(importance['feature'], importance['importance'])
        plt.xlabel('Importance')
        plt.title('Top 15 Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()

        print("\n" + "=" * 60)
        print("ТОП-10 ВАЖНЫХ ПРИЗНАКОВ")
        print("=" * 60)
        for idx, row in importance.head(10).iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")

    def save_models(self):
        """Сохранение обученных моделей"""

        logger.info("Сохранение моделей...")

        # Сохраняем модели
        joblib.dump(self.scaler, 'fraud_scaler.pkl')
        joblib.dump(self.isolation_forest, 'fraud_isolation_forest.pkl')
        joblib.dump(self.xgb_model, 'fraud_xgboost.pkl')

        logger.info("Модели сохранены")

    def run_full_pipeline(self):
        """Запуск полного пайплайна обучения"""

        print("\n🚀 Запуск Fraud Detection Pipeline...")

        # Подготовка данных
        X, y = self.prepare_final_dataset()

        # Обучение моделей
        self.train_isolation_forest(X)
        self.train_xgboost(X, y)

        # Визуализация
        self.plot_feature_importance(X)

        # Сохранение
        self.save_models()

        print("\n" + "=" * 60)
        print("✅ FRAUD DETECTION PIPELINE ЗАВЕРШЕН!")
        print("=" * 60)


def main():
    detector = FraudDetector()
    detector.run_full_pipeline()

    print("\n🎯 Следующие шаги:")
    print("  1. Используйте модель для скоринга новых транзакций")
    print("  2. Настройте пороги для алертов")
    print("  3. Интегрируйте в real-time систему")


if __name__ == "__main__":
    main()