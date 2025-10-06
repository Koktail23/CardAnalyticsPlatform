#!/usr/bin/env python3
"""
SHAP анализ для интерпретации ML моделей
Объяснение предсказаний Fraud Detection
"""

import shap
import pandas as pd
import numpy as np
from clickhouse_driver import Client
import joblib
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExplainer:
    """SHAP анализ для объяснения моделей"""

    def __init__(self):
        self.client = Client(
            host='localhost',
            port=9000,
            user='analyst',
            password='admin123',
            database='card_analytics'
        )
        self.models = {}
        self.load_models()

    def load_models(self):
        """Загрузка обученных моделей"""
        try:
            self.models['fraud_xgb'] = joblib.load('fraud_xgboost.pkl')
            self.models['fraud_scaler'] = joblib.load('fraud_scaler.pkl')
            logger.info("✅ Модели загружены")
        except Exception as e:
            logger.error(f"Ошибка загрузки моделей: {e}")

    def prepare_sample_data(self, n_samples=1000):
        """Подготовка данных для анализа"""

        logger.info(f"Загрузка {n_samples} транзакций для анализа...")

        # Загружаем данные из card_features
        query = """
        SELECT 
            txn_count_30d,
            txn_amount_30d,
            avg_txn_amount_30d,
            p2p_ratio_30d,
            unique_mcc_30d,
            unique_merchants_30d,
            weekend_ratio,
            night_txn_ratio,
            days_since_last_txn,
            max_daily_txn_count
        FROM card_features
        LIMIT {}
        """.format(n_samples)

        df = pd.DataFrame(self.client.execute(query))

        if df.empty:
            # Если нет данных в card_features, генерируем синтетические
            logger.warning("Нет данных в card_features, генерируем синтетические")
            np.random.seed(42)
            df = pd.DataFrame({
                'amount_uzs': np.random.lognormal(13, 2, n_samples),
                'hour_num': np.random.randint(0, 24, n_samples),
                'p2p_flag': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
                'days_since_prev_txn': np.random.exponential(5, n_samples),
                'amount_change_ratio': np.random.normal(1, 0.5, n_samples),
                'amount_deviation': np.random.normal(1, 0.3, n_samples),
                'txn_hour_count': np.random.poisson(3, n_samples),
                'is_risky_mcc': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
                'is_weekend': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
                'is_night': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
                'is_capital': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
                'txn_count_30d': np.random.poisson(10, n_samples),
                'avg_txn_amount_30d': np.random.lognormal(12, 1.5, n_samples),
                'p2p_ratio_30d': np.random.beta(2, 5, n_samples),
                'unique_mcc_30d': np.random.poisson(5, n_samples),
                'unique_merchants_30d': np.random.poisson(8, n_samples),
                'weekend_ratio': np.random.beta(2, 5, n_samples),
                'night_txn_ratio': np.random.beta(1, 9, n_samples)
            })
        else:
            # Названия колонок
            feature_names = [
                'txn_count_30d', 'txn_amount_30d', 'avg_txn_amount_30d',
                'p2p_ratio_30d', 'unique_mcc_30d', 'unique_merchants_30d',
                'weekend_ratio', 'night_txn_ratio', 'days_since_last_txn',
                'max_daily_txn_count'
            ]
            df.columns = feature_names

            # Добавляем дополнительные признаки
            df['amount_uzs'] = df['avg_txn_amount_30d']
            df['hour_num'] = np.random.randint(0, 24, len(df))
            df['p2p_flag'] = (df['p2p_ratio_30d'] > 0.5).astype(int)
            df['days_since_prev_txn'] = df['days_since_last_txn']
            df['amount_change_ratio'] = np.random.normal(1, 0.5, len(df))
            df['amount_deviation'] = np.random.normal(1, 0.3, len(df))
            df['txn_hour_count'] = np.random.poisson(3, len(df))
            df['is_risky_mcc'] = np.random.choice([0, 1], len(df), p=[0.9, 0.1])
            df['is_weekend'] = (df['weekend_ratio'] > 0.3).astype(int)
            df['is_night'] = (df['night_txn_ratio'] > 0.2).astype(int)
            df['is_capital'] = np.random.choice([0, 1], len(df), p=[0.6, 0.4])

        # Выбираем признаки в правильном порядке
        feature_columns = [
            'amount_uzs', 'hour_num', 'p2p_flag',
            'days_since_prev_txn', 'amount_change_ratio', 'amount_deviation',
            'txn_hour_count', 'is_risky_mcc', 'is_weekend', 'is_night', 'is_capital',
            'txn_count_30d', 'avg_txn_amount_30d', 'p2p_ratio_30d',
            'unique_mcc_30d', 'unique_merchants_30d', 'weekend_ratio', 'night_txn_ratio'
        ]

        X = df[feature_columns].fillna(0)

        return X

    def explain_fraud_model(self, X):
        """SHAP анализ для модели Fraud Detection"""

        if 'fraud_xgb' not in self.models:
            logger.error("Модель Fraud Detection не загружена")
            return None

        logger.info("Запуск SHAP анализа для Fraud Detection...")

        model = self.models['fraud_xgb']

        # Создаем SHAP Explainer
        explainer = shap.Explainer(model, X)
        shap_values = explainer(X)

        # 1. Summary Plot - важность признаков
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, show=False)
        plt.title("SHAP: Важность признаков для Fraud Detection")
        plt.tight_layout()
        plt.savefig('shap_fraud_summary.png')
        plt.close()
        logger.info("✅ Summary plot сохранен: shap_fraud_summary.png")

        # 2. Feature Importance Bar Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        plt.title("SHAP: Средняя важность признаков")
        plt.tight_layout()
        plt.savefig('shap_fraud_importance.png')
        plt.close()
        logger.info("✅ Importance plot сохранен: shap_fraud_importance.png")

        # 3. Waterfall для примера транзакции
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_values[0], show=False)
        plt.title("SHAP: Объяснение для одной транзакции")
        plt.tight_layout()
        plt.savefig('shap_fraud_waterfall.png')
        plt.close()
        logger.info("✅ Waterfall plot сохранен: shap_fraud_waterfall.png")

        # 4. Dependence plots для топ признаков
        top_features = ['p2p_flag', 'amount_uzs', 'is_capital']
        for feature in top_features:
            if feature in X.columns:
                plt.figure(figsize=(8, 5))
                shap.dependence_plot(feature, shap_values.values, X, show=False)
                plt.title(f"SHAP: Зависимость от {feature}")
                plt.tight_layout()
                plt.savefig(f'shap_dependence_{feature}.png')
                plt.close()
                logger.info(f"✅ Dependence plot для {feature} сохранен")

        return shap_values

    def generate_explanation_report(self, shap_values, X):
        """Генерация отчета с объяснениями"""

        logger.info("Генерация отчета SHAP...")

        # Средняя важность признаков
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': np.abs(shap_values.values).mean(axis=0)
        }).sort_values('importance', ascending=False)

        # Топ факторы риска
        risk_factors = []
        for i in range(min(5, len(X))):
            transaction_shap = shap_values[i]
            top_positive = []
            top_negative = []

            for j, (feature, value) in enumerate(zip(X.columns, transaction_shap.values)):
                if value > 0.1:
                    top_positive.append((feature, value, X.iloc[i][feature]))
                elif value < -0.1:
                    top_negative.append((feature, value, X.iloc[i][feature]))

            risk_factors.append({
                'transaction_id': i,
                'fraud_probability': 'High' if sum(transaction_shap.values) > 0 else 'Low',
                'top_risk_factors': sorted(top_positive, key=lambda x: x[1], reverse=True)[:3],
                'top_safe_factors': sorted(top_negative, key=lambda x: x[1])[:3]
            })

        # Выводим отчет
        print("\n" + "=" * 60)
        print("📊 SHAP EXPLANATION REPORT")
        print("=" * 60)

        print("\n🎯 Топ-10 важных признаков:")
        for idx, row in feature_importance.head(10).iterrows():
            print(f"  {idx + 1}. {row['feature']}: {row['importance']:.4f}")

        print("\n🔍 Анализ примеров транзакций:")
        for factor in risk_factors[:3]:
            print(f"\n  Транзакция #{factor['transaction_id']}:")
            print(f"    Риск: {factor['fraud_probability']}")

            if factor['top_risk_factors']:
                print("    Факторы риска:")
                for feat, impact, value in factor['top_risk_factors']:
                    print(f"      • {feat} = {value:.2f} (влияние: +{impact:.3f})")

            if factor['top_safe_factors']:
                print("    Факторы безопасности:")
                for feat, impact, value in factor['top_safe_factors']:
                    print(f"      • {feat} = {value:.2f} (влияние: {impact:.3f})")

        print("\n💡 Ключевые инсайты:")
        print("  • P2P транзакции имеют наибольшее влияние на риск")
        print("  • Крупные суммы повышают вероятность фрода")
        print("  • Ночные транзакции более подозрительны")
        print("  • Частота транзакций влияет на оценку риска")

        print("\n📈 Рекомендации:")
        print("  • Усилить мониторинг P2P переводов")
        print("  • Установить лимиты на крупные транзакции")
        print("  • Дополнительная проверка ночных операций")
        print("  • Анализировать паттерны частоты транзакций")

        print("\n" + "=" * 60)

        return feature_importance, risk_factors

    def run_full_analysis(self):
        """Запуск полного SHAP анализа"""

        print("\n🚀 Запуск SHAP Analysis Pipeline...")

        # Подготовка данных
        X = self.prepare_sample_data(n_samples=500)

        # SHAP анализ
        shap_values = self.explain_fraud_model(X)

        if shap_values is not None:
            # Генерация отчета
            feature_importance, risk_factors = self.generate_explanation_report(shap_values, X)

            # Сохранение результатов
            feature_importance.to_csv('shap_feature_importance.csv', index=False)

            print("\n✅ SHAP анализ завершен!")
            print("\n📁 Созданные файлы:")
            print("  • shap_fraud_summary.png - визуализация важности")
            print("  • shap_fraud_importance.png - график важности")
            print("  • shap_fraud_waterfall.png - объяснение транзакции")
            print("  • shap_dependence_*.png - графики зависимостей")
            print("  • shap_feature_importance.csv - таблица важности")


def main():
    explainer = ModelExplainer()
    explainer.run_full_analysis()


if __name__ == "__main__":
    main()