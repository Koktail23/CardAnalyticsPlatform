#!/usr/bin/env python3
"""
Data Quality Validation с Great Expectations и Pandera
Создает подробный отчет о качестве данных
"""

import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema, Check
from clickhouse_driver import Client
import json
from datetime import datetime
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityValidator:
    """Валидация качества данных транзакций"""

    def __init__(self):
        self.client = Client(
            host='localhost',
            port=9000,
            user='analyst',
            password='admin123',
            database='card_analytics'
        )
        self.validation_results = {}

    def create_transaction_schema(self):
        """Создаем схему валидации для транзакций с Pandera"""

        schema = DataFrameSchema({
            'hpan': Column(str, nullable=False),
            'transaction_code': Column(str, nullable=False, unique=True),
            'amount_uzs': Column(float, Check.ge(0), nullable=False),
            'mcc': Column(int, Check.in_range(0, 9999), nullable=True),
            'p2p_flag': Column(int, Check.isin([0, 1])),
            'hour_num': Column(int, Check.in_range(0, 23)),
            'merchant_name': Column(str, nullable=True),
            'emitent_bank': Column(str, nullable=False),
            'age': Column(str, nullable=True),
            'gender': Column(str, Check.isin(['М', 'Ж', '']), nullable=True),
            'respcode': Column(str, nullable=True),
            'reversal_flag': Column(str, nullable=True)
        })

        return schema

    def run_custom_validation(self, df: pd.DataFrame):
        """Запуск кастомных проверок вместо Great Expectations"""

        logger.info("Запуск валидации данных...")

        validation_results = {
            'completeness': {},
            'consistency': {},
            'uniqueness': {},
            'validity': {},
            'accuracy': {}
        }

        # COMPLETENESS проверки
        logger.info("1. Проверка полноты данных (Completeness)...")

        critical_columns = ['hpan', 'transaction_code', 'amount_uzs', 'emitent_bank']
        for col in critical_columns:
            if col in df.columns:
                null_count = df[col].isna().sum()
                empty_count = (df[col] == '').sum() if df[col].dtype == 'object' else 0
                missing_count = null_count + empty_count
                missing_percent = (missing_count / len(df)) * 100

                validation_results['completeness'][col] = {
                    'success': missing_percent < 5,  # Считаем успешным если пропусков < 5%
                    'missing_count': missing_count,
                    'missing_percent': round(missing_percent, 2)
                }

        # CONSISTENCY проверки
        logger.info("2. Проверка консистентности (Consistency)...")

        # Даты должны быть корректными
        if 'issue_date' in df.columns and 'expire_date' in df.columns:
            try:
                issue_dates = pd.to_datetime(df['issue_date'], errors='coerce')
                expire_dates = pd.to_datetime(df['expire_date'], errors='coerce')
                valid_dates = (expire_dates > issue_dates).fillna(False)
                validation_results['consistency']['dates_order'] = valid_dates.mean() > 0.95
            except:
                validation_results['consistency']['dates_order'] = False

        # Суммы должны быть согласованы
        if all(col in df.columns for col in ['reqamt', 'conamt', 'amount_uzs']):
            try:
                reqamt = pd.to_numeric(df['reqamt'], errors='coerce').fillna(0)
                amount_uzs = pd.to_numeric(df['amount_uzs'], errors='coerce').fillna(0)
                amounts_consistent = (amount_uzs <= reqamt * 1.1) | (reqamt == 0)
                validation_results['consistency']['amounts'] = amounts_consistent.mean()
            except:
                validation_results['consistency']['amounts'] = 0

        # UNIQUENESS проверки
        logger.info("3. Проверка уникальности (Uniqueness)...")

        if 'transaction_code' in df.columns:
            duplicates = df['transaction_code'].duplicated().sum()
            validation_results['uniqueness']['transaction_code'] = {
                'success': duplicates == 0,
                'duplicates': duplicates,
                'duplicate_percent': round((duplicates / len(df)) * 100, 2)
            }

        # VALIDITY проверки
        logger.info("4. Проверка валидности (Validity)...")

        # MCC должны быть в правильном диапазоне
        if 'mcc' in df.columns:
            mcc_numeric = pd.to_numeric(df['mcc'], errors='coerce')
            valid_mcc = ((mcc_numeric >= 0) & (mcc_numeric <= 9999)).fillna(False)
            validation_results['validity']['mcc_range'] = valid_mcc.mean() > 0.95

        # Возраст должен быть разумным
        if 'age' in df.columns:
            age_numeric = pd.to_numeric(df['age'], errors='coerce')
            valid_age = ((age_numeric >= 18) & (age_numeric <= 100)).fillna(False).mean()
            validation_results['validity']['age_range'] = valid_age

        # Hour должен быть в диапазоне 0-23
        if 'hour_num' in df.columns:
            hour_numeric = pd.to_numeric(df['hour_num'], errors='coerce')
            valid_hours = ((hour_numeric >= 0) & (hour_numeric <= 23)).fillna(False).mean()
            validation_results['validity']['hour_range'] = valid_hours

        # ACCURACY проверки
        logger.info("5. Проверка точности (Accuracy)...")

        # Проверяем распределение сумм на выбросы
        if 'amount_uzs' in df.columns:
            amount_numeric = pd.to_numeric(df['amount_uzs'], errors='coerce').dropna()
            if len(amount_numeric) > 0:
                q1 = amount_numeric.quantile(0.25)
                q3 = amount_numeric.quantile(0.75)
                iqr = q3 - q1
                outliers = ((amount_numeric < (q1 - 3 * iqr)) |
                            (amount_numeric > (q3 + 3 * iqr))).sum()

                validation_results['accuracy']['outliers'] = {
                    'count': int(outliers),
                    'percent': round((outliers / len(amount_numeric)) * 100, 2)
                }

        # Проверка на отрицательные суммы
        if 'amount_uzs' in df.columns:
            amount_numeric = pd.to_numeric(df['amount_uzs'], errors='coerce')
            negative_amounts = (amount_numeric < 0).sum()
            validation_results['accuracy']['negative_amounts'] = {
                'count': int(negative_amounts),
                'percent': round((negative_amounts / len(df)) * 100, 2)
            }

        return validation_results

    def analyze_data_patterns(self, df: pd.DataFrame):
        """Анализ паттернов в данных"""

        patterns = {}

        # Анализ временных паттернов
        if 'hour_num' in df.columns:
            hour_numeric = pd.to_numeric(df['hour_num'], errors='coerce')
            if not hour_numeric.isna().all():
                peak_hours = hour_numeric.value_counts().head(3).index.tolist()
                patterns['peak_hours'] = [int(h) for h in peak_hours]

        # Анализ P2P
        if 'p2p_flag' in df.columns:
            p2p_numeric = pd.to_numeric(df['p2p_flag'], errors='coerce')
            patterns['p2p_ratio'] = round(p2p_numeric.mean() * 100, 2) if not p2p_numeric.isna().all() else 0

        # Анализ топ MCC
        if 'mcc' in df.columns:
            mcc_counts = df['mcc'].value_counts().head(5)
            patterns['top_mcc'] = {str(mcc): int(count) for mcc, count in mcc_counts.items()}

        return patterns

    def generate_quality_report(self, table_name: str = 'transactions_optimized'):
        """Генерация полного отчета качества данных"""

        logger.info(f"Генерация Data Quality Report для {table_name}...")

        # Проверяем существование таблицы
        try:
            table_check = self.client.execute(f"EXISTS TABLE {table_name}")
            if not table_check[0][0]:
                logger.error(f"Таблица {table_name} не существует")
                # Пробуем альтернативные таблицы
                alt_tables = ['transactions_simple', 'transactions_main', 'card_transactions']
                for alt_table in alt_tables:
                    if self.client.execute(f"EXISTS TABLE card_analytics.{alt_table}")[0][0]:
                        table_name = f"card_analytics.{alt_table}"
                        logger.info(f"Используем таблицу {table_name}")
                        break
                else:
                    logger.error("Не найдено ни одной таблицы с транзакциями")
                    return None
        except:
            table_name = f"card_analytics.{table_name}"

        # Загружаем sample данных
        try:
            df = pd.DataFrame(self.client.execute(f"""
                SELECT *
                FROM {table_name}
                LIMIT 10000
            """))
        except Exception as e:
            logger.error(f"Ошибка при загрузке данных: {e}")
            return None

        if df.empty:
            logger.error("Нет данных для анализа")
            return None

        # Получаем названия колонок
        columns = [col[0] for col in self.client.execute(f"DESCRIBE {table_name}")]
        df.columns = columns[:len(df.columns)]

        # 1. Запускаем кастомную валидацию
        validation_results = self.run_custom_validation(df)

        # 2. Статистика по таблице
        try:
            table_stats = self.client.execute(f"""
                SELECT 
                    count() as total_rows,
                    uniq(hpan) as unique_cards,
                    uniq(transaction_code) as unique_transactions
                FROM {table_name}
            """)[0]

            # Пробуем получить даты, если они есть
            date_range = "N/A"
            try:
                dates = self.client.execute(f"""
                    SELECT 
                        min(transaction_date) as min_date,
                        max(transaction_date) as max_date
                    FROM {table_name}
                """)[0]
                if dates[0] and dates[1]:
                    date_range = f"{dates[0]} to {dates[1]}"
            except:
                pass

        except Exception as e:
            logger.warning(f"Не удалось получить статистику: {e}")
            table_stats = [len(df), df['hpan'].nunique() if 'hpan' in df.columns else 0,
                           df['transaction_code'].nunique() if 'transaction_code' in df.columns else 0]
            date_range = "N/A"

        # 3. Анализ пропущенных значений
        missing_analysis = {}
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if df[col].dtype == 'object':
                missing_count += (df[col] == '').sum()
            missing_percent = (missing_count / len(df)) * 100
            missing_analysis[col] = {
                'missing_count': int(missing_count),
                'missing_percent': round(missing_percent, 2)
            }

        # 4. Анализ паттернов
        patterns = self.analyze_data_patterns(df)

        # 5. Создаем финальный отчет
        report = {
            'metadata': {
                'table_name': table_name,
                'report_date': datetime.now().isoformat(),
                'sample_size': len(df),
                'total_rows': int(table_stats[0]),
                'unique_cards': int(table_stats[1]),
                'unique_transactions': int(table_stats[2]),
                'date_range': date_range
            },
            'validation_results': validation_results,
            'missing_analysis': missing_analysis,
            'data_patterns': patterns,
            'quality_scores': self.calculate_quality_scores(validation_results, missing_analysis)
        }

        # Сохраняем отчет
        with open('data_quality_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("✅ Data Quality Report сохранен в data_quality_report.json")

        return report

    def calculate_quality_scores(self, validation_results, missing_analysis):
        """Расчет общих метрик качества"""

        # Completeness Score
        critical_cols = ['hpan', 'transaction_code', 'amount_uzs', 'emitent_bank']
        completeness_scores = []
        for col in critical_cols:
            if col in missing_analysis:
                score = max(0, 100 - missing_analysis[col]['missing_percent'])
                completeness_scores.append(score)

        completeness_score = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0

        # Consistency Score
        consistency_checks = validation_results.get('consistency', {})
        consistency_scores = []
        for k, v in consistency_checks.items():
            if isinstance(v, bool):
                consistency_scores.append(100 if v else 0)
            elif isinstance(v, (int, float)):
                consistency_scores.append(v * 100)

        consistency_score = sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0

        # Uniqueness Score
        uniqueness_checks = validation_results.get('uniqueness', {})
        uniqueness_score = 100
        if 'transaction_code' in uniqueness_checks:
            if uniqueness_checks['transaction_code'].get('success', False):
                uniqueness_score = 100
            else:
                dup_percent = uniqueness_checks['transaction_code'].get('duplicate_percent', 0)
                uniqueness_score = max(0, 100 - dup_percent)

        # Validity Score
        validity_checks = validation_results.get('validity', {})
        validity_scores = []
        for k, v in validity_checks.items():
            if isinstance(v, bool):
                validity_scores.append(100 if v else 0)
            elif isinstance(v, (int, float)):
                validity_scores.append(v * 100)

        validity_score = sum(validity_scores) / len(validity_scores) if validity_scores else 0

        # Accuracy Score
        accuracy_checks = validation_results.get('accuracy', {})
        accuracy_score = 100
        if 'outliers' in accuracy_checks:
            outlier_percent = accuracy_checks['outliers'].get('percent', 0)
            accuracy_score -= min(outlier_percent * 2, 30)  # Максимум -30% за выбросы
        if 'negative_amounts' in accuracy_checks:
            neg_percent = accuracy_checks['negative_amounts'].get('percent', 0)
            accuracy_score -= neg_percent * 5  # -5% за каждый процент отрицательных сумм

        # Overall Score
        overall_score = (completeness_score + consistency_score + uniqueness_score +
                         validity_score + accuracy_score) / 5

        return {
            'completeness_score': round(completeness_score, 2),
            'consistency_score': round(consistency_score, 2),
            'uniqueness_score': round(uniqueness_score, 2),
            'validity_score': round(validity_score, 2),
            'accuracy_score': round(accuracy_score, 2),
            'overall_score': round(overall_score, 2),
            'grade': self.get_quality_grade(overall_score)
        }

    def get_quality_grade(self, score):
        """Определение оценки качества"""
        if score >= 95:
            return 'A+ (Excellent)'
        elif score >= 90:
            return 'A (Very Good)'
        elif score >= 80:
            return 'B (Good)'
        elif score >= 70:
            return 'C (Acceptable)'
        elif score >= 60:
            return 'D (Poor)'
        else:
            return 'F (Critical)'

    def print_report_summary(self, report):
        """Вывод краткой сводки отчета"""

        print("\n" + "=" * 60)
        print("📊 DATA QUALITY REPORT SUMMARY")
        print("=" * 60)

        metadata = report['metadata']
        print(f"\n📋 Метаданные:")
        print(f"  • Таблица: {metadata['table_name']}")
        print(f"  • Всего записей: {metadata['total_rows']:,}")
        print(f"  • Период: {metadata['date_range']}")
        print(f"  • Уникальных карт: {metadata['unique_cards']:,}")
        print(f"  • Уникальных транзакций: {metadata['unique_transactions']:,}")

        scores = report['quality_scores']
        print(f"\n📈 Оценки качества:")
        print(f"  • Полнота (Completeness): {scores['completeness_score']}%")
        print(f"  • Консистентность (Consistency): {scores['consistency_score']}%")
        print(f"  • Уникальность (Uniqueness): {scores['uniqueness_score']}%")
        print(f"  • Валидность (Validity): {scores['validity_score']}%")
        print(f"  • Точность (Accuracy): {scores['accuracy_score']}%")
        print(f"\n  🎯 ОБЩАЯ ОЦЕНКА: {scores['overall_score']}% - {scores['grade']}")

        # Паттерны данных
        if 'data_patterns' in report:
            patterns = report['data_patterns']
            print(f"\n📊 Паттерны данных:")
            if 'peak_hours' in patterns:
                print(f"  • Пиковые часы: {patterns['peak_hours']}")
            if 'p2p_ratio' in patterns:
                print(f"  • Доля P2P транзакций: {patterns['p2p_ratio']}%")
            if 'top_mcc' in patterns and patterns['top_mcc']:
                print(f"  • Топ MCC категории:")
                for mcc, count in list(patterns['top_mcc'].items())[:3]:
                    print(f"    - MCC {mcc}: {count:,} транзакций")

        # Топ проблемных полей
        print(f"\n⚠️ Топ-5 полей с пропусками:")
        missing = report['missing_analysis']
        sorted_missing = sorted(missing.items(), key=lambda x: x[1]['missing_percent'], reverse=True)[:5]
        for field, stats in sorted_missing:
            if stats['missing_percent'] > 0:
                print(f"  • {field}: {stats['missing_percent']}% пропущено ({stats['missing_count']:,} записей)")

        # Проблемы валидации
        validation = report['validation_results']
        print(f"\n🔍 Результаты валидации:")

        # Completeness
        completeness = validation.get('completeness', {})
        failed_completeness = [k for k, v in completeness.items() if not v.get('success', True)]
        if failed_completeness:
            print(f"  • Неполные поля: {', '.join(failed_completeness)}")

        # Uniqueness
        uniqueness = validation.get('uniqueness', {})
        if 'transaction_code' in uniqueness and uniqueness['transaction_code'].get('duplicates', 0) > 0:
            print(f"  • Дубликаты в transaction_code: {uniqueness['transaction_code']['duplicates']:,}")

        # Accuracy
        accuracy = validation.get('accuracy', {})
        if 'outliers' in accuracy and accuracy['outliers']['count'] > 0:
            print(f"  • Выбросы в суммах: {accuracy['outliers']['count']:,} ({accuracy['outliers']['percent']}%)")
        if 'negative_amounts' in accuracy and accuracy['negative_amounts']['count'] > 0:
            print(f"  • Отрицательные суммы: {accuracy['negative_amounts']['count']:,}")

        # Рекомендации
        print(f"\n💡 Рекомендации:")
        if scores['completeness_score'] < 90:
            print("  • Улучшить сбор данных для критических полей")
        if scores['consistency_score'] < 90:
            print("  • Добавить проверки консистентности на этапе загрузки")
        if scores['validity_score'] < 90:
            print("  • Внедрить валидацию диапазонов значений")
        if scores['accuracy_score'] < 90:
            print("  • Проверить выбросы и аномальные значения")
        if scores['overall_score'] >= 90:
            print("  • Качество данных на высоком уровне! Поддерживайте текущий уровень.")

        print("\n" + "=" * 60)


def main():
    """Запуск валидации качества данных"""

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--table", default="transactions_grade_a")
    args = parser.parse_args()

    validator = DataQualityValidator()

    # Генерируем отчет
    report = validator.generate_quality_report(args.table)

    if report:
        # Выводим сводку
        validator.print_report_summary(report)

        print("\n✅ Полный отчет сохранен в:")
        print("  • data_quality_report.json")

        print("\n🎯 Следующие шаги:")
        print("  1. Изучите детальный отчет")
        print("  2. Исправьте критические проблемы")
        print("  3. Настройте автоматическую валидацию")
        print("  4. Интегрируйте в ETL pipeline")


if __name__ == "__main__":
    main()