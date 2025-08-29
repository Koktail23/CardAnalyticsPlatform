# scripts/analyze_data.py
"""
Анализ реального датасета
Проверка структуры и качества данных перед загрузкой
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Добавляем корневую директорию в path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def analyze_csv(file_path: str = 'data_100k.csv'):
    """Детальный анализ CSV файла"""

    print("\n" + "=" * 70)
    print("📊 АНАЛИЗ ДАТАСЕТА")
    print("=" * 70)

    # Проверяем существование файла
    if not Path(file_path).exists():
        print(f"❌ Файл {file_path} не найден!")
        print("\nПопробуем найти другие файлы с данными...")

        # Ищем альтернативные файлы
        possible_files = ['output.csv', 'data/data_100k.csv', 'data/output.csv']
        for alt_file in possible_files:
            if Path(alt_file).exists():
                print(f"✅ Найден файл: {alt_file}")
                file_path = alt_file
                break
        else:
            print("❌ Файлы с данными не найдены")
            return None

    # Загружаем данные
    print(f"\n📁 Загрузка файла: {file_path}")
    df = pd.read_csv(file_path, low_memory=False)

    file_size_mb = Path(file_path).stat().st_size / 1024 / 1024

    print(f"📏 Размер: {df.shape[0]:,} строк × {df.shape[1]} колонок")
    print(f"💾 Размер файла: {file_size_mb:.2f} MB")
    print(f"💾 В памяти: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

    # Анализ колонок
    print("\n" + "-" * 70)
    print("📋 СТРУКТУРА ДАННЫХ:")
    print("-" * 70)

    # Группируем колонки по типам данных
    dtypes_summary = df.dtypes.value_counts()
    print("\nТипы данных:")
    for dtype, count in dtypes_summary.items():
        print(f"  • {dtype}: {count} колонок")

    # Список всех колонок
    print(f"\nВсего колонок: {len(df.columns)}")
    print("Колонки:", ", ".join(df.columns[:10]), "..." if len(df.columns) > 10 else "")

    # Анализ пропущенных значений
    print("\n" + "-" * 70)
    print("❗ ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ (топ-10):")
    print("-" * 70)

    null_counts = df.isnull().sum()
    null_percent = (null_counts / len(df)) * 100
    null_df = pd.DataFrame({
        'Колонка': null_counts.index,
        'Пропущено': null_counts.values,
        'Процент': null_percent.values
    })
    null_df = null_df[null_df['Пропущено'] > 0].sort_values('Процент', ascending=False).head(10)

    if not null_df.empty:
        for _, row in null_df.iterrows():
            print(f"  • {row['Колонка']}: {row['Пропущено']:,} ({row['Процент']:.1f}%)")
    else:
        print("  ✅ Пропущенных значений нет!")

    # Анализ ключевых полей
    print("\n" + "-" * 70)
    print("🔑 АНАЛИЗ КЛЮЧЕВЫХ ПОЛЕЙ:")
    print("-" * 70)

    # Транзакции
    if 'transaction_code' in df.columns:
        print(f"\n📝 transaction_code:")
        print(f"  • Уникальных: {df['transaction_code'].nunique():,}")
        print(f"  • Дубликатов: {df['transaction_code'].duplicated().sum():,}")
        if len(df) > 0:
            print(f"  • Пример: {df['transaction_code'].iloc[0]}")

    # Даты
    if 'rday' in df.columns:
        print(f"\n📅 rday (timestamp):")
        rday_numeric = pd.to_numeric(df['rday'], errors='coerce')
        if not rday_numeric.isna().all():
            print(f"  • Мин: {rday_numeric.min()}")
            print(f"  • Макс: {rday_numeric.max()}")
            # Попробуем конвертировать в дату
            try:
                min_date = pd.to_datetime(rday_numeric.min(), unit='s')
                max_date = pd.to_datetime(rday_numeric.max(), unit='s')
                days_range = (max_date - min_date).days
                print(f"  • Период: {min_date.date()} - {max_date.date()} ({days_range} дней)")
            except:
                pass

    # Суммы
    if 'amount_uzs' in df.columns:
        print(f"\n💰 amount_uzs:")
        amount_numeric = pd.to_numeric(df['amount_uzs'], errors='coerce')
        if not amount_numeric.isna().all():
            print(f"  • Мин: {amount_numeric.min():,.0f} UZS")
            print(f"  • Макс: {amount_numeric.max():,.0f} UZS")
            print(f"  • Средняя: {amount_numeric.mean():,.0f} UZS")
            print(f"  • Медиана: {amount_numeric.median():,.0f} UZS")
            print(f"  • Сумма: {amount_numeric.sum():,.0f} UZS")

    # MCC коды
    if 'mcc' in df.columns:
        print(f"\n🏪 MCC коды:")
        mcc_numeric = pd.to_numeric(df['mcc'], errors='coerce')
        if not mcc_numeric.isna().all():
            print(f"  • Уникальных: {mcc_numeric.nunique()}")
            print(f"  • Топ-5 категорий:")
            top_mcc = df['mcc'].value_counts().head(5)
            for mcc, count in top_mcc.items():
                percent = count / len(df) * 100
                print(f"    - MCC {mcc}: {count:,} транзакций ({percent:.1f}%)")

    # P2P
    if 'p2p_flag' in df.columns:
        print(f"\n💸 P2P переводы:")
        p2p_numeric = pd.to_numeric(df['p2p_flag'], errors='coerce')
        if not p2p_numeric.isna().all():
            p2p_count = p2p_numeric.sum()
            print(f"  • P2P транзакций: {p2p_count:,.0f} ({p2p_count / len(df) * 100:.1f}%)")
            print(f"  • Обычных транзакций: {len(df) - p2p_count:,.0f} ({(len(df) - p2p_count) / len(df) * 100:.1f}%)")

    # Банки
    if 'emitent_bank' in df.columns:
        print(f"\n🏦 Банки-эмитенты (топ-5):")
        top_banks = df['emitent_bank'].value_counts().head(5)
        for bank, count in top_banks.items():
            percent = count / len(df) * 100
            print(f"  • {bank}: {count:,} карт ({percent:.1f}%)")

    # Регионы
    if 'emitent_region' in df.columns:
        print(f"\n📍 Регионы (топ-5):")
        top_regions = df['emitent_region'].value_counts().head(5)
        for region, count in top_regions.items():
            percent = count / len(df) * 100
            print(f"  • {region}: {count:,} транзакций ({percent:.1f}%)")

    # Проверка уникальности
    print("\n" + "-" * 70)
    print("🔍 УНИКАЛЬНОСТЬ ДАННЫХ:")
    print("-" * 70)

    if 'hpan' in df.columns:
        unique_cards = df['hpan'].nunique()
        print(f"  • Уникальных карт (hpan): {unique_cards:,}")
        print(f"  • Среднее транзакций на карту: {len(df) / unique_cards:.1f}")

    if 'pinfl' in df.columns:
        unique_clients = df['pinfl'].nunique()
        print(f"  • Уникальных клиентов (pinfl): {unique_clients:,}")
        print(f"  • Среднее транзакций на клиента: {len(df) / unique_clients:.1f}")

    if 'merchant_name' in df.columns:
        unique_merchants = df['merchant_name'].nunique()
        print(f"  • Уникальных мерчантов: {unique_merchants:,}")
        print(f"  • Среднее транзакций на мерчанта: {len(df) / unique_merchants:.1f}")

    # Проблемы с данными
    print("\n" + "-" * 70)
    print("⚠️ ПОТЕНЦИАЛЬНЫЕ ПРОБЛЕМЫ:")
    print("-" * 70)

    issues = []

    # Проверка на полные дубликаты
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        issues.append(f"Найдено {duplicates:,} полных дубликатов строк")

    # Проверка дат
    date_columns = ['expire_date', 'issue_date']
    for date_col in date_columns:
        if date_col in df.columns:
            try:
                dates = pd.to_datetime(df[date_col], errors='coerce')
                invalid_dates = dates.isna().sum() - df[date_col].isna().sum()
                if invalid_dates > 0:
                    issues.append(f"Некорректных {date_col}: {invalid_dates:,}")
            except:
                issues.append(f"Проблемы с форматом {date_col}")

    if issues:
        for issue in issues:
            print(f"  ⚠️ {issue}")
    else:
        print("  ✅ Критических проблем не обнаружено")

    # Итоговая статистика
    print("\n" + "=" * 70)
    print("📈 ИТОГОВАЯ СТАТИСТИКА:")
    print("=" * 70)
    print(f"  • Всего транзакций: {len(df):,}")
    if 'amount_uzs' in df.columns:
        total_amount = pd.to_numeric(df['amount_uzs'], errors='coerce').sum()
        print(f"  • Общий объем: {total_amount:,.0f} UZS")
    if 'hpan' in df.columns:
        print(f"  • Уникальных карт: {df['hpan'].nunique():,}")
    if 'pinfl' in df.columns:
        print(f"  • Уникальных клиентов: {df['pinfl'].nunique():,}")
    print("\n" + "=" * 70)

    return df


def quick_info(file_path: str = 'data_100k.csv'):
    """Быстрая информация о файле"""

    if not Path(file_path).exists():
        # Ищем альтернативные файлы
        for alt_file in ['output.csv', 'data/data_100k.csv', 'data/output.csv']:
            if Path(alt_file).exists():
                file_path = alt_file
                break
        else:
            print("❌ Файлы с данными не найдены")
            return

    file_size_mb = Path(file_path).stat().st_size / 1024 / 1024

    # Подсчет строк без полной загрузки
    with open(file_path, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in f) - 1  # Минус заголовок

    print(f"\n📊 ФАЙЛ: {file_path}")
    print(f"   Размер: {file_size_mb:.2f} MB")
    print(f"   Строк: {line_count:,}")
    print(f"   Примерный размер строки: {file_size_mb * 1024 / line_count:.0f} KB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Анализ CSV данных')
    parser.add_argument('--file', type=str, default='data_100k.csv',
                        help='Путь к CSV файлу')
    parser.add_argument('--quick', action='store_true',
                        help='Быстрая информация без загрузки')

    args = parser.parse_args()

    if args.quick:
        quick_info(args.file)
    else:
        analyze_csv(args.file)