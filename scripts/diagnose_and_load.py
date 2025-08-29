# diagnose_and_load.py
"""
Диагностика структуры CSV и загрузка с правильными типами
"""

import pandas as pd
import numpy as np
from clickhouse_driver import Client
import sys
from pathlib import Path


def diagnose_csv_structure(csv_file='data_100k.csv'):
    """Анализ структуры CSV для создания правильной таблицы"""

    print("\n" + "=" * 60)
    print("ДИАГНОСТИКА СТРУКТУРЫ CSV")
    print("=" * 60)

    # Читаем первые строки для анализа
    df = pd.read_csv(csv_file, nrows=1000)
    print(f"\n📊 Анализ первых 1000 строк из {csv_file}")
    print(f"Найдено колонок: {len(df.columns)}")

    # Анализируем каждую колонку
    column_info = {}

    print("\n📋 АНАЛИЗ КОЛОНОК:")
    print("-" * 60)

    for col in df.columns:
        # Получаем информацию о колонке
        non_null_count = df[col].notna().sum()
        null_count = df[col].isna().sum()
        unique_count = df[col].nunique()

        # Определяем тип данных
        col_type = str(df[col].dtype)

        # Пробуем определить реальный тип
        sample_values = df[col].dropna().head(5).tolist()

        # Определяем рекомендуемый тип для ClickHouse
        ch_type = 'String'  # По умолчанию

        # Проверяем на числа
        try:
            numeric_vals = pd.to_numeric(df[col], errors='coerce')
            if numeric_vals.notna().sum() > non_null_count * 0.9:  # 90% числовые
                if (numeric_vals % 1 == 0).all():  # Целые числа
                    max_val = numeric_vals.max()
                    min_val = numeric_vals.min()

                    if min_val >= 0:
                        if max_val < 256:
                            ch_type = 'UInt8'
                        elif max_val < 65536:
                            ch_type = 'UInt16'
                        elif max_val < 4294967296:
                            ch_type = 'UInt32'
                        else:
                            ch_type = 'UInt64'
                    else:
                        if -128 <= min_val and max_val < 128:
                            ch_type = 'Int8'
                        elif -32768 <= min_val and max_val < 32768:
                            ch_type = 'Int16'
                        elif -2147483648 <= min_val and max_val < 2147483648:
                            ch_type = 'Int32'
                        else:
                            ch_type = 'Int64'
                else:  # Дробные числа
                    ch_type = 'Float64'

                # Проверяем на nullable
                if null_count > 0:
                    ch_type = f'Nullable({ch_type})'
        except:
            pass

        # Особые случаи
        if col in ['transaction_code', 'pinfl', 'terminal_id', 'ip', 'login']:
            ch_type = 'String'
        elif col in ['expire_date', 'issue_date']:
            ch_type = 'String'  # Будем хранить как строку из-за проблем с форматом
        elif col == 'hpan':
            ch_type = 'Float64'  # Хешированный PAN
        elif col in ['pinfl_flag', 'oked', 'respcode']:
            ch_type = 'Nullable(Float32)'

        column_info[col] = {
            'pandas_type': col_type,
            'clickhouse_type': ch_type,
            'nulls': null_count,
            'unique': unique_count,
            'samples': sample_values[:3]
        }

        if len(column_info) <= 10:  # Показываем первые 10 колонок
            print(f"\n{col}:")
            print(f"  Pandas тип: {col_type}")
            print(f"  ClickHouse тип: {ch_type}")
            print(f"  Nulls: {null_count}/{len(df)} ({null_count / len(df) * 100:.1f}%)")
            print(f"  Уникальных: {unique_count}")
            if sample_values:
                print(f"  Примеры: {sample_values[:3]}")

    if len(column_info) > 10:
        print(f"\n... и еще {len(column_info) - 10} колонок")

    return df, column_info


def create_table_from_analysis(client, column_info):
    """Создание таблицы на основе анализа"""

    print("\n" + "=" * 60)
    print("СОЗДАНИЕ ТАБЛИЦЫ")
    print("=" * 60)

    # Удаляем старую таблицу
    print("\n🗑️ Удаление старой таблицы...")
    try:
        client.execute('DROP TABLE IF EXISTS card_analytics.transactions_main')
        print("✅ Старая таблица удалена")
    except Exception as e:
        print(f"⚠️ {e}")

    # Формируем CREATE TABLE
    columns_sql = []
    for col_name, info in column_info.items():
        # Экранируем имя колонки
        safe_col_name = f"`{col_name}`"
        columns_sql.append(f"    {safe_col_name} {info['clickhouse_type']}")

    create_sql = f"""
    CREATE TABLE card_analytics.transactions_main
    (
{',\\n'.join(columns_sql)}
    )
    ENGINE = MergeTree()
    ORDER BY tuple()
    SETTINGS index_granularity = 8192
    """

    print("\n📝 Создание новой таблицы...")
    try:
        client.execute('CREATE DATABASE IF NOT EXISTS card_analytics')
        client.execute(create_sql)
        print("✅ Таблица создана успешно")
        return True
    except Exception as e:
        print(f"❌ Ошибка создания таблицы: {e}")
        return False


def load_data_smart(client, csv_file, column_info, batch_size=1000):
    """Умная загрузка с приведением типов"""

    print("\n" + "=" * 60)
    print("ЗАГРУЗКА ДАННЫХ")
    print("=" * 60)

    # Читаем весь CSV
    print(f"\n📊 Чтение {csv_file}...")
    df = pd.read_csv(csv_file, low_memory=False)
    print(f"✅ Прочитано {len(df):,} строк")

    # Приводим типы данных согласно анализу
    print("\n🔧 Приведение типов данных...")

    for col_name, info in column_info.items():
        if col_name not in df.columns:
            continue

        ch_type = info['clickhouse_type']

        try:
            if 'Int' in ch_type or 'UInt' in ch_type:
                # Целочисленные типы
                if 'Nullable' in ch_type:
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                else:
                    df[col_name] = pd.to_numeric(df[col_name], errors='coerce').fillna(0).astype('int64')

            elif 'Float' in ch_type:
                # Дробные типы
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
                if 'Nullable' not in ch_type:
                    df[col_name] = df[col_name].fillna(0.0)

            else:
                # Строковые типы
                df[col_name] = df[col_name].fillna('').astype(str)
                df[col_name] = df[col_name].replace('nan', '')

        except Exception as e:
            print(f"  ⚠️ Проблема с колонкой {col_name}: {e}")

    print("✅ Типы приведены")

    # Загрузка батчами
    print(f"\n📤 Загрузка батчами по {batch_size:,} записей...")

    total_loaded = 0
    failed_batches = []

    for i in range(0, len(df), batch_size):
        batch_num = i // batch_size + 1
        batch = df.iloc[i:i + batch_size]

        try:
            # Подготовка данных для вставки
            data = []
            for _, row in batch.iterrows():
                row_data = []
                for col_name in column_info.keys():
                    if col_name in row.index:
                        val = row[col_name]
                        # Заменяем pandas NaN на Python None для nullable полей
                        if pd.isna(val) and 'Nullable' in column_info[col_name]['clickhouse_type']:
                            row_data.append(None)
                        else:
                            row_data.append(val)
                    else:
                        row_data.append(None)
                data.append(tuple(row_data))

            # Вставка с проверкой типов
            client.execute(
                'INSERT INTO card_analytics.transactions_main VALUES',
                data,
                types_check=True  # Включаем проверку типов для диагностики
            )

            total_loaded += len(batch)

            # Показываем прогресс каждые 10 батчей
            if batch_num % 10 == 0 or batch_num == 1:
                print(f"  ✓ Батч {batch_num}: загружено {total_loaded:,} / {len(df):,}")

        except Exception as e:
            error_msg = str(e)[:200]
            print(f"  ✗ Батч {batch_num}: {error_msg}")
            failed_batches.append(batch_num)

            # Если много ошибок подряд, останавливаемся
            if len(failed_batches) > 5:
                print("\n❌ Слишком много ошибок, остановка загрузки")
                break

    print(f"\n{'=' * 60}")
    print(f"РЕЗУЛЬТАТ ЗАГРУЗКИ:")
    print(f"{'=' * 60}")
    print(f"✅ Успешно загружено: {total_loaded:,} записей")
    if failed_batches:
        print(f"❌ Проблемные батчи: {failed_batches[:10]}")

    return total_loaded


def main(csv_file='data_100k.csv'):
    """Основная функция"""

    if not Path(csv_file).exists():
        print(f"❌ Файл {csv_file} не найден!")
        return

    # Подключение к ClickHouse
    print("\n🔌 Подключение к ClickHouse...")
    try:
        client = Client(
            host='localhost',
            port=9000,
            user='analyst',
            password='admin123'
        )
        client.execute('SELECT 1')
        print("✅ Подключение успешно")
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        print("Запустите Docker: docker-compose up -d")
        return

    # 1. Анализ структуры CSV
    df_sample, column_info = diagnose_csv_structure(csv_file)

    # 2. Создание таблицы
    if not create_table_from_analysis(client, column_info):
        return

    # 3. Загрузка данных
    loaded = load_data_smart(client, csv_file, column_info)

    # 4. Проверка результата
    if loaded > 0:
        try:
            count = client.execute('SELECT count() FROM card_analytics.transactions_main')[0][0]
            print(f"\n✅ ПРОВЕРКА: В таблице {count:,} записей")

            # Простая статистика
            stats = client.execute('''
                SELECT 
                    count() as cnt,
                    count(DISTINCT hpan) as unique_cards,
                    avg(amount_uzs) as avg_amount
                FROM card_analytics.transactions_main
            ''')[0]

            print(f"\n📊 СТАТИСТИКА:")
            print(f"  • Транзакций: {stats[0]:,}")
            print(f"  • Уникальных карт: {stats[1]:,}")
            if stats[2]:
                print(f"  • Средняя сумма: {stats[2]:,.0f} UZS")

            print("\n🎉 ГОТОВО! Данные загружены успешно!")
            print("\nТеперь можно:")
            print("  1. Открыть ClickHouse UI: http://localhost:8123/play")
            print("  2. Запустить анализ: python check_data.py")
            print("  3. Запустить дашборд: streamlit run run_app.py")

        except Exception as e:
            print(f"\n⚠️ Ошибка при проверке: {e}")
    else:
        print("\n❌ Данные не загрузились. Проверьте ошибки выше.")


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'data_100k.csv'
    main(csv_file)