# clean_and_load.py
"""
Полная очистка и загрузка с нуля
"""

import pandas as pd
from clickhouse_driver import Client
import sys
from pathlib import Path


def clean_and_load(csv_file='data_100k.csv'):
    """Полная очистка БД и загрузка данных"""

    print("\n" + "=" * 60)
    print("ПОЛНАЯ ОЧИСТКА И ЗАГРУЗКА")
    print("=" * 60)

    if not Path(csv_file).exists():
        print(f"❌ Файл {csv_file} не найден!")
        return False

    # Подключение
    print("\n🔌 Подключение к ClickHouse...")
    try:
        client = Client(
            host='localhost',
            port=9000,
            user='analyst',
            password='admin123'
        )
        client.execute('SELECT 1')
        print("✅ Подключено")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

    # ПОЛНАЯ ОЧИСТКА
    print("\n🧹 ПОЛНАЯ ОЧИСТКА БД...")

    try:
        # Удаляем ВСЮ базу данных
        client.execute('DROP DATABASE IF EXISTS card_analytics')
        print("✅ БД card_analytics удалена")

        # Создаем новую чистую БД
        client.execute('CREATE DATABASE card_analytics')
        print("✅ БД card_analytics создана заново")

    except Exception as e:
        print(f"⚠️ {e}")

    # Читаем структуру CSV
    print(f"\n📊 Анализ {csv_file}...")
    df_header = pd.read_csv(csv_file, nrows=0)
    columns = list(df_header.columns)
    print(f"✅ Найдено {len(columns)} колонок")

    # Создаем ПРОСТУЮ таблицу в НОВОЙ БД
    print("\n📝 Создание простой таблицы...")

    # Создаем таблицу с уникальным именем
    table_name = 'transactions_simple'

    columns_sql = []
    for col in columns:
        # Очищаем имя колонки от спецсимволов
        safe_col = col.replace('`', '').replace('"', '').replace("'", '')
        columns_sql.append(f"    `{safe_col}` String")

    create_table = f"""
    CREATE TABLE card_analytics.{table_name}
    (
{','.join(columns_sql)}
    )
    ENGINE = MergeTree()
    ORDER BY tuple()
    SETTINGS index_granularity = 8192
    """

    try:
        client.execute(create_table)
        print(f"✅ Таблица {table_name} создана")
    except Exception as e:
        print(f"❌ Ошибка создания таблицы: {e}")
        return False

    # Загружаем данные БЕЗ батчей - построчно для надежности
    print(f"\n📤 Загрузка данных (может занять время)...")

    # Читаем весь CSV
    df = pd.read_csv(csv_file)
    total_rows = len(df)
    print(f"  Прочитано {total_rows:,} строк")

    # Очищаем данные
    df = df.fillna('')
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace('nan', '')

    # Загружаем батчами по 500 записей
    batch_size = 500
    loaded = 0
    failed = 0

    for i in range(0, total_rows, batch_size):
        batch = df.iloc[i:i + batch_size]

        # Готовим данные
        data = []
        for _, row in batch.iterrows():
            row_values = [str(row[col]) for col in columns]
            data.append(row_values)

        try:
            # Простая вставка
            query = f"INSERT INTO card_analytics.{table_name} VALUES"
            client.execute(query, data)
            loaded += len(batch)

            # Прогресс каждые 5000 записей
            if loaded % 5000 == 0:
                percent = (loaded / total_rows) * 100
                print(f"  ✓ {loaded:,} / {total_rows:,} ({percent:.1f}%)")

        except Exception as e:
            failed += len(batch)
            if failed < 1000:  # Показываем первые ошибки
                print(f"  ✗ Ошибка батча {i // batch_size}: {str(e)[:50]}")

    print(f"\n📊 РЕЗУЛЬТАТ:")
    print(f"  ✅ Загружено: {loaded:,} записей")
    if failed > 0:
        print(f"  ❌ Ошибок: {failed:,} записей")

    # Проверка
    if loaded > 0:
        try:
            count = client.execute(f'SELECT count() FROM card_analytics.{table_name}')[0][0]
            print(f"\n✅ ПРОВЕРКА: В таблице {count:,} записей")

            if count > 0:
                print("\n" + "=" * 60)
                print("🎉 УСПЕХ! Данные загружены!")
                print("=" * 60)

                print("\n📋 Проверьте данные в ClickHouse:")
                print("  http://localhost:8123/play")
                print()
                print("SQL запросы для проверки:")
                print(f"  SELECT count() FROM card_analytics.{table_name};")
                print(f"  SELECT * FROM card_analytics.{table_name} LIMIT 10;")

                return True

        except Exception as e:
            print(f"⚠️ Ошибка проверки: {e}")

    return False


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'data_100k.csv'
    success = clean_and_load(csv_file)

    if not success:
        print("\n❌ Загрузка не удалась")
        sys.exit(1)