# safe_load.py
"""
Безопасная загрузка - все колонки как String для начала
"""

import pandas as pd
from clickhouse_driver import Client
import sys
from pathlib import Path


def safe_load_to_clickhouse(csv_file='data_100k.csv'):
    """Безопасная загрузка - сначала всё как строки"""

    print("\n" + "=" * 60)
    print("БЕЗОПАСНАЯ ЗАГРУЗКА В CLICKHOUSE")
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

    # Читаем заголовки CSV
    print(f"\n📊 Анализ {csv_file}...")
    df_header = pd.read_csv(csv_file, nrows=0)
    columns = list(df_header.columns)
    print(f"✅ Найдено {len(columns)} колонок")

    # Создаем БД
    client.execute('CREATE DATABASE IF NOT EXISTS card_analytics')

    # Удаляем старую таблицу
    print("\n🗑️ Очистка...")
    client.execute('DROP TABLE IF EXISTS card_analytics.transactions_main')

    # Создаем таблицу - ВСЕ колонки как String для безопасности
    print("\n📝 Создание таблицы (все колонки String)...")

    columns_sql = []
    for col in columns:
        # Экранируем имена колонок
        safe_col = col.replace('`', '').replace('"', '')
        columns_sql.append(f"    `{safe_col}` String")

    create_table = f"""
    CREATE TABLE card_analytics.transactions_main
    (
{','.join(columns_sql)}
    )
    ENGINE = MergeTree()
    ORDER BY tuple()
    """

    try:
        client.execute(create_table)
        print("✅ Таблица создана")
    except Exception as e:
        print(f"❌ Ошибка создания таблицы: {e}")
        return False

    # Загружаем данные
    print(f"\n📤 Загрузка данных...")

    # Читаем CSV частями для экономии памяти
    batch_size = 1000
    total_loaded = 0

    for chunk in pd.read_csv(csv_file, chunksize=batch_size):
        # Заменяем все NaN на пустые строки
        chunk = chunk.fillna('')

        # Преобразуем всё в строки
        for col in chunk.columns:
            chunk[col] = chunk[col].astype(str)
            chunk[col] = chunk[col].replace('nan', '')

        # Готовим данные для вставки
        data = []
        for _, row in chunk.iterrows():
            # Берем значения в том же порядке, что и колонки
            row_values = [str(row[col]) for col in columns]
            data.append(row_values)

        try:
            # Вставляем данные
            client.execute(
                'INSERT INTO card_analytics.transactions_main VALUES',
                data
            )
            total_loaded += len(chunk)

            # Прогресс
            if total_loaded % 10000 == 0:
                print(f"  ✓ Загружено {total_loaded:,} записей...")

        except Exception as e:
            print(f"  ✗ Ошибка: {str(e)[:100]}")
            continue

    print(f"\n✅ Загружено {total_loaded:,} записей!")

    # Проверка
    try:
        count = client.execute('SELECT count() FROM card_analytics.transactions_main')[0][0]
        print(f"\n📊 ПРОВЕРКА: В таблице {count:,} записей")

        if count > 0:
            # Показываем пример данных
            print("\n📋 Пример загруженных данных:")
            sample = client.execute('SELECT * FROM card_analytics.transactions_main LIMIT 1')
            if sample:
                print(f"  Первая запись содержит {len(sample[0])} полей")

            print("\n" + "=" * 60)
            print("✅ УСПЕХ! Данные загружены!")
            print("=" * 60)

            print("\n🎯 Что дальше:")
            print("  1. Проверить данные:")
            print("     http://localhost:8123/play")
            print("     SELECT * FROM card_analytics.transactions_main LIMIT 10")
            print()
            print("  2. Посмотреть статистику:")
            print("     SELECT count(*) FROM card_analytics.transactions_main")
            print()
            print("  3. Создать оптимизированную таблицу с правильными типами")
            print("     (после анализа данных)")

            return True
    except Exception as e:
        print(f"\n⚠️ Ошибка проверки: {e}")

    return False


if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'data_100k.csv'

    success = safe_load_to_clickhouse(csv_file)

    if not success:
        print("\n❌ Загрузка не удалась")
        sys.exit(1)