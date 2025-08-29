# load_data_fixed.py
"""
Исправленный скрипт для загрузки данных в ClickHouse
Учитывает особенности структуры вашего CSV
"""

import pandas as pd
from clickhouse_driver import Client
from datetime import datetime
import sys
from pathlib import Path


def load_csv_to_clickhouse(csv_file='data_100k.csv'):
    """Загрузка CSV в ClickHouse с правильным маппингом колонок"""

    print(f"\n{'=' * 60}")
    print("ЗАГРУЗКА ДАННЫХ В CLICKHOUSE")
    print('=' * 60)

    # Проверяем файл
    if not Path(csv_file).exists():
        print(f"❌ Файл {csv_file} не найден!")
        return False

    print(f"📁 Файл: {csv_file}")
    file_size = Path(csv_file).stat().st_size / 1024 / 1024
    print(f"📏 Размер: {file_size:.2f} MB")

    # Подключение к ClickHouse
    print("\n🔌 Подключение к ClickHouse...")
    try:
        client = Client(
            host='localhost',
            port=9000,
            user='analyst',
            password='admin123'
        )

        # Проверка подключения
        client.execute('SELECT 1')
        print("✅ Подключение успешно")
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        print("\nПроверьте что Docker запущен:")
        print("  docker-compose up -d")
        return False

    # Создание БД
    print("\n📝 Создание базы данных...")
    try:
        client.execute('CREATE DATABASE IF NOT EXISTS card_analytics')
        client.execute('USE card_analytics')
        print("✅ База данных готова")
    except Exception as e:
        print(f"⚠️ Предупреждение: {e}")

    # Удаляем старую таблицу если есть проблемы
    print("\n🗑️ Очистка старых таблиц...")
    try:
        client.execute('DROP TABLE IF EXISTS card_analytics.transactions_main')
        print("✅ Старая таблица удалена")
    except:
        pass

    # Читаем первую строку CSV чтобы понять структуру
    print("\n📊 Анализ структуры CSV...")
    df_sample = pd.read_csv(csv_file, nrows=5)
    csv_columns = list(df_sample.columns)
    print(f"  Найдено {len(csv_columns)} колонок в CSV")

    # Проверяем наличие проблемных колонок
    if 'hour' in csv_columns and 'hour_num' not in csv_columns:
        print("  ⚠️ Найдена колонка 'hour' - будет переименована в 'hour_num'")
    if 'minute' in csv_columns and 'minute_num' not in csv_columns:
        print("  ⚠️ Найдена колонка 'minute' - будет переименована в 'minute_num'")

    # Создаем таблицу на основе реальной структуры CSV
    print("\n📝 Создание таблицы...")

    create_table = """
    CREATE TABLE IF NOT EXISTS transactions_main
    (
        hpan Float64,
        transaction_code String,
        rday UInt32,
        day_type String,
        issue_method String,
        card_product_type String,
        emitent_filial String,
        product_category String,
        emitent_region String,
        reissuing_flag String,
        expire_date String,
        issue_date String,
        card_type String,
        product_type String,
        card_bo_table String,
        pinfl String,
        pinfl_flag Nullable(Float32),
        gender String,
        birth_year String,
        age String,
        age_group String,
        iss_flag UInt8,
        emitent_net String,
        emitent_bank String,
        acq_flag UInt8,
        acquirer_net String,
        acquirer_bank String,
        mcc UInt16,
        acquirer_mfo String,
        merchant_name String,
        acquirer_branch String,
        acquirer_region String,
        merchant_type String,
        ip String,
        oked Nullable(Float32),
        login_category String,
        login_group String,
        login String,
        amount_uzs UInt64,
        record_state String,
        reqamt UInt64,
        conamt UInt64,
        match_num UInt32,
        reversal_flag UInt8,
        fe_trace UInt32,
        refnum UInt64,
        sttl_date UInt32,
        sttl_hour UInt8,
        sttl_minute UInt8,
        hour_num UInt8,
        minute_num UInt8,
        udatetime_month UInt8,
        merchant UInt32,
        respcode Nullable(Float32),
        terminal_id String,
        address_name String,
        address_country String,
        currency UInt16,
        merch_id UInt32,
        terminal_type String,
        credit_debit String,
        inst_id UInt32,
        inst_id2 UInt32,
        term_id_key String,
        bo_table String,
        data_flag String,
        trans_type_by_day_key String,
        emission_country String,
        sender_hpan String,
        sender_bank String,
        receiver_hpan String,
        receiver_bank String,
        p2p_flag UInt8,
        p2p_type String
    )
    ENGINE = MergeTree()
    ORDER BY (rday, transaction_code)
    SETTINGS index_granularity = 8192
    """

    try:
        client.execute(create_table)
        print("✅ Таблица создана")
    except Exception as e:
        print(f"⚠️ Предупреждение при создании таблицы: {e}")

    # Загрузка данных
    print(f"\n📊 Загрузка данных из CSV...")

    try:
        # Читаем CSV полностью
        df = pd.read_csv(csv_file, low_memory=False)
        print(f"  Прочитано {len(df):,} строк")

        # Переименовываем колонки если нужно
        rename_map = {}
        if 'hour' in df.columns and 'hour_num' not in df.columns:
            rename_map['hour'] = 'hour_num'
        if 'minute' in df.columns and 'minute_num' not in df.columns:
            rename_map['minute'] = 'minute_num'

        if rename_map:
            print(f"  Переименование колонок: {rename_map}")
            df = df.rename(columns=rename_map)

        # Очистка данных
        print("  Очистка данных...")

        # Список ожидаемых колонок в таблице
        expected_columns = [
            'hpan', 'transaction_code', 'rday', 'day_type', 'issue_method',
            'card_product_type', 'emitent_filial', 'product_category', 'emitent_region',
            'reissuing_flag', 'expire_date', 'issue_date', 'card_type', 'product_type',
            'card_bo_table', 'pinfl', 'pinfl_flag', 'gender', 'birth_year', 'age',
            'age_group', 'iss_flag', 'emitent_net', 'emitent_bank', 'acq_flag',
            'acquirer_net', 'acquirer_bank', 'mcc', 'acquirer_mfo', 'merchant_name',
            'acquirer_branch', 'acquirer_region', 'merchant_type', 'ip', 'oked',
            'login_category', 'login_group', 'login', 'amount_uzs', 'record_state',
            'reqamt', 'conamt', 'match_num', 'reversal_flag', 'fe_trace', 'refnum',
            'sttl_date', 'sttl_hour', 'sttl_minute', 'hour_num', 'minute_num',
            'udatetime_month', 'merchant', 'respcode', 'terminal_id', 'address_name',
            'address_country', 'currency', 'merch_id', 'terminal_type', 'credit_debit',
            'inst_id', 'inst_id2', 'term_id_key', 'bo_table', 'data_flag',
            'trans_type_by_day_key', 'emission_country', 'sender_hpan', 'sender_bank',
            'receiver_hpan', 'receiver_bank', 'p2p_flag', 'p2p_type'
        ]

        # Добавляем отсутствующие колонки
        for col in expected_columns:
            if col not in df.columns:
                print(f"  Добавление отсутствующей колонки: {col}")
                df[col] = 0 if col in ['rday', 'mcc', 'amount_uzs'] else ''

        # Числовые поля
        numeric_fields = {
            'hpan': 0.0,
            'rday': 0,
            'iss_flag': 0,
            'acq_flag': 0,
            'mcc': 0,
            'amount_uzs': 0,
            'reqamt': 0,
            'conamt': 0,
            'match_num': 0,
            'reversal_flag': 0,
            'fe_trace': 0,
            'refnum': 0,
            'sttl_date': 0,
            'sttl_hour': 0,
            'sttl_minute': 0,
            'hour_num': 0,
            'minute_num': 0,
            'udatetime_month': 0,
            'merchant': 0,
            'currency': 860,
            'merch_id': 0,
            'inst_id': 0,
            'inst_id2': 0,
            'p2p_flag': 0
        }

        for field, default in numeric_fields.items():
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce').fillna(default)

        # Nullable float поля
        nullable_float_fields = ['pinfl_flag', 'oked', 'respcode']
        for field in nullable_float_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')
                # Заменяем NaN на None для nullable полей
                df[field] = df[field].where(pd.notnull(df[field]), None)

        # Строковые поля
        string_fields = [col for col in expected_columns if
                         col not in numeric_fields and col not in nullable_float_fields]
        for field in string_fields:
            if field in df.columns:
                df[field] = df[field].fillna('').astype(str)

        # Оставляем только нужные колонки в правильном порядке
        df = df[expected_columns]

        # Загружаем батчами
        batch_size = 5000  # Уменьшаем размер батча
        total_loaded = 0

        print(f"  Загрузка батчами по {batch_size:,} записей...")

        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i + batch_size]

            try:
                # Преобразуем DataFrame в список кортежей
                data = []
                for _, row in batch.iterrows():
                    # Преобразуем строку в кортеж, заменяя NaN на None
                    row_data = []
                    for val in row.values:
                        if pd.isna(val):
                            row_data.append(None)
                        else:
                            row_data.append(val)
                    data.append(tuple(row_data))

                # Вставляем в ClickHouse
                client.execute(
                    'INSERT INTO card_analytics.transactions_main VALUES',
                    data
                )

                total_loaded += len(batch)
                print(f"  ✓ Загружено {total_loaded:,} / {len(df):,}")

            except Exception as e:
                print(f"  ✗ Ошибка при загрузке батча {i // batch_size + 1}: {str(e)[:100]}")
                # Продолжаем со следующим батчем
                continue

        print(f"\n✅ Успешно загружено {total_loaded:,} записей!")

        # Проверяем результат
        try:
            count = client.execute('SELECT count() FROM card_analytics.transactions_main')[0][0]
            print(f"\n📊 Проверка: в таблице {count:,} записей")

            # Показываем примеры данных
            print("\n📈 Примеры загруженных данных:")

            # Топ MCC
            mcc_stats = client.execute('''
                SELECT mcc, count() as cnt 
                FROM card_analytics.transactions_main 
                GROUP BY mcc 
                ORDER BY cnt DESC 
                LIMIT 5
            ''')

            if mcc_stats:
                print("\nТоп-5 MCC категорий:")
                for mcc, cnt in mcc_stats:
                    print(f"  MCC {mcc}: {cnt:,} транзакций")

            # P2P статистика
            p2p_stats = client.execute('''
                SELECT p2p_flag, count() as cnt, avg(amount_uzs) as avg_amount
                FROM card_analytics.transactions_main
                GROUP BY p2p_flag
            ''')

            if p2p_stats:
                print("\nP2P статистика:")
                for flag, cnt, avg_amt in p2p_stats:
                    p2p_type = "P2P переводы" if flag == 1 else "Обычные транзакции"
                    print(f"  {p2p_type}: {cnt:,} ({avg_amt:,.0f} UZS средняя сумма)")

        except Exception as e:
            print(f"⚠️ Ошибка при проверке: {e}")

        print("\n" + "=" * 60)
        print("ГОТОВО!")
        print("=" * 60)
        print("\n🎉 Данные успешно загружены! Теперь вы можете:")
        print("  1. Открыть ClickHouse UI: http://localhost:8123/play")
        print("  2. Запустить Streamlit: streamlit run run_app.py")
        print("  3. Использовать SQL запросы для анализа")

        return True

    except Exception as e:
        print(f"\n❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Проверяем аргументы
    csv_file = 'data_100k.csv'

    if len(sys.argv) > 1:
        csv_file = sys.argv[1]

    # Запускаем загрузку
    success = load_csv_to_clickhouse(csv_file)

    if not success:
        sys.exit(1)