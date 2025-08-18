from clickhouse_driver import Client
import pandas as pd


def test_connection():
    try:
        # Подключаемся к существующей базе datagate
        client = Client(
            host='localhost',
            port=9000,
            user='admin',
            password='admin123',
            database='datagate'  # Используем существующую базу
        )

        print(f"✅ Подключен к базе datagate")

        # Проверяем существующие данные
        print("\n📊 СУЩЕСТВУЮЩИЕ ДАННЫЕ:")
        print("-" * 40)

        # Смотрим sales_data (10,000 записей!)
        sample = client.execute('SELECT * FROM sales_data LIMIT 5')
        print(f"Пример из sales_data:")
        for row in sample:
            print(f"  {row}")

        # Структура таблицы sales_data
        columns = client.execute("DESCRIBE sales_data")
        print(f"\nСтруктура sales_data:")
        for col in columns:
            print(f"  - {col[0]}: {col[1]}")

        # Создаем таблицу для карточных транзакций
        print("\n🔨 СОЗДАНИЕ ТАБЛИЦ ДЛЯ ПРОЕКТА:")
        print("-" * 40)

        client.execute('''
            CREATE TABLE IF NOT EXISTS card_transactions (
                transaction_id String,
                company_id String,
                card_masked String,
                amount Decimal64(2),
                currency FixedString(3) DEFAULT 'USD',
                mcc_code FixedString(4),
                mcc_description String,
                merchant_name String,
                merchant_id String,
                transaction_date Date,
                transaction_time DateTime,
                status String DEFAULT 'completed',
                is_fraud UInt8 DEFAULT 0,
                fraud_score Float32 DEFAULT 0
            ) ENGINE = MergeTree()
            PARTITION BY toYYYYMM(transaction_date)
            ORDER BY (company_id, transaction_date)
        ''')
        print("✅ Таблица card_transactions создана")

        # Проверяем
        tables = client.execute('SHOW TABLES')
        print(f"\n📋 Все таблицы в базе datagate:")
        for table in tables:
            count = client.execute(f'SELECT count() FROM {table[0]}')[0][0]
            print(f"  - {table[0]}: {count:,} записей")

        print("\n🎉 ВСЕ ГОТОВО К РАБОТЕ!")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


def analyze_sales_data():
    """Анализ существующих данных sales_data"""
    print("\n" + "=" * 50)
    print("АНАЛИЗ СУЩЕСТВУЮЩИХ ДАННЫХ")
    print("=" * 50)

    client = Client(
        host='localhost',
        port=9000,
        user='admin',
        password='admin123',
        database='datagate'
    )

    # Базовая статистика - используем правильное поле date
    stats = client.execute('''
        SELECT 
            count() as total_records,
            min(date) as min_date,
            max(date) as max_date,
            sum(total) as total_amount,
            avg(total) as avg_amount
        FROM sales_data
    ''')[0]

    print(f"📊 Статистика sales_data:")
    print(f"  Записей: {stats[0]:,}")
    print(f"  Период: {stats[1]} - {stats[2]}")
    print(f"  Общая сумма: ${stats[3]:,.2f}")
    print(f"  Средняя сумма: ${stats[4]:,.2f}")


if __name__ == "__main__":
    test_connection()
    analyze_sales_data()