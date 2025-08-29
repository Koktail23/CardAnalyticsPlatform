#!/usr/bin/env python3
"""
Прямая проверка как ClickHouse рассчитывает даты
"""

from clickhouse_driver import Client


def direct_check():
    client = Client(
        host='localhost',
        port=9000,
        user='analyst',
        password='admin123',
        database='card_analytics'
    )

    print("\n" + "=" * 60)
    print("ПРЯМАЯ ПРОВЕРКА РАСЧЕТА ДАТ")
    print("=" * 60)

    # Проверяем разные базовые даты
    test_rday = 45658  # Минимальное значение rday из ваших данных

    print(f"\nТестируем rday = {test_rday}")
    print("-" * 40)

    bases = [
        ('1900-01-01', 1900),
        ('1970-01-01', 1970),
        ('2000-01-01', 2000)
    ]

    for base_str, base_year in bases:
        result = client.execute(f"""
            SELECT 
                toDate('{base_str}') as base_date,
                {test_rday} as rday,
                toDate('{base_str}') + {test_rday} as calculated_date,
                toDate('{base_str}') + toIntervalDay({test_rday}) as interval_date
        """)[0]

        print(f"\nБаза {base_str}:")
        print(f"  toDate('{base_str}') + {test_rday} = {result[2]}")
        print(f"  toDate('{base_str}') + toIntervalDay({test_rday}) = {result[3]}")

        # Ожидаемый год
        expected_year = base_year + (test_rday // 365)
        print(f"  Ожидаемый год: ~{expected_year}")

    # Проверяем что в таблице
    print("\n" + "=" * 60)
    print("ЧТО СЕЙЧАС В ТАБЛИЦЕ transactions_optimized:")
    print("-" * 40)

    sample = client.execute("""
        SELECT 
            rday,
            transaction_date,
            toDate('1900-01-01') + rday as calc_1900,
            toDate('1970-01-01') + rday as calc_1970,
            toDate('2000-01-01') + rday as calc_2000
        FROM transactions_optimized
        LIMIT 5
    """)

    print("\nПримеры записей:")
    print("rday  | transaction_date | база_1900   | база_1970   | база_2000")
    print("-" * 65)
    for row in sample:
        print(f"{row[0]:<6} | {row[1]} | {row[2]} | {row[3]} | {row[4]}")

    # Проверяем структуру таблицы
    print("\n" + "=" * 60)
    print("СТРУКТУРА ТАБЛИЦЫ:")
    print("-" * 40)

    structure = client.execute("""
        SELECT name, type, default_kind, default_expression
        FROM system.columns
        WHERE database = 'card_analytics' 
        AND table = 'transactions_optimized'
        AND name IN ('rday', 'transaction_date')
    """)

    for col_name, col_type, default_kind, default_expr in structure:
        print(f"{col_name}: {col_type}")
        if default_kind:
            print(f"  Default: {default_kind} = {default_expr}")


if __name__ == "__main__":
    direct_check()