# scripts/init_database.py
"""
Скрипт инициализации базы данных ClickHouse для Card Analytics Platform
Только загрузка реальных данных, без генерации
"""

import sys
import os
from pathlib import Path
import time
import logging
from typing import Dict, Any
import pandas as pd
from datetime import datetime

# Добавляем корневую директорию в path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.clickhouse_client import get_clickhouse_manager

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseInitializer:
    """Класс для инициализации и настройки БД"""

    def __init__(self):
        self.ch_manager = get_clickhouse_manager()
        self.migrations_dir = Path('database/migrations')

    def check_connection(self, max_retries: int = 5) -> bool:
        """
        Проверка подключения к ClickHouse с повторными попытками

        Args:
            max_retries: Максимальное количество попыток

        Returns:
            True если подключение успешно
        """
        logger.info("Проверка подключения к ClickHouse...")

        for attempt in range(max_retries):
            try:
                if self.ch_manager.test_connection():
                    logger.info("✅ Подключение к ClickHouse успешно!")
                    return True
            except Exception as e:
                logger.warning(f"Попытка {attempt + 1}/{max_retries} не удалась: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)

        logger.error("❌ Не удалось подключиться к ClickHouse")
        return False

    def run_migrations(self) -> bool:
        """
        Запуск SQL миграций

        Returns:
            True если все миграции выполнены успешно
        """
        logger.info("Запуск миграций...")

        # Получаем список файлов миграций
        migration_files = sorted(self.migrations_dir.glob('*.sql'))

        if not migration_files:
            logger.warning("Файлы миграций не найдены")
            # Используем встроенную миграцию
            return self.run_embedded_migration()

        for migration_file in migration_files:
            logger.info(f"Выполнение миграции: {migration_file.name}")

            try:
                with open(migration_file, 'r', encoding='utf-8') as f:
                    sql_content = f.read()

                # Разбиваем на отдельные команды
                commands = [cmd.strip() for cmd in sql_content.split(';') if cmd.strip()]

                for command in commands:
                    if command:
                        self.ch_manager.execute(command)

                logger.info(f"✅ Миграция {migration_file.name} выполнена успешно")

            except Exception as e:
                logger.error(f"❌ Ошибка при выполнении миграции {migration_file.name}: {e}")
                return False

        return True

    def run_embedded_migration(self) -> bool:
        """Запуск встроенной миграции если файлы не найдены"""
        logger.info("Запуск встроенной миграции...")

        try:
            # Создаем базу данных
            self.ch_manager.execute("CREATE DATABASE IF NOT EXISTS card_analytics")
            self.ch_manager.execute("USE card_analytics")

            # Проверяем существование основной таблицы
            tables = self.ch_manager.query("SHOW TABLES")
            existing_tables = [t[0] for t in tables]

            if 'transactions_main' not in existing_tables:
                logger.info("Создание таблицы transactions_main...")

                # Создаем основную таблицу для 74 колонок
                create_table_sql = """
                CREATE TABLE IF NOT EXISTS transactions_main
                (
                    -- Идентификаторы транзакции
                    transaction_code String,
                    rday UInt32,
                    day_type String,

                    -- Информация о карте
                    hpan Float64,
                    card_product_type String,
                    card_type String,
                    product_type String,
                    product_category String,
                    card_bo_table String,
                    issue_method String,
                    issue_date Date,
                    expire_date Date,
                    reissuing_flag String,

                    -- Информация о клиенте
                    pinfl String,
                    pinfl_flag Nullable(Float32),
                    gender String,
                    birth_year String,
                    age String,
                    age_group String,

                    -- Эмитент
                    iss_flag UInt8,
                    emitent_filial String,
                    emitent_region String,
                    emitent_net String,
                    emitent_bank String,
                    emission_country String,

                    -- Эквайер
                    acq_flag UInt8,
                    acquirer_net String,
                    acquirer_bank String,
                    acquirer_mfo String,
                    acquirer_branch String,
                    acquirer_region String,

                    -- Мерчант
                    mcc UInt16,
                    merchant_name String,
                    merchant_type String,
                    merchant UInt32,
                    merch_id UInt32,
                    oked Nullable(Float32),
                    terminal_id String,
                    terminal_type String,
                    term_id_key String,
                    address_name String,
                    address_country String,

                    -- IP и логин
                    ip String,
                    login_category String,
                    login_group String,
                    login String,

                    -- Суммы
                    amount_uzs UInt64,
                    reqamt UInt64,
                    conamt UInt64,
                    currency UInt16,

                    -- Статусы
                    record_state String,
                    match_num UInt32,
                    reversal_flag UInt8,
                    respcode Nullable(Float32),
                    credit_debit String,
                    data_flag String,
                    trans_type_by_day_key String,

                    -- Временные метки
                    fe_trace UInt32,
                    refnum UInt64,
                    sttl_date UInt32,
                    sttl_hour UInt8,
                    sttl_minute UInt8,
                    hour_num UInt8,
                    minute_num UInt8,
                    udatetime_month UInt8,

                    -- Инстанс данные
                    inst_id UInt32,
                    inst_id2 UInt32,
                    bo_table String,

                    -- P2P данные
                    p2p_flag UInt8,
                    p2p_type String,
                    sender_hpan String,
                    sender_bank String,
                    receiver_hpan String,
                    receiver_bank String,

                    -- Системные поля
                    inserted_at DateTime DEFAULT now()
                )
                ENGINE = MergeTree()
                PARTITION BY toYYYYMM(toDate(rday))
                ORDER BY (rday, transaction_code, hpan)
                SETTINGS index_granularity = 8192
                """

                self.ch_manager.execute(create_table_sql)
                logger.info("✅ Таблица transactions_main создана")

            return True

        except Exception as e:
            logger.error(f"❌ Ошибка при выполнении встроенной миграции: {e}")
            return False

    def load_data_from_csv(self, file_path: str, batch_size: int = 10000) -> bool:
        """
        Прямая загрузка данных из CSV в ClickHouse

        Args:
            file_path: Путь к CSV файлу
            batch_size: Размер батча для загрузки

        Returns:
            True если загрузка успешна
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"Файл {file_path} не найден")
            return False

        logger.info(f"📂 Загрузка данных из {file_path}...")

        try:
            # Загружаем CSV
            logger.info("Чтение CSV файла...")
            df = pd.read_csv(file_path, low_memory=False)
            logger.info(f"Загружено {len(df)} строк из CSV")

            # Базовая очистка данных
            logger.info("Очистка данных...")

            # Обработка дат
            date_columns = ['expire_date', 'issue_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    df[col] = df[col].fillna(pd.Timestamp('2024-01-01'))

            # Обработка числовых колонок
            numeric_columns = {
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
                'currency': 860,  # UZS по умолчанию
                'merch_id': 0,
                'inst_id': 0,
                'inst_id2': 0,
                'p2p_flag': 0
            }

            for col, default_val in numeric_columns.items():
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default_val)

            # Обработка строковых колонок
            string_columns = [col for col in df.columns if col not in numeric_columns and col not in date_columns]
            for col in string_columns:
                if col in df.columns:
                    df[col] = df[col].fillna('').astype(str).str.strip()
                    df[col] = df[col].replace('nan', '')

            # Добавляем системное поле если его нет
            if 'inserted_at' not in df.columns:
                df['inserted_at'] = datetime.now()

            # Убеждаемся что используем правильную БД
            self.ch_manager.execute("USE card_analytics")

            # Загружаем данные батчами
            logger.info(f"Загрузка {len(df)} записей в ClickHouse батчами по {batch_size}...")

            total_loaded = 0
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i + batch_size]

                try:
                    # Используем insert_df из clickhouse_client
                    self.ch_manager.insert_df('transactions_main', batch, batch_size=batch_size)
                    total_loaded += len(batch)
                    logger.info(f"Загружено {total_loaded}/{len(df)} записей...")
                except Exception as e:
                    logger.error(f"Ошибка при загрузке батча: {e}")
                    # Продолжаем загрузку остальных батчей
                    continue

            logger.info(f"✅ Успешно загружено {total_loaded} записей в transactions_main")
            return True

        except Exception as e:
            logger.error(f"❌ Ошибка при загрузке данных: {e}")
            return False

    def verify_tables(self) -> Dict[str, Any]:
        """
        Проверка созданных таблиц

        Returns:
            Словарь с информацией о таблицах
        """
        logger.info("Проверка таблиц...")

        try:
            self.ch_manager.execute("USE card_analytics")

            # Получаем список таблиц
            tables = self.ch_manager.query("SHOW TABLES")
            table_names = [t[0] for t in tables]

            table_info = {}
            for table_name in table_names:
                # Получаем количество записей
                count = self.ch_manager.query(f"SELECT count() FROM {table_name}")[0][0]

                # Получаем размер таблицы
                size_query = f"""
                SELECT 
                    formatReadableSize(sum(bytes_on_disk)) as size
                FROM system.parts
                WHERE database = 'card_analytics' 
                    AND table = '{table_name}'
                    AND active
                """
                size_result = self.ch_manager.query(size_query)
                size = size_result[0][0] if size_result else '0 B'

                table_info[table_name] = {
                    'records': count,
                    'size': size
                }

                logger.info(f"  📊 {table_name}: {count:,} записей, размер: {size}")

            return table_info

        except Exception as e:
            logger.error(f"Ошибка при проверке таблиц: {e}")
            return {}

    def print_summary(self):
        """Вывод итоговой информации"""
        print("\n" + "=" * 60)
        print("  CARD ANALYTICS PLATFORM - ИНИЦИАЛИЗАЦИЯ ЗАВЕРШЕНА")
        print("=" * 60)

        table_info = self.verify_tables()

        if table_info:
            print("\n📊 СТАТУС ТАБЛИЦ:")
            print("-" * 40)
            total_records = 0
            for table, info in table_info.items():
                print(f"  • {table}: {info['records']:,} записей ({info['size']})")
                if table == 'transactions_main':
                    total_records = info['records']

            if total_records > 0:
                # Показываем примеры запросов
                print("\n🔍 ПРИМЕРЫ ЗАПРОСОВ ДЛЯ АНАЛИЗА:")
                print("-" * 40)
                print("  -- Топ MCC категорий:")
                print("  SELECT mcc, count() as cnt")
                print("  FROM card_analytics.transactions_main")
                print("  GROUP BY mcc ORDER BY cnt DESC LIMIT 10;")
                print()
                print("  -- P2P статистика:")
                print("  SELECT p2p_flag, count(), avg(amount_uzs)")
                print("  FROM card_analytics.transactions_main")
                print("  GROUP BY p2p_flag;")

        print("\n🚀 СЛЕДУЮЩИЕ ШАГИ:")
        print("-" * 40)
        print("  1. Запустить Streamlit приложение:")
        print("     streamlit run run_app.py")
        print()
        print("  2. Проверить данные в ClickHouse:")
        print("     http://localhost:8123/play")
        print()
        print("  3. Открыть Jupyter для анализа:")
        print("     http://localhost:8888")
        print()
        print("=" * 60)


def main():
    """Основная функция"""
    import argparse

    parser = argparse.ArgumentParser(description='Инициализация БД Card Analytics Platform')
    parser.add_argument('--skip-migrations', action='store_true', help='Пропустить миграции')
    parser.add_argument('--file', type=str, default='data_100k.csv',
                        help='CSV файл для загрузки (по умолчанию data_100k.csv)')
    parser.add_argument('--batch-size', type=int, default=10000,
                        help='Размер батча для загрузки (по умолчанию 10000)')

    args = parser.parse_args()

    # Создаем инициализатор
    initializer = DatabaseInitializer()

    # Шаг 1: Проверка подключения
    print("\n🔌 Проверка подключения к ClickHouse...")
    if not initializer.check_connection():
        print("\n❌ Не удалось подключиться к ClickHouse!")
        print("Проверьте что Docker контейнер запущен:")
        print("  docker-compose up -d")
        sys.exit(1)

    # Шаг 2: Запуск миграций (создание таблиц)
    if not args.skip_migrations:
        print("\n📝 Создание таблиц...")
        if not initializer.run_migrations():
            print("❌ Ошибка при создании таблиц")
            sys.exit(1)

    # Шаг 3: Поиск и загрузка данных
    # Приоритет файлов для загрузки
    data_files_priority = [
        args.file,  # Указанный пользователем файл
        'data_100k.csv',  # Файл с 100к записей
        'output.csv',  # Оригинальный файл с 50 записями
        'data/data_100k.csv',  # В папке data
        'data/output.csv'
    ]

    data_loaded = False
    for data_file in data_files_priority:
        if data_file and Path(data_file).exists():
            file_size = Path(data_file).stat().st_size / 1024 / 1024  # В МБ
            print(f"\n📊 Найден файл: {data_file} ({file_size:.2f} MB)")

            # Подсчитываем примерное количество строк
            with open(data_file, 'r', encoding='utf-8') as f:
                first_line = f.readline()
                sample_lines = f.read(1024 * 100)  # Читаем 100KB для оценки
                lines_in_sample = sample_lines.count('\n')
                estimated_lines = int(lines_in_sample * file_size * 1024 / 100)
                print(f"   Примерно {estimated_lines:,} строк")

            print(f"📂 Загрузка данных из {data_file}...")
            if initializer.load_data_from_csv(data_file, batch_size=args.batch_size):
                data_loaded = True
                break

    if not data_loaded:
        print("\n❌ Не найдены файлы с данными!")
        print("Поместите файл data_100k.csv в корневую директорию проекта")
        sys.exit(1)

    # Шаг 4: Вывод итоговой информации
    initializer.print_summary()


if __name__ == "__main__":
    main()