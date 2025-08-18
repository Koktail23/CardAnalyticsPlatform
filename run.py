#!/usr/bin/env python3
"""
Главная точка входа для PyCharm
Настройте эту конфигурацию в PyCharm:
Run -> Edit Configurations -> Add Python Configuration
Script path: путь к этому файлу
"""

import sys
import os
from pathlib import Path
from loguru import logger
import warnings

# Добавляем корневую директорию в Python path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

# Настройка логирования
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/app_{time}.log",
    rotation="100 MB",
    retention="7 days",
    level="DEBUG"
)

# Игнорируем предупреждения
warnings.filterwarnings("ignore")


def check_clickhouse():
    """Проверка доступности ClickHouse"""
    try:
        from database.clickhouse_client import clickhouse
        result = clickhouse.execute("SELECT version()")
        logger.info(f"ClickHouse version: {result[0][0]}")
        return True
    except Exception as e:
        logger.error(f"ClickHouse is not available: {e}")
        logger.info("Please ensure ClickHouse is running:")
        logger.info("  Docker: docker run -d -p 9000:9000 -p 8123:8123 clickhouse/clickhouse-server")
        logger.info("  Or install locally: https://clickhouse.com/docs/en/install")
        return False


def init_database():
    """Инициализация таблиц в ClickHouse"""
    try:
        from database.clickhouse_client import clickhouse

        # Читаем и выполняем миграции
        migrations_dir = ROOT_DIR / "database" / "migrations"
        for migration_file in sorted(migrations_dir.glob("*.sql")):
            logger.info(f"Running migration: {migration_file.name}")

            with open(migration_file, 'r') as f:
                sql = f.read()

            # Разбиваем на отдельные команды
            commands = [cmd.strip() for cmd in sql.split(';') if cmd.strip()]
            for command in commands:
                clickhouse.execute(command)

        logger.info("Database initialized successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def load_sample_data():
    """Загрузка тестовых данных"""
    try:
        from etl.sample_data_generator import generate_sample_data

        logger.info("Generating sample data...")
        df = generate_sample_data(num_records=100_000)

        from database.clickhouse_client import clickhouse
        clickhouse.insert_df("transactions", df)

        logger.info("Sample data loaded successfully")
        return True

    except Exception as e:
        logger.error(f"Failed to load sample data: {e}")
        return False


def run_streamlit():
    """Запуск Streamlit приложения"""
    import streamlit.web.cli as stcli
    import sys

    app_path = str(ROOT_DIR / "app" / "main.py")

    sys.argv = [
        "streamlit",
        "run",
        app_path,
        "--server.port=8501",
        "--server.address=localhost",
        "--browser.gatherUsageStats=false",
        "--theme.base=dark",
        "--theme.primaryColor=#FF4B4B"
    ]

    logger.info("Starting Streamlit app at http://localhost:8501")
    sys.exit(stcli.main())


def main():
    """Главная функция"""

    print("""
    ╔══════════════════════════════════════════════════╗
    ║   CARD ANALYTICS PLATFORM WITH CLICKHOUSE       ║
    ║   Optimized for PyCharm Development             ║
    ╚══════════════════════════════════════════════════╝
    """)

    # Проверка ClickHouse
    if not check_clickhouse():
        print("\n❌ Please start ClickHouse first!")
        sys.exit(1)

    # Инициализация БД
    if not init_database():
        print("\n❌ Database initialization failed!")
        sys.exit(1)

    # Проверка данных
    from database.clickhouse_client import clickhouse
    count = clickhouse.execute("SELECT count() FROM transactions")[0][0]

    if count == 0:
        print("\n📊 No data found. Loading sample data...")
        load_sample_data()
    else:
        print(f"\n✅ Found {count:,} transactions in database")

    # Запуск Streamlit
    print("\n🚀 Launching application...")
    run_streamlit()


if __name__ == "__main__":
    main()