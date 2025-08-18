import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    # ClickHouse - используем ваши настройки
    CLICKHOUSE_HOST: str = os.getenv("CLICKHOUSE_HOST", "clickhouse")
    CLICKHOUSE_PORT: int = 9000
    CLICKHOUSE_HTTP_PORT: int = 8123
    CLICKHOUSE_USER: str = "analyst"          # Ваш пользователь
    CLICKHOUSE_PASSWORD: str = "admin123"   # Ваш пароль
    CLICKHOUSE_DATABASE: str = "card_analytics"  # Новая БД для проекта

    # Настройки производительности ClickHouse
    CLICKHOUSE_MAX_THREADS: int = 8
    CLICKHOUSE_MAX_MEMORY_USAGE: int = 10_000_000_000  # 10GB
    CLICKHOUSE_MAX_BLOCK_SIZE: int = 65_536
    CLICKHOUSE_INSERT_BATCH_SIZE: int = 100_000

    # AI Services
    CLAUDE_API_KEY: str = "your_claude_api_key_here"
    OPENAI_API_KEY: str = ""  # Опционально для GPT

    # Пути к данным
    DATA_RAW_PATH: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_PATH: Path = BASE_DIR / "data" / "processed"

    # Streamlit
    STREAMLIT_PORT: int = 8501
    STREAMLIT_THEME: str = "dark"

    # Приложение
    APP_NAME: str = "Card Analytics Platform"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # ETL настройки
    ETL_BATCH_SIZE: int = 50_000
    ETL_PARALLEL_WORKERS: int = 4
    ETL_CHUNK_SIZE_MB: int = 100

    class Config:
        env_file = ".env"
        case_sensitive = True


# Синглтон настроек
settings = Settings()

# Создаем директории если не существуют
settings.DATA_RAW_PATH.mkdir(parents=True, exist_ok=True)
settings.DATA_PROCESSED_PATH.mkdir(parents=True, exist_ok=True)