from database.clickhouse_client import get_clickhouse_manager
from pathlib import Path


def run_migrations(migrations_dir='database/migrations'):
    manager = get_clickhouse_manager()
    migrations_path = Path(migrations_dir)

    # Сортировка файлов по имени (001_, 002_ и т.д.)
    sql_files = sorted(migrations_path.glob('*.sql'))

    for sql_file in sql_files:
        with open(sql_file, 'r', encoding='utf-8') as f:
            sql = f.read()
        manager.execute(sql)
        print(f"Applied migration: {sql_file}")


if __name__ == "__main__":
    run_migrations()