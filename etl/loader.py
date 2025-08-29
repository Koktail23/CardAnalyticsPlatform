import polars as pl
from database.clickhouse_client import get_clickhouse_manager
from config.settings import settings
import argparse

def load_to_clickhouse(file_path, table_name, batch_size=None):
    if batch_size is None:
        batch_size = settings.ETL_BATCH_SIZE  # 50_000 из settings.py
    
    # Чтение CSV с помощью Polars (поддерживает большие файлы)
    df = pl.read_csv(file_path, infer_schema_length=10000)  # Автоопределение типов
    
    # Подключение к ClickHouse
    manager = get_clickhouse_manager()
    
    # Batch-вставка
    for i in range(0, len(df), batch_size):
        batch = df[i:i + batch_size]
        # Вставка через Arrow (эффективно)
        manager.client.insert_arrow(table_name, batch.to_arrow())
        print(f"Inserted batch {i // batch_size + 1} of {len(df) // batch_size + 1}")
    
    print("Data loaded successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load CSV to ClickHouse")
    parser.add_argument("--file", required=True, help="Path to CSV file")
    parser.add_argument("--table", required=True, help="Target table name")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size for insertion")
    
    args = parser.parse_args()
    load_to_clickhouse(args.file, args.table, args.batch_size)