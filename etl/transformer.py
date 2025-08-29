import polars as pl
from datetime import datetime
from config.settings import settings


def transform_data(file_path, output_path='data/processed/cleaned_transactions.csv'):
    df = pl.read_csv(file_path, infer_schema_length=10000)

    # Стандартизация дат (пример: expire_date, issue_date в формат Date)
    df = df.with_columns([
        pl.col('expire_date').str.strptime(pl.Date, '%d.%m.%Y').alias('expire_date'),
        pl.col('issue_date').str.strptime(pl.Date, '%d.%m.%Y').alias('issue_date'),
        # Добавьте для других дат, если нужно
    ])

    # Заполнение null (медиана для чисел, 'Unknown' для строк)
    for col in df.columns:
        if df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64]:
            median_val = df[col].median()
            df = df.with_columns(pl.col(col).fill_null(median_val))
        elif df[col].dtype == pl.Utf8:
            df = df.with_columns(pl.col(col).fill_null('Unknown'))

    # Нормализация строк (upper case для merchant_name, etc.)
    df = df.with_columns([
        pl.col('merchant_name').str.to_uppercase(),
        pl.col('emitent_region').str.strip_chars(),
        # Добавьте для других строковых колонок
    ])

    # Обогащение: Добавьте производные фичи (например, is_weekend)
    df = df.with_columns([
        pl.col('rday').apply(lambda x: 1 if datetime.fromtimestamp(x).weekday() >= 5 else 0).alias('is_weekend')
    ])

    # Сохранение очищенных данных
    df.write_csv(output_path)
    print(f"Transformed data saved to {output_path}")
    return df


if __name__ == "__main__":
    transform_data('output.csv')