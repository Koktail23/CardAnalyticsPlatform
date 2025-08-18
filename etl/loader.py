"""
ETL загрузчик оптимизированный для ClickHouse и больших локальных файлов
"""
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, Any, Optional, List
import glob
from loguru import logger
from tqdm import tqdm
import hashlib
from datetime import datetime
import pyarrow.parquet as pq
import pyarrow as pa

from database.clickhouse_client import clickhouse
from config.settings import settings


class DataLoader:
    """Загрузчик данных в ClickHouse"""

    def __init__(self):
        self.clickhouse = clickhouse
        self.processed_files = set()
        self.load_processed_files()

    def load_processed_files(self):
        """Загрузка списка обработанных файлов"""
        try:
            result = self.clickhouse.execute("""
                SELECT DISTINCT source_file 
                FROM transactions 
                WHERE source_file != ''
            """)
            self.processed_files = {row[0] for row in result}
            logger.info(f"Found {len(self.processed_files)} processed files")
        except:
            logger.warning("Could not load processed files list")

    def load_csv(self,
                 file_path: Path,
                 company_id: str,
                 column_mapping: Optional[Dict[str, str]] = None,
                 chunk_size: int = 100_000) -> int:
        """Загрузка CSV файла по частям"""

        file_hash = self._get_file_hash(file_path)

        if file_hash in self.processed_files:
            logger.info(f"File {file_path.name} already processed, skipping")
            return 0

        logger.info(f"Loading CSV: {file_path.name} for company {company_id}")

        # Дефолтный маппинг колонок
        if not column_mapping:
            column_mapping = {
                'transaction_id': 'transaction_id',
                'card_number': 'card_masked',
                'amount': 'amount',
                'currency': 'currency',
                'mcc': 'mcc_code',
                'merchant': 'merchant_name',
                'date': 'transaction_date',
                'time': 'transaction_time',
                'status': 'status'
            }

        total_rows = 0

        try:
            # Используем Polars для больших файлов
            if file_path.stat().st_size > 100_000_000:  # >100MB
                logger.info("Using Polars for large file processing")
                df_lazy = pl.scan_csv(file_path, low_memory=True)

                # Обработка по батчам
                for batch_df in self._read_csv_in_batches_polars(df_lazy, chunk_size):
                    processed_df = self._process_dataframe(
                        batch_df.to_pandas(),
                        company_id,
                        column_mapping,
                        file_hash
                    )

                    self.clickhouse.insert_df('transactions', processed_df)
                    total_rows += len(processed_df)

            else:
                # Pandas для небольших файлов
                for chunk_df in pd.read_csv(file_path, chunksize=chunk_size):
                    processed_df = self._process_dataframe(
                        chunk_df,
                        company_id,
                        column_mapping,
                        file_hash
                    )

                    self.clickhouse.insert_df('transactions', processed_df)
                    total_rows += len(processed_df)

            self.processed_files.add(file_hash)
            logger.info(f"Successfully loaded {total_rows:,} rows from {file_path.name}")

        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise

        return total_rows

    def load_parquet(self,
                     file_path: Path,
                     company_id: str,
                     batch_size: int = 100_000) -> int:
        """Загрузка Parquet файла"""

        file_hash = self._get_file_hash(file_path)

        if file_hash in self.processed_files:
            logger.info(f"File {file_path.name} already processed")
            return 0

        logger.info(f"Loading Parquet: {file_path.name}")

        total_rows = 0

        try:
            # Читаем Parquet по батчам
            parquet_file = pq.ParquetFile(file_path)

            for batch in parquet_file.iter_batches(batch_size=batch_size):
                df = batch.to_pandas()

                # Добавляем служебные поля
                df['company_id'] = company_id
                df['source_file'] = file_hash
                df['loaded_at'] = datetime.now()

                # Маскируем номера карт
                if 'card_number' in df.columns:
                    df['card_masked'] = df['card_number'].apply(self._mask_card)
                    df.drop('card_number', axis=1, inplace=True)

                self.clickhouse.insert_df('transactions', df)
                total_rows += len(df)

            self.processed_files.add(file_hash)
            logger.info(f"Loaded {total_rows:,} rows from Parquet")

        except Exception as e:
            logger.error(f"Failed to load Parquet: {e}")
            raise

        return total_rows

    def load_directory(self,
                       directory: Path,
                       company_id: str,
                       pattern: str = "*.csv") -> Dict[str, int]:
        """Загрузка всех файлов из директории"""

        logger.info(f"Loading files from {directory} with pattern {pattern}")

        results = {}
        files = list(directory.glob(pattern))

        if not files:
            logger.warning(f"No files found matching pattern {pattern}")
            return results

        for file_path in tqdm(files, desc="Loading files"):
            try:
                if file_path.suffix.lower() == '.csv':
                    rows = self.load_csv(file_path, company_id)
                elif file_path.suffix.lower() in ['.parquet', '.pq']:
                    rows = self.load_parquet(file_path, company_id)
                else:
                    logger.warning(f"Unsupported file type: {file_path.suffix}")
                    continue

                results[file_path.name] = rows

            except Exception as e:
                logger.error(f"Failed to load {file_path.name}: {e}")
                results[file_path.name] = 0

        # Оптимизация таблиц после загрузки
        logger.info("Optimizing tables...")
        self.clickhouse.optimize_table('transactions')

        return results

    def _process_dataframe(self,
                           df: pd.DataFrame,
                           company_id: str,
                           column_mapping: Dict[str, str],
                           file_hash: str) -> pd.DataFrame:
        """Обработка и очистка DataFrame"""

        # Переименование колонок
        df = df.rename(columns=column_mapping)

        # Добавление обязательных полей
        df['company_id'] = company_id
        df['source_file'] = file_hash
        df['loaded_at'] = datetime.now()
        df['batch_id'] = hashlib.md5(f"{file_hash}_{datetime.now()}".encode()).hexdigest()[:8]

        # Обработка дат
        if 'transaction_date' in df.columns:
            df['transaction_date'] = pd.to_datetime(df['transaction_date']).dt.date

        if 'transaction_time' in df.columns:
            df['transaction_time'] = pd.to_datetime(df['transaction_time'])
        elif 'transaction_date' in df.columns:
            df['transaction_time'] = pd.to_datetime(df['transaction_date'])

        # Маскирование карт
        if 'card_masked' in df.columns:
            df['card_masked'] = df['card_masked'].apply(self._mask_card)

        # Очистка сумм
        if 'amount' in df.columns:
            df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0)
            df = df[df['amount'] > 0]  # Убираем нулевые транзакции

        # Заполнение пропущенных значений
        defaults = {
            'currency': 'USD',
            'status': 'completed',
            'channel': 'other',
            'is_fraud': 0,
            'fraud_score': 0.0,
            'mcc_code': '0000',
            'mcc_description': 'Unknown'
        }

        for col, default_value in defaults.items():
            if col in df.columns:
                df[col] = df[col].fillna(default_value)

        return df

    def _mask_card(self, card_number: Any) -> str:
        """Маскирование номера карты"""
        if pd.isna(card_number):
            return "****"

        card_str = str(card_number).replace(' ', '').replace('-', '')

        if len(card_str) >= 12:
            return f"{card_str[:4]}****{card_str[-4:]}"
        elif len(card_str) >= 8:
            return f"****{card_str[-4:]}"
        else:
            return "****"

    def _get_file_hash(self, file_path: Path) -> str:
        """Получение хеша файла"""
        return hashlib.md5(f"{file_path.name}_{file_path.stat().st_size}".encode()).hexdigest()

    def _read_csv_in_batches_polars(self, df_lazy: pl.LazyFrame, batch_size: int):
        """Чтение CSV батчами через Polars"""
        offset = 0
        while True:
            batch = df_lazy.slice(offset, batch_size).collect()
            if batch.is_empty():
                break
            yield batch
            offset += batch_size