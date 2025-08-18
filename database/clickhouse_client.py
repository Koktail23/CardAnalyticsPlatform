"""
ClickHouse client module for Card Analytics Platform
Provides connection management and basic operations
"""

import os
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, date
import pandas as pd
import polars as pl
from contextlib import contextmanager
from loguru import logger
from clickhouse_driver import Client, connect
from clickhouse_driver.errors import Error as ClickHouseError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class ClickHouseManager:
    """
    ClickHouse connection and query manager
    """

    def __init__(self,
                 host: Optional[str] = None,
                 port: Optional[int] = None,
                 user: Optional[str] = None,
                 password: Optional[str] = None,
                 database: Optional[str] = None):
        """
        Initialize ClickHouse connection manager

        Args:
            host: ClickHouse host
            port: ClickHouse port
            user: Username
            password: Password
            database: Default database
        """
        self.host = host or os.getenv('CLICKHOUSE_HOST', 'localhost')
        self.port = port or int(os.getenv('CLICKHOUSE_PORT', 9000))
        self.user = user or os.getenv('CLICKHOUSE_USER', 'analyst')
        self.password = password or os.getenv('CLICKHOUSE_PASSWORD', '')
        self.database = database or os.getenv('CLICKHOUSE_DATABASE', 'card_analytics')

        self.connection_params = {
            'host': self.host,
            'port': self.port,
            'user': self.user,
            'password': self.password,
            'database': self.database,
            'connect_timeout': 10,
            'send_receive_timeout': 300,
            'compression': True,
            'secure': False
        }

        self._client = None
        logger.info(f"ClickHouse manager initialized for {self.host}:{self.port}/{self.database}")

    @property
    def client(self) -> Client:
        """Get or create ClickHouse client"""
        if self._client is None:
            self._client = Client(**self.connection_params)
        return self._client

    @contextmanager
    def get_connection(self):
        """Context manager for database connection"""
        conn = connect(**self.connection_params)
        try:
            yield conn
        finally:
            conn.close()

    def execute(self, query: str, params: Optional[Dict] = None) -> Any:
        """
        Execute a query without returning results

        Args:
            query: SQL query
            params: Query parameters

        Returns:
            Query result
        """
        try:
            logger.debug(f"Executing query: {query[:100]}...")
            result = self.client.execute(query, params or {})
            logger.debug("Query executed successfully")
            return result
        except ClickHouseError as e:
            logger.error(f"ClickHouse error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    def query(self, query: str, params: Optional[Dict] = None) -> List[tuple]:
        """
        Execute a SELECT query and return results

        Args:
            query: SELECT query
            params: Query parameters

        Returns:
            List of tuples with results
        """
        try:
            logger.debug(f"Executing SELECT query: {query[:100]}...")
            result = self.client.execute(query, params or {}, with_column_types=False)
            logger.debug(f"Query returned {len(result)} rows")
            return result
        except ClickHouseError as e:
            logger.error(f"ClickHouse error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    def query_df(self, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
        """
        Execute query and return results as pandas DataFrame

        Args:
            query: SELECT query
            params: Query parameters

        Returns:
            pandas DataFrame with results
        """
        try:
            logger.debug(f"Executing query for DataFrame: {query[:100]}...")
            result = self.client.execute(query, params or {}, with_column_types=True)

            if not result:
                return pd.DataFrame()

            data, columns = result[0], result[1]
            column_names = [col[0] for col in columns]

            df = pd.DataFrame(data, columns=column_names)
            logger.debug(f"DataFrame created with shape {df.shape}")
            return df

        except ClickHouseError as e:
            logger.error(f"ClickHouse error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

    def query_polars(self, query: str, params: Optional[Dict] = None) -> pl.DataFrame:
        """
        Execute query and return results as Polars DataFrame

        Args:
            query: SELECT query
            params: Query parameters

        Returns:
            Polars DataFrame with results
        """
        try:
            logger.debug(f"Executing query for Polars DataFrame: {query[:100]}...")
            pandas_df = self.query_df(query, params)
            polars_df = pl.from_pandas(pandas_df)
            logger.debug(f"Polars DataFrame created with shape {polars_df.shape}")
            return polars_df

        except Exception as e:
            logger.error(f"Error creating Polars DataFrame: {e}")
            raise

    def insert_df(self, table: str, df: Union[pd.DataFrame, pl.DataFrame],
                  batch_size: int = 10000) -> int:
        """
        Insert DataFrame into ClickHouse table

        Args:
            table: Table name
            df: DataFrame to insert (pandas or polars)
            batch_size: Batch size for insertion

        Returns:
            Number of inserted rows
        """
        try:
            # Convert polars to pandas if needed
            if isinstance(df, pl.DataFrame):
                df = df.to_pandas()

            total_rows = len(df)
            logger.info(f"Inserting {total_rows} rows into {table}")

            # Insert in batches
            for i in range(0, total_rows, batch_size):
                batch = df.iloc[i:i + batch_size]
                data = batch.to_dict('records')

                self.client.execute(
                    f'INSERT INTO {table} VALUES',
                    data,
                    types_check=True
                )

                logger.debug(f"Inserted batch {i // batch_size + 1}/{(total_rows - 1) // batch_size + 1}")

            logger.info(f"Successfully inserted {total_rows} rows into {table}")
            return total_rows

        except ClickHouseError as e:
            logger.error(f"ClickHouse error during insertion: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during insertion: {e}")
            raise

    def create_table_from_df(self, table: str, df: Union[pd.DataFrame, pl.DataFrame],
                             primary_key: Optional[List[str]] = None,
                             partition_by: Optional[str] = None,
                             order_by: Optional[List[str]] = None) -> bool:
        """
        Create ClickHouse table from DataFrame schema

        Args:
            table: Table name
            df: DataFrame to use for schema
            primary_key: Primary key columns
            partition_by: Partition expression
            order_by: Order by columns

        Returns:
            True if successful
        """
        try:
            # Convert polars to pandas if needed
            if isinstance(df, pl.DataFrame):
                df = df.to_pandas()

            # Map pandas dtypes to ClickHouse types
            type_mapping = {
                'int64': 'Int64',
                'int32': 'Int32',
                'int16': 'Int16',
                'int8': 'Int8',
                'uint64': 'UInt64',
                'uint32': 'UInt32',
                'uint16': 'UInt16',
                'uint8': 'UInt8',
                'float64': 'Float64',
                'float32': 'Float32',
                'bool': 'UInt8',
                'object': 'String',
                'datetime64[ns]': 'DateTime',
                'datetime64[ns, UTC]': 'DateTime',
            }

            # Build column definitions
            columns = []
            for col, dtype in df.dtypes.items():
                ch_type = type_mapping.get(str(dtype), 'String')
                columns.append(f"`{col}` {ch_type}")

            columns_str = ",\n    ".join(columns)

            # Build CREATE TABLE query
            query = f"""
            CREATE TABLE IF NOT EXISTS {table}
            (
                {columns_str}
            )
            ENGINE = MergeTree()
            """

            if partition_by:
                query += f"\nPARTITION BY {partition_by}"

            if order_by:
                query += f"\nORDER BY ({', '.join(order_by)})"
            elif primary_key:
                query += f"\nORDER BY ({', '.join(primary_key)})"
            else:
                # Use first column as default order
                query += f"\nORDER BY (`{df.columns[0]}`)"

            self.execute(query)
            logger.info(f"Table {table} created successfully")
            return True

        except Exception as e:
            logger.error(f"Error creating table: {e}")
            raise

    def get_table_info(self, table: str) -> pd.DataFrame:
        """
        Get table structure information

        Args:
            table: Table name

        Returns:
            DataFrame with column information
        """
        query = f"""
        SELECT 
            name,
            type,
            default_type,
            default_expression,
            comment
        FROM system.columns
        WHERE database = '{self.database}'
            AND table = '{table}'
        ORDER BY position
        """
        return self.query_df(query)

    def get_table_stats(self, table: str) -> Dict[str, Any]:
        """
        Get table statistics

        Args:
            table: Table name

        Returns:
            Dictionary with table statistics
        """
        stats_query = f"""
        SELECT
            count() as row_count,
            sum(bytes_on_disk) as bytes_on_disk,
            sum(data_compressed_bytes) as compressed_bytes,
            sum(data_uncompressed_bytes) as uncompressed_bytes,
            min(min_date) as min_date,
            max(max_date) as max_date
        FROM system.parts
        WHERE database = '{self.database}'
            AND table = '{table}'
            AND active
        """

        result = self.query(stats_query)
        if result:
            row = result[0]
            return {
                'row_count': row[0],
                'bytes_on_disk': row[1],
                'compressed_bytes': row[2],
                'uncompressed_bytes': row[3],
                'compression_ratio': row[3] / row[2] if row[2] > 0 else 0,
                'min_date': row[4],
                'max_date': row[5],
                'size_mb': row[1] / (1024 * 1024) if row[1] else 0
            }
        return {}

    def optimize_table(self, table: str, final: bool = False) -> bool:
        """
        Optimize table (merge parts)

        Args:
            table: Table name
            final: Force final merge

        Returns:
            True if successful
        """
        try:
            query = f"OPTIMIZE TABLE {table}"
            if final:
                query += " FINAL"

            self.execute(query)
            logger.info(f"Table {table} optimized successfully")
            return True

        except Exception as e:
            logger.error(f"Error optimizing table: {e}")
            return False

    def truncate_table(self, table: str) -> bool:
        """
        Truncate table (delete all data)

        Args:
            table: Table name

        Returns:
            True if successful
        """
        try:
            self.execute(f"TRUNCATE TABLE {table}")
            logger.info(f"Table {table} truncated successfully")
            return True

        except Exception as e:
            logger.error(f"Error truncating table: {e}")
            return False

    def drop_table(self, table: str) -> bool:
        """
        Drop table

        Args:
            table: Table name

        Returns:
            True if successful
        """
        try:
            self.execute(f"DROP TABLE IF EXISTS {table}")
            logger.info(f"Table {table} dropped successfully")
            return True

        except Exception as e:
            logger.error(f"Error dropping table: {e}")
            return False

    def test_connection(self) -> bool:
        """
        Test database connection

        Returns:
            True if connection successful
        """
        try:
            result = self.query("SELECT 1")
            logger.info("ClickHouse connection test successful")
            return bool(result)

        except Exception as e:
            logger.error(f"ClickHouse connection test failed: {e}")
            return False

    def close(self):
        """Close the connection"""
        if self._client:
            self._client.disconnect()
            self._client = None
            logger.info("ClickHouse connection closed")


# Singleton instance
_clickhouse_manager = None


def get_clickhouse_manager() -> ClickHouseManager:
    """
    Get singleton ClickHouse manager instance

    Returns:
        ClickHouseManager instance
    """
    global _clickhouse_manager
    if _clickhouse_manager is None:
        _clickhouse_manager = ClickHouseManager()
    return _clickhouse_manager


# Convenience functions
def query_clickhouse(query: str, params: Optional[Dict] = None) -> pd.DataFrame:
    """Quick function to query ClickHouse and get DataFrame"""
    manager = get_clickhouse_manager()
    return manager.query_df(query, params)


def insert_to_clickhouse(table: str, df: Union[pd.DataFrame, pl.DataFrame]) -> int:
    """Quick function to insert DataFrame to ClickHouse"""
    manager = get_clickhouse_manager()
    return manager.insert_df(table, df)