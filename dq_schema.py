#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Quality Schema definitions and business rules for transaction data validation.
"""
from __future__ import annotations
import datetime as _dt
from typing import List

import pandas as pd

try:
    import pandera.pandas as pa
    from pandera import Column, Check

    HAS_PANDERA = True
except ImportError:
    HAS_PANDERA = False
    print("[WARN] Pandera not installed")

if HAS_PANDERA:
    TransactionsSchema = pa.DataFrameSchema(
        {
            # Card/transaction identifiers
            "hpan": Column(
                pa.String,
                required=False,
                nullable=True,
                coerce=True
            ),
            "transaction_code": Column(
                pa.String,
                required=False,
                nullable=True,
                coerce=True
            ),

            # Date/time fields
            "rday": Column(
                pa.Int,
                required=False,
                nullable=True,
                coerce=True,
                checks=Check.in_range(min_value=0, max_value=500000, name="valid_rday_range")
            ),
            "transaction_date": Column(
                pa.DateTime,
                required=False,
                nullable=True,
                coerce=True,
                checks=Check.le(_dt.datetime.now() + _dt.timedelta(days=1), name="not_future_date")
            ),

            # Amount fields
            "amount_uzs": Column(
                pa.Float,
                required=False,
                nullable=True,
                coerce=True,
                checks=Check.greater_than_or_equal_to(0, name="non_negative_amount")
            ),
            "reqamt": Column(
                pa.Float,
                required=False,
                nullable=True,
                coerce=True
            ),
            "conamt": Column(
                pa.Float,
                required=False,
                nullable=True,
                coerce=True
            ),

            # Merchant fields
            "mcc": Column(
                pa.Int,
                required=False,
                nullable=True,
                coerce=True,
                checks=Check.in_range(min_value=0, max_value=9999, name="valid_mcc_range")
            ),
            "merchant_name": Column(
                pa.String,
                required=False,
                nullable=True,
                coerce=True
            ),
            "merchant_type": Column(
                pa.String,
                required=False,
                nullable=True,
                coerce=True
            ),
            "merchant": Column(
                pa.Int,
                required=False,
                nullable=True,
                coerce=True,
                checks=Check.greater_than_or_equal_to(0, name="valid_merchant_id")
            ),

            # P2P fields
            "p2p_flag": Column(
                pa.Int,
                required=False,
                nullable=True,
                coerce=True,
                checks=Check.isin([0, 1], name="binary_p2p_flag")
            ),
            "p2p_type": Column(
                pa.String,
                required=False,
                nullable=True,
                coerce=True
            ),

            # Response/status
            "respcode": Column(
                pa.String,
                required=False,
                nullable=True,
                coerce=True,
                checks=Check.str_length(min_value=0, max_value=8, name="valid_respcode_length")
            ),

            # Hour field
            "hour_num": Column(
                pa.Int,
                required=False,
                nullable=True,
                coerce=True,
                checks=Check.in_range(min_value=0, max_value=23, name="valid_hour_range")
            ),

            # Bank field
            "emitent_bank": Column(
                pa.String,
                required=False,
                nullable=True,
                coerce=True
            ),
        },
        coerce=True,
        strict=False,  # Allow extra columns not in schema
    )


def subset_schema_for(df_cols: List[str]) -> pa.DataFrameSchema:
    """
    Create a schema only for columns that actually exist in the DataFrame.
    This prevents validation errors for missing columns.
    """
    if not HAS_PANDERA:
        return None

    # Get only columns that exist in both schema and dataframe
    cols = {
        col_name: col_obj
        for col_name, col_obj in TransactionsSchema.columns.items()
        if col_name in df_cols
    }

    return pa.DataFrameSchema(cols, coerce=True, strict=False)


def business_checks(df) -> List[str]:
    """
    Perform business logic checks on the data.
    Returns list of issues found.
    """
    issues: List[str] = []

    # Check 1: Negative amounts
    if "amount_uzs" in df.columns:
        try:
            neg_count = int((df["amount_uzs"].fillna(0) < 0).sum())
            if neg_count > 0:
                issues.append(f"amount_uzs: {neg_count:,} negative values (should be >= 0)")
        except Exception:
            pass

    # Check 2: Invalid hours
    if "hour_num" in df.columns:
        try:
            # Handle both numeric and string hour_num
            hour_series = df["hour_num"]
            if hour_series.dtype == 'object':
                # Try to convert string to numeric
                hour_numeric = pd.to_numeric(hour_series, errors='coerce')
            else:
                hour_numeric = hour_series

            bad_hour = int((~hour_numeric.between(0, 23, inclusive='both')).sum())
            if bad_hour > 0:
                issues.append(f"hour_num: {bad_hour:,} values outside [0..23] range")
        except Exception:
            pass

    # Check 3: Both MCC and merchant_name empty
    if "mcc" in df.columns and "merchant_name" in df.columns:
        try:
            both_empty = int(
                ((df["mcc"].fillna(0) == 0) &
                 (df["merchant_name"].fillna('').str.strip() == '')).sum()
            )
            if both_empty > 0:
                issues.append(f"MCC and merchant_name: {both_empty:,} rows with both empty")
        except Exception:
            pass

    # Check 4: P2P flag set but no type
    if "p2p_flag" in df.columns and "p2p_type" in df.columns:
        try:
            bad_p2p = int(
                ((df["p2p_flag"].fillna(0) == 1) &
                 (df["p2p_type"].fillna('').str.strip() == '')).sum()
            )
            if bad_p2p > 0:
                issues.append(f"p2p_flag=1 but p2p_type empty: {bad_p2p:,} rows")
        except Exception:
            pass

    # Check 5: Response code too long
    if "respcode" in df.columns:
        try:
            too_long = int(
                (df["respcode"].fillna('').astype(str).str.len() > 8).sum()
            )
            if too_long > 0:
                issues.append(f"respcode: {too_long:,} values longer than 8 characters")
        except Exception:
            pass

    # Check 6: Suspicious amount patterns
    if "amount_uzs" in df.columns:
        try:
            # Check for too many exact duplicates of large amounts
            amounts = df["amount_uzs"].fillna(0)
            large_amounts = amounts[amounts > 1000000]  # Amounts over 1M UZS
            if len(large_amounts) > 0:
                value_counts = large_amounts.value_counts()
                suspicious = value_counts[value_counts > 10]  # Same large amount appears > 10 times
                if len(suspicious) > 0:
                    issues.append(f"Suspicious pattern: {len(suspicious)} large amounts repeated >10 times")
        except Exception:
            pass

    # Check 7: Date consistency
    if "transaction_date" in df.columns:
        try:
            # Convert to datetime if not already
            dates = pd.to_datetime(df["transaction_date"], errors='coerce')

            # Check for future dates
            future = dates > pd.Timestamp.now()
            if future.sum() > 0:
                issues.append(f"transaction_date: {future.sum():,} future dates detected")

            # Check for very old dates (before 2000)
            very_old = dates < pd.Timestamp('2000-01-01')
            if very_old.sum() > 0:
                issues.append(f"transaction_date: {very_old.sum():,} dates before year 2000")
        except Exception:
            pass

    # Check 8: MCC validity
    if "mcc" in df.columns:
        try:
            # Common valid MCC codes (simplified list)
            common_mcc = {
                5411, 5541, 5812, 5912, 5999,  # Retail/Food
                6011, 6012,  # Financial
                4814, 4815,  # Telecom
                7011, 7512,  # Travel
                8011, 8021,  # Professional
            }

            mcc_values = df["mcc"].dropna()
            if len(mcc_values) > 0:
                # Check if majority of MCCs are in reasonable range
                valid_range = (mcc_values > 0) & (mcc_values < 10000)
                if valid_range.mean() < 0.95:  # Less than 95% in valid range
                    issues.append(f"MCC: {(~valid_range).sum():,} values outside [1..9999] range")
        except Exception:
            pass

    return issues