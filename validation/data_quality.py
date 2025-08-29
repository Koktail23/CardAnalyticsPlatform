import polars as pl
import pandera as pa
from pandera import DataFrameSchema, Column, Check
import great_expectations as ge


def validate_data(df_path):
    df = pl.read_csv(df_path).to_pandas()  # Конверт в Pandas для GE/Pandera

    # Pandera схема (базовая валидация типов и чеков)
    schema = DataFrameSchema({
        'hpan': Column(str, Check.str_length(min_value=1), nullable=False),
        'mcc': Column(int, Check.in_range(1000, 9999)),
        'amount_uzs': Column(float, Check.ge(0)),
        # Добавьте чеки для других колонок: 'expire_date': Column('datetime', Check.lt('issue_date', strict=False)),
    })

    try:
        schema.validate(df)
        print("Pandera validation passed")
    except pa.errors.SchemaError as e:
        print(f"Pandera validation failed: {e}")

    # Great Expectations (ожидания)
    gx_df = ge.from_pandas(df)
    expectations = [
        gx_df.expect_column_values_to_not_be_null('hpan'),
        gx_df.expect_column_values_to_be_between('amount_uzs', 0, 1e9),
        gx_df.expect_column_values_to_be_in_set('gender', ['М', 'Ж', 'Unknown']),
        # Добавьте больше: expect_column_values_to_be_unique('refnum') и т.д.
    ]

    results = gx_df.validate(
        expectation_suite=ge.core.ExpectationSuite(expectation_suite_name="transactions", expectations=expectations))
    print(results)  # Или сохраните в JSON/report

    return results.success  # True если все ок


if __name__ == "__main__":
    validate_data('data/processed/cleaned_transactions.csv')