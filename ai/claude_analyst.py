"""
Claude интеграция специально для ClickHouse аналитики
"""
from anthropic import Anthropic
import pandas as pd
from typing import Dict, Any, Optional
import json
from database.clickhouse_client import clickhouse
from config.settings import settings


class ClaudeAnalyst:
    def __init__(self):
        self.client = Anthropic(api_key=settings.CLAUDE_API_KEY) if settings.CLAUDE_API_KEY else None
        self.ch = clickhouse

    def analyze_with_sql(self, user_question: str) -> Dict[str, Any]:
        """Генерация SQL и анализ результатов"""

        if not self.client:
            return {"error": "Claude API key not configured"}

        # Получаем схему таблиц
        schema = self.ch.execute("""
            SELECT table, name, type 
            FROM system.columns 
            WHERE database = 'card_analytics'
            AND table IN ('transactions', 'companies', 'daily_metrics')
        """)

        schema_text = "\n".join([f"{t[0]}.{t[1]} ({t[2]})" for t in schema])

        # Генерируем SQL через Claude
        sql_prompt = f"""You are a ClickHouse SQL expert. Generate SQL query for this question:

Question: {user_question}

Available tables and columns:
{schema_text}

ClickHouse specific functions you can use:
- quantile(0.5)(column) for median
- topK(5)(column) for top 5 values  
- uniq(column) for unique count
- countIf(condition) for conditional count
- sumIf(column, condition) for conditional sum

Return ONLY the SQL query, no explanations."""

        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=500,
            messages=[{"role": "user", "content": sql_prompt}]
        )

        sql = response.content[0].text.strip()

        # Выполняем запрос
        try:
            result_df = self.ch.query_df(sql)

            # Анализируем результаты
            analysis = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1000,
                messages=[{
                    "role": "user",
                    "content": f"""Analyze these query results and answer: {user_question}

Results:
{result_df.to_string() if len(result_df) < 100 else result_df.head(100).to_string()}

Provide insights in Russian, be concise and highlight key findings."""
                }]
            )

            return {
                "success": True,
                "sql": sql,
                "data": result_df.to_dict('records'),
                "analysis": analysis.content[0].text,
                "rows": len(result_df)
            }

        except Exception as e:
            return {
                "success": False,
                "sql": sql,
                "error": str(e)
            }