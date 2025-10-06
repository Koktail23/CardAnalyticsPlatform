#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced RAG system with:
- Multi-document context (last N reports)
- TF-IDF based retrieval for relevant chunks
- Source citations
- Structured responses
- Trend awareness
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class EnhancedRAG:
    """Enhanced RAG system with semantic retrieval and citations."""

    def __init__(self, reports_dir: str = "./reports", models_dir: str = "./ml"):
        self.reports_dir = Path(reports_dir)
        self.models_dir = Path(models_dir)
        self.vectorizer = TfidfVectorizer(
            max_features=500,
            ngram_range=(1, 2),
            stop_words='english',
            min_df=1,
            max_df=0.95
        )
        self.documents = []
        self.document_chunks = []
        self.chunk_sources = []

    def load_documents(self, last_n_weeks: int = 4) -> Dict[str, str]:
        """Load multiple document types for context."""
        docs = {}

        # 1. Load last N weekly reports
        weekly_reports = sorted(
            self.reports_dir.glob("weekly_report_*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )[:last_n_weeks]

        for i, report in enumerate(weekly_reports):
            try:
                content = report.read_text(encoding="utf-8", errors="ignore")
                docs[f"weekly_report_{i}"] = {
                    "content": content,
                    "path": str(report),
                    "type": "weekly_report",
                    "date": report.stem.split("_")[-1]
                }
            except Exception as e:
                print(f"[WARN] Failed to load {report}: {e}")

        # 2. Load latest data dictionary
        dict_files = sorted(
            self.reports_dir.glob("data_dictionary_*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if dict_files:
            try:
                docs["data_dictionary"] = {
                    "content": dict_files[0].read_text(encoding="utf-8", errors="ignore"),
                    "path": str(dict_files[0]),
                    "type": "dictionary"
                }
            except Exception:
                pass

        # 3. Load latest DQ report
        dq_files = sorted(
            self.reports_dir.glob("dq_report_*.md"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
        if dq_files:
            try:
                docs["dq_report"] = {
                    "content": dq_files[0].read_text(encoding="utf-8", errors="ignore"),
                    "path": str(dq_files[0]),
                    "type": "dq_report"
                }
            except Exception:
                pass

        # 4. Load fraud model artifacts
        for name in ["thresholds.json", "fraud_summary.json", "feature_importance.csv"]:
            path = self.models_dir / name
            if path.exists():
                try:
                    if name.endswith(".json"):
                        content = json.dumps(json.loads(path.read_text()), indent=2)
                    else:
                        content = path.read_text(encoding="utf-8", errors="ignore")
                    docs[name.split(".")[0]] = {
                        "content": content,
                        "path": str(path),
                        "type": "model_artifact"
                    }
                except Exception:
                    pass

        # 5. Load MLflow metrics if available
        mlflow_metrics = self._get_mlflow_metrics()
        if mlflow_metrics:
            docs["mlflow_metrics"] = {
                "content": mlflow_metrics,
                "path": "mlflow",
                "type": "metrics"
            }

        return docs

    def _get_mlflow_metrics(self) -> Optional[str]:
        """Get MLflow metrics summary."""
        try:
            import mlflow
            from mlflow.tracking import MlflowClient

            mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
            client = MlflowClient()

            metrics_summary = []
            for exp_name in ["fraud_detection", "forecasting", "segmentation"]:
                try:
                    exp = next((e for e in client.search_experiments() if e.name == exp_name), None)
                    if not exp:
                        continue

                    runs = client.search_runs(
                        exp.experiment_id,
                        order_by=["attributes.start_time DESC"],
                        max_results=3
                    )

                    if runs:
                        metrics_summary.append(f"\n## {exp_name}")
                        for run in runs:
                            metrics_summary.append(
                                f"- Run {run.info.run_id[:8]} ({run.info.start_time}): "
                                f"{', '.join(f'{k}={v:.3f}' for k, v in run.data.metrics.items())}"
                            )
                except Exception:
                    pass

            return "\n".join(metrics_summary) if metrics_summary else None
        except Exception:
            return None

    def chunk_documents(self, chunk_size: int = 500) -> None:
        """Split documents into chunks for retrieval."""
        self.document_chunks = []
        self.chunk_sources = []

        for doc_id, doc_info in self.documents.items():
            content = doc_info["content"]

            # Split by paragraphs/sections
            if doc_info["type"] == "weekly_report":
                sections = re.split(r'\n## ', content)
                for i, section in enumerate(sections):
                    if section.strip():
                        self.document_chunks.append(section[:chunk_size])
                        self.chunk_sources.append({
                            "doc_id": doc_id,
                            "section": i,
                            "type": doc_info["type"],
                            "path": doc_info["path"]
                        })

            elif doc_info["type"] in ["dictionary", "dq_report"]:
                # Split by lines/tables
                lines = content.split("\n")
                chunk = []
                for line in lines:
                    chunk.append(line)
                    if len("\n".join(chunk)) > chunk_size:
                        self.document_chunks.append("\n".join(chunk))
                        self.chunk_sources.append({
                            "doc_id": doc_id,
                            "type": doc_info["type"],
                            "path": doc_info["path"]
                        })
                        chunk = []
                if chunk:
                    self.document_chunks.append("\n".join(chunk))
                    self.chunk_sources.append({
                        "doc_id": doc_id,
                        "type": doc_info["type"],
                        "path": doc_info["path"]
                    })

            else:
                # For JSON/CSV, keep as single chunk
                self.document_chunks.append(content[:chunk_size * 2])
                self.chunk_sources.append({
                    "doc_id": doc_id,
                    "type": doc_info["type"],
                    "path": doc_info["path"]
                })

    def retrieve_relevant_chunks(self, query: str, top_k: int = 5) -> List[Tuple[str, Dict]]:
        """Retrieve most relevant chunks using TF-IDF."""
        if not self.document_chunks:
            return []

        try:
            # Fit vectorizer if needed
            if not hasattr(self.vectorizer, 'vocabulary_'):
                self.vectorizer.fit(self.document_chunks)

            # Transform query and documents
            query_vec = self.vectorizer.transform([query])
            doc_vecs = self.vectorizer.transform(self.document_chunks)

            # Calculate similarities
            similarities = cosine_similarity(query_vec, doc_vecs)[0]

            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            # Return chunks with sources
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    results.append((
                        self.document_chunks[idx],
                        self.chunk_sources[idx]
                    ))

            return results
        except Exception as e:
            print(f"[WARN] Retrieval error: {e}")
            # Fallback: return latest chunks
            return [(chunk, source) for chunk, source in
                    zip(self.document_chunks[:top_k], self.chunk_sources[:top_k])]

    def get_trend_summary(self, table: str = "transactions_optimized") -> str:
        """Generate trend summary for last 4 weeks."""
        try:
            from clickhouse_driver import Client

            client = Client(
                host=os.getenv("CLICKHOUSE_HOST", "localhost"),
                port=int(os.getenv("CLICKHOUSE_PORT", "9000")),
                user=os.getenv("CLICKHOUSE_USER", "analyst"),
                password=os.getenv("CLICKHOUSE_PASSWORD", "admin123"),
                database=os.getenv("CLICKHOUSE_DATABASE", "card_analytics")
            )

            # Get weekly KPIs for last 4 weeks
            query = f"""
            SELECT 
                toMonday(transaction_date) AS week,
                sum(amount_uzs) AS volume,
                count() AS transactions,
                avg(amount_uzs) AS avg_amount
            FROM {table}
            WHERE transaction_date >= toDate(now()) - INTERVAL 28 DAY
            GROUP BY week
            ORDER BY week DESC
            LIMIT 4
            """

            rows = client.execute(query)
            if rows:
                df = pd.DataFrame(rows, columns=["week", "volume", "transactions", "avg_amount"])

                # Calculate trends
                df["volume_change"] = df["volume"].pct_change(-1) * 100
                df["tx_change"] = df["transactions"].pct_change(-1) * 100

                # Format as markdown table
                trend_summary = "## Тренд за 4 недели\n\n"
                trend_summary += "| Неделя | Объём (UZS) | Δ% | Транзакций | Δ% | Средний чек |\n"
                trend_summary += "|--------|-------------|-----|------------|-----|-------------|\n"

                for _, row in df.iterrows():
                    vol_change = f"{row['volume_change']:+.1f}%" if pd.notna(row['volume_change']) else "-"
                    tx_change = f"{row['tx_change']:+.1f}%" if pd.notna(row['tx_change']) else "-"
                    trend_summary += f"| {row['week']:%Y-%m-%d} | {row['volume']:,.0f} | {vol_change} | "
                    trend_summary += f"{row['transactions']:,} | {tx_change} | {row['avg_amount']:,.0f} |\n"

                return trend_summary

        except Exception as e:
            print(f"[WARN] Failed to get trend summary: {e}")

        return ""

    def format_response(
            self,
            raw_response: str,
            sources: List[Dict],
            include_sql: bool = False
    ) -> Dict:
        """Format response with structure and citations."""

        # Try to extract structured parts from response
        insights = []
        risks = []
        actions = []
        sql_query = None

        # Simple parsing (can be improved with better prompting)
        lines = raw_response.split("\n")
        current_section = "general"

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect sections
            if any(word in line.lower() for word in ["insight", "вывод", "показател"]):
                current_section = "insights"
            elif any(word in line.lower() for word in ["risk", "риск", "проблем"]):
                current_section = "risks"
            elif any(word in line.lower() for word in ["action", "рекоменд", "действи"]):
                current_section = "actions"
            elif line.upper().startswith("SELECT"):
                sql_query = line
                current_section = "sql"
            else:
                # Add to appropriate section
                if current_section == "insights" and line.startswith(("-", "•", "*")):
                    insights.append(line.lstrip("-•* "))
                elif current_section == "risks" and line.startswith(("-", "•", "*")):
                    risks.append(line.lstrip("-•* "))
                elif current_section == "actions" and line.startswith(("-", "•", "*")):
                    actions.append(line.lstrip("-•* "))

        # Format sources
        formatted_sources = []
        for source in sources:
            source_str = f"{source.get('doc_id', 'unknown')}"
            if 'section' in source:
                source_str += f" §{source['section']}"
            formatted_sources.append(source_str)

        return {
            "summary": raw_response,
            "insights": insights or ["См. полный ответ выше"],
            "risks": risks or [],
            "actions": actions or [],
            "sql_query": sql_query,
            "sources": formatted_sources,
            "timestamp": datetime.now().isoformat()
        }


def create_enhanced_prompt(
        query: str,
        context_chunks: List[Tuple[str, Dict]],
        trend_summary: str = ""
) -> str:
    """Create enhanced prompt with context and instructions."""

    # Build context from chunks
    context_parts = []
    for i, (chunk, source) in enumerate(context_chunks):
        context_parts.append(f"[Источник {i + 1}: {source['doc_id']}]\n{chunk}\n")

    prompt = f"""
Ты опытный аналитик данных. Ответь на вопрос пользователя, используя ТОЛЬКО предоставленный контекст.

ВАЖНО:
1. Отвечай строго по контексту ниже
2. Структурируй ответ по категориям: Insights (выводы), Risks (риски), Actions (рекомендации)
3. Если нужен SQL - предложи SELECT запрос
4. Если данных недостаточно - честно скажи об этом

Вопрос пользователя: {query}

{"=" * 50}
КОНТЕКСТ:
{"=" * 50}

{trend_summary}

{"".join(context_parts)}

{"=" * 50}

Формат ответа:
- Начни с краткого резюме
- Выдели ключевые инсайты (если есть)
- Укажи риски (если есть)
- Предложи действия (если применимо)
- SQL запрос (если требуется)
"""

    return prompt


# CLI interface for testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Enhanced RAG system")
    parser.add_argument("--query", type=str, required=True, help="User query")
    parser.add_argument("--weeks", type=int, default=4, help="Number of weeks to load")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    args = parser.parse_args()

    # Initialize RAG
    rag = EnhancedRAG()

    # Load documents
    print("[INFO] Loading documents...")
    rag.documents = rag.load_documents(last_n_weeks=args.weeks)
    print(f"[INFO] Loaded {len(rag.documents)} documents")

    # Chunk documents
    print("[INFO] Chunking documents...")
    rag.chunk_documents()
    print(f"[INFO] Created {len(rag.document_chunks)} chunks")

    # Get trend summary
    print("[INFO] Getting trend summary...")
    trend = rag.get_trend_summary()

    # Retrieve relevant chunks
    print(f"[INFO] Retrieving top-{args.top_k} chunks for query...")
    chunks = rag.retrieve_relevant_chunks(args.query, top_k=args.top_k)

    # Create prompt
    prompt = create_enhanced_prompt(args.query, chunks, trend)

    # Output
    print("\n" + "=" * 50)
    print("ENHANCED PROMPT:")
    print("=" * 50)
    print(prompt)

    # Format response (mock)
    mock_response = "Based on the context, here are the findings..."
    result = rag.format_response(mock_response, [c[1] for c in chunks])

    print("\n" + "=" * 50)
    print("STRUCTURED RESPONSE:")
    print("=" * 50)
    print(json.dumps(result, indent=2, ensure_ascii=False))