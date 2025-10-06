#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced RAG Helpers - Caching and optimization utilities for RAG system
"""

import os
import hashlib
import joblib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class RAGCache:
    """Advanced caching system for RAG with TF-IDF matrix persistence."""

    def __init__(self, cache_dir: str = "./reports/.rag_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_metadata_file = self.cache_dir / "cache_metadata.json"

    def get_cache_key(self, files: List[Path], config: Dict) -> str:
        """Generate unique cache key based on files and configuration."""
        # Create hash from file metadata
        file_info = []
        for f in files:
            if f.exists():
                file_info.append({
                    "name": f.name,
                    "size": f.stat().st_size,
                    "mtime": f.stat().st_mtime
                })

        # Include config in hash
        cache_data = {
            "files": file_info,
            "config": config
        }

        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()

    def is_cache_valid(self, cache_key: str, max_age_hours: int = 24) -> bool:
        """Check if cache is valid and not expired."""
        cache_file = self.cache_dir / f"rag_{cache_key}.pkl"

        if not cache_file.exists():
            return False

        # Check age
        age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        if age > timedelta(hours=max_age_hours):
            return False

        # Check metadata
        if self._cache_metadata_file.exists():
            metadata = json.loads(self._cache_metadata_file.read_text())
            if cache_key in metadata:
                return metadata[cache_key].get("valid", False)

        return True

    def save_cache(self, cache_key: str, rag_instance: any) -> bool:
        """Save RAG instance to cache."""
        cache_file = self.cache_dir / f"rag_{cache_key}.pkl"

        try:
            # Save the RAG instance
            joblib.dump(rag_instance, cache_file, compress=3)

            # Update metadata
            metadata = {}
            if self._cache_metadata_file.exists():
                metadata = json.loads(self._cache_metadata_file.read_text())

            metadata[cache_key] = {
                "created": datetime.now().isoformat(),
                "valid": True,
                "file": str(cache_file.name)
            }

            self._cache_metadata_file.write_text(json.dumps(metadata, indent=2))

            # Clean old cache files
            self.cleanup_old_cache(keep_recent=5)

            return True
        except Exception as e:
            print(f"[WARN] Failed to save cache: {e}")
            return False

    def load_cache(self, cache_key: str) -> Optional[any]:
        """Load RAG instance from cache."""
        cache_file = self.cache_dir / f"rag_{cache_key}.pkl"

        if not self.is_cache_valid(cache_key):
            return None

        try:
            return joblib.load(cache_file)
        except Exception as e:
            print(f"[WARN] Failed to load cache: {e}")
            return None

    def cleanup_old_cache(self, keep_recent: int = 5):
        """Remove old cache files, keeping only the most recent ones."""
        cache_files = sorted(
            self.cache_dir.glob("rag_*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        # Remove old files
        for old_file in cache_files[keep_recent:]:
            try:
                old_file.unlink()
                print(f"[INFO] Removed old cache: {old_file.name}")
            except Exception:
                pass

        # Update metadata
        if self._cache_metadata_file.exists():
            metadata = json.loads(self._cache_metadata_file.read_text())
            existing_keys = set()

            for cache_file in cache_files[:keep_recent]:
                # Extract key from filename
                key = cache_file.stem.replace("rag_", "")
                existing_keys.add(key)

            # Remove metadata for deleted files
            for key in list(metadata.keys()):
                if key not in existing_keys:
                    del metadata[key]

            self._cache_metadata_file.write_text(json.dumps(metadata, indent=2))


class OptimizedRetriever:
    """Optimized retriever with query caching and batch processing."""

    def __init__(self, cache_dir: str = "./reports/.rag_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._query_cache = {}
        self._similarity_cache = {}

    def cache_query_result(self, query: str, chunks: List[Tuple], ttl_seconds: int = 300):
        """Cache query results with TTL."""
        cache_key = hashlib.md5(query.encode()).hexdigest()
        self._query_cache[cache_key] = {
            "chunks": chunks,
            "timestamp": datetime.now(),
            "ttl": ttl_seconds
        }

        # Clean expired entries
        self._clean_expired_cache()

    def get_cached_query(self, query: str) -> Optional[List[Tuple]]:
        """Get cached query result if available and not expired."""
        cache_key = hashlib.md5(query.encode()).hexdigest()

        if cache_key in self._query_cache:
            entry = self._query_cache[cache_key]
            age = (datetime.now() - entry["timestamp"]).total_seconds()

            if age < entry["ttl"]:
                return entry["chunks"]
            else:
                del self._query_cache[cache_key]

        return None

    def _clean_expired_cache(self):
        """Remove expired entries from cache."""
        now = datetime.now()
        expired_keys = []

        for key, entry in self._query_cache.items():
            age = (now - entry["timestamp"]).total_seconds()
            if age >= entry["ttl"]:
                expired_keys.append(key)

        for key in expired_keys:
            del self._query_cache[key]

    def batch_similarity_search(self,
                                queries: List[str],
                                vectorizer: TfidfVectorizer,
                                doc_vectors: np.ndarray,
                                top_k: int = 5) -> Dict[str, List[int]]:
        """Batch similarity search for multiple queries."""
        try:
            # Transform all queries at once
            query_vectors = vectorizer.transform(queries)

            # Compute similarities for all queries
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(query_vectors, doc_vectors)

            # Get top-k for each query
            results = {}
            for i, query in enumerate(queries):
                sim_scores = similarities[i]
                top_indices = np.argsort(sim_scores)[-top_k:][::-1]
                results[query] = top_indices.tolist()

            return results

        except Exception as e:
            print(f"[WARN] Batch similarity search failed: {e}")
            return {}


class SmartChunker:
    """Smart document chunking with semantic boundaries."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_with_context(self, text: str, doc_type: str = "generic") -> List[Dict]:
        """Create chunks with preserved context and metadata."""
        chunks = []

        if doc_type == "weekly_report":
            # Split by sections for reports
            sections = text.split("\n## ")
            for i, section in enumerate(sections):
                if section.strip():
                    # Keep section header
                    if i > 0:
                        section = "## " + section

                    # Further split if section is too long
                    if len(section) > self.chunk_size:
                        sub_chunks = self._split_with_overlap(section)
                        for j, sub_chunk in enumerate(sub_chunks):
                            chunks.append({
                                "text": sub_chunk,
                                "section": i,
                                "subsection": j,
                                "type": doc_type
                            })
                    else:
                        chunks.append({
                            "text": section,
                            "section": i,
                            "subsection": 0,
                            "type": doc_type
                        })

        elif doc_type == "table" or doc_type == "csv":
            # Keep table rows together
            lines = text.split("\n")
            current_chunk = []
            current_size = 0

            for line in lines:
                line_size = len(line)
                if current_size + line_size > self.chunk_size and current_chunk:
                    chunks.append({
                        "text": "\n".join(current_chunk),
                        "type": doc_type
                    })
                    # Keep overlap
                    if self.overlap > 0:
                        overlap_lines = int(len(current_chunk) * 0.1)
                        current_chunk = current_chunk[-overlap_lines:] if overlap_lines > 0 else []
                        current_size = sum(len(l) for l in current_chunk)
                    else:
                        current_chunk = []
                        current_size = 0

                current_chunk.append(line)
                current_size += line_size

            if current_chunk:
                chunks.append({
                    "text": "\n".join(current_chunk),
                    "type": doc_type
                })

        else:
            # Generic chunking with overlap
            chunks_list = self._split_with_overlap(text)
            for i, chunk_text in enumerate(chunks_list):
                chunks.append({
                    "text": chunk_text,
                    "position": i,
                    "type": doc_type
                })

        return chunks

    def _split_with_overlap(self, text: str) -> List[str]:
        """Split text with overlap between chunks."""
        chunks = []
        words = text.split()

        if not words:
            return []

        # Calculate words per chunk
        avg_word_len = sum(len(w) for w in words[:100]) / min(100, len(words))
        words_per_chunk = int(self.chunk_size / (avg_word_len + 1))  # +1 for spaces

        # Create chunks with overlap
        start = 0
        while start < len(words):
            end = start + words_per_chunk
            chunk = " ".join(words[start:end])
            chunks.append(chunk)

            # Move start with overlap
            if self.overlap > 0:
                overlap_words = int(words_per_chunk * (self.overlap / self.chunk_size))
                start = end - overlap_words
            else:
                start = end

        return chunks


class QueryExpander:
    """Query expansion for better retrieval."""

    def __init__(self):
        self.synonyms = {
            "fraud": ["мошенничество", "фрод", "подозрительный"],
            "transaction": ["транзакция", "платеж", "операция"],
            "volume": ["объем", "объём", "сумма", "amount"],
            "merchant": ["мерчант", "продавец", "магазин"],
            "card": ["карта", "карточка", "hpan"],
            "trend": ["тренд", "динамика", "изменение"],
            "risk": ["риск", "угроза", "проблема"],
            "quality": ["качество", "валидность", "корректность"],
            "p2p": ["п2п", "peer-to-peer", "перевод"],
            "mcc": ["мкк", "merchant category code"],
        }

    def expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms."""
        query_lower = query.lower()
        expanded_terms = []

        # Add original query
        expanded_terms.append(query)

        # Add synonyms
        for key, synonyms in self.synonyms.items():
            if key in query_lower:
                for syn in synonyms:
                    if syn not in query_lower:
                        expanded_terms.append(syn)

        # Add date variations if dates are mentioned
        import re
        date_patterns = [
            r'\b\d{4}-\d{2}-\d{2}\b',
            r'\b\d{1,2}\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\b',
            r'\b(понедельник|вторник|среда|четверг|пятница|суббота|воскресенье)\b',
            r'\b(вчера|сегодня|завтра|позавчера)\b',
        ]

        for pattern in date_patterns:
            if re.search(pattern, query_lower):
                expanded_terms.extend(["дата", "период", "время"])
                break

        return " ".join(expanded_terms)


class CitationManager:
    """Manage citations and source references."""

    def __init__(self):
        self.citations = []
        self.source_map = {}

    def add_citation(self, text: str, source: Dict, relevance: float) -> str:
        """Add citation mark to text."""
        citation_id = len(self.citations) + 1
        self.citations.append({
            "id": citation_id,
            "source": source,
            "relevance": relevance,
            "text_snippet": text[:100]
        })

        # Store source mapping
        source_key = f"{source.get('doc_id', 'unknown')}_{source.get('section', 0)}"
        if source_key not in self.source_map:
            self.source_map[source_key] = []
        self.source_map[source_key].append(citation_id)

        return f'<span class="citation">[{citation_id}]</span>'

    def format_citations_html(self) -> str:
        """Format citations as HTML."""
        if not self.citations:
            return ""

        html = '<div class="citations-container">\n'
        html += '<h4>📌 Источники:</h4>\n'
        html += '<ol class="citations-list">\n'

        for citation in self.citations:
            source = citation['source']
            html += f'<li class="citation-item">\n'
            html += f'  <span class="citation-id">[{citation["id"]}]</span>\n'
            html += f'  <span class="citation-doc">{source.get("doc_id", "unknown")}</span>\n'
            if 'section' in source:
                html += f'  <span class="citation-section">§{source["section"]}</span>\n'
            html += f'  <span class="citation-relevance">({citation["relevance"]:.2f})</span>\n'
            html += '</li>\n'

        html += '</ol>\n'
        html += '</div>\n'

        return html

    def format_citations_markdown(self) -> str:
        """Format citations as Markdown."""
        if not self.citations:
            return ""

        md = "\n### 📌 Источники:\n\n"

        for citation in self.citations:
            source = citation['source']
            doc_id = source.get("doc_id", "unknown")
            section = f"§{source['section']}" if 'section' in source else ""
            relevance = f"(релевантность: {citation['relevance']:.2f})"

            md += f"- [{citation['id']}] {doc_id} {section} {relevance}\n"

        return md


# Export main classes
__all__ = [
    'RAGCache',
    'OptimizedRetriever',
    'SmartChunker',
    'QueryExpander',
    'CitationManager'
]

if __name__ == "__main__":
    # Test caching
    cache = RAGCache()

    # Test files
    test_files = [
        Path("./reports/weekly_report_20250101.md"),
        Path("./reports/dq_report_20250101.md")
    ]

    config = {"weeks": 4, "chunk_size": 500}

    key = cache.get_cache_key(test_files, config)
    print(f"Cache key: {key}")

    # Test chunking
    chunker = SmartChunker(chunk_size=200, overlap=20)

    test_text = """
    ## Section 1
    This is a test section with some content.
    It has multiple lines and paragraphs.

    ## Section 2
    Another section with different content.
    This section is longer and contains more detailed information.
    We want to test how the chunker handles different section sizes.
    """

    chunks = chunker.chunk_with_context(test_text, doc_type="weekly_report")
    print(f"\nCreated {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i}: {chunk['text'][:50]}...")

    # Test query expansion
    expander = QueryExpander()

    test_queries = [
        "fraud detection metrics",
        "p2p transaction volume",
        "качество данных за вчера"
    ]

    print("\nQuery expansion:")
    for query in test_queries:
        expanded = expander.expand_query(query)
        print(f"Original: {query}")
        print(f"Expanded: {expanded}\n")