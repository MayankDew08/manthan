# Manthan: LLM-Driven Workflow Pipeline

Manthan is an automated pipeline for ingesting, grading, enriching, and storing data derived from chat communications. It processes raw chat exports (currently supporting WhatsApp) to extract insights, resources, and topics, leveraging LLMs for automated analysis and classification.

## Key Features

- **Automated Ingestion**: Parses raw chat exports.
- **Grading & Classification**: Uses LLMs to grade message quality and categorize them into topics.
- **Link Enrichment**: Extracts links, scrapes content, and summarizes them for knowledge graph storage.
- **Knowledge Graph Storage**: Stores structured entities, topics, and relationships in Neo4j and Qdrant.

## Metrics Overview

The pipeline's performance is tracked via comprehensive metric reports.

### Actual Data Performance (2026-08-18)
- **Ingestion**: 146 messages parsed, 131 kept.
- **Grading**: Strong quality distribution (Mean confidence: 0.893).
- **Link Processing**: 57 total links extracted from 49 records.
- **Storage**: 116+ relationships in Neo4j, 16+ vector points in Qdrant.

### Synthetic Data Performance (2026-08-17)
- **Ingestion**: 44 messages parsed, 30 kept.
- **Grading**: High confidence metrics (Mean confidence: 0.947).
- **Link Processing**: 9 total links extracted from 8 records.

*(See `metrics/manthan_report.md` and `metrics/synthetic_report.md` for full details.)*

## Getting Started

1.  **Environment Setup**: Install dependencies using `uv`. Ensure `.env` is configured for necessary APIs (LLMs, Databases).
2.  **Pipeline Execution**: Run the ingestion and processing scripts (e.g., `main.py`).
3.  **Analysis**: Generate reports using `scripts/metrics_report.py`.
