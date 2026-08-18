# Manthan: Local-First LLM-Driven Workflow Pipeline

Manthan is a **local-first, automated pipeline** designed to ingest, grade, enrich, and map knowledge derived from chat communications (initially WhatsApp). By running LLM analysis locally, it transforms fragmented chat history into structured insights, resources, and relationships without relying on external cloud processing.

## Core Philosophy
- **Privacy-First**: All LLM processing occurs locally. No chat history leaves your machine.
- **Local-First Knowledge**: Insights are stored in local graph (Neo4j) and vector (Qdrant) databases.
- **Reproducible**: Every pipeline run generates metric reports, allowing for performance tracking across different LLM models and versions.

## Pipeline Performance Metrics

We track our pipeline's health through automated metric generation.

| Metric | Actual Run (2026-08-18) | Synthetic Run (2026-08-17) |
| :--- | :--- | :--- |
| **Ingestion** | 146 msgs (131 kept) | 44 msgs (30 kept) |
| **Grading Confidence (Mean)** | **0.893** | **0.947** |
| **Enriched Records** | 49 | 8 |
| **Total Links Extracted** | 57 | 9 |
| **Relationship Density** | 116 relationships | N/A (Node Error) |

*(Full reports available in `metrics/manthan_report.md` and `metrics/synthetic_report.md`)*

## Architecture

1.  **Ingestion Layer**: Sanitizes and parses raw chat exports.
2.  **Grading Engine**: Uses locally hosted LLMs to grade message quality (1–5) and extract topical categories.
3.  **Enrichment Pipeline**: Extracts links, scrapes URL content, and generates summaries.
4.  **Graph Storage**:
    *   **Neo4j**: Maps entities (topics, people, organizations, links).
    *   **Qdrant**: Vector-based semantic search across all enriched records.

## Local Setup

### Prerequisites
- `uv` for environment/dependency management.
- Local LLM backend (e.g., llama.cpp).
- Databases: Neo4j (Graph), Qdrant (Vector).

### Quick Start
1.  **Configure**: Clone `.env.example` to `.env` and set your local DB endpoints.
2.  **Sync**: Run `uv sync` to install dependencies.
3.  **Process**: Execute `python main.py` to run the full pipeline.
4.  **Evaluate**: Run `python scripts/metrics_report.py` to regenerate the metrics report after a run.

## License & Privacy
Manthan is designed for individual knowledge management. As a local-first tool, ensure your own local databases (Neo4j/Qdrant) are secured according to your data sensitivity requirements.
