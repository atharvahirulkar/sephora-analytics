# CosmeTik

> Multi-database skincare analytics pipeline - PostgreSQL · Neo4j · Qdrant · Hybrid ML Recommender · Streamlit

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14+-336791?style=flat-square&logo=postgresql)
![Neo4j](https://img.shields.io/badge/Neo4j-5.x-008CC1?style=flat-square&logo=neo4j)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-red?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat-square&logo=streamlit)

---

## Overview

End-to-end data engineering pipeline analyzing relationships between cosmetic ingredients, products, and user skin types across **8,500+ products** and **1M+ reviews** using a three-database architecture.

- **PostgreSQL** - normalized 3NF relational schema for structured product and review data
- **Neo4j** - ingredient co-occurrence graph with PageRank + Louvain community detection
- **Qdrant** - 100K review embeddings (384-dim) for semantic similarity search
- **Hybrid Recommender** - combines graph signals (Neo4j PageRank) + semantic signals (Qdrant cosine similarity)

---

## Architecture

```
Raw CSV Data (Kaggle - 1M+ reviews, 8.5K products)
        │
        ▼
Python ETL Pipeline (Pandas)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│                                                       │
│  PostgreSQL            Neo4j              Qdrant      │
│  ──────────            ─────              ──────      │
│  Normalized            Ingredient         100K review │
│  relational            co-occurrence      embeddings  │
│  schema (3NF)          graph              (384-dim)   │
│  + SQL analysis        PageRank +         semantic    │
│  1M+ reviews           30 communities     search      │
│                                                       │
└───────────────────────────────────────────────────────┘
        │
        ▼
Hybrid ML Recommender
(Graph signals + Semantic signals)
        │
        ▼
Streamlit Web Application
(Live semantic search + ingredient-aware recommendations)
```

---

## Key Findings

| Finding | Detail |
|---------|--------|
| **Most universal ingredient** | Glycerin - appears in 1,005 products, ranked #1 by PageRank |
| **30 ingredient communities** | Louvain algorithm independently discovered functional groups (fragrance, hydration, emollient) without chemical knowledge |
| **Top preservative** | Phenoxyethanol (645 products) - always co-occurs with citric acid |
| **Semantic search quality** | Cosine similarity scores 0.70+ across 100K stratified embeddings |
| **Hybrid recommender** | 50/50 weighted combination of ingredient PageRank score + review semantic similarity |

---

## Database Statistics

| Database | Contents |
|----------|----------|
| PostgreSQL | 8,494 products · 8,156 ingredients · 81,486 product-ingredient relationships · 1,094,411 reviews |
| Neo4j | 8,156 ingredient nodes · 2,351 product nodes · 162,972 `FOUND_IN` edges · PageRank + community scores |
| Qdrant | 100,000 embeddings · stratified 20K per rating level (1–5) · 384 dimensions · Docker server |

---

## Normalized Schema (3NF)

```
products
├── product_id (PK)
├── product_name
├── brand_name
├── price_usd
└── primary_category

ingredients
├── ingredient_id (PK)
└── ingredient_name

product_ingredients          reviews
├── product_id (FK)          ├── review_id (PK)
└── ingredient_id (FK)       ├── product_id (FK)
                             ├── skin_type
product_skin_types           ├── rating
├── product_id (FK)          └── review_text
└── skin_type
```

---

## ML Recommendation Engine

Three-strategy hybrid using all three databases:

**Strategy 1 - Ingredient-Based (PostgreSQL + Neo4j)**
- Fetches shared ingredients from PostgreSQL
- Weights by Neo4j PageRank score - higher PageRank = more influential shared ingredient

**Strategy 2 - Review-Based (Qdrant)**
- Embeds product reviews with `all-MiniLM-L6-v2`
- Averages into a single product vector
- Nearest neighbours via cosine similarity

**Strategy 3 - Hybrid (Final)**
```
hybrid_score = 0.5 × ingredient_score + 0.5 × semantic_score
```
Optional skin type filter applied post-scoring.

---

## Graph Analysis

PageRank applied to ingredient co-occurrence graph (two ingredients connected if they appear in the same product):

| Community | Ingredients |
|-----------|------------|
| Community 0 - Fragrance | limonene, linalool, citronellol |
| Community 3 - Hydration | butylene glycol, sodium hyaluronate |
| Community 6 - Core formulation | glycerin, phenoxyethanol, citric acid |

30 communities total detected without any domain knowledge input.

---

## Streamlit Application

Live semantic search and recommendation interface:

- Free-text search - any concern, ingredient, or product type
- Skin type filter - dry, oily, combination, normal, sensitive
- **Read Reviews** - semantically matched reviews from Qdrant, enriched with full text from PostgreSQL
- **Find Similar** - full hybrid Neo4j + Qdrant pipeline on any result
- Allergy filter - flag or hide products containing specified ingredients

```bash
streamlit run app.py
# → http://localhost:8501
```

---

## Repository Structure

```
cosmetik/
├── app.py                          # Streamlit web application
├── src/
│   └── recommender.py              # Hybrid recommendation engine
├── notebooks/
│   ├── 00_etl.ipynb                # Raw data cleaning + normalization
│   ├── 01_ingest.ipynb             # PostgreSQL ingestion + constraints
│   ├── 02_sql_analysis.ipynb       # SQL queries + analysis
│   ├── 03_neo4j.ipynb              # Graph loading + PageRank + communities
│   ├── 04_qdrant.ipynb             # Vector indexing + semantic search
│   ├── 05_recommender.ipynb        # Hybrid ML recommender
│   └── 06_summary.ipynb            # Final findings + visualizations
├── data/
│   ├── raw/                        # Kaggle CSVs (gitignored)
│   └── processed/                  # Normalized CSVs
│       ├── products.csv
│       ├── ingredients.csv
│       ├── product_ingredients.csv
│       ├── product_skin_types.csv
│       ├── reviews.csv
│       └── neo4j_edges.csv
├── reports/                        # Generated plots
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

```bash
git clone https://github.com/atharvahirulkar/cosmetik.git
cd cosmetik
python -m venv venv && source venv/bin/activate
pip install pandas sqlalchemy psycopg2-binary neo4j qdrant-client \
            sentence-transformers networkx python-louvain \
            matplotlib seaborn plotly scikit-learn \
            python-dotenv streamlit
cp .env.example .env   # fill in credentials
```

Start services:

```bash
# PostgreSQL
brew services start postgresql

# Neo4j
neo4j start

# Qdrant
docker run -d --name qdrant -p 6333:6333 \
  -v ~/cosmetik/data/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

Run notebooks in order `00 → 06`, then:

```bash
streamlit run app.py
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ETL | Python · Pandas |
| Relational | PostgreSQL 14+ · SQLAlchemy |
| Graph | Neo4j 5.x · NetworkX · python-louvain |
| Vector | Qdrant · Sentence Transformers (`all-MiniLM-L6-v2`) |
| Application | Streamlit |
| Visualization | Matplotlib · Seaborn · Plotly |

---

## Dataset

[Sephora Products and Skincare Reviews - Kaggle](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)
- 8K+ products · ~1M reviews · ingredients · skin type labels
