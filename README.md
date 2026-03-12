# Sephora Skincare Analytics - Multi-Database Data Engineering Pipeline

A production-grade data engineering project that analyzes relationships between cosmetic ingredients, products, and user skin types using a **multi-database architecture** combining PostgreSQL, Neo4j, and Qdrant.

---

## Architecture

```
Raw CSV Data (Kaggle - 1M+ reviews, 8k+ products)
        ↓
Python ETL Pipeline (Pandas)
        ↓
┌─────────────────────────────────────────────────────┐
│                                                     │
│  PostgreSQL          Neo4j            Qdrant        │
│  ──────────          ─────            ──────        │
│  Normalized          Ingredient       100k review   │
│  relational          co-occurrence    embeddings    │
│  schema (3NF)        graph +          (384-dim)     │
│  + SQL analysis      PageRank +       semantic      │
│                      communities      search        │
│                                                     │
└─────────────────────────────────────────────────────┘
        ↓
ML Recommendation Engine
(Hybrid: Graph signals + Semantic signals)
```

---

## Research Questions

| # | Question | Database Used |
|---|---|---|
| 1 | Which ingredients appear most frequently across products? | PostgreSQL |
| 2 | Which ingredients are associated with multiple skin types? | PostgreSQL |
| 3 | Which ingredients are most influential in the product network? | Neo4j (PageRank) |
| 4 | Can review text support semantic similarity search? | Qdrant |

---

## Key Findings

- **Glycerin** is the most universal skincare ingredient - appears in 1,005 products and ranked #1 by PageRank across the ingredient network
- **30 ingredient communities** detected by Louvain algorithm - independently discovered functional groups (fragrance cluster, hydration cluster, emollient cluster) without any chemical knowledge
- **Phenoxyethanol** is the most common preservative (645 products), always co-occurring with citric acid in the same community
- **Semantic search** retrieves contextually relevant reviews across 100,000 indexed embeddings with cosine similarity scores of 0.70+
- **Hybrid recommender** combines graph-weighted ingredient similarity with semantic review matching to produce explainable product recommendations

---

## Dataset

**Source:** [Sephora Products and Skincare Reviews - Kaggle](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)

| File | Description | Size |
|---|---|---|
| `product_info.csv` | 8,000+ products with ingredients, price, category | ~50MB |
| `sephora_full.csv` | 1,094,411 reviews with skin type, rating, text | ~377MB |

---

## Database Statistics

| Database | Contents |
|---|---|
| PostgreSQL | 411 products, 8,156 ingredients, 81,486 product-ingredient relationships, 1,094,411 reviews |
| Neo4j | 6,347 ingredient nodes, 1,288 product nodes, 568,225 co-occurrence edges, PageRank + community scores |
| Qdrant | 100,000 review embeddings, 384 dimensions, persistent disk storage |

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

product_ingredients
├── product_id (FK → products)
└── ingredient_id (FK → ingredients)

product_skin_types
├── product_id (FK → products)
└── skin_type

reviews
├── review_id (PK)
├── product_id (FK → products)
├── skin_type
├── rating
└── review_text
```

---

## ML Recommendation Engine

Three-strategy hybrid recommender using all 3 databases:

**Strategy 1 - Ingredient Based (PostgreSQL + Neo4j)**
- Fetches product ingredients from PostgreSQL
- Weights shared ingredients by Neo4j PageRank score
- Higher PageRank = more influential shared ingredient

**Strategy 2 - Review Based (Qdrant)**
- Embeds product reviews using `all-MiniLM-L6-v2`
- Averages embeddings into a single product vector
- Finds semantically similar products via cosine similarity

**Strategy 3 - Hybrid (Final)**
- Normalizes both scores to [0, 1]
- Combines with equal weighting (50/50)
- Optionally filters by skin type

```python
hybrid_score = 0.5 × ingredient_score + 0.5 × semantic_score
```

---

## Graph Analysis

**PageRank** applied to ingredient co-occurrence graph:
- Nodes: 6,347 unique ingredients
- Edges: 568,225 co-occurrence relationships
- Two ingredients connected if they appear in the same product

**Louvain Community Detection:**
- 30 communities detected
- Community 0: Fragrance cluster (limonene, linalool, citronellol)
- Community 3: Hydration cluster (butylene glycol, sodium hyaluronate)
- Community 6: Core formulation cluster (glycerin, phenoxyethanol, citric acid)

---

## Project Structure

```
sephora-analytics/
├── data/
│   ├── raw/                    # Original Kaggle CSVs (gitignored)
│   ├── processed/              # Normalized tables
│   │   ├── products.csv
│   │   ├── ingredients.csv
│   │   ├── product_ingredients.csv
│   │   ├── product_skin_types.csv
│   │   ├── reviews.csv
│   │   └── neo4j_edges.csv
│   └── qdrant_storage/         # Persistent Qdrant vectors (gitignored)
├── notebooks/
│   ├── 01_ingest.ipynb         # Data cleaning + PostgreSQL ingestion
│   ├── 02_sql_analysis.ipynb   # SQL queries + advanced analysis
│   ├── 03_neo4j.ipynb          # Graph loading + PageRank + communities
│   ├── 04_qdrant.ipynb         # Vector indexing + semantic search
│   ├── 05_recommender.ipynb    # Hybrid ML recommender
│   └── 06_summary.ipynb        # Final findings across all databases
├── .env.example                # Environment variable template
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.9+
- PostgreSQL 18+
- Neo4j 2025+
- Git

### 1. Clone the repository
```bash
git clone https://github.com/atharvahirulkar/sephora-analytics.git
cd sephora-analytics
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install pandas sqlalchemy psycopg2-binary neo4j qdrant-client \
            sentence-transformers networkx python-louvain \
            matplotlib seaborn scikit-learn python-dotenv
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Edit .env with your credentials
```

`.env` format:
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
PG_CONNECTION=postgresql://your_user@localhost:5432/sephora_db
```

### 5. Start databases
```bash
# PostgreSQL (via Homebrew)
brew services start postgresql

# Neo4j
neo4j start

# Qdrant (Docker)
docker start qdrant

# First time only - pulls and starts Qdrant container
docker run -d --name qdrant -p 6333:6333 -v ~/your-project-path/data/qdrant_storage:/qdrant/storage qdrant/qdrant
```

### 6. Run notebooks in order
```
01_ingest.ipynb       → cleans data + loads PostgreSQL
02_sql_analysis.ipynb → SQL analysis + visualizations
03_neo4j.ipynb        → graph analysis + PageRank
04_qdrant.ipynb       → vector indexing + semantic search
05_recommender.ipynb  → hybrid ML recommender
06_summary.ipynb      → final findings
```

---

## Technologies

| Technology | Role |
|---|---|
| Python 3.14 | ETL pipeline + analysis |
| PostgreSQL 18 | Relational database (3NF schema) |
| Neo4j 2025 | Graph database |
| Qdrant | Vector database |
| SQLAlchemy | PostgreSQL ORM |
| NetworkX | Graph algorithms (PageRank, Louvain) |
| Sentence Transformers | Review embeddings (all-MiniLM-L6-v2) |
| Pandas | Data manipulation |
| Matplotlib + Seaborn | Visualization |
| python-dotenv | Credential management |

---

## Environment Variables

| Variable | Description |
|---|---|
| `PG_CONNECTION` | PostgreSQL connection string |
| `NEO4J_URI` | Neo4j bolt URI |
| `NEO4J_USER` | Neo4j username |
| `NEO4J_PASSWORD` | Neo4j password |

---

## Course Context

Built as a graduate-level Data Management course project at Universtity of California - San Diego.

Demonstrates: relational database design, data normalization, graph modeling,
vector search, and ML-powered recommendation systems.
