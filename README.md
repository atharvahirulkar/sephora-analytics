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
        ↓
Streamlit Web Application
(Live semantic search + hybrid recommendations)
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

- **Glycerin** is the most universal skincare ingredient - appears in 1,005 products and ranked #1 by PageRank across the ingredient network.
- **30 ingredient communities** detected by Louvain algorithm - independently discovered functional groups (fragrance cluster, hydration cluster, emollient cluster) without any chemical knowledge.
- **Phenoxyethanol** is the most common preservative (645 products), always co-occurring with citric acid in the same community.
- **Semantic search** retrieves contextually relevant reviews across 100,000 stratified embeddings (balanced across rating levels 1–5) with cosine similarity scores of 0.70+.
- **Hybrid recommender** combines graph-weighted ingredient similarity with semantic review matching to produce explainable product recommendations.

---

## Dataset

**Source:** [Sephora Products and Skincare Reviews - Kaggle](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews)

| File | Description | Size |
|---|---|---|
| `product_info.csv` | 8K+ products with ingredients, price, category | ~50MB |
| `sephora_full.csv` | ~1M reviews with skin type, rating, text | ~377MB |

---

## Database Statistics

| Database | Contents |
|---|---|
| PostgreSQL | 8494 products, 8156 ingredients, 81486 product-ingredient relationships, 1094411 reviews |
| Neo4j | 8156 ingredient nodes, 2351 product nodes, 162972 FOUND_IN edges, PageRank + community scores |
| Qdrant | 100000 review embeddings (stratified - 20000 per rating level, 1–5), 384 dimensions, running via Docker server |

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
- Nodes: 8156 unique ingredients
- Edges: 162972 co-occurrence relationships
- Two ingredients connected if they appear in the same product

**Louvain Community Detection:**
- 30 communities detected
- Community 0: Fragrance cluster (limonene, linalool, citronellol)
- Community 3: Hydration cluster (butylene glycol, sodium hyaluronate)
- Community 6: Core formulation cluster (glycerin, phenoxyethanol, citric acid)

---

## Streamlit Application

A live semantic search and recommendation interface powered by all 3 databases.

**Features:**
- Free text search - type any concern, ingredient, or product type
- Skin type filter - dry, oily, combination, normal, sensitive
- Relevance score, average rating, price, and top ingredients per result
- **Read Reviews** - fetches semantically matched reviews from Qdrant, enriched with full text from PostgreSQL
- **Find Similar** - runs full hybrid Neo4j + Qdrant pipeline on any result
- Allergy filter - flag or hide products containing specified ingredients

**Run the app:**
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## Project Structure
```
sephora-analytics/
├── app.py                      # Streamlit web application
├── src/
│   └── recommender.py          # Reusable recommendation engine
├── data/
│   ├── raw/                    # Original Kaggle CSVs (gitignored)
│   ├── processed/              # Normalized tables
│   │   ├── products.csv
│   │   ├── ingredients.csv
│   │   ├── product_ingredients.csv
│   │   ├── product_skin_types.csv
│   │   ├── reviews.csv
│   │   └── neo4j_edges.csv     # Full ingredient-product export (no LIMIT)
│   └── qdrant_storage/         # Persistent Qdrant vectors (gitignored)
├── notebooks/
│   ├── 00_etl.ipynb            # Raw data cleaning + normalization
│   ├── 01_ingest.ipynb         # PostgreSQL ingestion + constraints
│   ├── 02_sql_analysis.ipynb   # SQL queries + advanced analysis
│   ├── 03_neo4j.ipynb          # Graph loading + PageRank + communities
│   ├── 04_qdrant.ipynb         # Vector indexing + semantic search
│   ├── 05_recommender.ipynb    # Hybrid ML recommender
│   └── 06_summary.ipynb        # Final findings + visualizations
├── reports/                    # Generated plot images
├── .env.example                # Environment variable template
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- PostgreSQL 14+
- Neo4j 5.x
- Docker Desktop

### 1. Clone the repository
```bash
git clone https://github.com/atharvahirulkar/sephora-analytics.git
cd sephora-analytics
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install pandas sqlalchemy psycopg2-binary neo4j qdrant-client \
            sentence-transformers networkx python-louvain \
            matplotlib seaborn plotly scikit-learn \
            python-dotenv streamlit
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

### 5. Start all services
```bash
# PostgreSQL
brew services start postgresql        # macOS
sudo service postgresql start         # Linux

# Neo4j
neo4j start

# Qdrant - first time setup
docker run -d --name qdrant -p 6333:6333 \
  -v ~/your-project-path/data/qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# Qdrant - subsequent runs
docker start qdrant
```

### 6. Run notebooks in order
```
00_etl.ipynb          → cleans raw data + builds normalized CSVs
01_ingest.ipynb       → loads PostgreSQL + adds constraints
02_sql_analysis.ipynb → SQL analysis + visualizations
03_neo4j.ipynb        → graph analysis + PageRank + communities
04_qdrant.ipynb       → vector indexing + semantic search
05_recommender.ipynb  → hybrid ML recommender
06_summary.ipynb      → final findings + report plots
```

### 7. Launch the app
```bash
streamlit run app.py
```

---

## Technologies

| Technology | Role |
|---|---|
| Python 3.10+ | ETL pipeline + analysis |
| PostgreSQL 14+ | Relational database (3NF schema) |
| Neo4j 5.x | Graph database |
| Qdrant (Docker) | Vector database server |
| Streamlit | Web application |
| SQLAlchemy | PostgreSQL ORM |
| NetworkX | Graph algorithms (PageRank, Louvain) |
| Sentence Transformers | Review embeddings (all-MiniLM-L6-v2) |
| Pandas | Data manipulation |
| Matplotlib + Seaborn + Plotly | Visualization |
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

Built as a graduate-level Data Management course project at University of California, San Diego.

Demonstrates: relational database design, data normalization, graph modeling, vector search, ML-powered recommendation systems, and production-grade application deployment.
