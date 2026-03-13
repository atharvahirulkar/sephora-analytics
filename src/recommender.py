import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(find_dotenv(), override=True)

# --- Connections ---
engine = create_engine(os.getenv("PG_CONNECTION"))
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))
)
qdrant = QdrantClient(host="localhost", port=6333)
model = SentenceTransformer("all-MiniLM-L6-v2")


# --- Core Functions ---

def get_product_info(product_id):
    return pd.read_sql(f"""
        SELECT product_id, product_name, brand_name, price_usd, primary_category
        FROM products
        WHERE product_id = '{product_id}';
    """, engine)


def search_products(query_text):
    """Search products by text using PostgreSQL full text."""
    return pd.read_sql(f"""
        SELECT product_id, product_name, brand_name, price_usd, primary_category
        FROM products
        WHERE LOWER(product_name) LIKE LOWER('%{query_text}%')
           OR LOWER(brand_name)   LIKE LOWER('%{query_text}%')
        LIMIT 20;
    """, engine)


def get_all_products():
    return pd.read_sql("""
        SELECT product_id, product_name, brand_name, price_usd, primary_category
        FROM products
        ORDER BY product_name
        LIMIT 500;
    """, engine)


def ingredient_based_recommend(product_id, top_k=20):
    """PostgreSQL + Neo4j ingredient similarity."""
    ingredients = pd.read_sql(f"""
        SELECT i.ingredient_name
        FROM ingredients i
        JOIN product_ingredients pi ON i.ingredient_id = pi.ingredient_id
        WHERE pi.product_id = '{product_id}';
    """, engine)

    if ingredients.empty:
        return None

    ing_list = ingredients["ingredient_name"].tolist()

    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH (i:Ingredient)-[:FOUND_IN]->(p:Product)
            WHERE i.name IN $ingredients
            WITH p, SUM(COALESCE(i.pagerank, 0.001)) AS pagerank_score
            RETURN p.id AS product_id, pagerank_score
            ORDER BY pagerank_score DESC
            LIMIT $top_k
        """, ingredients=ing_list, top_k=top_k)
        recs = pd.DataFrame([dict(r) for r in result])

    if recs.empty:
        return None

    recs = recs[recs["product_id"] != product_id]

    product_details = pd.read_sql(f"""
        SELECT product_id, product_name, brand_name, price_usd
        FROM products
        WHERE product_id IN ({','.join([f"'{p}'" for p in recs['product_id'].tolist()])});
    """, engine)

    return recs.merge(product_details, on="product_id", how="left")


def review_based_recommend(product_id, skin_type=None, top_k=20):
    """Qdrant semantic similarity on reviews."""
    reviews = pd.read_sql(f"""
        SELECT review_text FROM reviews
        WHERE product_id = '{product_id}'
        AND review_text IS NOT NULL
        LIMIT 10;
    """, engine)

    if reviews.empty:
        return None

    combined_text = " ".join(reviews["review_text"].tolist())
    query_vector = model.encode(combined_text).tolist()

    must_not_filter = Filter(
        must_not=[FieldCondition(key="product_id", match=MatchValue(value=product_id))]
    )

    results = qdrant.query_points(
        collection_name="sephora_reviews",
        query=query_vector,
        query_filter=must_not_filter,
        limit=top_k
    ).points

    if not results:
        return None

    rows = [{"product_id": r.payload["product_id"], "semantic_score": r.score}
            for r in results]
    recs = pd.DataFrame(rows).groupby("product_id")["semantic_score"].mean().reset_index()

    product_details = pd.read_sql(f"""
        SELECT product_id, product_name, brand_name, price_usd
        FROM products
        WHERE product_id IN ({','.join([f"'{p}'" for p in recs['product_id'].tolist()])});
    """, engine)

    return recs.merge(product_details, on="product_id", how="left")


def query_based_recommend(query_text, skin_type=None, top_k=10):
    """
    Free text query → semantic search → ranked products.
    New function for the Streamlit app.
    """
    query_vector = model.encode(query_text).tolist()

    # Optional skin type filter
    qdrant_filter = None
    if skin_type and skin_type != "All":
        qdrant_filter = Filter(
            must=[FieldCondition(key="skin_type", match=MatchValue(value=skin_type.lower()))]
        )

    results = qdrant.query_points(
        collection_name="sephora_reviews",
        query=query_vector,
        query_filter=qdrant_filter,
        limit=50
    ).points

    if not results:
        return None

    # Aggregate scores per product
    rows = [{"product_id": r.payload["product_id"], "semantic_score": r.score}
            for r in results]
    recs = (pd.DataFrame(rows)
            .groupby("product_id")["semantic_score"]
            .mean()
            .reset_index()
            .sort_values("semantic_score", ascending=False)
            .head(top_k))

    # Fetch product details
    product_details = pd.read_sql(f"""
        SELECT product_id, product_name, brand_name, price_usd, primary_category
        FROM products
        WHERE product_id IN ({','.join([f"'{p}'" for p in recs['product_id'].tolist()])});
    """, engine)

    return recs.merge(product_details, on="product_id", how="left")


def hybrid_recommend(product_id, skin_type=None, top_k=5):
    """Full hybrid: PostgreSQL + Neo4j + Qdrant."""
    ing_recs = ingredient_based_recommend(product_id, top_k=20)
    rev_recs = review_based_recommend(product_id, skin_type=skin_type, top_k=20)

    if ing_recs is None or rev_recs is None:
        return None

    ing_recs["ing_score_norm"] = (
        ing_recs["pagerank_score"] / ing_recs["pagerank_score"].max()
    )
    rev_recs["sem_score_norm"] = (
        rev_recs["semantic_score"] / rev_recs["semantic_score"].max()
    )

    hybrid = ing_recs[["product_id", "product_name", "brand_name",
                        "price_usd", "ing_score_norm"]].merge(
        rev_recs[["product_id", "sem_score_norm"]],
        on="product_id", how="outer"
    ).fillna(0)

    # Fix ghost rows
    missing = hybrid[hybrid["product_name"] == 0]["product_id"].tolist()
    if missing:
        missing_details = pd.read_sql(f"""
            SELECT product_id, product_name, brand_name, price_usd
            FROM products
            WHERE product_id IN ({','.join([f"'{p}'" for p in missing])});
        """, engine)
        for _, row in missing_details.iterrows():
            mask = hybrid["product_id"] == row["product_id"]
            hybrid.loc[mask, "product_name"] = row["product_name"]
            hybrid.loc[mask, "brand_name"] = row["brand_name"]
            hybrid.loc[mask, "price_usd"] = row["price_usd"]

    hybrid["hybrid_score"] = (
        0.5 * hybrid["ing_score_norm"] +
        0.5 * hybrid["sem_score_norm"]
    )

    if skin_type and skin_type != "All":
        skin_products = pd.read_sql(f"""
            SELECT DISTINCT product_id FROM product_skin_types
            WHERE LOWER(skin_type) = LOWER('{skin_type}');
        """, engine)
        hybrid = hybrid[hybrid["product_id"].isin(skin_products["product_id"].tolist())]

    return hybrid.sort_values("hybrid_score", ascending=False).head(top_k)