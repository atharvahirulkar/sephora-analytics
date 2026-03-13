import streamlit as st
from sqlalchemy import create_engine
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(__file__))
from src.recommender import query_based_recommend, hybrid_recommend, get_product_info

load_dotenv(find_dotenv(), override=True)
engine = create_engine(os.getenv("PG_CONNECTION"))

st.set_page_config(
    page_title="Sephora Analytics",
    page_icon="✦",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;600&family=DM+Mono:wght@300;400&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Mono', monospace;
    background-color: #0a0a0a;
    color: #d4cfc9;
}
.block-container { padding-top: 2.5rem; max-width: 960px; }

h1, h2, h3, h4 {
    font-family: 'Cormorant Garamond', serif;
    color: #e8c89a;
    font-weight: 300;
    letter-spacing: 0.04em;
}

/* Search bar */
.stTextInput > div > div > input {
    background-color: #111;
    color: #e8c89a;
    border: 1px solid #2a2a2a;
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
    font-size: 0.95rem;
    padding: 0.6rem 1rem;
}
.stTextInput > div > div > input:focus {
    border-color: #e8c89a;
    box-shadow: none;
}

/* Selectbox */
.stSelectbox > div > div {
    background-color: #111;
    border: 1px solid #2a2a2a;
    color: #d4cfc9;
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
}

/* Slider */
.stSlider > div { padding: 0; }

/* Metric */
[data-testid="metric-container"] {
    background: #111;
    border: 1px solid #1e1e1e;
    border-radius: 2px;
    padding: 0.5rem 0.8rem;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace;
    color: #e8c89a;
    font-size: 0.9rem;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace;
    color: #666;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Result card */
.result-card {
    background: #0f0f0f;
    border: 1px solid #1e1e1e;
    border-left: 2px solid #e8c89a;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    border-radius: 2px;
}
.result-card:hover { border-left-color: #fff; }

.result-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.25rem;
    color: #e8e0d5;
    font-weight: 400;
    margin-bottom: 0.25rem;
}
.result-meta {
    font-size: 0.78rem;
    color: #888;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.tag {
    display: inline-block;
    background: #1a1a1a;
    border: 1px solid #333;
    color: #c8b99a;
    font-size: 0.78rem;
    padding: 4px 12px;
    border-radius: 20px;
    margin-right: 6px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500;
}
.result-ingredients {
    font-size: 0.75rem;
    color: #666;
    margin-top: 0.7rem;
    font-style: italic;
}

/* Button */
.stButton > button {
    background: transparent;
    border: 1px solid #2a2a2a;
    color: #888;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-radius: 2px;
    padding: 0.3rem 1rem;
    transition: all 0.2s;
}
.stButton > button:hover {
    border-color: #e8c89a;
    color: #e8c89a;
}

/* Divider */
hr { border-color: #1a1a1a; }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────
st.markdown("## ✦ Lookin for somethin' good? ✦")
st.markdown(
    "<p style='color:#555; font-size:0.78rem; letter-spacing:0.08em; "
    "text-transform:uppercase; margin-top:-0.8rem;'>"
    "PostgreSQL · Neo4j · Qdrant · Semantic Search</p>",
    unsafe_allow_html=True
)
st.divider()


# ── Controls ─────────────────────────────────────────────
col_q, col_skin, col_n = st.columns([5, 2, 1])
with col_q:
    query = st.text_input("", placeholder="Search - e.g. hydrating serum for sensitive skin, SPF moisturizer...")
with col_skin:
    skin_type = st.selectbox("Skin Type", ["All", "Dry", "Oily", "Combination", "Normal"])
with col_n:
    top_k = st.slider("Results", 3, 15, 6)

st.divider()


# ── Helpers ──────────────────────────────────────────────
def fetch_product_extras(product_id):
    rating = pd.read_sql(f"""
        SELECT ROUND(AVG(rating)::numeric, 1) AS avg_rating
        FROM reviews WHERE product_id = '{product_id}';
    """, engine)
    ingredients = pd.read_sql(f"""
        SELECT i.ingredient_name
        FROM ingredients i
        JOIN product_ingredients pi ON i.ingredient_id = pi.ingredient_id
        WHERE pi.product_id = '{product_id}'
        LIMIT 5;
    """, engine)
    avg = rating.iloc[0]['avg_rating'] if not rating.empty else "N/A"
    ings = " · ".join(ingredients["ingredient_name"].str.title().tolist()) if not ingredients.empty else ""
    return avg, ings


def render_result(row, score_label, score_value, idx):
    avg_rating, ings = fetch_product_extras(row['product_id'])
    st.markdown(f"""
    <div class="result-card">
        <div class="result-title">{row['product_name']}</div>
        <div class="result-meta">{row['brand_name']} &nbsp;·&nbsp; {row.get('primary_category', '')}</div>
        <span class="tag">⭐ {avg_rating}</span>
        <span class="tag">↯ {score_label} {score_value}</span>
        <span class="tag">$ {row['price_usd']}</span>
        <div class="result-ingredients">🧪 {ings}</div>
    </div>
    """, unsafe_allow_html=True)

    if st.button(f"Find Similar →", key=f"sim_{row['product_id']}_{idx}"):
        st.session_state["similar_product_id"] = row['product_id']
        st.session_state["similar_product_name"] = row['product_name']


# ── Main Search ──────────────────────────────────────────
if query:
    with st.spinner("Searching..."):
        results = query_based_recommend(
            query_text=query,
            skin_type=skin_type if skin_type != "All" else None,
            top_k=top_k
        )

    if results is not None and not results.empty:
        st.markdown(
            f"<p style='color:#555; font-size:0.75rem; letter-spacing:0.06em; "
            f"text-transform:uppercase;'>{len(results)} results for \"{query}\"</p>",
            unsafe_allow_html=True
        )
        for idx, row in results.iterrows():
            render_result(row, "relevance", f"{row['semantic_score']:.0%}", idx)
    else:
        st.markdown("<p style='color:#555;'>No results found. Try different terms.</p>",
                    unsafe_allow_html=True)


# ── Similar Products Panel ───────────────────────────────
if "similar_product_id" in st.session_state:
    pid = st.session_state["similar_product_id"]
    pname = st.session_state["similar_product_name"]

    st.divider()
    st.markdown(f"#### Similar to - *{pname}*")

    with st.spinner("Running hybrid pipeline..."):
        similar = hybrid_recommend(
            product_id=pid,
            skin_type=skin_type if skin_type != "All" else None,
            top_k=5
        )

    if similar is not None and not similar.empty:
        for idx, row in similar.iterrows():
            render_result(row, "score", f"{row['hybrid_score']:.3f}", f"s_{idx}")
    else:
        st.markdown("<p style='color:#555;'>No similar products found.</p>",
                    unsafe_allow_html=True)

    if st.button("✕ Clear"):
        del st.session_state["similar_product_id"]
        del st.session_state["similar_product_name"]
        st.rerun()