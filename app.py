import streamlit as st
from sqlalchemy import create_engine, text
from dotenv import load_dotenv, find_dotenv
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
import pandas as pd
import os, sys

sys.path.append(os.path.dirname(__file__))
from src.recommender import query_based_recommend, hybrid_recommend

load_dotenv(find_dotenv(), override=True)

engine = create_engine(os.getenv("PG_CONNECTION"))
neo4j_driver = GraphDatabase.driver(
    os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", ""))
)
qdrant = QdrantClient(host="localhost", port=6333)

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")
embedder = load_embedder()

st.set_page_config(
    page_title="CosmeTik",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;1,9..144,300;1,9..144,400&family=Outfit:wght@300;400;500;600&display=swap');

:root {
    --bg:       #faf7f4;
    --bg-card:  #ffffff;
    --bg-side:  #f3ede8;
    --border:   #e2d5cc;
    --coral:    #e8614f;
    --coral-d:  #c44535;
    --coral-bg: #fdecea;
    --sage:     #2d9967;
    --sage-bg:  #e6f7f0;
    --gold:     #b8800a;
    --gold-bg:  #fef5e0;
    --lav:      #7c52c8;
    --lav-bg:   #f0ebff;
    --rose:     #d4607a;
    --rose-bg:  #fdeef3;
    --tx:       #2a1f1a;
    --tx2:      #5c4a42;
    --tx3:      #8a7068;
    --tx4:      #c0aea4;
}

/* ── Force light background ── */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main .block-container { background-color: var(--bg) !important; }

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    color: var(--tx);
}
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 840px; }

/* ── Sidebar ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background-color: #e8ddd5 !important;
    border-right: 1px solid #cfc0b5;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; max-width: 100%; background: transparent !important; }
[data-testid="stSidebar"] label { color: #2a1f1a !important; font-weight: 500 !important; font-size: 0.88rem !important; }
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span:not(.stat-num) { color: #2a1f1a !important; }
[data-testid="stSidebar"] .stSelectbox > div > div { background: #fff !important; border-color: #cfc0b5 !important; }
[data-testid="stSidebar"] .stTextInput > div > div > input { background: #fff !important; border-color: #cfc0b5 !important; }
[data-testid="stSidebar"] .stCheckbox label { color: #2a1f1a !important; font-weight: 500 !important; }

/* ── Text input ── */
.stTextInput > div > div > input {
    background: #fff !important; color: var(--tx) !important;
    border: 1.5px solid var(--border) !important; border-radius: 12px;
    font-family: 'Outfit', sans-serif; font-size: 1rem;
    padding: 0.7rem 1.2rem; caret-color: var(--coral);
    box-shadow: 0 1px 4px #00000010;
    transition: border-color 0.2s, box-shadow 0.2s;
}
.stTextInput > div > div > input:focus {
    border-color: var(--coral) !important;
    box-shadow: 0 0 0 3px #e8614f18 !important;
}
.stTextInput > div > div > input::placeholder { color: var(--tx4); font-style: italic; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: #fff !important; border: 1.5px solid var(--border) !important;
    border-radius: 10px; color: var(--tx) !important;
    font-family: 'Outfit', sans-serif; font-size: 0.88rem;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] div[role="slider"] { background-color: var(--coral) !important; }

/* ── Buttons ── */
.stButton > button {
    background: #fff !important; border: 1.5px solid var(--border) !important;
    color: var(--tx2) !important; font-family: 'Outfit', sans-serif;
    font-size: 0.82rem; font-weight: 500; border-radius: 20px;
    padding: 0.38rem 1.1rem; transition: all 0.18s ease;
    white-space: normal; box-shadow: 0 1px 3px #00000010;
}
.stButton > button:hover {
    border-color: var(--coral) !important; color: var(--coral) !important;
    background: var(--coral-bg) !important; transform: translateY(-1px);
    box-shadow: 0 4px 12px #e8614f18;
}

hr { border-color: var(--border); margin: 1.2rem 0; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Hide sidebar collapse button ── */
[data-testid="collapsedControl"],
button[kind="header"],
[data-testid="stSidebarCollapseButton"] { display: none !important; }

/* ── App header ── */
.app-eyebrow { font-size:.72rem; font-weight:600; color:var(--coral); text-transform:uppercase; letter-spacing:.18em; margin-bottom:.4rem; }
.app-title { font-family:'Fraunces',serif; font-size:2.3rem; color:var(--tx); font-weight:300; line-height:1.15; margin-bottom:.4rem; }
.app-title em { color:var(--coral); font-style:italic; }
.app-subtitle { font-size:.88rem; color:var(--tx2); line-height:1.7; }
.db-tag { background:#fff; border:1px solid var(--border); border-radius:6px; padding:2px 9px; font-size:.78rem; margin:0 3px; color:var(--tx3); }

/* ── Chips ── */
.chips-label { font-size:.7rem; font-weight:600; color:var(--tx3); text-transform:uppercase; letter-spacing:.12em; margin-bottom:.4rem; }

/* ── Results header ── */
.results-header { display:flex; align-items:baseline; gap:10px; margin-bottom:.9rem; flex-wrap:wrap; }
.results-count { font-family:'Fraunces',serif; font-size:1.6rem; color:var(--coral); font-weight:300; }
.results-label { font-size:.9rem; color:var(--tx2); }
.results-query { font-family:'Fraunces',serif; font-style:italic; color:var(--tx); }

/* ── Result card ── */
.result-card {
    background:#fff; border:1.5px solid var(--border); border-radius:16px;
    padding:1.2rem 1.4rem 1rem; margin-bottom:.4rem; position:relative;
    overflow:hidden; box-shadow:0 2px 10px #2a1f1a09;
    transition:border-color .2s, transform .15s, box-shadow .2s;
}
.result-card::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background:linear-gradient(90deg,var(--coral),var(--rose),var(--lav));
    opacity:0; transition:opacity .2s;
}
.result-card:hover { border-color:#e8614f60; transform:translateY(-2px); box-shadow:0 8px 24px #e8614f12; }
.result-card:hover::before { opacity:1; }
.result-rank { position:absolute; top:1rem; right:1.3rem; font-family:'Fraunces',serif; font-size:2.6rem; color:#eeddd6; font-weight:300; line-height:1; user-select:none; }
.result-title { font-family:'Fraunces',serif; font-size:1.15rem; color:var(--tx); font-weight:400; margin-bottom:.15rem; padding-right:3.2rem; line-height:1.3; }
.result-brand { font-size:.76rem; color:var(--tx3); font-weight:500; text-transform:uppercase; letter-spacing:.08em; margin-bottom:.75rem; }

/* ── Similar card ── */
.similar-card {
    background:#fdfaf8; border:1.5px solid var(--border); border-radius:12px;
    padding:.95rem 1.1rem .75rem; margin-bottom:.5rem; position:relative;
    overflow:hidden; box-shadow:0 1px 4px #00000008;
    transition:border-color .2s;
}
.similar-card:hover { border-color:var(--sage); }

/* ── Badges ── */
.badge-row { display:flex; flex-wrap:wrap; gap:5px; margin-bottom:.7rem; }
.badge { display:inline-flex; align-items:center; gap:4px; font-size:.76rem; font-weight:500; padding:3px 10px; border-radius:20px; }
.badge-rating { background:var(--gold-bg); border:1px solid #b8800a25; color:var(--gold); }
.badge-price  { background:var(--lav-bg);  border:1px solid #7c52c825; color:var(--lav); }
.badge-score  { background:var(--coral-bg);border:1px solid #e8614f25; color:var(--coral); }
.badge-cat    { background:var(--sage-bg); border:1px solid #2d996725; color:var(--sage); }

/* ── Score bar ── */
.score-row { display:flex; align-items:center; gap:10px; margin-bottom:5px; }
.score-name { font-size:.7rem; font-weight:500; color:var(--tx3); text-transform:uppercase; letter-spacing:.08em; width:72px; flex-shrink:0; }
.score-bar-bg { flex:1; height:5px; background:#ede5df; border-radius:3px; overflow:hidden; }
.score-bar-fill { height:100%; border-radius:3px; background:linear-gradient(90deg,var(--coral-d),var(--coral)); }
.score-pct { font-size:.8rem; font-weight:600; color:var(--coral); width:38px; text-align:right; flex-shrink:0; }

/* ── Ingredients ── */
.ing-section { margin-top:.8rem; padding-top:.8rem; border-top:1px solid var(--border); }
.ing-label { font-size:.66rem; font-weight:600; color:var(--tx3); text-transform:uppercase; letter-spacing:.1em; margin-bottom:.4rem; }
.ing-pill { display:inline-block; background:var(--sage-bg); border:1px solid #2d996720; color:var(--sage); font-size:.76rem; padding:2px 9px; border-radius:6px; margin-right:4px; margin-bottom:4px; }

/* ── Review panel ── */
.review-panel { background:var(--lav-bg); border:1.5px solid #7c52c818; border-radius:12px; padding:.9rem 1.1rem; margin:.3rem 0 .7rem; }
.review-card { background:#fff; border:1px solid #7c52c815; border-left:3px solid var(--lav); border-radius:10px; padding:.85rem 1rem; margin-bottom:.5rem; }
.review-meta { display:flex; gap:5px; flex-wrap:wrap; margin-bottom:.5rem; }
.rev-tag  { background:var(--lav-bg); border:1px solid #7c52c820; color:var(--lav); font-size:.7rem; padding:2px 8px; border-radius:20px; }
.rev-match{ background:var(--rose-bg);border:1px solid #d4607a20;color:var(--rose);font-size:.7rem; padding:2px 8px; border-radius:20px; }
.review-text { font-size:.86rem; color:var(--tx2); line-height:1.75; font-style:italic; }

/* ── Similar panel ── */
.similar-panel { background:var(--sage-bg); border:1.5px solid #2d996718; border-radius:12px; padding:.9rem 1.1rem; margin:.3rem 0 .7rem; }
.similar-heading { font-family:'Fraunces',serif; font-size:.95rem; color:var(--tx2); font-style:italic; font-weight:300; margin-bottom:.7rem; padding-bottom:.55rem; border-bottom:1px solid #2d996725; }

/* ── Allergy tag ── */
.allergy-tag { display:inline-block; background:var(--coral-bg); border:1px solid #e8614f25; color:var(--coral); font-size:.7rem; padding:2px 8px; border-radius:6px; margin:2px; }

/* ── Sidebar stat ── */
.sidebar-logo { font-family:'Fraunces',serif; font-size:1.45rem; color:#2a1f1a; font-weight:300; margin-bottom:.1rem; }
.sidebar-tag  { font-size:.72rem; color:#4a3830; margin-bottom:1.1rem; line-height:1.6; }
.sidebar-sec  { font-size:.68rem; font-weight:700; color:#4a3830; text-transform:uppercase; letter-spacing:.14em; margin:1rem 0 .45rem; }
.stat-row { display:flex; align-items:center; justify-content:space-between; background:#fff; border:1px solid #cfc0b5; border-radius:10px; padding:.5rem .85rem; margin-bottom:.35rem; box-shadow:0 1px 3px #00000008; }
.stat-name { font-size:.76rem; color:#2a1f1a; font-weight:500; }
.stat-num  { font-family:'Fraunces',serif; font-size:1.05rem; color:var(--coral); font-weight:300; }

/* ── Empty ── */
.empty-wrap { text-align:center; padding:4rem 2rem; }
.empty-emoji { font-size:2.8rem; margin-bottom:.7rem; }
.empty-heading { font-family:'Fraunces',serif; font-size:1.4rem; color:var(--tx2); font-weight:300; font-style:italic; margin-bottom:.4rem; }
.empty-body { font-size:.87rem; color:var(--tx3); line-height:1.7; max-width:400px; margin:0 auto; }

@keyframes fadeUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
.result-card { animation:fadeUp .22s ease both; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────

@st.cache_data(ttl=600)
def get_db_stats():
    try:
        with engine.connect() as conn:
            p  = conn.execute(text("SELECT COUNT(*) FROM products")).scalar()
            i  = conn.execute(text("SELECT COUNT(*) FROM ingredients")).scalar()
            pi = conn.execute(text("SELECT COUNT(*) FROM product_ingredients")).scalar()
            r  = conn.execute(text("SELECT COUNT(*) FROM reviews")).scalar()
        return p, i, pi, r
    except:
        return 0, 0, 0, 0


@st.cache_data(ttl=300)
def fetch_product_extras(product_id):
    try:
        with engine.connect() as conn:
            rating = pd.read_sql(
                text("SELECT ROUND(AVG(rating)::numeric,1) AS avg_rating FROM reviews WHERE product_id = :pid"),
                conn, params={"pid": product_id}
            )
            ings = pd.read_sql(
                text("""
                    SELECT i.ingredient_name FROM ingredients i
                    JOIN product_ingredients pi ON i.ingredient_id = pi.ingredient_id
                    WHERE pi.product_id = :pid LIMIT 6
                """),
                conn, params={"pid": product_id}
            )
        avg = rating.iloc[0]['avg_rating'] if not rating.empty else "—"
        return avg, ings["ingredient_name"].str.title().tolist() if not ings.empty else []
    except:
        return "—", []


@st.cache_data(ttl=300)
def fetch_all_ingredients(product_id):
    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text("""
                    SELECT i.ingredient_name FROM ingredients i
                    JOIN product_ingredients pi ON i.ingredient_id = pi.ingredient_id
                    WHERE pi.product_id = :pid
                """),
                conn, params={"pid": product_id}
            )
        return set(df["ingredient_name"].str.lower().tolist())
    except:
        return set()


def has_allergen(product_id, allergens):
    if not allergens:
        return []
    ings = fetch_all_ingredients(product_id)
    return [a for a in allergens if any(a in ing for ing in ings)]


def fetch_reviews_for_product(product_id: str, query_text: str, top_k: int = 5):
    """
    Uses query_points (qdrant-client >= 1.7).
    product_id stored as str in payload (confirmed from notebook).
    review_text truncated to 300 in payload — always enriches from PostgreSQL.
    """
    vec = embedder.encode([query_text])[0].tolist()

    # Try product-filtered search first
    hits = []
    try:
        res = qdrant.query_points(
            collection_name="sephora_reviews",
            query=vec,
            query_filter=Filter(must=[
                FieldCondition(key="product_id", match=MatchValue(value=str(product_id)))
            ]),
            limit=top_k,
            with_payload=True
        )
        hits = res.points
    except Exception:
        pass

    source = "product"
    if not hits:
        # Fallback: global unfiltered
        try:
            res = qdrant.query_points(
                collection_name="sephora_reviews",
                query=vec,
                limit=top_k,
                with_payload=True
            )
            hits = res.points
            source = "global"
        except Exception:
            return [], "error"

    enriched = []
    for hit in hits:
        p           = hit.payload or {}
        review_text = p.get("review_text", "")
        # Always fetch full text from PostgreSQL (payload is truncated to 300)
        try:
            rid = p.get("review_id") or hit.id
            with engine.connect() as conn:
                row_df = pd.read_sql(
                    text("SELECT review_text FROM reviews WHERE review_id = :rid LIMIT 1"),
                    conn, params={"rid": int(rid)}
                )
            if not row_df.empty and row_df.iloc[0]["review_text"]:
                review_text = str(row_df.iloc[0]["review_text"])
        except Exception:
            pass
        enriched.append({
            "score":       hit.score,
            "skin_type":   p.get("skin_type", "unknown"),
            "rating":      p.get("rating", "?"),
            "recommended": p.get("is_recommended", None),
            "review_text": review_text,
        })
    return enriched, source


def row_to_dict(row):
    """Safely convert pandas Series or dict to plain dict."""
    if hasattr(row, 'to_dict'):
        return row.to_dict()
    return dict(row)


def card_html(row, score_col, rank=None, card_class="result-card"):
    d          = row_to_dict(row)
    avg, ings  = fetch_product_extras(d.get('product_id', ''))
    try:    sf = float(d.get(score_col) or 0)
    except: sf = 0.0
    pct        = min(int(sf * 100), 100)
    rank_html  = f'<div class="result-rank">{rank:02d}</div>' if rank else ''
    ing_pills  = "".join(f'<span class="ing-pill">🌿 {i}</span>' for i in ings)
    ing_block  = (f'<div class="ing-section"><div class="ing-label">Key Ingredients</div>'
                  f'{ing_pills}</div>') if ings else ""
    sname      = "Relevance" if score_col == "semantic_score" else "Match"
    cat        = d.get('primary_category') or ''
    cat_badge  = f'<span class="badge badge-cat">🗂 {cat}</span>' if cat else ''

    return (
        f'<div class="{card_class}">'
        f'{rank_html}'
        f'<div class="result-title">{d.get("product_name","—")}</div>'
        f'<div class="result-brand">{d.get("brand_name","")}</div>'
        f'<div class="badge-row">'
        f'<span class="badge badge-rating">⭐ {avg}</span>'
        f'<span class="badge badge-price">💰 ${d.get("price_usd","—")}</span>'
        f'<span class="badge badge-score">✦ {pct}% {sname}</span>'
        f'{cat_badge}'
        f'</div>'
        f'<div class="score-row">'
        f'<span class="score-name">{sname}</span>'
        f'<div class="score-bar-bg"><div class="score-bar-fill" style="width:{pct}%"></div></div>'
        f'<span class="score-pct">{pct}%</span>'
        f'</div>'
        f'{ing_block}'
        f'</div>'
    )


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="sidebar-logo">🌿 CosmeTik </div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="sidebar-sec">🎛 Filters</div>', unsafe_allow_html=True)
    skin_type      = st.selectbox("Type", ["All Skin Types","Dry","Oily","Combination","Normal","Sensitive"])
    top_k          = st.slider("Results", 3, 15, 6)

    st.markdown('<div class="sidebar-sec">⚠️ Allergy Filter</div>', unsafe_allow_html=True)
    allergy_input  = st.text_input("Ingredients to avoid", placeholder="e.g. retinol, fragrance")
    hide_allergens = st.checkbox("Hide products with allergens", value=False)
    allergen_list  = [a.strip().lower() for a in allergy_input.split(",") if a.strip()] if allergy_input else []

    st.markdown('<div class="sidebar-sec">📊 Database</div>', unsafe_allow_html=True)
    n_p, n_i, n_pi, n_r = get_db_stats()
    for name, val in [("Products", f"{n_p:,}"), ("Ingredients", f"{n_i:,}"),
                      ("Relationships", f"{n_pi:,}"), ("Reviews", f"{n_r:,}")]:
        st.markdown(
            f'<div class="stat-row"><span class="stat-name">{name}</span>'
            f'<span class="stat-num">{val}</span></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="sidebar-sec">🔬 How it works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:.76rem;color:#8a7068;line-height:2;">'
        '<b style="color:#5c4a42;">Search</b> → Qdrant semantic<br>'
        '<b style="color:#5c4a42;">Reviews</b> → Qdrant + PostgreSQL<br>'
        '<b style="color:#5c4a42;">Similar</b> → Neo4j + Qdrant hybrid'
        '</div>',
        unsafe_allow_html=True
    )


# ── Page header ───────────────────────────────────────────────────
st.markdown(
    '<div class="app-eyebrow">🌿 CosmeTik </div>'
    '<div class="app-title">Find what actually works<br><em>for you !!</em></div>',
    unsafe_allow_html=True
)
st.divider()

skin_filter = None if skin_type == "All Skin Types" else skin_type.lower()

# ── Search + chips ────────────────────────────────────────────────
if "pending_chip" in st.session_state:
    st.session_state["main_search"] = st.session_state.pop("pending_chip")

query = st.text_input(
    "",
    placeholder="✦  Try: 'calming serum for redness'  ·  'retinol for beginners'  ·  'SPF moisturizer oily skin'",
    key="main_search"
)

CHIPS = [
    ("🧴", "hydrating toner dry skin"),
    ("🌸", "gentle cleanser sensitive skin"),
    ("✨", "vitamin C brightening serum"),
    ("🌙", "retinol anti-aging"),
    ("💧", "niacinamide pore minimizer"),
    ("🛡️", "barrier repair eczema"),
]
st.markdown('<div class="chips-label">✦ Popular searches</div>', unsafe_allow_html=True)
# 3 per row — enough space so text never truncates
for row_chips in [CHIPS[:3], CHIPS[3:6], CHIPS[6:]]:
    cols = st.columns([1, 1, 1])
    for col, (emoji, chip_text) in zip(cols, row_chips):
        with col:
            if st.button(f"{emoji}  {chip_text}", key=f"chip_{chip_text}", use_container_width=True):
                st.session_state["pending_chip"] = chip_text
                st.rerun()

st.divider()

# ── Results ───────────────────────────────────────────────────────
if query:
    with st.spinner("🔍 Searching..."):
        results = query_based_recommend(query_text=query, skin_type=skin_filter, top_k=top_k + 5)

    filtered_results = []
    if results is not None and not results.empty:
        for _, row in results.iterrows():
            found = has_allergen(row['product_id'], allergen_list)
            if found and hide_allergens:
                continue
            filtered_results.append((row_to_dict(row), found))
    filtered_results = filtered_results[:top_k]

    if filtered_results:
        st.markdown(
            f'<div class="results-header">'
            f'<span class="results-count">{len(filtered_results)}</span>'
            f'<span class="results-label">results for</span>'
            f'<span class="results-query">"{query}"</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        if allergen_list:
            tags = "".join(f'<span class="allergy-tag">⚠️ {a}</span>' for a in allergen_list)
            st.markdown(
                f'<div style="margin-bottom:.7rem;font-size:.78rem;color:var(--tx3);">Filtering for: {tags}</div>',
                unsafe_allow_html=True
            )

        for rank, (row_d, allergens_found) in enumerate(filtered_results, 1):
            pid = row_d['product_id']

            if allergens_found:
                warn = "".join(f'<span class="allergy-tag">⚠️ {a}</span>' for a in allergens_found)
                st.markdown(
                    f'<div style="margin-bottom:3px;font-size:.76rem;">Contains: {warn}</div>',
                    unsafe_allow_html=True
                )

            # Product card
            st.markdown(card_html(row_d, "semantic_score", rank=rank), unsafe_allow_html=True)

            # Mutual-exclusion: one panel per product at a time
            active = st.session_state.get(f"panel_{pid}")  # None | "reviews" | "similar"

            b1, b2, _ = st.columns([1.8, 1.8, 4])
            with b1:
                if st.button(
                    "✕ Hide Reviews" if active == "reviews" else "💬 Read Reviews",
                    key=f"rev_{pid}_{rank}"
                ):
                    st.session_state[f"panel_{pid}"] = None if active == "reviews" else "reviews"
                    st.rerun()
            with b2:
                if st.button(
                    "✕ Close Similar" if active == "similar" else "🔍 Find Similar",
                    key=f"sim_{pid}_{rank}"
                ):
                    st.session_state[f"panel_{pid}"] = None if active == "similar" else "similar"
                    st.rerun()

            # ── Reviews panel ──────────────────────────────────
            if active == "reviews":
                st.markdown('<div class="review-panel">', unsafe_allow_html=True)
                with st.spinner("Fetching from Qdrant + PostgreSQL..."):
                    reviews, source = fetch_reviews_for_product(pid, query)

                if not reviews:
                    st.markdown(
                        "<p style='color:var(--tx3);font-size:.84rem;margin:0;'>"
                        "No reviews found for this product in Qdrant.</p>",
                        unsafe_allow_html=True
                    )
                else:
                    note = ("" if source == "product"
                            else " · <em style='color:var(--tx4);'>globally similar (product not indexed)</em>")
                    st.markdown(
                        f"<div style='font-size:.73rem;color:var(--tx3);margin-bottom:.7rem;'>"
                        f"💬 {len(reviews)} reviews matching <em>\"{query}\"</em>{note}</div>",
                        unsafe_allow_html=True
                    )
                    for rev in reviews:
                        sim_pct  = int(rev["score"] * 100)
                        rec_text = ("✅ Recommends"    if rev["recommended"] is True  else
                                    "❌ Doesn't recommend" if rev["recommended"] is False else "")
                        txt      = str(rev["review_text"] or "").strip()
                        preview  = (txt[:450] + "…") if len(txt) > 450 else txt
                        if not preview:
                            preview = "<em style='color:var(--tx4);'>Text not available</em>"
                        rec_badge = f'<span class="rev-tag">{rec_text}</span>' if rec_text else ''
                        st.markdown(
                            f'<div class="review-card">'
                            f'<div class="review-meta">'
                            f'<span class="rev-tag">🧴 {rev["skin_type"]} skin</span>'
                            f'<span class="rev-tag">⭐ {rev["rating"]}/5</span>'
                            f'{rec_badge}'
                            f'<span class="rev-match">🎯 {sim_pct}% match</span>'
                            f'</div>'
                            f'<div class="review-text">"{preview}"</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Similar panel ──────────────────────────────────
            if active == "similar":
                st.markdown('<div class="similar-panel">', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="similar-heading">✦ Similar to <em>{row_d.get("product_name","")}</em></div>',
                    unsafe_allow_html=True
                )
                with st.spinner("🧪 Running Neo4j + Qdrant hybrid..."):
                    sim_df = hybrid_recommend(product_id=pid, skin_type=skin_filter, top_k=5)

                if sim_df is not None and not sim_df.empty:
                    for _, s_row in sim_df.iterrows():
                        s_dict      = row_to_dict(s_row)          # Series → dict
                        s_allergens = has_allergen(s_dict.get('product_id', ''), allergen_list)
                        if s_allergens:
                            warn = "".join(f'<span class="allergy-tag">⚠️ {a}</span>' for a in s_allergens)
                            st.markdown(
                                f'<div style="margin-bottom:3px;font-size:.74rem;">{warn}</div>',
                                unsafe_allow_html=True
                            )
                        # card_html returns an HTML string — render with unsafe_allow_html
                        st.markdown(
                            card_html(s_dict, "hybrid_score", card_class="similar-card"),
                            unsafe_allow_html=True
                        )
                else:
                    st.markdown(
                        "<p style='color:var(--tx3);font-size:.84rem;margin:0;'>"
                        "No similar products found.</p>",
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<div style='margin-bottom:.6rem'></div>", unsafe_allow_html=True)

    else:
        st.markdown(
            '<div class="empty-wrap">'
            '<div class="empty-emoji">🤔</div>'
            '<div class="empty-heading">Nothing matched.</div>'
            '<div class="empty-body">Try different words or adjust the skin type filter.</div>'
            '</div>',
            unsafe_allow_html=True
        )

else:
    st.markdown(
        '<div class="empty-wrap">'
        '<div class="empty-emoji">🌿</div>'
        '<div class="empty-heading">What is your skin craving today?</div>'
        '<div class="empty-body">'
        'Type a concern, ingredient, or product type above.<br>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )