import streamlit as st
import pandas as pd
import numpy as np
import re
import sqlite3
import hashlib
import json
from datetime import datetime

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy
import altair as alt

# ============================================================
# CONFIG & THEME (DARK)
# ============================================================

st.set_page_config(
    page_title="ABSA Dashboard - Milestone 4",
    layout="wide"
)

PRIMARY_COLOR = "#20808D"
BACKGROUND_COLOR = "#020617"
TEXT_COLOR = "#E5E7EB"

CUSTOM_CSS = f"""
<style>
    :root {{
        color-scheme: dark;
    }}

    body, .stApp, .block-container, .main {{
        background-color: {BACKGROUND_COLOR} !important;
        color: {TEXT_COLOR} !important;
        font-family: "system-ui", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}

    h1, h2, h3, h4, h5, h6, label, p {{
        color: {TEXT_COLOR} !important;
    }}

    .stButton > button {{
        background-color: {PRIMARY_COLOR} !important;
        color: #FBFAF4 !important;
        border-radius: 999px !important;
        border: none !important;
        padding: 0.5rem 1.1rem !important;
        font-weight: 600 !important;
    }}
    .stButton > button:hover {{
        filter: brightness(1.05) !important;
    }}

    textarea, .stTextInput > div > div > input {{
        background-color: #020617 !important;
        color: {TEXT_COLOR} !important;
        border-radius: 10px !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: #020617;
        color: {TEXT_COLOR};
        border-radius: 10px;
        border: 1px solid {PRIMARY_COLOR}33;
        padding: 0.3rem 0.9rem;
        margin-right: 0.35rem;
        font-size: 0.9rem;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {PRIMARY_COLOR};
        color: #FBFAF4 !important;
        border: 1px solid {PRIMARY_COLOR};
    }}

    .card {{
        background-color: #020617;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid #1f2937;
        box-shadow: 0 8px 18px rgba(0,0,0,0.40);
        margin-bottom: 0.75rem;
    }}

    .review-card {{
        background-color: #020617;
        color: {TEXT_COLOR};
        border-radius: 10px;
        padding: 0.9rem 1rem;
        border: 1px solid #1f2937;
        margin-bottom: 0.75rem;
        min-height: 3rem;
        display: flex;
        align-items: center;
    }}

    .aspect-positive {{
        background-color: rgba(22, 163, 74, 0.30);
        color: #bbf7d0;
        padding: 2px 8px;
        border-radius: 999px;
        font-weight: 500;
    }}
    .aspect-negative {{
        background-color: rgba(220, 38, 38, 0.30);
        color: #fecaca;
        padding: 2px 8px;
        border-radius: 999px;
        font-weight: 500;
    }}
    .aspect-neutral {{
        background-color: rgba(234, 179, 8, 0.30);
        color: #fef3c7;
        padding: 2px 8px;
        border-radius: 999px;
        font-weight: 500;
    }}

    .sentiment-pill {{
        display: inline-block;
        padding: 2px 8px;
        border-radius: 999px;
        font-size: 0.75rem;
        margin-right: 4px;
        border: 1px solid #1f2937;
        background-color: #020617;
    }}
    .pill-pos {{ color: #22c55e; }}
    .pill-neg {{ color: #f97373; }}
    .pill-neu {{ color: #eab308; }}

    .muted-text {{
        color: #9ca3af;
        font-size: 0.8rem;
    }}

    .auth-card {{
        background-color: #020617;
        border-radius: 16px;
        padding: 1.5rem 1.5rem 1.2rem 1.5rem;
        border: 1px solid #1f2937;
        box-shadow: 0 16px 30px rgba(0, 0, 0, 0.70);
        max-width: 540px;
        margin: 1.2rem auto;
    }}

    .auth-tabs .stRadio > div {{
        display: flex;
        justify-content: center;
        gap: 0.7rem;
    }}
    .auth-tabs .stRadio label {{
        background-color: #020617;
        border-radius: 10px;
        border: 1px solid #1f2937;
        padding: 0.35rem 1.3rem;
        cursor: pointer;
    }}
    .auth-tabs .stRadio [role="radio"][aria-checked="true"] label {{
        background-color: {PRIMARY_COLOR};
        border-color: {PRIMARY_COLOR};
        color: #FBFAF4;
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ============================================================
# DATABASE
# ============================================================

DB_PATH = "absa_app.db"
UNCERTAINTY_THRESHOLD_DB = 0.5  # auto-queue threshold


def get_db_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def hash_password(p: str) -> str:
    return hashlib.sha256(p.encode("utf-8")).hexdigest()


def add_column_if_not_exists(cur, table, col_name, col_def):
    """
    Safely add a column to an existing table if it does not already exist.
    col_def includes type and default, e.g. 'TEXT', 'REAL', 'INTEGER DEFAULT 0'.
    """
    cur.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cur.fetchall()]
    if col_name not in cols:
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")


def init_db():
    conn = get_db_connection()
    cur = conn.cursor()

    # Users
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            email TEXT
        )
        """
    )

    # Saved single analyses
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS saved_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            input_text TEXT NOT NULL,
            aspect_results TEXT,
            created_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    # Dataset results
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            dataset_name TEXT,
            n_rows INTEGER,
            aspect_agg_json TEXT,
            created_at TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )

    # Admins
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS admins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT,
            email TEXT
        )
        """
    )

    # Active learning queue (created if not exists, then migrated)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS active_learning_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT
        )
        """
    )
    # Migrate / ensure required columns exist
    add_column_if_not_exists(cur, "active_learning_samples", "user_id", "INTEGER NOT NULL DEFAULT 0")
    add_column_if_not_exists(cur, "active_learning_samples", "username", "TEXT")
    add_column_if_not_exists(cur, "active_learning_samples", "source", "TEXT")
    add_column_if_not_exists(cur, "active_learning_samples", "dataset_name", "TEXT")
    add_column_if_not_exists(cur, "active_learning_samples", "review_id", "INTEGER")
    add_column_if_not_exists(cur, "active_learning_samples", "review_text", "TEXT")
    add_column_if_not_exists(cur, "active_learning_samples", "aspect", "TEXT")
    add_column_if_not_exists(cur, "active_learning_samples", "opinion", "TEXT")
    add_column_if_not_exists(cur, "active_learning_samples", "predicted_label", "TEXT")
    add_column_if_not_exists(cur, "active_learning_samples", "predicted_confidence", "REAL")
    add_column_if_not_exists(cur, "active_learning_samples", "adjusted_confidence", "REAL")
    add_column_if_not_exists(cur, "active_learning_samples", "corrected_label", "TEXT")
    add_column_if_not_exists(cur, "active_learning_samples", "feedback", "TEXT")
    add_column_if_not_exists(cur, "active_learning_samples", "is_applied", "INTEGER DEFAULT 0")
    add_column_if_not_exists(cur, "active_learning_samples", "created_at", "TEXT")
    add_column_if_not_exists(cur, "active_learning_samples", "corrected_at", "TEXT")

    # Activity logs
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS activity_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT
        )
        """
    )
    add_column_if_not_exists(cur, "activity_logs", "user_id", "INTEGER")
    add_column_if_not_exists(cur, "activity_logs", "username", "TEXT")
    add_column_if_not_exists(cur, "activity_logs", "action", "TEXT")
    add_column_if_not_exists(cur, "activity_logs", "details", "TEXT")
    add_column_if_not_exists(cur, "activity_logs", "created_at", "TEXT")

    # Retraining logs
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS model_retraining_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT
        )
        """
    )
    add_column_if_not_exists(cur, "model_retraining_logs", "user_id", "INTEGER")
    add_column_if_not_exists(cur, "model_retraining_logs", "username", "TEXT")
    add_column_if_not_exists(cur, "model_retraining_logs", "n_samples_used", "INTEGER")
    add_column_if_not_exists(cur, "model_retraining_logs", "notes", "TEXT")
    add_column_if_not_exists(cur, "model_retraining_logs", "created_at", "TEXT")

    # Default admin
    cur.execute("SELECT COUNT(*) FROM admins")
    if cur.fetchone()[0] == 0:
        cur.execute(
            "INSERT INTO admins (username, password_hash, full_name, email) VALUES (?, ?, ?, ?)",
            ("admin", hash_password("admin123"), "Default Admin", "admin@example.com"),
        )

    conn.commit()
    conn.close()


# ------------------------------------------------------------
# USER / ADMIN HELPERS
# ------------------------------------------------------------

def create_user(username, password, full_name="", email="") -> bool:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, full_name, email) VALUES (?, ?, ?, ?)",
            (username, hash_password(password), full_name, email),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def authenticate_user(username, password):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, username, password_hash, full_name, email FROM users WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    if row and row[2] == hash_password(password):
        return {"id": row[0], "username": row[1], "full_name": row[3], "email": row[4]}
    return None


def authenticate_admin(username, password):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, username, password_hash, full_name, email FROM admins WHERE username=?", (username,))
    row = cur.fetchone()
    conn.close()
    if row and row[2] == hash_password(password):
        return {"id": row[0], "username": row[1], "full_name": row[3], "email": row[4]}
    return None


def update_user_profile(user_id, full_name, email):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("UPDATE users SET full_name=?, email=? WHERE id=?", (full_name, email, user_id))
    conn.commit()
    conn.close()


def update_password(user_id, new_password):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("UPDATE users SET password_hash=? WHERE id=?", (hash_password(new_password), user_id))
    conn.commit()
    conn.close()


def save_single_analysis(user_id, input_text, aspect_results):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO saved_analyses (user_id, input_text, aspect_results, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (user_id, input_text, json.dumps(aspect_results), datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def save_dataset_analysis(user_id, dataset_name, n_rows, agg_df):
    conn = get_db_connection()
    cur = conn.cursor()
    agg_json = agg_df.to_json(orient="records")
    cur.execute(
        """
        INSERT INTO dataset_results (user_id, dataset_name, n_rows, aspect_agg_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, dataset_name, n_rows, agg_json, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def get_user_dataset_history(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, dataset_name, n_rows, created_at FROM dataset_results WHERE user_id=? ORDER BY created_at DESC",
        (user_id,),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def get_user_saved_analyses(user_id, limit=10):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, input_text, created_at FROM saved_analyses WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
        (user_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def delete_user(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()


def log_activity(user_id, username, action, details=""):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO activity_logs (user_id, username, action, details, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, username, action, details, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


def create_retraining_log(user_id, username, n_samples, notes=""):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO model_retraining_logs (user_id, username, n_samples_used, notes, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, username, n_samples, notes, datetime.utcnow().isoformat()),
    )
    conn.commit()
    conn.close()


# ------------------------------------------------------------
# ACTIVE LEARNING HELPERS
# ------------------------------------------------------------

def enqueue_uncertain_samples(user, pair_df, source, dataset_name=None):
    """
    Auto-queue predictions with confidence < UNCERTAINTY_THRESHOLD_DB.
    """
    if pair_df.empty:
        return 0

    uncertain = pair_df[pair_df["confidence"] < UNCERTAINTY_THRESHOLD_DB].copy()
    if uncertain.empty:
        return 0

    conn = get_db_connection()
    cur = conn.cursor()
    for _, r in uncertain.iterrows():
        cur.execute(
            """
            INSERT INTO active_learning_samples (
                user_id, username, source, dataset_name, review_id, review_text,
                aspect, opinion, predicted_label, predicted_confidence,
                adjusted_confidence, corrected_label, feedback,
                is_applied, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                user["username"],
                source,
                dataset_name,
                int(r.get("review_id", -1)) if "review_id" in r else -1,
                str(r.get("review_text", "")),
                str(r.get("aspect", "")),
                str(r.get("opinion", "")),
                str(r.get("sentiment_label", "")),
                float(r.get("confidence", 0.0)),
                None,
                None,
                None,
                0,
                datetime.utcnow().isoformat(),
            ),
        )
    conn.commit()
    conn.close()

    log_activity(
        user["id"],
        user["username"],
        "enqueue_active_learning_auto",
        f"source={source}, dataset={dataset_name}, threshold={UNCERTAINTY_THRESHOLD_DB}, samples={len(uncertain)}",
    )

    return len(uncertain)


def enqueue_manual_samples(user, pair_df, source, dataset_name=None):
    """
    Manually send ALL predictions to active learning (currently unused, but kept for flexibility).
    """
    if pair_df.empty:
        return 0

    conn = get_db_connection()
    cur = conn.cursor()
    for _, r in pair_df.iterrows():
        cur.execute(
            """
            INSERT INTO active_learning_samples (
                user_id, username, source, dataset_name, review_id, review_text,
                aspect, opinion, predicted_label, predicted_confidence,
                adjusted_confidence, corrected_label, feedback,
                is_applied, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user["id"],
                user["username"],
                source,
                dataset_name,
                int(r.get("review_id", -1)) if "review_id" in r else -1,
                str(r.get("review_text", "")),
                str(r.get("aspect", "")),
                str(r.get("opinion", "")),
                str(r.get("sentiment_label", "")),
                float(r.get("confidence", 0.0)),
                None,
                None,
                None,
                0,
                datetime.utcnow().isoformat(),
            ),
        )
    conn.commit()
    conn.close()

    log_activity(
        user["id"],
        user["username"],
        "enqueue_active_learning_manual",
        f"source={source}, dataset={dataset_name}, samples={len(pair_df)}",
    )

    return len(pair_df)


def get_uncorrected_active_learning_samples(user, limit=50):
    """
    Fetch all non-applied samples (uncertain + manual) for this user.
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, source, dataset_name, review_text, aspect, opinion,
               predicted_label, predicted_confidence, adjusted_confidence,
               corrected_label, feedback, created_at
        FROM active_learning_samples
        WHERE user_id=?
          AND is_applied=0
          AND (corrected_label IS NULL OR corrected_label='')
        ORDER BY predicted_confidence ASC, created_at ASC
        LIMIT ?
        """,
        (user["id"], limit),
    )
    rows = cur.fetchall()
    conn.close()
    return rows


def apply_corrections_to_df(review_text, df):
    """
    Override sentiment_label using applied corrections for same review_text + aspect.
    """
    if df.empty:
        return df

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT aspect, corrected_label
        FROM active_learning_samples
        WHERE review_text=? AND corrected_label IS NOT NULL AND is_applied=1
        """,
        (review_text,),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        return df

    corrections = {(a.lower() if a else ""): lab for a, lab in rows}

    def override_row(row):
        asp = str(row["aspect"]).lower()
        if asp in corrections:
            row["sentiment_label"] = corrections[asp]
        return row

    return df.apply(override_row, axis=1)


# ============================================================
# NLP & ASPECTS
# ============================================================

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")


@st.cache_resource
def load_vader():
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon")
        return SentimentIntensityAnalyzer()


nlp = load_nlp()
sia = load_vader()

# Expanded opinion lexicons
POSITIVE_OPINIONS = {
    "good", "great", "amazing", "excellent", "fantastic", "impressive",
    "smooth", "fast", "bright", "vibrant", "comfortable", "tasty",
    "engaging", "neat", "perfect", "awesome", "superb", "nice",
    "responsive", "reliable", "satisfying", "lovely", "pleasant"
}
NEGATIVE_OPINIONS = {
    "bad", "worst", "poor", "terrible", "awful", "slow", "confusing",
    "damaged", "torn", "rude", "average", "frustrating", "delayed",
    "noisy", "inaccurate", "salty", "bland", "cheap", "crashing",
    "under cooked", "undercooked", "missing", "loud", "buggy",
    "laggy", "disappointing", "oily", "greasy", "overpriced", "stale"
}

# Expanded aspect keywords
aspect_keywords = {
    "battery": [
        "battery", "battery life", "backup", "drains", "standby", "charge",
        "charging", "optimization", "power", "screen on time", "drain"
    ],
    "camera": [
        "camera", "front camera", "selfie", "photo", "picture", "night mode",
        "video stabilization", "shutter", "shutter lag", "focus", "focusing",
        "colors", "images", "shots", "zoom"
    ],
    "display": [
        "display", "screen", "screen brightness", "brightness", "resolution",
        "viewing angles", "picture quality", "yellow tint", "panel", "touchscreen"
    ],
    "performance": [
        "performance", "smooth", "fast", "slow", "lag", "laggy", "heats up", "heat",
        "gaming", "gaming performance", "crashes", "freezing", "multitasking",
        "speed", "response time"
    ],
    "delivery": [
        "delivery", "courier", "delivered", "arrival", "tracking",
        "tracking information", "partner", "delayed", "shipping",
        "logistics", "arrival time"
    ],
    "packaging": [
        "packaging", "package", "box", "parcel", "torn", "damaged",
        "eco-friendly", "neat", "bubble wrap", "seal", "cover"
    ],
    "product": [
        "product", "item", "design", "quality", "defective", "premium",
        "cheap", "tv", "laptop", "headphones", "smartwatch", "ebook reader",
        "build", "build quality", "material", "finish"
    ],
    "customer_service": [
        "customer service", "customer support", "support", "customer care",
        "agents", "helpdesk", "live chat", "call", "billing issue",
        "helpline", "representative"
    ],
    "app": [
        "app", "application", "mobile app", "navigation", "interface", "ui",
        "dark mode", "login", "shuffle", "buttons", "features", "page",
        "keeps logging", "verification emails", "app performance"
    ],
    "website": [
        "website", "site", "order", "order page", "description", "misleading",
        "layout", "crashing", "webpage", "checkout", "cart"
    ],
    "restaurant": [
        "restaurant", "staff", "service", "ambience", "seating",
        "background music", "waiting time", "host", "waiter", "server"
    ],
    "food": [
        "food", "meal", "desserts", "main course", "taste", "salty", "bland",
        "portions", "dish", "curry", "pizza", "burger", "snacks"
    ],
    "audio": [
        "headphones", "sound", "sound quality", "volume", "leakage", "audio",
        "bass", "treble", "mic", "microphone"
    ],
    "watch": [
        "smartwatch", "step tracking", "steps", "strap", "watch", "dial"
    ],
    "media": [
        "movie", "film", "story", "storyline", "ending", "playlist",
        "music", "songs", "ads", "series", "episode"
    ],
    "book": [
        "book", "novel", "chapters", "story", "writing"
    ],
}


def map_token_to_canonical(word: str):
    lw = word.lower()
    for canonical, synonyms in aspect_keywords.items():
        if lw == canonical:
            return canonical
        for s in synonyms:
            if lw == s.lower():
                return canonical
    return None


def detect_rule_based_aspects(text: str):
    text_low = text.lower()
    detected = set()
    for canonical, synonyms in aspect_keywords.items():
        for w in synonyms:
            if w.lower() in text_low:
                detected.add(canonical)
                break
    return list(detected)


def dependency_aspect_opinion_pairs(text: str):
    doc = nlp(text)
    pairs = []

    for token in doc:
        if token.dep_ == "amod" and token.head.pos_ in ("NOUN", "PROPN"):
            aspect_candidate = token.head.lemma_
            canonical = map_token_to_canonical(aspect_candidate)
            if canonical:
                pairs.append((canonical, token.text))

        if token.pos_ == "ADJ":
            for child in token.children:
                if child.dep_ in ("nsubj", "nsubjpass") and child.pos_ in ("NOUN", "PROPN"):
                    canonical = map_token_to_canonical(child.lemma_)
                    if canonical:
                        pairs.append((canonical, token.text))

    uniq = []
    seen = set()
    for a, o in pairs:
        key = (a.lower(), o.lower())
        if key not in seen:
            seen.add(key)
            uniq.append((a, o))
    return uniq


def lexicon_aspect_opinion_pairs(text: str):
    doc = nlp(text)
    aspects = []
    opinions = []

    for token in doc:
        canon = map_token_to_canonical(token.lemma_)
        if canon:
            aspects.append((token.i, canon))

        lemma = token.lemma_.lower()
        if lemma in POSITIVE_OPINIONS or lemma in NEGATIVE_OPINIONS:
            opinions.append(token)

    pairs = []
    for idx, asp in aspects:
        best_op = None
        best_dist = 999
        for op_tok in opinions:
            dist = abs(op_tok.i - idx)
            if dist <= 5 and dist < best_dist:
                best_dist = dist
                best_op = op_tok
        if best_op is not None:
            pairs.append((asp, best_op.text))

    uniq = []
    seen = set()
    for a, o in pairs:
        key = (a.lower(), o.lower())
        if key not in seen:
            seen.add(key)
            uniq.append((a, o))
    return uniq


def get_sentiment_label(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


def sentiment_for_phrase(aspect: str, opinion: str | None, full_text: str):
    phrase = f"{aspect} {opinion}" if opinion and opinion != "(keyword only)" else full_text
    vs = sia.polarity_scores(phrase)
    score = vs["compound"]

    if opinion:
        lower = opinion.lower()
        if score == 0 and lower in NEGATIVE_OPINIONS:
            score = -0.8
        elif score == 0 and lower in POSITIVE_OPINIONS:
            score = 0.8

    label = get_sentiment_label(score)
    confidence = max(vs["pos"], vs["neg"], vs["neu"])
    return score, confidence, label


def run_absa_on_text(text: str):
    text = str(text)

    rb_aspects = detect_rule_based_aspects(text)
    dep_pairs = dependency_aspect_opinion_pairs(text)
    lex_pairs = lexicon_aspect_opinion_pairs(text)

    combined_pairs = []
    seen = set()
    for a, o in dep_pairs + lex_pairs:
        key = (a.lower(), o.lower())
        if key not in seen:
            seen.add(key)
            combined_pairs.append((a, o))

    records = []

    if combined_pairs:
        for asp, opinion in combined_pairs:
            score, conf, label = sentiment_for_phrase(asp, opinion, text)
            records.append({
                "aspect": asp,
                "opinion": opinion,
                "sentiment_score": score,
                "confidence": conf,
                "sentiment_label": label,
                "method": "pair"
            })
    elif rb_aspects:
        for asp in rb_aspects:
            score, conf, label = sentiment_for_phrase(asp, "(keyword only)", text)
            records.append({
                "aspect": asp,
                "opinion": "(keyword only)",
                "sentiment_score": score,
                "confidence": conf,
                "sentiment_label": label,
                "method": "keyword"
            })

    pair_df = pd.DataFrame(records)
    pair_df = apply_corrections_to_df(text, pair_df)
    return pair_df, rb_aspects


def run_absa_on_dataframe(df: pd.DataFrame, text_col: str):
    all_records = []

    for idx, row in df.iterrows():
        text = str(row[text_col])
        pair_df, rb_aspects = run_absa_on_text(text)

        if pair_df.empty and rb_aspects:
            for asp in rb_aspects:
                score, conf, label = sentiment_for_phrase(asp, "(keyword only)", text)
                all_records.append({
                    "review_id": idx,
                    "review_text": text,
                    "aspect": asp,
                    "opinion": "(keyword only)",
                    "sentiment_score": score,
                    "confidence": conf,
                    "sentiment_label": label,
                    "method": "keyword"
                })
        else:
            for _, r in pair_df.iterrows():
                all_records.append({
                    "review_id": idx,
                    "review_text": text,
                    "aspect": r["aspect"],
                    "opinion": r["opinion"],
                    "sentiment_score": r["sentiment_score"],
                    "confidence": r["confidence"],
                    "sentiment_label": r["sentiment_label"],
                    "method": r["method"]
                })

    pair_df_all = pd.DataFrame(all_records)

    if pair_df_all.empty:
        agg_df = pd.DataFrame(columns=["aspect", "mentions", "positive", "negative", "neutral", "avg_score"])
        return pair_df_all, agg_df

    agg_rows = []
    for aspect, group in pair_df_all.groupby("aspect"):
        mentions = len(group)
        pos = (group["sentiment_label"] == "Positive").sum()
        neg = (group["sentiment_label"] == "Negative").sum()
        neu = (group["sentiment_label"] == "Neutral").sum()
        avg_score = group["sentiment_score"].mean()
        agg_rows.append({
            "aspect": aspect,
            "mentions": mentions,
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "avg_score": avg_score
        })

    agg_df = pd.DataFrame(agg_rows).sort_values("mentions", ascending=False)
    return pair_df_all, agg_df


def highlight_review(text: str, pair_df_for_review: pd.DataFrame) -> str:
    if pair_df_for_review.empty:
        return text

    # Prefer Negative if any negative for that aspect, else Positive, else Neutral
    aspect_sentiment = {}
    for aspect, group in pair_df_for_review.groupby("aspect"):
        labels = list(group["sentiment_label"])
        if "Negative" in labels:
            label = "Negative"
        elif "Positive" in labels:
            label = "Positive"
        else:
            label = "Neutral"
        aspect_sentiment[aspect] = label

    highlighted = text
    for aspect, synonyms in aspect_keywords.items():
        if aspect not in aspect_sentiment:
            continue
        label = aspect_sentiment[aspect]
        css_class = (
            "aspect-positive" if label == "Positive"
            else "aspect-negative" if label == "Negative"
            else "aspect-neutral"
        )
        for word in sorted(synonyms, key=len, reverse=True):
            pattern = r"\b" + re.escape(word) + r"\b"

            def repl(m):
                return f'<span class="{css_class}">{m.group(0)}</span>'

            highlighted = re.sub(pattern, repl, highlighted, flags=re.IGNORECASE)
    return highlighted


# ============================================================
# AUTH (Login / Sign Up / Admin)
# ============================================================

def show_auth_page():
    st.markdown(
        """
        <div style="text-align:center;margin-top:1rem;">
            <h2 style="margin-bottom:0.2rem;">Aspect-Based Sentiment Analysis</h2>
            <p class="muted-text">Milestone 4 • Dashboard · Dataset Analysis · Active Learning · Admin Panel</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    options = ["Login", "Sign Up", "Admin"]
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "Login"
    index = options.index(st.session_state["auth_mode"])

    st.markdown('<div class="auth-card auth-tabs">', unsafe_allow_html=True)
    mode = st.radio(
        "Choose mode",
        options=options,
        index=index,
        horizontal=True,
        label_visibility="collapsed",
        key="auth_mode_radio",
    )
    st.session_state["auth_mode"] = mode

    if mode == "Login":
        st.markdown("#### User Login")
        u = st.text_input("Username", key="login_username")
        p = st.text_input("Password", type="password", key="login_password")
        if st.button("Login to Dashboard", key="login_btn"):
            user = authenticate_user(u, p)
            if user:
                st.session_state["user"] = user
                st.session_state["admin"] = None
                log_activity(user["id"], user["username"], "user_login", "success")
                st.success(f"Logged in as **{user['username']}**")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    elif mode == "Sign Up":
        st.markdown("#### Sign Up")
        full_name = st.text_input("Full Name", key="signup_full_name")
        email = st.text_input("Email", key="signup_email")
        username = st.text_input("Choose a Username", key="signup_username")
        pw = st.text_input("Choose a Password", type="password", key="signup_password")
        pw2 = st.text_input("Confirm Password", type="password", key="signup_password2")
        if st.button("Create Account", key="signup_btn"):
            if not username or not pw:
                st.error("Username and password are required.")
            elif pw != pw2:
                st.error("Passwords do not match.")
            else:
                created = create_user(username, pw, full_name, email)
                if created:
                    st.success("Account created. You can now login.")
                    st.session_state["auth_mode"] = "Login"
                else:
                    st.error("Username already exists. Choose another one.")

    else:  # Admin
        st.markdown("#### Admin Login")
        u = st.text_input("Admin Username", key="admin_username")
        p = st.text_input("Admin Password", type="password", key="admin_password")
        if st.button("Login as Admin", key="admin_login_btn"):
            admin = authenticate_admin(u, p)
            if admin:
                st.session_state["admin"] = admin
                st.session_state["user"] = None
                log_activity(None, admin["username"], "admin_login", "success")
                st.success(f"Logged in as Admin **{admin['username']}**")
                st.rerun()
            else:
                st.error("Invalid admin credentials.")

    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# USER TABS
# ============================================================

def show_dashboard_tab(user):
    st.subheader("🧠 Dashboard — Manual Aspect Analysis")

    st.markdown(
        '<div class="card"><span class="muted-text">'
        "Enter a review. The system detects aspects and opinion words, classifies sentiment "
        "and shows confidence. Low-confidence predictions (confidence < 0.50) are "
        "automatically sent to the Active Learning page for manual correction and retraining."
        "</span></div>",
        unsafe_allow_html=True,
    )

    user_text = st.text_area(
        "Enter a sentence or review:",
        height=140,
        key="dashboard_input"
    )

    analyze_clicked = st.button("Analyze Review", key="analyze_single")

    if not analyze_clicked:
        return

    if not user_text.strip():
        st.warning("Please enter some text.")
        return

    with st.spinner("Analyzing..."):
        pair_df, rb_aspects = run_absa_on_text(user_text)

    if pair_df.empty and not rb_aspects:
        st.info("No clear aspects detected. Try a more detailed sentence.")
        return

    col_a, col_b = st.columns([1.4, 1])

    with col_a:
        st.markdown("#### Highlighted Review")
        html = highlight_review(user_text, pair_df)
        st.markdown(f'<div class="review-card">{html}</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown("#### Aspect–Sentiment Pairs")
        if not pair_df.empty:
            display_df = pair_df[["aspect", "opinion", "sentiment_label", "confidence", "sentiment_score"]].copy()
            display_df["confidence"] = display_df["confidence"].round(3)
            display_df["sentiment_score"] = display_df["sentiment_score"].round(3)
            st.dataframe(display_df, height=220)
        else:
            st.info("No aspect–opinion pairs detected. Only keywords found.")

        if not pair_df.empty:
            counts = pair_df["sentiment_label"].value_counts()
            pos = counts.get("Positive", 0)
            neg = counts.get("Negative", 0)
            neu = counts.get("Neutral", 0)
            st.markdown(
                f"""
                <div>
                    <span class="sentiment-pill pill-pos">Positive: {pos}</span>
                    <span class="sentiment-pill pill-neg">Negative: {neg}</span>
                    <span class="sentiment-pill pill-neu">Neutral: {neu}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not pair_df.empty:
        st.markdown("#### Aspect Confidence Overview")
        chart_df = pair_df.copy()
        chart_df["confidence"] = chart_df["confidence"].astype(float)
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("aspect:N", title="Aspect"),
                y=alt.Y("confidence:Q", title="Confidence (0–1)"),
                color=alt.Color(
                    "sentiment_label:N",
                    scale=alt.Scale(
                        domain=["Positive", "Negative", "Neutral"],
                        range=["#22c55e", "#ef4444", "#eab308"],
                    ),
                    legend=alt.Legend(title="Sentiment", labelColor=TEXT_COLOR, titleColor=TEXT_COLOR),
                ),
                tooltip=["aspect", "opinion", "sentiment_label", "confidence"],
            )
            .properties(height=260, background=BACKGROUND_COLOR)
            .configure_axis(labelColor=TEXT_COLOR, titleColor=TEXT_COLOR, gridColor="#1f2937")
            .configure_view(strokeOpacity=0)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("---")
    if st.button("Save this Analysis to My Profile"):
        result_dict = pair_df.to_dict(orient="records")
        save_single_analysis(user["id"], user_text, result_dict)
        st.success("Analysis saved to your profile.")

    # AUTO-QUEUE low confidence samples ONLY
    if not pair_df.empty:
        q_df = pair_df.copy()
        q_df["review_text"] = user_text
        q_df["review_id"] = -1

        added = enqueue_uncertain_samples(user, q_df, source="single", dataset_name=None)
        if added > 0:
            st.info(f"{added} low-confidence predictions (confidence < 0.50) were auto-sent to Active Learning.")
        else:
            st.info("No predictions were below 0.50 confidence.")


def show_dataset_tab(user):
    st.subheader("📊 Dataset Analysis")

    st.markdown(
        '<div class="card"><span class="muted-text">'
        "Upload a CSV with reviews. Aspect-based sentiment analysis is run on every row. "
        "Aggregated aspect statistics and charts are shown. Predictions with confidence < 0.50 "
        "are automatically queued to the Active Learning page for manual correction."
        "</span></div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload CSV dataset", type=["csv"], key="dataset_uploader")

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return

        object_cols = [c for c in df.columns if df[c].dtype == "object"]
        if not object_cols:
            st.error("No text columns detected. At least one column must contain text.")
            return

        text_col = st.selectbox("Select text column:", options=object_cols, index=0)

        if st.button("Run Dataset Analysis"):
            with st.spinner("Running ABSA over dataset..."):
                pair_df_all, agg_df = run_absa_on_dataframe(df, text_col)

            st.session_state["dataset_pair_df"] = pair_df_all
            st.session_state["dataset_agg_df"] = agg_df
            st.session_state["dataset_name"] = getattr(uploaded, "name", "uploaded_dataset.csv")
            st.session_state["dataset_n_rows"] = len(df)

            log_activity(
                user["id"],
                user["username"],
                "run_dataset_analysis",
                f"dataset={st.session_state['dataset_name']} rows={len(df)}",
            )

            if not pair_df_all.empty:
                added = enqueue_uncertain_samples(
                    user,
                    pair_df_all,
                    source="dataset",
                    dataset_name=st.session_state["dataset_name"],
                )
                if added > 0:
                    st.info(f"{added} low-confidence predictions (confidence < 0.50) queued to Active Learning.")
                else:
                    st.info("No predictions were below 0.50 confidence.")

    if (
        "dataset_pair_df" in st.session_state
        and isinstance(st.session_state["dataset_pair_df"], pd.DataFrame)
        and not st.session_state["dataset_pair_df"].empty
    ):
        pair_df_all = st.session_state["dataset_pair_df"]
        agg_df = st.session_state["dataset_agg_df"]

        st.markdown("### Aspect Summary")

        col1, col2 = st.columns([1.3, 1])

        with col1:
            st.markdown("#### Aggregated Aspect Table")
            display_agg = agg_df.copy()
            display_agg["avg_score"] = display_agg["avg_score"].round(3)
            st.dataframe(display_agg.reset_index(drop=True))

        with col2:
            st.markdown("#### Mentions per Aspect")
            chart1 = (
                alt.Chart(agg_df)
                .mark_bar()
                .encode(
                    x=alt.X("aspect:N", title="Aspect"),
                    y=alt.Y("mentions:Q", title="Mentions"),
                )
                .properties(height=220, background=BACKGROUND_COLOR)
                .configure_axis(labelColor=TEXT_COLOR, titleColor=TEXT_COLOR, gridColor="#1f2937")
                .configure_view(strokeOpacity=0)
            )
            st.altair_chart(chart1, use_container_width=True)

            st.markdown("#### Positive / Negative / Neutral Counts")
            counts_long = agg_df.melt(
                id_vars=["aspect"],
                value_vars=["positive", "negative", "neutral"],
                var_name="sentiment",
                value_name="count",
            )
            chart2 = (
                alt.Chart(counts_long)
                .mark_bar()
                .encode(
                    x=alt.X("aspect:N", title="Aspect"),
                    y=alt.Y("count:Q", title="Count"),
                    color=alt.Color(
                        "sentiment:N",
                        scale=alt.Scale(
                            domain=["positive", "negative", "neutral"],
                            range=["#22c55e", "#ef4444", "#eab308"],
                        ),
                        legend=alt.Legend(title="Sentiment", labelColor=TEXT_COLOR, titleColor=TEXT_COLOR),
                    ),
                )
                .properties(height=220, background=BACKGROUND_COLOR)
                .configure_axis(labelColor=TEXT_COLOR, titleColor=TEXT_COLOR, gridColor="#1f2937")
                .configure_view(strokeOpacity=0)
            )
            st.altair_chart(chart2, use_container_width=True)

        st.markdown("### Detailed Aspect–Sentiment Pairs")
        st.dataframe(
            pair_df_all[["review_id", "aspect", "opinion", "sentiment_label", "confidence", "sentiment_score", "review_text"]],
            height=320
        )

        st.markdown("---")
        if st.button("Save Aggregated Results to Database"):
            save_dataset_analysis(
                user["id"],
                st.session_state.get("dataset_name", "uploaded_dataset.csv"),
                st.session_state.get("dataset_n_rows", 0),
                agg_df,
            )
            log_activity(
                user["id"],
                user["username"],
                "save_dataset_summary",
                f"dataset={st.session_state.get('dataset_name')}",
            )
            st.success("Dataset analysis summary saved to your profile history.")

        st.markdown("### My Saved Dataset Analyses")
        history_rows = get_user_dataset_history(user["id"])
        if history_rows:
            hist_df = pd.DataFrame(history_rows, columns=["ID", "Dataset Name", "Rows", "Created At"])
            st.dataframe(hist_df)
        else:
            st.info("No dataset analyses saved yet.")


def show_active_learning_tab(user):
    st.subheader("🔁 Active Learning — Correct & Adjust Predictions")

    st.markdown(
        '<div class="card"><span class="muted-text">'
        "This page shows predictions that were auto-queued as uncertain (confidence < 0.50) "
        "from the Dashboard and Dataset Analysis pages. "
        "For uncertain predictions, you can adjust the confidence and correct the sentiment "
        "to simulate retraining and make the model smarter."
        "</span></div>",
        unsafe_allow_html=True,
    )

    rows = get_uncorrected_active_learning_samples(user, limit=50)

    if not rows:
        st.info("No pending samples. Run new analyses or upload datasets.")
        return

    st.markdown(f"**Pending samples:** {len(rows)} (showing up to 50)")

    with st.form("active_learning_form"):
        updated_items = []

        for idx, row in enumerate(rows):
            (
                sample_id,
                source,
                dataset_name,
                review_text,
                aspect,
                opinion,
                pred_label,
                pred_conf,
                adjusted_conf,
                corrected_label,
                feedback,
                created_at,
            ) = row

            is_uncertain = (pred_conf is not None) and (pred_conf < UNCERTAINTY_THRESHOLD_DB)

            st.markdown("---")
            header_badge = "Uncertain" if is_uncertain else "Manual / High confidence"
            st.markdown(
                f"**Sample #{idx+1}** • Source: `{source}` • Dataset: `{dataset_name or 'N/A'}` "
                f"• <span class='muted-text'>{header_badge}</span>",
                unsafe_allow_html=True,
            )
            st.markdown(f"<div class='review-card'>{review_text}</div>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns([1, 1, 1.8])
            with c1:
                st.markdown(f"**Aspect:** `{aspect}`")
                st.markdown(f"**Opinion:** `{opinion}`")
            with c2:
                st.markdown(f"**Predicted Sentiment:** `{pred_label}`")
                st.markdown(f"**Original Confidence:** `{pred_conf:.3f}`")
            with c3:
                corr_key = f"corr_label_{sample_id}"
                fb_key = f"feedback_{sample_id}"
                conf_key = f"adj_conf_{sample_id}"

                corr_default = corrected_label if corrected_label else "Keep prediction"
                options = ["Keep prediction", "Positive", "Negative", "Neutral"]
                corr_choice = st.selectbox(
                    "Correct sentiment",
                    options=options,
                    index=options.index(corr_default) if corr_default in options else 0,
                    key=corr_key,
                )

                if is_uncertain:
                    default_conf = adjusted_conf if adjusted_conf is not None else max(pred_conf, 0.1)
                    adj_conf = st.slider(
                        "Adjusted confidence after retraining (only for uncertain samples)",
                        min_value=0.0,
                        max_value=1.0,
                        value=float(default_conf),
                        step=0.05,
                        key=conf_key,
                    )
                else:
                    st.caption(
                        "This is a high-confidence/manual sample. "
                        "Confidence adjustment is not required; only label correction is used."
                    )
                    adj_conf = adjusted_conf if adjusted_conf is not None else pred_conf

                fb_value = st.text_input("Feedback / remarks (optional)", value=feedback or "", key=fb_key)

                updated_items.append((sample_id, corr_choice, fb_value, adj_conf))

        submitted = st.form_submit_button("💾 Save Corrections")
        if submitted:
            conn = get_db_connection()
            cur = conn.cursor()
            n_changed = 0
            for sample_id, corr_choice, fb_value, adj_conf in updated_items:
                corrected_label = None if corr_choice == "Keep prediction" else corr_choice
                now = datetime.utcnow().isoformat()
                cur.execute(
                    """
                    UPDATE active_learning_samples
                    SET corrected_label=?, feedback=?, adjusted_confidence=?, corrected_at=?
                    WHERE id=?
                    """,
                    (corrected_label, fb_value, float(adj_conf) if adj_conf is not None else None, now, sample_id),
                )
                n_changed += 1
            conn.commit()
            conn.close()

            if n_changed > 0:
                log_activity(user["id"], user["username"], "correct_feedback", f"samples_corrected={n_changed}")
                st.success(f"Saved corrections for {n_changed} samples.")
                st.rerun()

    st.markdown("---")
    st.markdown("### 🔄 Retrain Model with Corrected Samples")

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COUNT(*) FROM active_learning_samples
        WHERE user_id=? AND corrected_label IS NOT NULL AND is_applied=0
        """,
        (user["id"],),
    )
    pending_for_retrain = cur.fetchone()[0]
    conn.close()

    st.markdown(
        f"<span class='muted-text'>Corrected samples not yet used in retraining: <b>{pending_for_retrain}</b></span>",
        unsafe_allow_html=True,
    )

    if pending_for_retrain > 0:
        if st.button("🚀 Retrain Model Using Corrected Samples"):
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute(
                """
                UPDATE active_learning_samples
                SET is_applied=1,
                    predicted_confidence = COALESCE(adjusted_confidence, predicted_confidence)
                WHERE user_id=? AND corrected_label IS NOT NULL AND is_applied=0
                """,
                (user["id"],),
            )
            conn.commit()
            conn.close()

            create_retraining_log(user["id"], user["username"], pending_for_retrain,
                                  "Applied corrected labels & adjusted confidence.")
            log_activity(user["id"], user["username"], "retrain_model",
                         f"samples_used={pending_for_retrain}")

            st.success("Retraining recorded. Adjusted confidence applied to corrected samples.")
    else:
        st.info("No corrected samples ready for retraining yet.")


def show_profile_tab(user):
    st.subheader("👤 Profile")

    st.markdown(
        '<div class="card"><span class="muted-text">'
        "Update your profile, change your password and view saved analyses."
        "</span></div>",
        unsafe_allow_html=True,
    )

    with st.form("profile_form"):
        full_name = st.text_input("Full Name", value=user.get("full_name") or "")
        email = st.text_input("Email", value=user.get("email") or "")
        submitted = st.form_submit_button("Update Profile")
        if submitted:
            update_user_profile(user["id"], full_name, email)
            st.success("Profile updated.")

    st.markdown("### Change Password")
    with st.form("password_form"):
        new_pwd = st.text_input("New Password", type="password")
        new_pwd2 = st.text_input("Confirm New Password", type="password")
        pwd_submitted = st.form_submit_button("Update Password")
        if pwd_submitted:
            if not new_pwd:
                st.error("Password cannot be empty.")
            elif new_pwd != new_pwd2:
                st.error("Passwords do not match.")
            else:
                update_password(user["id"], new_pwd)
                st.success("Password updated successfully.")

    st.markdown("### Recently Saved Manual Analyses")
    saved = get_user_saved_analyses(user["id"])
    if saved:
        saved_df = pd.DataFrame(saved, columns=["ID", "Input Text", "Created At"])
        st.dataframe(saved_df)
    else:
        st.info("No saved manual analyses yet.")


# ============================================================
# ADMIN PANEL
# ============================================================

def compute_admin_sentiment_trends():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT dataset_name, n_rows, aspect_agg_json, created_at FROM dataset_results", conn)
    conn.close()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    rows_trend = []
    rows_aspect = []
    for _, row in df.iterrows():
        created = row["created_at"][:10]
        agg_json = row["aspect_agg_json"]
        aspect_df = pd.read_json(agg_json)
        totals = {
            "positive": aspect_df["positive"].sum(),
            "negative": aspect_df["negative"].sum(),
            "neutral": aspect_df["neutral"].sum(),
        }
        rows_trend.append({
            "date": created,
            "positive": totals["positive"],
            "negative": totals["negative"],
            "neutral": totals["neutral"],
        })
        aspect_df["date"] = created
        rows_aspect.append(aspect_df)

    trend_df = pd.DataFrame(rows_trend)
    aspects_df = pd.concat(rows_aspect, ignore_index=True) if rows_aspect else pd.DataFrame()
    return trend_df, aspects_df


def show_admin_panel(admin):
    st.subheader(f"🔧 Admin Panel — {admin['username']}")

    tabs = st.tabs(["Dashboard", "Analytics", "Users", "Logs & Activity"])

    # Dashboard
    with tabs[0]:
        trend_df, aspects_df = compute_admin_sentiment_trends()

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT SUM(n_rows) FROM dataset_results")
        total_reviews = cur.fetchone()[0] or 0
        cur.execute("SELECT COUNT(DISTINCT aspect) FROM active_learning_samples WHERE aspect IS NOT NULL")
        aspect_count = cur.fetchone()[0] or 0
        cur.execute("SELECT COUNT(*) FROM active_learning_samples")
        total_al = cur.fetchone()[0] or 0   # kept for accuracy calc
        cur.execute("SELECT COUNT(*) FROM active_learning_samples WHERE corrected_label IS NOT NULL")
        corrected = cur.fetchone()[0] or 0
        cur.execute("SELECT COUNT(*) FROM users")
        total_users = cur.fetchone()[0] or 0
        cur.execute("SELECT COUNT(*) FROM dataset_results")
        dataset_count = cur.fetchone()[0] or 0
        conn.close()

        accuracy = 1.0
        if total_al > 0:
            accuracy = max(0.0, 1.0 - corrected / total_al)

        top1, top2, top3, top4 = st.columns(4)
        top1.metric("Total Reviews Analyzed", int(total_reviews))
        top2.metric("Distinct Aspects", int(aspect_count))
        top3.metric("Total Users", int(total_users))
        top4.metric("Approx. Accuracy", f"{accuracy*100:.1f}%")

        c1, c2 = st.columns([1.4, 1])

        with c1:
            st.markdown("#### Sentiment Trends Over Time")
            if not trend_df.empty:
                trend_long = trend_df.melt(
                    id_vars=["date"],
                    value_vars=["positive", "negative", "neutral"],
                    var_name="sentiment",
                    value_name="count",
                )
                chart = (
                    alt.Chart(trend_long)
                    .mark_line(point=True)
                    .encode(
                        x=alt.X("date:T", title="Date"),
                        y=alt.Y("count:Q", title="Mentions"),
                        color=alt.Color(
                            "sentiment:N",
                            scale=alt.Scale(
                                domain=["positive", "negative", "neutral"],
                                range=["#22c55e", "#ef4444", "#eab308"],
                            ),
                            legend=alt.Legend(title="Sentiment", labelColor=TEXT_COLOR, titleColor=TEXT_COLOR),
                        ),
                        tooltip=["date", "sentiment", "count"],
                    )
                    .properties(height=260, background=BACKGROUND_COLOR)
                    .configure_axis(labelColor=TEXT_COLOR, titleColor=TEXT_COLOR, gridColor="#1f2937")
                    .configure_view(strokeOpacity=0)
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No dataset analyses yet to show trends.")

        with c2:
            st.markdown("#### Admin Overview")
            st.write(f"- **Users:** {total_users}")
            st.write(f"- **Datasets:** {dataset_count}")

            if st.button("Export Active Learning Samples (CSV)"):
                conn = get_db_connection()
                al_df = pd.read_sql_query("SELECT * FROM active_learning_samples", conn)
                conn.close()
                csv = al_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Active Learning CSV",
                    data=csv,
                    file_name="active_learning_samples.csv",
                    mime="text/csv",
                )

        st.markdown("### Aspect Sentiment Distribution")
        if not aspects_df.empty:
            aspects_agg = aspects_df.groupby("aspect")[["positive", "negative"]].sum().reset_index()
            aspects_long = aspects_agg.melt(
                id_vars=["aspect"],
                value_vars=["positive", "negative"],
                var_name="sentiment",
                value_name="count",
            )
            chart = (
                alt.Chart(aspects_long)
                .mark_bar()
                .encode(
                    x=alt.X("aspect:N", title="Aspect"),
                    y=alt.Y("count:Q", title="Mentions"),
                    color=alt.Color(
                        "sentiment:N",
                        scale=alt.Scale(domain=["positive", "negative"], range=["#22c55e", "#ef4444"]),
                        legend=alt.Legend(title="Sentiment", labelColor=TEXT_COLOR, titleColor=TEXT_COLOR),
                    ),
                    tooltip=["aspect", "sentiment", "count"],
                )
                .properties(height=260, background=BACKGROUND_COLOR)
                .configure_axis(labelColor=TEXT_COLOR, titleColor=TEXT_COLOR, gridColor="#1f2937")
                .configure_view(strokeOpacity=0)
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No aspect distribution available yet.")

        # NEW: Uploaded datasets overview
        st.markdown("### Uploaded Datasets")
        conn = get_db_connection()
        datasets_df = pd.read_sql_query(
            """
            SELECT d.id,
                   COALESCE(u.username, 'unknown') AS username,
                   d.dataset_name,
                   d.n_rows,
                   d.created_at
            FROM dataset_results d
            LEFT JOIN users u ON d.user_id = u.id
            ORDER BY d.created_at DESC
            """,
            conn,
        )
        conn.close()
        if datasets_df.empty:
            st.info("No datasets uploaded yet.")
        else:
            st.dataframe(datasets_df)

    # Analytics
    with tabs[1]:
        st.markdown("### Analytics — Aspect-Level Insights")

        _, aspects_df = compute_admin_sentiment_trends()
        if aspects_df.empty:
            st.info("No dataset analyses yet.")
        else:
            aspects_agg = aspects_df.groupby("aspect")[["mentions", "positive", "negative", "neutral"]].sum().reset_index()
            aspects_agg["avg_score"] = (
                aspects_df.groupby("aspect")["avg_score"].mean().reindex(aspects_agg["aspect"]).values
            )
            aspects_agg = aspects_agg.sort_values("mentions", ascending=False)

            st.markdown("#### Top Aspects by Mentions")
            st.dataframe(aspects_agg.head(20))

            st.markdown("#### Positive vs Negative by Aspect")
            posneg_long = aspects_agg.melt(
                id_vars=["aspect"],
                value_vars=["positive", "negative"],
                var_name="sentiment",
                value_name="count",
            )
            chart = (
                alt.Chart(posneg_long)
                .mark_bar()
                .encode(
                    x=alt.X("aspect:N", title="Aspect"),
                    y=alt.Y("count:Q", title="Mentions"),
                    color=alt.Color(
                        "sentiment:N",
                        scale=alt.Scale(domain=["positive", "negative"], range=["#22c55e", "#ef4444"]),
                        legend=alt.Legend(title="Sentiment", labelColor=TEXT_COLOR, titleColor=TEXT_COLOR),
                    ),
                )
                .properties(height=260, background=BACKGROUND_COLOR)
                .configure_axis(labelColor=TEXT_COLOR, titleColor=TEXT_COLOR, gridColor="#1f2937")
                .configure_view(strokeOpacity=0)
            )
            st.altair_chart(chart, use_container_width=True)

    # Users
    with tabs[2]:
        st.markdown("### Users Management")
        conn = get_db_connection()
        users_df = pd.read_sql_query("SELECT id, username, full_name, email FROM users ORDER BY id", conn)
        conn.close()

        if users_df.empty:
            st.info("No registered users yet.")
        else:
            st.dataframe(users_df)

            labels = [f"{row['id']} - {row['username']}" for _, row in users_df.iterrows()]
            to_delete = st.selectbox("Select user to remove", options=["None"] + labels)
            if to_delete != "None":
                if st.button("Remove Selected User"):
                    user_id = int(to_delete.split(" - ")[0])
                    delete_user(user_id)
                    log_activity(None, admin["username"], "admin_delete_user", f"user_id={user_id}")
                    st.success("User removed.")
                    st.rerun()

    # Logs & Activity
    with tabs[3]:
        st.markdown("### Activity Logs")
        conn = get_db_connection()
        logs_df = pd.read_sql_query("SELECT * FROM activity_logs ORDER BY created_at DESC", conn)
        retrain_df = pd.read_sql_query("SELECT * FROM model_retraining_logs ORDER BY created_at DESC", conn)
        conn.close()

        if logs_df.empty:
            st.info("No logs yet.")
        else:
            st.dataframe(logs_df)

            logs_df["username"] = logs_df["username"].fillna("system")

            st.markdown("#### User & Admin Activity (Events per Account)")
            user_agg = logs_df.groupby("username").size().reset_index(name="events")
            user_chart = (
                alt.Chart(user_agg)
                .mark_bar()
                .encode(
                    x=alt.X("username:N", title="User/Admin"),
                    y=alt.Y("events:Q", title="Events"),
                    tooltip=["username", "events"],
                )
                .properties(height=260, background=BACKGROUND_COLOR)
                .configure_axis(labelColor=TEXT_COLOR, titleColor=TEXT_COLOR, gridColor="#1f2937")
                .configure_view(strokeOpacity=0)
            )
            st.altair_chart(user_chart, use_container_width=True)

            st.markdown("#### Activity by Type")
            action_agg = logs_df.groupby("action").size().reset_index(name="events")
            action_chart = (
                alt.Chart(action_agg)
                .mark_bar()
                .encode(
                    x=alt.X("action:N", title="Action"),
                    y=alt.Y("events:Q", title="Events"),
                    tooltip=["action", "events"],
                )
                .properties(height=260, background=BACKGROUND_COLOR)
                .configure_axis(labelColor=TEXT_COLOR, titleColor=TEXT_COLOR, gridColor="#1f2937")
                .configure_view(strokeOpacity=0)
            )
            st.altair_chart(action_chart, use_container_width=True)

        st.markdown("### Model Retraining History")
        if retrain_df.empty:
            st.info("No retraining events yet.")
        else:
            st.dataframe(retrain_df)
            retrain_df["date"] = retrain_df["created_at"].str.slice(0, 10)
            chart = (
                alt.Chart(retrain_df)
                .mark_bar()
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("n_samples_used:Q", title="Corrected samples used"),
                    tooltip=["date", "n_samples_used", "username"],
                )
                .properties(height=260, background=BACKGROUND_COLOR)
                .configure_axis(labelColor=TEXT_COLOR, titleColor=TEXT_COLOR, gridColor="#1f2937")
                .configure_view(strokeOpacity=0)
            )
            st.altair_chart(chart, use_container_width=True)


# ============================================================
# MAIN
# ============================================================

def main():
    init_db()

    st.markdown(
        """
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.8rem;">
            <div>
                <h2 style="margin-bottom:0;">Aspect-Based Sentiment Analysis (ABSA) — Milestone 4</h2>
                <p class="muted-text" style="margin-top:4px;">
                    Milestone 3 + Active Learning + Admin Dashboard • Interactive Insights & Feedback-driven Improvement
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if ("user" not in st.session_state or st.session_state["user"] is None) and \
       ("admin" not in st.session_state or st.session_state["admin"] is None):
        show_auth_page()
        return

    # Admin session
    if st.session_state.get("admin") is not None:
        admin = st.session_state["admin"]
        c1, c2 = st.columns([3, 1])
        with c1:
            st.markdown(
                f"<span class='muted-text'>Logged in as Admin <b>{admin['username']}</b></span>",
                unsafe_allow_html=True,
            )
        with c2:
            if st.button("Logout"):
                log_activity(None, admin["username"], "admin_logout", "")
                st.session_state["admin"] = None
                st.rerun()
        show_admin_panel(admin)
        return

    # User session
    user = st.session_state["user"]
    c1, c2 = st.columns([3, 1])
    with c1:
        st.markdown(
            f"<span class='muted-text'>Logged in as <b>{user['username']}</b></span>",
            unsafe_allow_html=True,
        )
    with c2:
        if st.button("Logout"):
            log_activity(user["id"], user["username"], "user_logout", "")
            st.session_state["user"] = None
            st.rerun()

    tabs = st.tabs(["Dashboard", "Dataset Analysis", "Active Learning", "Profile"])

    with tabs[0]:
        show_dashboard_tab(user)
    with tabs[1]:
        show_dataset_tab(user)
    with tabs[2]:
        show_active_learning_tab(user)
    with tabs[3]:
        show_profile_tab(user)


if __name__ == "__main__":
    main()
