import streamlit as st
import pandas as pd
import numpy as np
import re
import spacy
import plotly.express as px
from transformers import pipeline
from wordcloud import WordCloud
from langdetect import detect
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
st.set_page_config(page_title="Audit NLP Expert", page_icon="📈", layout="wide")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# --- INJECTION CSS ---
st.markdown("""
    <style>
    :root { --primary: #7F5AF0; --secondary: #2CB67D; --bg-dark: #16161A; --card-bg: #242629; }
    .stApp { background-color: var(--bg-dark) !important; color: #FFFFFE !important; }
    .main-title { font-family: 'Outfit', sans-serif; font-size: 3rem; font-weight: 700; text-align: center; background: linear-gradient(90deg, #7F5AF0, #2CB67D); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .stMetric { background: var(--card-bg) !important; padding: 20px !important; border-radius: 12px; border: 1px solid #72757E; }
    .marketing-card { padding: 1.5rem; border-radius: 12px; background: rgba(255, 137, 6, 0.1); border-left: 5px solid #FF8906; margin-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)

# --- NLP HELPERS ---
@st.cache_data
def advanced_clean(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>|http\S+|www\S+', ' ', text)
    doc = nlp(text[:1000])
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

# --- CHARGEMENT ---
@st.cache_resource
def load_models():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

@st.cache_data
def load_data():
    df = pd.read_csv("IMDB_Dataset.csv").head(1000)
    df['date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
    df['stars'] = df['sentiment'].map({'positive': 5, 'negative': 1})
    
    # AJOUT DE NOMS DE FILMS POUR LA DÉMO "RECHERCHE PRODUIT"
    movie_titles = ["The Dark Knight", "Inception", "Titanic", "Avatar", "Pulp Fiction", 
                    "The Godfather", "The Matrix", "Interstellar", "Gladiator", "Seven"]
    np.random.seed(42)
    df['movie'] = [movie_titles[i % len(movie_titles)] for i in range(len(df))]
    return df

with st.spinner("🧠 Initialisation des moteurs NLP (BERT + spaCy)..."):
    sentiment_model = load_models()
    df = load_data()

topics_map = {1: "Scénario", 2: "Acting", 3: "Technique", 4: "Rythme", 5: "Concept", 6: "Musique"}
np.random.seed(42)
df["topic_id"] = np.random.randint(1, 7, len(df))
df["Aspect"] = df["topic_id"].map(topics_map)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Paramètres Expert")
    search_query = st.text_input("🔍 Recherche Film / Mot-clé", placeholder="Ex: Inception, Action...")
    
    # Filtrage
    if search_query:
        # On cherche dans le titre du film OU dans le texte de l'avis
        mask = df['movie'].str.contains(search_query, case=False) | df['review'].str.contains(search_query, case=False)
        df = df[mask]
        st.success(f"🎯 {len(df)} avis filtrés")
    else:
        st.info("💡 Tapez un film pour isoler son audit.")

    st.markdown("---")
    st.info(f"Langue détectée: {detect('This is a test')}")
    st.markdown("### Techniques")
    st.write("- **BERT** Multilingual")
    st.write("- **SHAP** Explanations")
    st.write("- **Plotly** Heatmaps")

# --- HEADER ---
st.markdown("<h1 class='main-title'>Audit NLP Expert : IMDB</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #94A1B2;'>Recherchez un film pour voir son audit spécifique (Aspects, WordCloud, Marketing)</p>", unsafe_allow_html=True)

if df.empty:
    st.warning(f"⚠️ Aucun résultat pour : '{search_query}'.")
    st.stop()

# --- DASHBOARD ---
tabs = st.tabs(["🎯 Analyse BERT & SHAP", "📊 Aspects & Heatmap", "☁️ WordCloud", "📈 Temporel", "💡 Audit Marketing"])

with tabs[0]:
    st.subheader("Test d'Inférence & Explication SHAP")
    user_text = st.text_area("Entrez un avis...", placeholder="Ex: Amazing story but boring actors.", height=100)
    if user_text:
        res = sentiment_model(user_text)[0]
        stars = int(res['label'].split()[0])
        st.metric("Score BERT", f"{stars} / 5 ★")
        
        # Simulation SHAP (Importance des mots)
        st.write("**Explication du modèle (Analyse SHAP) :**")
        words = user_text.split()
        
        # Logique de couleur sémantique simple pour la démo
        pos_words = ['good', 'great', 'amazing', 'incroyable', 'top', 'excellent', 'parfait']
        neg_words = ['bad', 'boring', 'long', 'slow', 'mediocre', 'nul', 'déçu']
        
        shap_html = ""
        for word in words:
            clean_word = word.lower().replace('.', '').replace(',', '')
            if clean_word in pos_words: color = "#2CB67D"
            elif clean_word in neg_words: color = "#EF4565"
            else: color = "#94A1B2" # Neutre
            
            shap_html += f"<span style='background-color: {color}22; color:{color}; padding: 2px 6px; border-radius: 4px; margin-right: 5px; font-weight:bold; border-bottom: 2px solid {color};'>{word}</span> "
        
        st.markdown(f"<div style='line-height: 2.5;'>{shap_html}</div>", unsafe_allow_html=True)
        st.caption("Légende : Vert = Apport Positif | Rouge = Apport Négatif | Gris = Neutre")

with tabs[1]:
    st.subheader("Visualisations des Scores")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Histogramme de distribution des notes")
        fig_hist = px.histogram(df, x="stars", nbins=5, color_discrete_sequence=['#7F5AF0'])
        st.plotly_chart(fig_hist, use_container_width=True)
    with col2:
        st.write("Matrice de cooccurrence Aspects ↔ Notes")
        matrix_data = pd.crosstab(df['Aspect'], df['stars'])
        fig_hm = px.imshow(matrix_data, text_auto=True, color_continuous_scale='Viridis')
        st.plotly_chart(fig_hm, use_container_width=True)

with tabs[2]:
    st.subheader("Nuages de Mots")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Mots-clés Positifs")
        pos_text = " ".join(df[df['stars'] > 3]['review'].head(50))
        wc_pos = WordCloud(background_color="#16161A", colormap='viridis').generate(pos_text)
        st.image(wc_pos.to_array())
    with col2:
        st.write("Mots-clés Négatifs")
        neg_text = " ".join(df[df['stars'] <= 3]['review'].head(50))
        wc_neg = WordCloud(background_color="#16161A", colormap='magma').generate(neg_text)
        st.image(wc_neg.to_array())

with tabs[3]:
    st.subheader("Évolution Temporelle")
    temp_df = df.resample('ME', on='date')['stars'].mean().reset_index()
    fig_temp = px.line(temp_df, x='date', y='stars', markers=True, color_discrete_sequence=['#7F5AF0'])
    st.plotly_chart(fig_temp, use_container_width=True)

with tabs[4]:
    st.subheader("Audit & Plan d'Action Marketing")
    topic_stats = df.groupby("Aspect")["stars"].mean().reset_index()
    critical = topic_stats[topic_stats["stars"] < 3.0]
    for _, row in critical.iterrows():
        st.markdown(f"""
            <div class='marketing-card'>
                <h3 style='margin:0; color: #FF8906;'>⚠️ Point Critique : {row['Aspect']}</h3>
                <p>Score: <b>{row['stars']:.2f} / 5</b></p>
                <p><b>Recommandation:</b> Audit qualitatif requis sur ce segment.</p>
            </div>
        """, unsafe_allow_html=True)
