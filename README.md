# 📊 Projet Web Mining — Analyse d'opinions E-commerce

> Pipeline NLP complète : analyse de sentiments avec BERT, extraction d'aspects et visualisation interactive pour l'analyse marketing.

## 🎯 Objectif du Projet
Ce projet a pour but d'analyser automatiquement des avis clients (E-commerce / IMDB) pour en extraire les sentiments et identifier les thèmes (aspects) les plus critiqués ou appréciés. Il permet de fournir des recommandations marketing concrètes basées sur les données.

## 📁 Structure du projet
- `Projet_Web_Mining_Final.ipynb` — Notebook principal (Prétraitement NLP, EDA, Analyse de sentiment BERT, NMF).
- `streamlit_app.py` — Dashboard interactif premium (Streamlit, Plotly, WordCloud).
- `presentation_notes.md` — Notes de soutenance et justification des choix techniques.

## 🚀 Installation
```bash
# 1. Cloner le dépôt
git clone https://github.com/Maaneeeeel67/Projet_Web_Mining.git
cd Projet_Web_Mining

# 2. Créer un environnement virtuel (recommandé)
python -m venv .venv
.venv\Scripts\activate  # Sur Windows
source .venv/bin/activate  # Sur Mac/Linux

# 3. Installer les dépendances
pip install pandas numpy scikit-learn transformers streamlit spacy plotly wordcloud langdetect seaborn matplotlib
python -m spacy download en_core_web_sm
