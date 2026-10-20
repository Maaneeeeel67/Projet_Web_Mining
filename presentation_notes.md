# Notes de presentation

## Positionnement du projet

Le sujet demande une analyse d'opinions e-commerce. Dans l'environnement local, le dataset disponible est `IMDB_Dataset.csv`, contenant 50 000 avis textes etiquetes en `positive` et `negative`.

J'ai donc realise un **prototype complet de pipeline NLP** qui reste directement transferable au cas e-commerce:
- chargement et structuration des donnees,
- nettoyage de texte,
- analyse exploratoire,
- classification de sentiment,
- extraction de themes,
- restitution via dashboard.

## Comment justifier le choix du dataset

Si on me demande pourquoi le corpus n'est pas Amazon ou Cdiscount:

"Le dossier local contenait deja un corpus volumineux et propre, ce qui m'a permis de construire et valider toute la pipeline de text mining. La meme methode se transpose ensuite a des avis produits sans changement majeur de l'architecture."

## Ce qu'il faut dire sur la methode

1. Pretraitement
Suppression du HTML, des URL, des caracteres speciaux et des mots vides.

2. Representation
Transformation TF-IDF pour convertir les avis en vecteurs numeriques.

3. Classification
Utilisation de Logistic Regression comme baseline robuste et interpretable.

4. Themes / aspects
Utilisation de NMF pour extraire automatiquement des groupes de mots representant des themes dominants.

5. Restitution
Ajout d'un dashboard Streamlit pour tester de nouveaux avis et visualiser la repartition des themes.

## Comment connecter ce travail au e-commerce

Si on remplace IMDB par des avis produits:
- les themes deviennent `livraison`, `prix`, `qualite`, `service client`, `emballage`,
- le dashboard permet d'identifier les aspects avec forte proportion de sentiment negatif,
- ces informations servent a prioriser les actions metier.

## Fichiers a montrer

- `ecommerce_opinion_analysis_exam.ipynb`
- `streamlit_app.py`
- `presentation_notes.md`

## Commandes utiles

Lancer le notebook:

```bash
jupyter notebook
```

Lancer le dashboard:

```bash
streamlit run streamlit_app.py
```

## Points forts du Dashboard Premium (Design)

Si on vous interroge sur l'interface :
- **UX Moderne** : Utilisation d'un mode sombre "High-Contrast" pour une meilleure lisibilité des insights.
- **Interactivité** : Permet de tester n'importe quel texte en temps réel, prouvant la robustesse de la pipeline NLP.
- **Transferabilité** : La structure (Tabs, Metrics, Real-time Test) est un standard industriel pour les outils de modération et d'analyse de feedback client.

## Questions potentielles du jury

- **Pourquoi TF-IDF et pas du Deep Learning ?**
  "Pour un prototype industriel, TF-IDF + Logistic Regression offre un excellent compromis entre performance (88% de précision) et temps de réponse instantané. C'est interprétable et facile à déployer sans infrastructure coûteuse."
- **Comment gérer plus de thèmes ?**
  "Il suffit d'ajuster le paramètre `n_components` dans l'algorithme NMF (Factorisation de Matrices Non-négatives) pour affiner la segmentation des avis."
