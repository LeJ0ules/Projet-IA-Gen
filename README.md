# AISCA — Agent Intelligent Sémantique et Génératif pour la Cartographie des Compétences

## Description

AISCA est une application prototype basée sur Streamlit qui analyse un profil utilisateur, détecte les compétences les plus pertinentes avec SBERT, puis génère une biographie et un plan de progression à l'aide de l'API Google Generative AI.

Le projet combine :
- une interface Web Streamlit (`app.py`)
- un moteur de matching sémantique avec SBERT (`src/skill_matcher.py`)
- une génération IA avec Gemini (`src/genai_client.py`)
- une pipeline de lancement via `src/launcher.py`


## Structure du projet

- `app.py` : interface Streamlit principale et gestion de l'état utilisateur
- `assets/style.css` : styles personnalisés pour l'interface
- `data/` : fichiers de données (compétences, jobs, cache, profils)
- `src/genai_client.py` : génération de biographie et plan avec cache
- `src/skill_matcher.py` : encodage SBERT et détection de compétences
- `src/launcher.py` : exécute la pipeline de traitement
- `src/traitementRep.py` : script de traitement des données utilisateur


## Prérequis

- Python 3.10+ recommandé
- `pip`


## Installation

1. Créez un environnement virtuel (recommandé) :

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Installez les dépendances :

```bash
pip install -r requirements.txt
```


## Configuration

L'application utilise une clé API Gemini via `streamlit` secrets.

Créez le fichier `.streamlit/secrets.toml` avec :

```toml
GEMINI_API_KEY = "votre_cle_api_gemini_ici"
```


## Exécution

Lancez l'application Streamlit :

```bash
streamlit run app.py
```


## Utilisation

1. Remplissez le formulaire de compétences et d'expérience.
2. Cliquez sur le bouton pour analyser votre profil.
3. Utilisez les boutons de génération pour créer une biographie et un plan de progression.


## Notes importantes

- Le modèle SBERT chargé dans `src/skill_matcher.py` est `paraphrase-multilingual-MiniLM-L12-v2`.
- La génération IA utilise la bibliothèque `google-generativeai` et le modèle `gemini-2.5-flash`.
- Les données attendues incluent `data/competencs2.csv` et `data/jobs.json`.


## Commandes utiles

- Exécuter la pipeline complète :

```bash
python src/launcher.py
```

- Vérifier les dépendances :

```bash
pip install -r requirements.txt
```


## Dépendances principales

- `streamlit`
- `plotly`
- `pandas`
- `numpy`
- `google-generativeai`
- `sentence-transformers`
- `scikit-learn`


## Licence

À adapter selon votre projet.
