import os
import json
import pandas as pd
import google.generativeai as genai
import streamlit as st

# Chemins vers les fichiers de données
MATCH_RESULTS_FILE = "data/match_results.json"
COMPETENCES_CSV = "data/competencs2.csv"
CACHE_FILE = "data/db_genai_cache.json"

# ==========================================
# FONCTIONS UTILITAIRES (I/O & Données)
# ==========================================

def _load_json(filepath, default_value):
    """Charge un fichier JSON en toute sécurité."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return default_value
    return default_value

def _save_cache(db_data):
    """Sauvegarde les données dans le cache JSON."""
    os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(db_data, f, indent=4, ensure_ascii=False)

def _get_competences_mapping():
    """Crée un dictionnaire { 'C001': 'Nom de la compétence' } depuis le CSV."""
    mapping = {}
    if os.path.exists(COMPETENCES_CSV):
        try:
            df = pd.read_csv(COMPETENCES_CSV)
            # On s'assure que les colonnes existent bien dans votre CSV
            if 'CompetencyID' in df.columns and 'Competency' in df.columns:
                mapping = dict(zip(df['CompetencyID'], df['Competency']))
        except Exception as e:
            print(f"Erreur lors de la lecture du CSV des compétences : {e}")
    return mapping

# ==========================================
# LOGIQUE DE CACHE AVANCÉE
# ==========================================

def _trouver_entree_cache(metier_cible, score_global):
    """Cherche une entrée similaire (+/- 0.05) dans le cache."""
    db_cache = _load_json(CACHE_FILE, default_value=[])
    for entry in db_cache:
        if entry.get("metier_cible") == metier_cible:
            if abs(entry.get("score_global", 0) - score_global) <= 0.05:
                return entry
    return None

def _mettre_a_jour_cache(metier_cible, score_global, cle, valeur):
    """Met à jour uniquement la bio ou le plan pour un profil donné."""
    db_cache = _load_json(CACHE_FILE, default_value=[])
    trouve = False
    
    for entry in db_cache:
        if entry.get("metier_cible") == metier_cible and abs(entry.get("score_global", 0) - score_global) <= 0.05:
            entry[cle] = valeur
            trouve = True
            break
            
    if not trouve:
        nouvelle_entree = {"metier_cible": metier_cible, "score_global": score_global}
        nouvelle_entree[cle] = valeur
        db_cache.append(nouvelle_entree)
        
    _save_cache(db_cache)

# ==========================================
# FONCTIONS DE GÉNÉRATION IA (Les 2 boutons)
# ==========================================

def generer_bio():
    """Phase 1 : Génération de la Biographie basée sur les forces."""
    match_data = _load_json(MATCH_RESULTS_FILE, default_value=None)
    if not match_data:
        return "Erreur : Résultats SBERT introuvables."

    meilleur_job = match_data["top_jobs"][0]
    meilleure_competence = match_data["top_skills"][0]
    
    metier_cible = meilleur_job["job_title"]
    score_global = meilleur_job["score"]
    skill_phare = meilleure_competence["skill_name"]

    # 1. Vérification du Cache
    entree_cache = _trouver_entree_cache(metier_cible, score_global)
    if entree_cache and entree_cache.get("bio"):
        print(f"✅ [CACHE HIT] Bio récupérée pour '{metier_cible}'.")
        return entree_cache["bio"]

    # 2. Appel API si absent du cache
    print(f"🌐 [API CALL] Génération Bio pour '{metier_cible}'...")
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""<context>
L'utilisateur vise le poste de '{metier_cible}' (Score : {score_global * 100:.0f}%). Sa compétence la plus forte est : '{skill_phare}'.
</context>

<instructions>
Rédige une biographie professionnelle mettant en valeur ce profil et sa compétence phare.
</instructions>

<do_and_dont>
- DO : Utiliser un style "Executive Summary" percutant.
- DON'T : N'invente pas d'expériences qui ne sont pas liées à la compétence phare.
</do_and_dont>

<format>
Courte biographie de 4 à 5 lignes maximum. Pas de formatage markdown (pas d'étoiles).
</format>"""
        
        res = model.generate_content(prompt)
        bio_text = res.text
        
        _mettre_a_jour_cache(metier_cible, score_global, "bio", bio_text)
        return bio_text
    except Exception as e:
        return f"Erreur API Bio : {str(e)}"


def generer_plan():
    """Phase 2 : Génération du Plan de Progression basé sur les lacunes (Skill Gap)."""
    match_data = _load_json(MATCH_RESULTS_FILE, default_value=None)
    if not match_data:
        return "Erreur : Résultats SBERT introuvables."

    meilleur_job = match_data["top_jobs"][0]
    metier_cible = meilleur_job["job_title"]
    score_global = meilleur_job["score"]
    
    # --- CALCUL DU RAG (Skill Gap) ---
    required_ids = meilleur_job.get("required_competencies", [])
    acquired_ids = [s["skill_id"] for s in match_data.get("top_skills", [])]
    
    missing_ids = [req for req in required_ids if req not in acquired_ids]
    
    mapping_competences = _get_competences_mapping()
    missing_names = [mapping_competences.get(c_id, c_id) for c_id in missing_ids]

    # 1. Vérification du Cache
    entree_cache = _trouver_entree_cache(metier_cible, score_global)
    if entree_cache and entree_cache.get("plan"):
        print(f"✅ [CACHE HIT] Plan récupéré pour '{metier_cible}'.")
        return entree_cache["plan"]

    # 2. Appel API si absent du cache
    print(f"🌐 [API CALL] Génération Plan pour '{metier_cible}'...")
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        lacunes = "\n- ".join(missing_names) if missing_names else "Aucune lacune majeure identifiée."
        
        prompt = f"""<context>
L'utilisateur vise le poste de '{metier_cible}' (Score actuel : {score_global * 100:.0f}%).
Voici les compétences critiques qui lui manquent actuellement (Skill Gap) :
- {lacunes}
</context>

<instructions>
En te basant UNIQUEMENT sur ces lacunes, propose un plan de progression ultra-personnalisé pour permettre à l'utilisateur de combler ce retard.
</instructions>

<do_and_dont>
- DO : Être précis, actionnable et cibler uniquement les compétences manquantes.
- DON'T : Ne pas proposer de travailler sur des compétences qui ne sont pas dans la liste des lacunes.
</do_and_dont>

<format>
Une liste à puces en 4 étapes claires.
</format>"""
        
        res = model.generate_content(prompt)
        plan_text = res.text
        
        _mettre_a_jour_cache(metier_cible, score_global, "plan", plan_text)
        return plan_text
    except Exception as e:
        return f"Erreur API Plan : {str(e)}"