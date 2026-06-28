#pip install sentence-transformers pandas scikit-learn numpy

import pandas as pd
import json
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# 1. Charger les datasets
# -----------------------------

#@
skills_df = pd.read_csv("data/competencs2.csv")
#@
with open("data/jobs.json", "r", encoding="utf-8") as f:
    jobs_data = json.load(f)

jobs = jobs_data["job_profiles"]


# -----------------------------
# 2. Initialiser SBERT
# -----------------------------

print("Loading SBERT model...")
#model = SentenceTransformer("all-MiniLM-L6-v2")
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")


# -----------------------------
# 3. Préparer les skills
# -----------------------------

skill_ids = skills_df["CompetencyID"].tolist()
skill_texts = skills_df["Competency"].tolist()
skill_block_map = skills_df.set_index("CompetencyID")["BlockName"].to_dict()

print("Encoding skills...")

skill_embeddings = model.encode(skill_texts)


# -----------------------------
# 4. Fonction analyse utilisateur
# -----------------------------

def analyze_user_texts(user_texts, facts=None, fact_weight=0.4):

    if isinstance(user_texts, str):
        user_texts = [user_texts]

    rich_weight = 1.5  # Pondération pour Q5, Q9, Q10
    normal_weight = 1.0

    text_scores = {skill_id: 0.0 for skill_id in skill_ids}
    fact_scores = {skill_id: 0.0 for skill_id in skill_ids}

    if user_texts:
        user_embeddings = model.encode(user_texts)
        similarities = cosine_similarity(user_embeddings, skill_embeddings)
        for i, user_text in enumerate(user_texts):
            weight = rich_weight if any(w in user_text.lower() for w in ["projet", "défi", "justification"]) else normal_weight
            for j, skill_id in enumerate(skill_ids):
                text_scores[skill_id] += float(similarities[i, j]) * weight
        for skill_id in skill_ids:
            text_scores[skill_id] /= (len(user_texts) * normal_weight)

    if facts:
        if isinstance(facts, str):
            facts = [facts]

        fact_embeddings = model.encode(facts)
        fact_similarities = cosine_similarity(fact_embeddings, skill_embeddings)
        for i in range(len(facts)):
            for j, skill_id in enumerate(skill_ids):
                fact_scores[skill_id] += float(fact_similarities[i, j])
        for skill_id in skill_ids:
            fact_scores[skill_id] /= len(facts)

    aggregated_scores = {}
    for skill_id in skill_ids:
        t = text_scores.get(skill_id, 0.0)
        f = fact_scores.get(skill_id, 0.0)
        if facts:
            score = (t + fact_weight * f) / (1 + fact_weight)
        else:
            score = t
        aggregated_scores[skill_id] = {
            "skill_id": skill_id,
            "skill_name": skill_texts[skill_ids.index(skill_id)],
            "score": float(score)
        }

    # Normalisation min-max
    scores_arr = np.array([v["score"] for v in aggregated_scores.values()])
    min_score = scores_arr.min()
    max_score = scores_arr.max()
    for skill_id in skill_ids:
        s = aggregated_scores[skill_id]["score"]
        if max_score > min_score:
            aggregated_scores[skill_id]["score"] = (s - min_score) / (max_score - min_score)
        else:
            aggregated_scores[skill_id]["score"] = s

    skill_scores = sorted(aggregated_scores.values(), key=lambda x: x["score"], reverse=True)
    return skill_scores


# -----------------------------
# 5. Détection des skills
# -----------------------------

def detect_user_skills(user_texts, facts=None, threshold=0.45):

    skill_scores = analyze_user_texts(user_texts, facts=facts)

    # debug : affichage des tops
    top_debug = skill_scores[:20]
    print('\nDEBUG: top 20 skill scores (analyse):')
    for item in top_debug:
        print(f"  {item['skill_id']} - {item['skill_name']} -> {item['score']:.4f}")

    detected_skills = [skill for skill in skill_scores if skill["score"] >= threshold]

    print(f"DEBUG: {len(detected_skills)} skills détectés >= {threshold}")
    if len(detected_skills) == 0:
        print("DEBUG: vérifier si user_inputs et facts sont non vides:")
        print('  user_inputs count', len(user_texts) if user_texts else 0)
        print('  facts count', len(facts) if facts else 0)

    return detected_skills


# -----------------------------
# 6. Calcul score métier
# -----------------------------

def compute_job_scores(detected_skills=None, all_skill_scores=None):

    if all_skill_scores is not None:
        user_skill_ids = {skill["skill_id"]: skill["score"] for skill in all_skill_scores}
    elif detected_skills is not None:
        user_skill_ids = {skill["skill_id"]: skill["score"] for skill in detected_skills}
    else:
        user_skill_ids = {}

    job_results = []

    for job in jobs:
        required_skills = job["Required_Competencies"]
        scores = []
        for skill_id in required_skills:
            score = user_skill_ids.get(skill_id, 0.0)
            scores.append(score)
        if len(scores) > 0:
            coverage = sum([1 for s in scores if s > 0.5]) / len(scores)
            job_score = np.mean(scores) + 0.15 * coverage
        else:
            job_score = 0.0
        job_results.append({
            "job_title": job["JobTitle"],
            "score": float(job_score)
        })
    scores_arr = np.array([j["score"] for j in job_results])
    min_score = scores_arr.min()
    max_score = scores_arr.max()
    for j in job_results:
        s = j["score"]
        if max_score > min_score:
            j["score"] = (s - min_score) / (max_score - min_score)
        else:
            j["score"] = s
    job_results = sorted(job_results, key=lambda x: x["score"], reverse=True)
    return job_results


# -----------------------------
# 7. Chargement des text_entries + facts depuis user_profile.json
# -----------------------------

#@
def get_profile_entries(path="data/user_profile.json"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            profile_data = json.load(f)
    except FileNotFoundError:
        return None, None, "data/user_profile.json introuvable"

    text_entries = []
    facts = []

    direct_text = profile_data.get("text_entries")
    if isinstance(direct_text, list):
        text_entries.extend([t for t in direct_text if isinstance(t, str) and t.strip()])
    elif isinstance(direct_text, dict):
        text_entries.extend([v for v in direct_text.values() if isinstance(v, str) and v.strip()])

    for section in profile_data.get("sections", {}).values():
        section_text = section.get("text_entries")
        if isinstance(section_text, list):
            text_entries.extend([t for t in section_text if isinstance(t, str) and t.strip()])
        elif isinstance(section_text, dict):
            text_entries.extend([v for v in section_text.values() if isinstance(v, str) and v.strip()])

        section_facts = section.get("facts")
        if isinstance(section_facts, list):
            facts.extend([t for t in section_facts if isinstance(t, str) and t.strip()])
        elif isinstance(section_facts, dict):
            facts.extend([v for v in section_facts.values() if isinstance(v, str) and v.strip()])

    if not text_entries and not facts:
        return None, None, "Aucune text_entries/n facts trouvée dans Dataset/user_profile.json"

    return text_entries or None, facts or None, None


# -----------------------------
# 8. Détection des skills
# -----------------------------

def detect_user_skills(user_texts, facts=None, threshold=0.45):

    skill_scores = analyze_user_texts(user_texts, facts=facts)

    # debug : affichage des tops
    top_debug = skill_scores[:20]
    print('\nDEBUG: top 20 skill scores (analyse):')
    for item in top_debug:
        print(f"  {item['skill_id']} - {item['skill_name']} -> {item['score']:.4f}")

    detected_skills = [skill for skill in skill_scores if skill["score"] >= threshold]

    print(f"DEBUG: {len(detected_skills)} skills détectés >= {threshold}")
    if len(detected_skills) == 0:
        print("DEBUG: vérifier si user_inputs et facts sont non vides:")
        print('  user_inputs count', len(user_texts) if user_texts else 0)
        print('  facts count', len(facts) if facts else 0)

    return detected_skills


# -----------------------------
# 9. Main
# -----------------------------

#top_skills = sorted(all_skill_scores, key=lambda x: x['score'], reverse=True)[:7]

user_inputs, user_facts, error = get_profile_entries()
if error:
    user_inputs = [
        "J'ai programmé des applications en Python",
        "J'ai conçu un modèle de machine learning",
        "J'ai travaillé avec des bases de données SQL",
        "J'ai de l'expérience avec les plateformes cloud comme AWS",
    ]
    user_facts = None
    print(f"\nAvertissement : {error}. Utilisation d'un jeu d'exemple.")
else:
    print("\nUtilisation de text_entries + facts depuis Dataset/user_profile.json")
    for u in (user_inputs or []):
        print("text:", u)
    for f in (user_facts or []):
        print("fact:", f)

# --- Ajustement Q1/Q2/Q3 AVANT skills ---
q1_vals = []
q2_vals = []
q3_vals = []
q3_map = {
    "débutant": 1,
    "notions": 2,
    "intermédiaire": 3,
    "avancé": 4,
    "expert": 5
}
try:
    with open("Dataset/user_profile.json", "r", encoding="utf-8") as f:
        profile_data = json.load(f)
    for section in profile_data.get("sections", {}).values():
        scores = section.get("scores", {})
        if "q1" in scores:
            try:
                q1_vals.append(float(scores["q1"]))
            except:
                pass
        if "q2" in scores:
            try:
                q2_vals.append(float(scores["q2"]))
            except:
                pass
        if "q3" in scores:
            val_txt = str(scores["q3"]).strip().lower()
            val_num = q3_map.get(val_txt, 3)  # par défaut intermédiaire
            q3_vals.append(val_num)
except Exception as e:
    print(f"Erreur extraction Q1/Q2/Q3: {e}")

if q1_vals and q2_vals and q3_vals:
    avg_q = (np.mean(q1_vals) + np.mean(q2_vals) + np.mean(q3_vals)) / 3
    print(f"!!!!!!!!!!!!\nMoyenne Q1: {np.mean(q1_vals):.2f}, Q2: {np.mean(q2_vals):.2f}, Q3: {np.mean(q3_vals):.2f} → avg_q: {avg_q:.2f}")
else:
    avg_q = 0.7  # valeur par défaut si non renseigné
    print("!!!!!!!!!!!!!!!!!!Avertissement: Q1/Q2/Q3 non trouvés, utilisation d'une valeur par défaut pour l'ajustement des scores métiers.")

# détecter skills (agrégation)
all_skill_scores = analyze_user_texts(user_inputs, facts=user_facts)
for skill in all_skill_scores:
    skill['score'] = 3.5 * (skill['score'] * avg_q / 5)  # division par 5 pour ramener sur [0,1]
detected_skills = [skill for skill in all_skill_scores if skill['score'] >= 0.45]

top_skills = sorted(all_skill_scores, key=lambda x: x['score'], reverse=True)[:7]
print("\nTop 7 Skills (après ajustement avg_q):")
for skill in top_skills:
    print(f"{skill['skill_name']} ({skill['skill_id']}) → {round(skill['score'],2):.2f}")

# calcul métiers (avec score pour toutes compétences requises, inclus non-detectées)
job_scores = compute_job_scores(all_skill_scores=all_skill_scores)

# --- Ajustement des scores métiers avec Q1, Q2, Q3 depuis user_profile.json ---
q1_vals = []
q2_vals = []
q3_vals = []
q3_map = {
    "débutant": 1,
    "notions": 2,
    "intermédiaire": 3,
    "avancé": 4,
    "expert": 5
}
# Charger les valeurs depuis le JSON user_profile.json
try:
    #@
    with open("data/user_profile.json", "r", encoding="utf-8") as f:
        profile_data = json.load(f)
    for section in profile_data.get("sections", {}).values():
        scores = section.get("scores", {})
        if "q1" in scores:
            try:
                q1_vals.append(float(scores["q1"]))
            except:
                pass
        if "q2" in scores:
            try:
                q2_vals.append(float(scores["q2"]))
            except:
                pass
        if "q3" in scores:
            val_txt = str(scores["q3"]).strip().lower()
            val_num = q3_map.get(val_txt, 3)  # par défaut intermédiaire
            q3_vals.append(val_num)
except Exception as e:
    print(f"Erreur extraction Q1/Q2/Q3: {e}")

# Moyenne des valeurs Q1, Q2, Q3
if q1_vals and q2_vals and q3_vals:
    avg_q = (np.mean(q1_vals) + np.mean(q2_vals) + np.mean(q3_vals)) / 3
    print(f"!!!!!!!!!!!!\nMoyenne Q1: {np.mean(q1_vals):.2f}, Q2: {np.mean(q2_vals):.2f}, Q3: {np.mean(q3_vals):.2f} → avg_q: {avg_q:.2f}")
else:
    avg_q = 0.7  # valeur par défaut si non renseigné
    print("!!!!!!!!!!!!!!!!!!Avertissement: Q1/Q2/Q3 non trouvés, utilisation d'une valeur par défaut pour l'ajustement des scores métiers.")

# Ajustement des scores métiers
for job in job_scores:
    job['score'] = job['score'] * avg_q / 5  # division par 5 pour ramener sur [0,1]

print("\nTop 3 Recommended Jobs (après ajustement):")
for job in job_scores[:3]:
    print(f"{job['job_title']} → {round(job['score'],2):.2f}")

# Prépare la sortie JSON
output = {
    "top_skills": [
        {
            "skill_id": s["skill_id"],
            "skill_name": s["skill_name"],
            "block_name": skill_block_map.get(s["skill_id"], ""),
            "score": round(s["score"], 2)
        }
        for s in top_skills
    ],
    "top_jobs": [
        {
            "job_title": j["job_title"],
            "score": round(j["score"], 2),
            "required_competencies": next((job["Required_Competencies"] for job in jobs if job["JobTitle"] == j["job_title"]), [])
        }
        for j in job_scores[:3]
    ]
}

#@
output_path = "data/match_results.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\nRésultat écrit dans {output_path}")