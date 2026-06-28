import json


def clean_section_name(section):
    """Transforme marketing_vente -> marketing vente"""
    return section.replace("_", " ")


# ------------------------------------------
# Template de génération par domaine / question
# ------------------------------------------
DOMAIN_TEMPLATES = {
    "marketing_vente": {
        "q1": "niveau global en stratégie marketing et développement commercial : {value}",
        "q2": "degré de maîtrise des outils digitaux et de la data marketing (CRM, SEO/SEM, analytics) : {value}",
        "q3": "capacité à piloter une stratégie de vente et à gérer un cycle commercial : {value}",
        "q4": "compétences confirmées en marketing vente : {value}",
        "q6": "{value} années d'expérience en marketing vente",
        "q7": "ma dernière fonction était {value}",
    },
    "relation_client": {
        "q1": "niveau global en gestion de la relation client : {value}",
        "q2": "degré de maîtrise des outils de support et de ticketing (Zendesk, Jira Service Desk, RDP) : {value}",
        "q3": "capacité à analyser les retours clients et à proposer des améliorations de l'expérience utilisateur : {value}",
        "q4": "compétences confirmées en relation client : {value}",
        "q6": "{value} années d'expérience en relation client",
        "q7": "ma dernière fonction était {value}",
    },
    "finance_legal_rh": {
        "q1": "niveau global en finance d'entreprise (modélisation, budgétisation, reporting, audit) : {value}",
        "q2": "degré de maîtrise des aspects juridiques et de conformité réglementaire (contrats, droit des sociétés, RGPD) : {value}",
        "q3": "capacité en gestion des ressources humaines (recrutement, GPEC, relations sociales, formation) : {value}",
        "q4": "compétences confirmées en finance/legal/rh : {value}",
        "q6": "{value} années d'expérience en finance/legal/rh",
        "q7": "ma dernière fonction était {value}",
    },
    "autres_fonctions": {
        "q1": "niveau global en leadership, management d'équipe et coordination interfonctionnelle : {value}",
        "q2": "degré de maîtrise des compétences de communication professionnelle (storytelling, négociation, présentations) : {value}",
        "q3": "capacité à prendre des décisions complexes dans un environnement incertain : {value}",
        "q4": "compétences confirmées en autres fonctions clés : {value}",
        "q6": "{value} années d'expérience en autres fonctions clés",
        "q7": "ma dernière fonction était {value}",
    },
    "produit_business": {
        "q1": "niveau global en gestion de produit (discovery, roadmap, priorisation, go-to-market) : {value}",
        "q2": "degré de maîtrise des méthodes d'optimisation des processus métier (Lean, Six Sigma, BPM) : {value}",
        "q3": "capacité à définir une stratégie produit et à en piloter l'exécution : {value}",
        "q4": "compétences confirmées en produit business : {value}",
        "q6": "{value} années d'expérience en produit business",
        "q7": "ma dernière fonction était {value}",
    },
    "logistique_supply": {
        "q1": "niveau global en gestion de la chaîne d'approvisionnement (planification, approvisionnement, gestion des stocks) : {value}",
        "q2": "degré de maîtrise des outils logistiques et ERP (SAP, Oracle TMS, WMS) : {value}",
        "q3": "capacité à optimiser les flux logistiques et à garantir la conformité réglementaire (HSE, transport) : {value}",
        "q4": "compétences confirmées en logistique supply chain : {value}",
        "q6": "{value} années d'expérience en logistique supply chain",
        "q7": "ma dernière fonction était {value}",
    },
    "infra_ops": {
        "q1": "niveau global en architecture et administration d'infrastructures Cloud/On-premise : {value}",
        "q2": "degré de maîtrise des pratiques DevOps/MLOps (CI/CD, IaC, conteneurisation, observabilité) : {value}",
        "q3": "capacité à concevoir et opérer des architectures résilientes, sécurisées et scalables : {value}",
        "q4": "compétences confirmées en infrastructure operations : {value}",
        "q6": "{value} années d'expérience en infrastructure operations",
        "q7": "ma dernière fonction était {value}",
    },
    "data_ia": {
        "q1": "niveau global en préparation et manipulation de données (Nettoyage, pipelines ETL/ELT, statistiques) : {value}",
        "q2": "degré de maîtrise en modélisation algorithmique (Machine Learning, Deep Learning, NLP ou Computer Vision) : {value}",
        "q3": "capacité à restituer la donnée pour orienter les décisions stratégiques : {value}",
        "q4": "compétences confirmées en data ia : {value}",
        "q6": "{value} années d'expérience en data ia",
        "q7": "ma dernière fonction était {value}",
    },
    "tech_it": {
        "q1": "niveau global en développement logiciel (algorithmique, architecture, clean code, tests) : {value}",
        "q2": "degré de maîtrise du développement web et/ou mobile (front-end, back-end, API, frameworks) : {value}",
        "q3": "capacité à concevoir des architectures logicielles robustes, scalables et maintenables : {value}",
        "q4": "compétences confirmées en tech it : {value}",
        "q6": "{value} années d'expérience en tech it",
        "q7": "ma dernière fonction était {value}",
    },
}


def process_section(section_name, section_data):
    templates = DOMAIN_TEMPLATES.get(section_name, {})

    # Groupes requis
    scores_group = {}
    facts_group = {"q4": [], "q6": None, "q7": None}
    text_group = {}

    # Sentences pour profil global
    sentences = []

    # q1/q2/q3 (scores) -> groupe scores
    for question_key in ["q1", "q2", "q3"]:
        full_key = f"{section_name}_{question_key}"
        if full_key in section_data:
            scores_group[question_key] = section_data[full_key]
            if question_key in templates:
                sentences.append(templates[question_key].format(value=section_data[full_key]))

    # q4/q6/q7 -> groupe faits
    q4_key = f"{section_name}_q4"
    if q4_key in section_data and isinstance(section_data[q4_key], list):
        facts_group["q4"] = section_data[q4_key]
        if "q4" in templates:
            for item in section_data[q4_key]:
                sentences.append(templates["q4"].format(value=item))

    for question_key in ["q6", "q7"]:
        full_key = f"{section_name}_{question_key}"
        if full_key in section_data:
            facts_group[question_key] = section_data[full_key]
            if question_key in templates:
                sentences.append(templates[question_key].format(value=section_data[full_key]))

    # q5/q8/q9/q10 -> texte libre (groupe sémantique)
    for optional_q in ["q5", "q8", "q9", "q10"]:
        full_key = f"{section_name}_{optional_q}"
        if full_key in section_data and isinstance(section_data[full_key], str) and section_data[full_key].strip():
            text_group[optional_q] = section_data[full_key].strip()
            sentences.append(f"{clean_section_name(section_name)} - {optional_q} : {section_data[full_key].strip()}")

    return {
        "scores": scores_group,
        "facts": facts_group,
        "text_entries": text_group,
        "sentences": sentences,
    }


def process_user_profile(data):
    profile = {
        "sections": {},
        "domaines_actifs": data.get("domaines_actifs", []),
    }

    all_sentences = []

    for section_name, section_data in data.items():
        if section_name == "domaines_actifs":
            continue
        if not isinstance(section_data, dict):
            continue
        if section_data.get("_status") == "ignoré":
            continue

        section_result = process_section(section_name, section_data)
        profile["sections"][section_name] = section_result
        all_sentences.extend(section_result.get("sentences", []))

    profile["sentences"] = all_sentences
    profile["profile_text"] = " ".join(all_sentences)

    return profile


# -------- UTILISATION --------

if __name__ == "__main__":
    with open("data/user_answers.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    profile_data = process_user_profile(data)

    output_path = "data/user_profile.json"
    with open(output_path, "w", encoding="utf-8") as out_file:
        json.dump(profile_data, out_file, ensure_ascii=False, indent=2)

    print(f"Résultats enregistrés dans {output_path}")
