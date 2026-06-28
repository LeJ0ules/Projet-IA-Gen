"""
AISCA - Agent Intelligent Sémantique et Génératif pour la Cartographie des Compétences
================================================================================================
Application Streamlit - Interface Frontend (Mock Data / Prototype UX)

GESTION D'ÉTAT (st.session_state) :
-------------------------------------
Trois clés principales pilotent toute la navigation :

1. `st.session_state["form_submitted"]` (bool)
   → Passe à True quand l'utilisateur clique sur "Lancer l'Analyse Sémantique".
   → Déverrouillerait les onglets 2 et 3.
   → Contiendrait les réponses brutes du formulaire.

2. `st.session_state["answers"]` (dict)
   → Stocke toutes les réponses du questionnaire (sliders, radios, text_areas).
   → Sera passé à votre pipeline SBERT/NLP pour le calcul des scores.

3. `st.session_state["genai_triggered"]` (bool)
   → Passe à True quand l'utilisateur clique sur "Générer mon profil".
   → Déclenche l'affichage du mock-up de la biographie et du plan de progression.
   → En production : remplacer par un appel à l'API Gemini/OpenAI.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
import json
import streamlit as st
import os
from src.genai_client import generer_bio, generer_plan
import subprocess

# ──────────────────────────────────────────────────────────────
# CONFIG & THÈME
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AISCA · Cartographie des Compétences",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# CSS PERSONNALISÉ (assets/style.css)
# ──────────────────────────────────────────────────────────────
def charger_css(fichier_css):
    with open(fichier_css, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Configuration de la page (doit toujours être le premier appel Streamlit)
st.set_page_config(page_title="AISCA", layout="wide")

# Chargement de votre superbe CSS
charger_css("assets/style.css")


# ──────────────────────────────────────────────────────────────
# SESSION STATE — Initialisation
# ──────────────────────────────────────────────────────────────
if "form_submitted" not in st.session_state:
    st.session_state["form_submitted"] = False
if "answers" not in st.session_state:
    st.session_state["answers"] = {}
if "genai_triggered" not in st.session_state:
    st.session_state["genai_triggered"] = False


# ──────────────────────────────────────────────────────────────
# HERO HEADER
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">EFREI · Master Data Engineering & AI</div>
    <h1 class="hero-title">AISCA : Découvrez le poste en lien avec vos compétences. </h1>
    <p class="hero-subtitle">Agent Intelligent Sémantique et Génératif pour la Cartographie des Compétences</p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# RÉFÉRENTIEL DES 9 DOMAINES DE COMPÉTENCES
# ──────────────────────────────────────────────────────────────
# Chaque domaine contient :
#   - key        : identifiant unique snake_case (préfixe des clés session_state)
#   - label      : intitulé affiché dans l'expander
#   - competences: liste des compétences pour le st.multiselect (Q4)
#                  ⚠️ REMPLACER par la liste réelle issue de `competences_par_domaine`
#   - q1/q2      : libellés des sliders Likert (Q1 & Q2)
#   - q3_options : libellés de l'échelle radio (Q3)
#   - q7_options : options du selectbox "Dernier poste" (Q7)

DOMAINES = [
    {
        "key": "marketing_vente",
        "label": "📈 Marketing / Vente",
        # ── BRANCHER : remplacer par competences_par_domaine["📈 Marketing / Vente"] ──
        "competences": [
            "Développement de stratégies de marketing et de croissance",
            "Gestion de la marque et positionnement",
            "Marketing sur les moteurs de recherche (SEO & SEM/PPC)",
            "Réseaux sociaux et marketing de contenu (Copywriting)",
            "Email marketing et automatisation du marketing (HubSpot, Mailchimp)",
            "Stratégies d'acquisition, de rétention et d'engagement des utilisateurs",
            "Expérimentation basée sur les données (Tests A/B) et CRO",
            "Stratégie de vente et planification des comptes",
            "Gestion de la relation client (CRM)",
            "Gestion des relations fournisseurs",
            "Négociation, contrats et conclusion de ventes",
            "Étude de marché et analyse de la concurrence",
            "Analyse des données de vente, prévisions et pipeline",
            "Outils et automatisation (Google Analytics, Salesforce, SQL, BI)",
        ],
        "q1": "Évaluez votre niveau global en stratégie marketing et développement commercial (acquisition, rétention, branding).",
        "q2": "Quel est votre degré de maîtrise des outils digitaux et de la data marketing (CRM, SEO/SEM, analytics) ?",
        "q3_label": "Comment jugez-vous votre capacité à piloter une stratégie de vente et à gérer un cycle commercial de bout en bout ?",
        "q7_options": ["Growth Marketer", "Account Manager", "Sales Manager", "Digital Marketing Manager", "CMO", "Business Developer", "Autre"],
        "q7_label": "Quel titre correspondait le mieux à votre dernière fonction ?",
        "q8_label": "Décrivez brièvement votre dernier environnement de travail (secteur, taille d'entreprise, outils utilisés).",
        "q9_label": "Détaillez une campagne ou un projet commercial majeur mené de bout en bout. Précisez la problématique initiale, votre rôle, les leviers activés et les résultats obtenus (CA, leads, ROI).",
        "q10_label": "Quels ont été les principaux défis (budget limité, concurrence, changement d'algorithme, cycle de vente long...) et comment les avez-vous surmontés ?",
    },
    {
        "key": "relation_client",
        "label": "🤝 Relation client / Support",
        # ── BRANCHER : remplacer par competences_par_domaine["🤝 Relation client / Support"] ──
        "competences": [
            "Dépannage matériel et logiciel",
            "Systèmes de billetterie et flux de travail d'assistance",
            "Service client et compétences en communication",
            "Dépannage technique (Matériel/Logiciel)",
            "Diagnostic des problèmes et analyse des causes profondes",
            "Gestion des problèmes clients complexes et prioritaires",
            "Processus d'escalade et stratégie de résolution",
            "Connaissance technique des produits/services",
            "Cartographie du parcours client",
            "Analyse des retours clients et des sentiments",
            "Systèmes de billetterie et outils de support à distance (RDP)",
            "Documentation des problèmes, des solutions et reporting",
        ],
        "q1": "Évaluez votre niveau global en gestion de la relation client et résolution de problèmes (support technique, escalades).",
        "q2": "Quel est votre degré de maîtrise des outils de support et de ticketing (Zendesk, Jira Service Desk, RDP) ?",
        "q3_label": "Comment jugez-vous votre capacité à analyser les retours clients et à proposer des améliorations de l'expérience utilisateur ?",
        "q7_options": ["Customer Support Specialist", "Customer Success Manager", "Technical Support Engineer", "Service Desk Analyst", "Head of Customer Experience", "Autre"],
        "q7_label": "Quel titre correspondait le mieux à votre dernière fonction ?",
        "q8_label": "Décrivez votre dernier environnement de travail (volume de tickets, outils utilisés, type de clientèle : B2B/B2C).",
        "q9_label": "Décrivez un cas client complexe ou un projet d'amélioration du support que vous avez piloté. Précisez la problématique, votre rôle et l'impact mesuré (CSAT, NPS, délai de résolution).",
        "q10_label": "Quels ont été les principaux défis (clients difficiles, volume de tickets, manque d'outils, documentation insuffisante) et comment les avez-vous surmontés ?",
    },
    {
        "key": "finance_legal_rh",
        "label": "💼 Finance / Legal / RH",
        # ── BRANCHER : remplacer par competences_par_domaine["💼 Finance / Legal / RH"] ──
        "competences": [
            "Planification financière, modélisation et stratégie",
            "Budgétisation, prévisions et analyse des écarts",
            "Analyse des performances de l'entreprise et aide à la décision",
            "Comptabilité générale (Clients/Fournisseurs, Rapprochements)",
            "Conformité fiscale, déclarations et reporting légal",
            "Planification, exécution et tests d'audit interne",
            "Évaluation des risques et des contrôles",
            "Rédaction, révision et gestion des contrats",
            "Connaissance du droit des sociétés et du droit commercial",
            "Conformité réglementaire et veille juridique",
            "Évaluation et atténuation des risques d'entreprise",
            "Alignement de la stratégie RH et partenariat d'affaires",
            "Relations avec les employés et droit du travail",
            "Gestion des talents et planification de la relève",
            "Stratégie de recrutement et planification des effectifs",
            "Stratégie de formation et analyse des besoins",
        ],
        "q1": "Évaluez votre niveau global en finance d'entreprise (modélisation, budgétisation, reporting, audit).",
        "q2": "Quel est votre degré de maîtrise des aspects juridiques et de conformité réglementaire (contrats, droit des sociétés, RGPD) ?",
        "q3_label": "Comment jugez-vous votre expertise en gestion des ressources humaines (recrutement, GPEC, relations sociales, formation) ?",
        "q7_options": ["Financial Analyst", "CFO / DAF", "Juriste / Counsel", "DRH / HRBP", "Auditeur Interne", "Contrôleur de Gestion", "Autre"],
        "q7_label": "Quel titre correspondait le mieux à votre dernière fonction ?",
        "q8_label": "Décrivez votre dernier environnement de travail (secteur, taille d'entreprise, outils : ERP SAP, Workday, etc.).",
        "q9_label": "Décrivez un projet majeur (clôture comptable complexe, restructuration RH, contentieux juridique, audit) que vous avez mené. Précisez votre rôle, les enjeux et l'impact.",
        "q10_label": "Quels ont été les principaux défis (contraintes réglementaires, délais serrés, résistance au changement) et comment les avez-vous surmontés ?",
    },
    {
        "key": "autres_fonctions",
        "label": "🏢 Autres fonctions clés",
        # ── BRANCHER : remplacer par competences_par_domaine["🏢 Autres fonctions clés"] ──
        "competences": [
            "Sécurité, Normes et Procédures",
            "Communication (Client, Storytelling, Publication)",
            "Collaboration d'équipe et Gestion du temps",
            "Leadership d'équipe, supervision et coaching",
            "Formation et développement du personnel",
            "Collaboration interfonctionnelle et négociation",
            "Prise de décision en situation d'incertitude",
            "Éthique, intégrité et normes professionnelles",
            "Conformité aux politiques, procédures et réglementations",
        ],
        "q1": "Évaluez votre niveau global en leadership, management d'équipe et coordination interfonctionnelle.",
        "q2": "Quel est votre degré de maîtrise des compétences de communication professionnelle (storytelling, négociation, présentations) ?",
        "q3_label": "Comment jugez-vous votre capacité à prendre des décisions complexes dans un environnement incertain et à garantir la conformité éthique ?",
        "q7_options": ["Manager / Team Lead", "Directeur Général / COO", "Chef de Projet", "Consultant", "Coordinateur", "Autre"],
        "q7_label": "Quel titre correspondait le mieux à votre dernière fonction ?",
        "q8_label": "Décrivez votre dernier environnement de travail (taille d'équipe, contexte : startup, grand groupe, secteur public).",
        "q9_label": "Décrivez un projet transversal ou une initiative de transformation majeure que vous avez piloté(e). Précisez la problématique, votre rôle de leadership et les résultats obtenus.",
        "q10_label": "Quels ont été les principaux défis (résistance au changement, conflits d'équipe, ressources limitées) et comment les avez-vous gérés ?",
    },
    {
        "key": "produit_business",
        "label": "🛒 Produit / Business",
        # ── BRANCHER : remplacer par competences_par_domaine["🛒 Produit / Business"] ──
        "competences": [
            "Gestion de projet et conduite du changement",
            "Gestion de budget et contrôle des coûts",
            "Cartographie et analyse des processus",
            "Méthodologies Lean et Six Sigma",
            "Analyse des causes profondes et résolution de problèmes",
            "Mise en œuvre d'améliorations de processus",
            "Assortiment de produits et planification des catégories",
            "Stratégie de prix et optimisation des marges",
            "Expertise produit/service et connaissance de l'industrie",
        ],
        "q1": "Évaluez votre niveau global en gestion de produit (discovery, roadmap, priorisation, go-to-market).",
        "q2": "Quel est votre degré de maîtrise des méthodes d'optimisation des processus métier (Lean, Six Sigma, BPM) ?",
        "q3_label": "Comment jugez-vous votre capacité à définir une stratégie produit et à en piloter l'exécution jusqu'à la livraison ?",
        "q7_options": ["Product Manager", "Product Owner", "Business Analyst", "Chief Product Officer", "Category Manager", "Autre"],
        "q7_label": "Quel titre correspondait le mieux à votre dernière fonction ?",
        "q8_label": "Décrivez votre dernier environnement de travail (stack produit, méthodologie Agile/Scrum, taille de l'équipe produit).",
        "q9_label": "Détaillez un produit ou une feature majeure que vous avez lancé(e) de bout en bout. Précisez la problématique utilisateur initiale, votre rôle, les arbitrages réalisés et les métriques d'impact.",
        "q10_label": "Quels ont été les principaux défis (alignement des parties prenantes, pivots, contraintes techniques, time-to-market) et comment les avez-vous surmontés ?",
    },
    {
        "key": "logistique_supply",
        "label": "📦 Logistique / Supply Chain",
        # ── BRANCHER : remplacer par competences_par_domaine["📦 Logistique / Supply Chain"] ──
        "competences": [
            "Prévision de la demande et planification",
            "Suivi, contrôle et réapprovisionnement des stocks",
            "Gestion et optimisation des stocks de sécurité",
            "Gestion des opérations d'entrepôt",
            "Logistique et planification/suivi des expéditions",
            "Planification et optimisation des itinéraires",
            "Sélection des transporteurs et gestion des fournisseurs",
            "Gestion de la flotte et des actifs",
            "Traitement des commandes et documentation",
            "Utilisation de logiciels ERP, TMS et Supply Chain",
            "Conformité Hygiène, Sécurité et Environnement (HSE)",
            "Conformité réglementaire (Lois sur les transports, normes)",
        ],
        "q1": "Évaluez votre niveau global en gestion de la chaîne d'approvisionnement (planification, approvisionnement, gestion des stocks).",
        "q2": "Quel est votre degré de maîtrise des outils logistiques et ERP (SAP, Oracle TMS, WMS) ?",
        "q3_label": "Comment jugez-vous votre capacité à optimiser les flux logistiques et à garantir la conformité réglementaire (HSE, transport) ?",
        "q7_options": ["Responsable Logistique", "Supply Chain Manager", "Gestionnaire de Stocks", "Acheteur / Procurement", "Responsable Entrepôt", "Autre"],
        "q7_label": "Quel titre correspondait le mieux à votre dernière fonction ?",
        "q8_label": "Décrivez votre dernier environnement de travail (type de supply chain : B2B/B2C, volumétrie, outils, périmètre géographique).",
        "q9_label": "Décrivez un projet d'optimisation logistique ou supply chain majeur que vous avez conduit. Précisez la problématique initiale, votre rôle et les gains obtenus (coûts, délais, fiabilité).",
        "q10_label": "Quels ont été les principaux défis (ruptures de stock, perturbations fournisseurs, réglementation, pics d'activité) et comment les avez-vous résolus ?",
    },
    {
        "key": "infra_ops",
        "label": "🏗 Infrastructure / Opérations techniques",
        # ── BRANCHER : remplacer par competences_par_domaine["🏗 Infrastructure / Opérations techniques"] ──
        "competences": [
            "Installation et raccordement (Fibre, Cuivre, Coaxial)",
            "Maintenance et dépannage pannes client",
            "MLOps et Déploiement (APIs, CI/CD, Monitoring)",
            "Plateformes Cloud (AWS, GCP, Azure)",
            "Architecture Big Data (Spark, Hadoop, NoSQL)",
            "Infrastructure as Code (Terraform, CloudFormation)",
            "Conteneurisation et Orchestration (Docker, Kubernetes)",
            "Pipelines CI/CD (Jenkins, GitLab CI, GitHub Actions)",
            "Administration système (Windows, Linux, macOS)",
            "Configuration réseau (Routage, Switching, VPC)",
            "Virtualisation de serveurs (VMware, Hyper-V)",
            "Sécurité réseau (Pare-feu, IDS/IPS, VPN)",
            "Sécurité des applications et Pentesting (OWASP)",
            "Gestion des identités et des accès (IAM)",
            "Chiffrement et cryptographie",
            "Surveillance et journalisation (Prometheus, ELK, CloudWatch)",
            "Stratégies de reprise après sinistre et de sauvegarde",
            "Optimisation des performances et réglages",
        ],
        "q1": "Évaluez votre niveau global en architecture et administration d'infrastructures Cloud/On-premise (déploiement, sécurité, réseaux).",
        "q2": "Quel est votre degré de maîtrise des pratiques DevOps/MLOps (CI/CD, IaC, conteneurisation, observabilité) ?",
        "q3_label": "Comment jugez-vous votre capacité à concevoir et opérer des architectures résilientes, sécurisées et scalables en production ?",
        "q7_options": ["DevOps / SRE Engineer", "Cloud Architect", "System Administrator", "Network Engineer", "Security Engineer", "Infrastructure Lead", "Autre"],
        "q7_label": "Quel titre correspondait le mieux à votre dernière fonction ?",
        "q8_label": "Décrivez votre dernier environnement de travail (cloud provider, stack technique, volumétrie infrastructure, périmètre de sécurité).",
        "q9_label": "Détaillez un projet d'infrastructure ou de migration cloud majeur que vous avez mené. Précisez la problématique initiale, votre rôle, les choix d'architecture et l'impact opérationnel.",
        "q10_label": "Quels ont été les principaux défis techniques (incidents de production, dette technique, contraintes de sécurité, coûts cloud) et comment les avez-vous adressés ?",
    },
    {
        "key": "data_ia",
        "label": "📊 Data / IA",
        # ── BRANCHER : remplacer par competences_par_domaine["📊 Data / IA"]
        # Liste exacte des 13 compétences du référentiel :
        "competences": [
            "Préparation de données (Nettoyage, Valeurs manquantes, ETL)",
            "Analyse de données (EDA, Statistiques descriptives)",
            "Statistiques avancées et Mathématiques",
            "Visualisation de données et Tableaux de bord",
            "Apprentissage automatique (Algorithmes, Théorie, Évaluation)",
            "Apprentissage profond et Architectures (Réseaux de neurones)",
            "IA Spécialisée (NLP, Vision par ordinateur)",
            "Recherche Scientifique et Prototypage",
            "Expérimentation et Tests A/B",
            "Conception et gestion de bases de données (SQL/NoSQL)",
            "Suivi des performances (KPIs, métriques, reporting)",
            "Prise de décision basée sur les données",
            "Analyse de données (Excel, SQL, outils BI)",
        ],
        # Questions spécifiques Data/IA (libellés exacts fournis dans le brief)
        "q1": "Évaluez votre niveau global en préparation et manipulation de données (Nettoyage, pipelines ETL/ELT, statistiques).",
        "q2": "Quel est votre degré de maîtrise en modélisation algorithmique (Machine Learning, Deep Learning, NLP ou Computer Vision) ?",
        "q3_label": "Comment jugez-vous votre capacité à restituer la donnée pour orienter les décisions stratégiques (Data-driven decision making et Dashboards) ?",
        "q7_options": ["Data Analyst", "Data Scientist", "Data Engineer", "Machine Learning Engineer", "BI Analyst", "Chercheur / Applied Scientist", "Autre"],
        "q7_label": "Quel titre correspondait le mieux à votre dernière fonction dans le domaine Data / IA ?",
        "q8_label": "Décrivez brièvement votre dernier environnement de travail (Stack technique, architecture cloud, volumétrie des données traitées).",
        "q9_label": "Détaillez un projet Data/IA majeur que vous avez mené de bout en bout. Précisez la problématique métier initiale, votre rôle exact, les modèles déployés et l'impact final.",
        "q10_label": "Quels ont été les principaux défis techniques (ex : mauvaise qualité des données, optimisation des hyperparamètres, mise en production) ou méthodologiques rencontrés lors de ce projet, et comment les avez-vous surmontés ?",
    },
    {
        "key": "tech_it",
        "label": "🧑‍💻 Tech / IT / Engineering",
        # ── BRANCHER : remplacer par competences_par_domaine["🧑‍💻 Tech / IT / Engineering"] ──
        "competences": [
            "Programmation (Python, C++, SQL)",
            "Modèles de conception et d'architecture logicielle",
            "Programmation orientée objet (POO)",
            "Structures de données et algorithmes",
            "Concepts des systèmes distribués",
            "HTML/CSS et Responsive Design",
            "JavaScript et gestion d'état (Redux, Context API)",
            "Frameworks Front-end (React, Angular, Vue)",
            "Programmation mobile (Swift, Kotlin, Java)",
            "Frameworks mobiles (Flutter, React Native)",
            "Déploiement d'applications (App Store, Google Play)",
            "Programmation côté serveur (Java, Python, Node.js)",
            "Conception et développement d'API (REST, GraphQL)",
            "Tests (Unitaires, Intégration, UI, E2E)",
            "Contrôle de version (Git)",
            "Scripting et automatisation (Python, Bash, PowerShell)",
        ],
        "q1": "Évaluez votre niveau global en développement logiciel (algorithmique, architecture, clean code, tests).",
        "q2": "Quel est votre degré de maîtrise du développement web et/ou mobile (front-end, back-end, API, frameworks) ?",
        "q3_label": "Comment jugez-vous votre capacité à concevoir des architectures logicielles robustes, scalables et maintenables ?",
        "q7_options": ["Software Engineer", "Full Stack Developer", "Back-end Developer", "Front-end Developer", "Mobile Developer", "Tech Lead / Architect", "Autre"],
        "q7_label": "Quel titre correspondait le mieux à votre dernière fonction ?",
        "q8_label": "Décrivez votre dernier environnement de travail (langages, frameworks, méthodologie Agile, taille de l'équipe technique).",
        "q9_label": "Détaillez un projet technique majeur que vous avez conçu ou largement contribué. Précisez la problématique, votre rôle, les choix technologiques effectués et l'impact livré.",
        "q10_label": "Quels ont été les principaux défis techniques (dette technique, performance, sécurité, scalabilité, bugs critiques) et comment les avez-vous résolus ?",
    },
]

# Échelle Likert commune (Q3 radio)
LIKERT_RADIO = ["Débutant", "Notions", "Intermédiaire", "Avancé", "Expert"]

# Fonction utilitaire : détecte si l'utilisateur a interagi avec un domaine
def _domaine_est_rempli(answers: dict, key: str) -> bool:
    """
    Retourne True si l'utilisateur a modifié au moins un champ du domaine :
      - slider Q1 ou Q2 différent de la valeur par défaut (3)
      - radio Q3 différent de 'Débutant' (index 0)
      - multiselect Q4 non vide
      - text_area Q5/Q9/Q10 non vide
      - number_input Q6 > 0
      - selectbox Q7 différent de 'Autre'
      - text_input Q8 non vide
    """
    defaults_unchanged = (
        answers.get(f"{key}_q1", 3) == 3
        and answers.get(f"{key}_q2", 3) == 3
        and answers.get(f"{key}_q3", LIKERT_RADIO[0]) == LIKERT_RADIO[0]
        and not answers.get(f"{key}_q4", [])
        and not answers.get(f"{key}_q5", "").strip()
        and answers.get(f"{key}_q6", 0) == 0
        and answers.get(f"{key}_q7", "Autre") == "Autre"
        and not answers.get(f"{key}_q8", "").strip()
        and not answers.get(f"{key}_q9", "").strip()
        and not answers.get(f"{key}_q10", "").strip()
    )
    return not defaults_unchanged


# ──────────────────────────────────────────────────────────────
# ONGLETS PRINCIPAUX
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "📝  Évaluation (Questionnaire)",
    "📊  Dashboard Analytique (Résultats NLP)",
    "✨  Agent GenAI (Plan & Bio)",
])


# ══════════════════════════════════════════════════════════════
# ONGLET 1 — QUESTIONNAIRE DYNAMIQUE (9 DOMAINES)
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="step-badge">⬤ ÉTAPE 1 / 3 · Auto-évaluation</div>', unsafe_allow_html=True)
    st.markdown("### Cartographiez vos compétences professionnelles")
    st.markdown(
        "Ouvrez les domaines qui correspondent à votre parcours et répondez aux 10 questions de chaque bloc. "
        "**Seuls les domaines remplis seront analysés.** Vos réponses textuelles alimentent le moteur SBERT pour "
        "la recommandation de métiers.",
        help="Vous pouvez ne remplir qu'un seul domaine — les blocs non ouverts seront ignorés dans l'analyse.",
    )

    # Indicateur de progression
    nb_domaines = len(DOMAINES)
    col_prog1, col_prog2, col_prog3 = st.columns([1, 1, 1])
    with col_prog1:
        st.markdown(
            f"<p style='font-size:0.82rem;color:#6B7280;margin:0;'>"
            f"📂 <strong>{nb_domaines} domaines</strong> disponibles</p>",
            unsafe_allow_html=True,
        )
    with col_prog2:
        st.markdown(
            "<p style='font-size:0.82rem;color:#6B7280;margin:0;'>"
            "❓ <strong>10 questions</strong> par domaine</p>",
            unsafe_allow_html=True,
        )
    with col_prog3:
        st.markdown(
            "<p style='font-size:0.82rem;color:#6B7280;margin:0;'>"
            "🔍 Analyse <strong>NLP/SBERT</strong> sur vos réponses libres</p>",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────
    # FORMULAIRE UNIQUE — toute la logique dans un st.form
    # ──────────────────────────────────────────────────────────
    with st.form(key="competency_form", clear_on_submit=False):

        raw_answers = {}  # Collecte brute de TOUS les champs

        # ── Boucle principale sur les 9 domaines ──────────────
        for domaine in DOMAINES:
            k = domaine["key"]

            with st.expander(f"**{domaine['label']}**", expanded=False):

                # ── BLOC 1 : Évaluation du niveau (Q1–Q3) ─────
                st.markdown(
                    "<p style='font-size:0.8rem;font-weight:600;color:#6B7280;"
                    "text-transform:uppercase;letter-spacing:1px;margin-bottom:0.8rem;'>"
                    "📊 Évaluation du niveau</p>",
                    unsafe_allow_html=True,
                )
                col_q1, col_q2 = st.columns(2, gap="large")

                with col_q1:
                    # Q1 — Slider Likert (1–5)
                    st.markdown(f"**Q1 ·** {domaine['q1']}")
                    q1_val = st.slider(
                        label="Q1",
                        min_value=1, max_value=5, value=3, step=1,
                        key=f"{k}_q1",
                        format="%d ⭐",
                        help="1 = Débutant · 3 = Intermédiaire · 5 = Expert",
                        label_visibility="collapsed",
                    )
                    likert_map = {1: "Débutant", 2: "Notions", 3: "Intermédiaire", 4: "Avancé", 5: "Expert"}
                    st.caption(f"Niveau : **{likert_map[q1_val]}**")
                    raw_answers[f"{k}_q1"] = q1_val

                with col_q2:
                    # Q2 — Slider Likert (1–5)
                    st.markdown(f"**Q2 ·** {domaine['q2']}")
                    q2_val = st.slider(
                        label="Q2",
                        min_value=1, max_value=5, value=3, step=1,
                        key=f"{k}_q2",
                        format="%d ⭐",
                        help="1 = Débutant · 3 = Intermédiaire · 5 = Expert",
                        label_visibility="collapsed",
                    )
                    st.caption(f"Niveau : **{likert_map[q2_val]}**")
                    raw_answers[f"{k}_q2"] = q2_val

                # Q3 — Radio Likert verbal
                st.markdown(f"**Q3 ·** {domaine['q3_label']}")
                q3_val = st.radio(
                    label="Q3",
                    options=LIKERT_RADIO,
                    index=0,
                    horizontal=True,
                    key=f"{k}_q3",
                    label_visibility="collapsed",
                )
                raw_answers[f"{k}_q3"] = q3_val

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                # ── BLOC 2 : Compétences & Justification (Q4–Q5) ──
                st.markdown(
                    "<p style='font-size:0.8rem;font-weight:600;color:#6B7280;"
                    "text-transform:uppercase;letter-spacing:1px;margin-bottom:0.8rem;'>"
                    "🎯 Compétences & Justification sémantique</p>",
                    unsafe_allow_html=True,
                )

                # Q4 — Multiselect compétences
                # ── BRANCHER : `domaine["competences"]` sera remplacé par la
                # liste réelle issue de `competences_par_domaine[domaine["label"]]`
                st.markdown(f"**Q4 ·** Sélectionnez vos compétences confirmées en **{domaine['label']}** :")
                q4_val = st.multiselect(
                    label="Q4",
                    options=domaine["competences"],
                    default=[],
                    key=f"{k}_q4",
                    label_visibility="collapsed",
                    placeholder="Cliquez pour sélectionner vos compétences...",
                )
                raw_answers[f"{k}_q4"] = q4_val

                # Q5 — Text area justification (obligatoire sémantiquement)
                st.markdown(
                    "**Q5 ·** *Justification requise :* Décrivez concrètement dans quel cadre vous avez "
                    "mis en pratique les compétences cochées ci-dessus. Précisez les outils utilisés et les résultats obtenus."
                )
                q5_val = st.text_area(
                    label="Q5",
                    placeholder=(
                        "Ex : « Dans le cadre de mon stage chez X, j'ai utilisé pandas et scikit-learn pour "
                        "nettoyer un dataset de 2M de lignes et entraîner un modèle de classification, "
                        "atteignant 91% d'accuracy sur le jeu de test... »"
                    ),
                    key=f"{k}_q5",
                    height=120,
                    label_visibility="collapsed",
                )
                raw_answers[f"{k}_q5"] = q5_val
                # ── Cette réponse sera encodée par SBERT pour le calcul de similarité cosine ──
                # from backend.nlp_engine import encode_text
                # raw_answers[f"{k}_q5_embedding"] = encode_text(q5_val)

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                # ── BLOC 3 : Expérience professionnelle (Q6–Q8) ───
                st.markdown(
                    "<p style='font-size:0.8rem;font-weight:600;color:#6B7280;"
                    "text-transform:uppercase;letter-spacing:1px;margin-bottom:0.8rem;'>"
                    "💼 Expérience professionnelle</p>",
                    unsafe_allow_html=True,
                )
                col_q6, col_q7 = st.columns([1, 1.5], gap="large")

                with col_q6:
                    # Q6 — Number input années d'expérience
                    st.markdown(f"**Q6 ·** Combien d'années d'expérience en entreprise cumulez-vous spécifiquement dans ce secteur ?")
                    q6_val = st.number_input(
                        label="Q6",
                        min_value=0, max_value=40, value=0, step=1,
                        key=f"{k}_q6",
                        help="Comptez uniquement l'expérience directement liée à ce domaine.",
                        label_visibility="collapsed",
                    )
                    raw_answers[f"{k}_q6"] = q6_val

                with col_q7:
                    # Q7 — Selectbox dernier poste
                    st.markdown(f"**Q7 ·** {domaine['q7_label']}")
                    q7_val = st.selectbox(
                        label="Q7",
                        options=domaine["q7_options"],
                        index=len(domaine["q7_options"]) - 1,  # "Autre" par défaut
                        key=f"{k}_q7",
                        label_visibility="collapsed",
                    )
                    raw_answers[f"{k}_q7"] = q7_val

                # Q8 — Text input environnement de travail
                st.markdown(f"**Q8 ·** {domaine['q8_label']}")
                q8_val = st.text_input(
                    label="Q8",
                    placeholder="Ex : Stack Python/AWS/Snowflake, équipe de 5 data scientists, volumétrie ~10M lignes/jour...",
                    key=f"{k}_q8",
                    label_visibility="collapsed",
                )
                raw_answers[f"{k}_q8"] = q8_val

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                # ── BLOC 4 : Projets & Résolution (Q9–Q10) ────────
                st.markdown(
                    "<p style='font-size:0.8rem;font-weight:600;color:#6B7280;"
                    "text-transform:uppercase;letter-spacing:1px;margin-bottom:0.8rem;'>"
                    "🚀 Projets & Résolution de problèmes</p>",
                    unsafe_allow_html=True,
                )
                # ── Q9 et Q10 sont les champs les plus riches pour SBERT ──────
                # from backend.nlp_engine import encode_text
                # raw_answers[f"{k}_q9_embedding"] = encode_text(q9_val)
                # raw_answers[f"{k}_q10_embedding"] = encode_text(q10_val)

                # Q9 — Projet majeur
                st.markdown(f"**Q9 ·** {domaine['q9_label']}")
                q9_val = st.text_area(
                    label="Q9",
                    placeholder=(
                        "Décrivez : la problématique métier initiale, votre rôle exact, "
                        "les solutions/modèles mis en œuvre et l'impact final mesuré..."
                    ),
                    key=f"{k}_q9",
                    height=130,
                    label_visibility="collapsed",
                )
                raw_answers[f"{k}_q9"] = q9_val

                # Q10 — Défis surmontés
                st.markdown(f"**Q10 ·** {domaine['q10_label']}")
                q10_val = st.text_area(
                    label="Q10",
                    placeholder=(
                        "Décrivez les obstacles rencontrés (techniques, organisationnels, "
                        "humains) et votre démarche pour les surmonter..."
                    ),
                    key=f"{k}_q10",
                    height=130,
                    label_visibility="collapsed",
                )
                raw_answers[f"{k}_q10"] = q10_val

        # ── BOUTON DE SOUMISSION UNIQUE ────────────────────────
        submitted = st.form_submit_button(
            "🚀  Lancer l'Analyse Sémantique",
            type="primary",
            use_container_width=False,
        )

        if submitted:
            # ── Filtrage : ne conserver que les domaines interagis ──
            active_answers = {
                "domaines_actifs": [],
            }

            for domaine in DOMAINES:
                k = domaine["key"]
                # Extraire les réponses de ce domaine
                domaine_data = {
                    field: raw_answers[field]
                    for field in raw_answers
                    if field.startswith(f"{k}_")
                }
                if _domaine_est_rempli(raw_answers, k):
                    active_answers[k] = domaine_data
                    active_answers["domaines_actifs"].append(domaine["label"])
                else:
                    # Domaine non rempli → score = 0, ignoré par SBERT
                    active_answers[k] = {"_status": "ignoré", "_score_global": 0}

            # ── BRANCHER ICI : appel SBERT / calcul des scores de similarité ──
            # from backend.nlp_engine import compute_similarity_scores
            # nlp_scores = compute_similarity_scores(active_answers)
            # st.session_state["nlp_scores"] = nlp_scores
            # ──────────────────────────────────────────────────────────────────

            st.session_state["form_submitted"] = True
            st.session_state["answers"] = active_answers
            st.session_state["genai_triggered"] = False  # reset si re-soumission

            # Enregistrer les réponses dans un fichier JSON pour persistance / debug
            with open("data/user_answers.json", "w", encoding="utf-8") as out_file:
                json.dump(active_answers, out_file, ensure_ascii=False, indent=2)

            nb_actifs = len(active_answers["domaines_actifs"])
            st.success(
                f"✅ Analyse lancée sur **{nb_actifs} domaine(s) actif(s)** ! "
                "Rendez-vous dans l'onglet **Dashboard Analytique** pour vos résultats.",
                icon="✅",
            )
            if nb_actifs == 0:
                st.warning(
                    "⚠️ Aucun domaine rempli détecté. Ouvrez au moins un bloc et modifiez un champ pour que l'analyse soit significative.",
                    icon="⚠️",
                )
            else:
                st.balloons()

            subprocess.run(["python3", "src/launcher.py"])
            print(">>> launcher.py a bien été appelé !")

    # ── Résumé session post-soumission ────────────────────────
    if st.session_state["form_submitted"]:
        with st.expander("ℹ️  Résumé des données enregistrées en session", expanded=False):
            actifs = st.session_state["answers"].get("domaines_actifs", [])
            if actifs:
                st.markdown(f"**Domaines analysés ({len(actifs)}) :** {' · '.join(actifs)}")
            st.json(st.session_state["answers"])


# ══════════════════════════════════════════════════════════════
# ONGLET 2 — DASHBOARD NLP
# ══════════════════════════════════════════════════════════════
with tab2:
    if not st.session_state["form_submitted"]:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>Formulaire non soumis.</strong><br>
            Complétez et soumettez le questionnaire de l'onglet <strong>Évaluation</strong>
            pour débloquer le dashboard analytique.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    st.markdown('<div class="step-badge">⬤ ÉTAPE 2 / 3 · Résultats NLP</div>', unsafe_allow_html=True)
    st.markdown("### Résultats de l'analyse sémantique")
    st.caption("Les scores ci-dessous simulent la sortie du moteur SBERT. Branchez `backend.nlp_engine` pour les données réelles.")

    # ── Top 3 recommandations dynamiques ───────────────────────
    st.markdown("#### 🏆 Top 3 des métiers recommandés")
    try:
        with open("data/match_results.json", "r", encoding="utf-8") as f:
            res_data = json.load(f)
        match_jobs = res_data.get("top_jobs", [])
    except Exception as e:
        st.error(f"Erreur lors du chargement des résultats : {e}")
        match_jobs = []

    cols_jobs = st.columns(3, gap="medium")
    for i, (col, job) in enumerate(zip(cols_jobs, match_jobs)):
        with col:
            tags_html = "".join(f'<span class="tag">{t}</span>' for t in job.get("tags", []))
            # Charger le mapping CompetencyID -> Competency
            try:
                comp_df = pd.read_csv("data/competencs2.csv")
                comp_map = dict(zip(comp_df["CompetencyID"], comp_df["Competency"]))
            except Exception as e:
                comp_map = {}
            # Afficher les noms des compétences
            comp_names = [comp_map.get(cid, cid) for cid in job.get("required_competencies", [])[:5]]
            st.markdown(f"""
            <div class="job-card">
                <div class="job-card-rank">{job.get('rank', f'# {i+1}')}</div>
                <div class="job-card-title">{job.get('job_title', 'Métier inconnu')}</div>
                <div class="job-card-score">{round(job.get('score', 0) * 100)}%</div>
                <div class="job-card-label">{', '.join(comp_names)}</div>
                <div class="job-card-tags">{tags_html}</div>
                <p style="font-size:0.82rem;color:#6B7280;margin-top:0.9rem;line-height:1.5;">{job.get('desc', '')}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # ── Graphiques ─────────────────────────────────────────────
    col_radar, col_bar = st.columns([1, 1], gap="large")

    # --- Définition des labels pour l'affichage ---
    ID_TO_NAME = {
        "C091": "Sales Strategy",
        "C099": "Process Optiz",
        "C097": "Sales Data",
        "C093": "Vendor Mgmt",
        "C082": "Brand & Pos.",
        "C081": "Growth Strategy",
        "C086": "SEO & SEM"
    }
    
    with col_radar:
        st.markdown("#### 🕸️ Radar — Profil vs Métier cible")
        
        try:
            # 1. Chargement des données
            path_results = "data/match_results.json"
            path_ref_secteur = "data/referentiel_secteurs.json"

            if os.path.exists(path_results) and os.path.exists(path_ref_secteur):
                with open(path_results, "r") as f1:
                    res_data = json.load(f1)
                with open(path_ref_secteur, "r") as f2:
                    ref_secteurs = json.load(f2)
                
                # 2. Identification du métier cible (Top 1)
                target_job = res_data["top_jobs"][0]["job_title"]
                st.caption(f"Comparaison de votre profil avec le référentiel : **{target_job}**")

                # 3. Extraction des IDs de compétences du secteur
                ids_cles = list(ID_TO_NAME.keys())
                labels = list(ID_TO_NAME.values())

                # 4. Préparation des scores Utilisateur (Scale 0-1 -> 0-5)
                # On cherche le score SBERT dans top_skills pour chaque ID du secteur
                user_scores_raw = {s['skill_id']: s['score'] for s in res_data.get("top_skills", [])}
                val_user = [user_scores_raw.get(cid, 0) * 5 for cid in ids_cles]

                # 5. Préparation des scores Cibles (Scale 0-5)
                job_ref = ref_secteurs.get(target_job, {})
                val_ref = [job_ref.get(cid, 0) for cid in ids_cles]

                # 6. Création du Radar Chart
                fig_radar = go.Figure()

                # Trace Profil Utilisateur
                fig_radar.add_trace(go.Scatterpolar(
                    r=val_user + [val_user[0]],
                    theta=labels + [labels[0]],
                    fill='toself',
                    fillcolor='rgba(27,58,92,0.15)',
                    line=dict(color='#1B3A5C', width=2),
                    name='Votre profil'
                ))

                # Trace Référentiel Métier
                fig_radar.add_trace(go.Scatterpolar(
                    r=val_ref + [val_ref[0]],
                    theta=labels + [labels[0]],
                    fill='toself',
                    fillcolor='rgba(245,158,11,0.10)',
                    line=dict(color='#F59E0B', width=2, dash='dash'),
                    name=f'Cible : {target_job}'
                ))

                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 5], gridcolor="#E4E7F0"),
                        angularaxis=dict(tickfont=dict(size=10)),
                        bgcolor="#F7F8FC",
                    ),
                    showlegend=True,
                    legend=dict(orientation="h", y=-0.2),
                    margin=dict(l=60, r=60, t=20, b=50),
                    height=400
                )
                
                st.plotly_chart(fig_radar, use_container_width=True)
                
            else:
                st.warning("Fichiers de données introuvables. Veuillez lancer l'analyse.")
                
        except Exception as e:
            st.error(f"Erreur lors de la génération du radar : {e}")

    with col_bar:
        st.markdown("#### 📊 Couverture des compétences par bloc")
        
        if "top_skills" in res_data and "top_jobs" in res_data:
            # 1. Préparation des données (mêmes IDs que le radar)
            ids_cles = list(ID_TO_NAME.keys())
            
            # Récupération des scores (User = 0-1, Ref = 0-5)
            user_scores_raw = {s['skill_id']: s['score'] for s in res_data.get("top_skills", [])}
            target_job = res_data["top_jobs"][0]["job_title"]
            job_ref = ref_secteurs.get(target_job, {})

            # 2. Calcul du pourcentage de couverture réel
            # Formule : (Score User * 5 / Score Cible) * 100
            rows = []
            for cid in ids_cles:
                score_user_5 = user_scores_raw.get(cid, 0) * 5
                score_cible = job_ref.get(cid, 0)
                
                # Calcul du pourcentage (max 100% pour la lisibilité)
                if score_cible > 0:
                    coverage = min((score_user_5 / score_cible) * 100, 100.0)
                else:
                    coverage = 0.0
                
                # Détermination du statut pour la couleur
                if coverage >= 75:
                    statut = "Fort"
                elif coverage >= 45:
                    statut = "Moyen"
                else:
                    statut = "Faible"
                    
                rows.append({
                    "Compétence": ID_TO_NAME[cid],
                    "Couverture (%)": round(coverage, 1),
                    "Statut": statut
                })

            # Charger le mapping CompetencyID -> BlockID et BlockName
            try:
                comp_df = pd.read_csv("data/competencs2.csv")
                blockid_map = dict(zip(comp_df["CompetencyID"], comp_df["BlockID"]))
                blockname_map = dict(zip(comp_df["CompetencyID"], comp_df["BlockName"]))
            except Exception as e:
                blockid_map = {}
                blockname_map = {}

            # Palette de couleurs pour BlockID (ajustez si besoin)
            block_colors = [
                "#1B3A5C", "#F59E0B", "#EF4444", "#10B981", "#6366F1", "#F472B6", "#FBBF24", "#3B82F6",
                "#8B5CF6", "#EC4899", "#22D3EE", "#A3E635", "#F87171", "#FCD34D", "#6EE7B7", "#818CF8",
                "#FDE68A", "#FCA5A5", "#C084FC", "#F9A8D4", "#34D399", "#60A5FA", "#FACC15", "#A78BFA"
            ]

            # Associer chaque BlockID à une couleur unique
            unique_blockids = list({blockid_map.get(cid, 0) for cid in ids_cles})
            blockid_to_color = {bid: block_colors[i % len(block_colors)] for i, bid in enumerate(sorted(unique_blockids))}

            # Ajouter Bloc et BlockID à coverage_data
            for row in rows:
                # On suppose que la clé de compétence est unique
                cid = next((k for k, v in ID_TO_NAME.items() if v == row["Compétence"]), None)
                row["Bloc"] = blockname_map.get(cid, "Bloc inconnu") if cid else "Bloc inconnu"
                row["BlockID"] = blockid_map.get(cid, 0) if cid else 0

            coverage_data = pd.DataFrame(rows)

            fig_bar = px.bar(
                coverage_data,
                x="Couverture (%)",
                y="Compétence",
                orientation="h",
                color="Bloc",
                color_discrete_map={row["Bloc"]: blockid_to_color.get(row["BlockID"], "#888888") for _, row in coverage_data.iterrows()},
                text="Couverture (%)",
            )

            fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
            fig_bar.update_layout(
                xaxis=dict(range=[0, 115], ticksuffix="%", gridcolor="#E4E7F0"),
                yaxis=dict(categoryorder="total ascending"),
                legend=dict(orientation="h", y=-0.2, font=dict(size=11)),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=30, t=20, b=50),
                height=400,
                showlegend=True,
            )

            st.caption(f"Adéquation de votre profil face aux exigences de : **{target_job}**")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Données insuffisantes pour calculer la couverture.")
            
    # ── Gaps identifiés & priorités de développement (DYNAMIQUE) ──────────────
    st.markdown("#### 🔍 Gaps identifiés & priorités de développement")

    def get_bloc_name(skill_id):
        """
        Mappe l'ID de la compétence vers le label du domaine correspondant.
        """
        try:
            num = int(skill_id.strip('C'))
            
            # --- MAPPING PRÉCIS BASÉ SUR TON RÉFÉRENTIEL ---
            if num in [6, 7, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]:
                return "Développement"
            elif num in [8, 9, 10, 74, 75, 76, 95, 96, 97, 117, 141, 142, 172]:
                return "Data Analysis & BI"
            elif num in [11, 12, 13]:
                return "Intelligence Artificielle"
            elif num in [34, 35]:
                return "Stockage de Données"
            elif num in [1, 2, 3, 14, 15, 16, 36, 37, 38, 39, 40, 41, 42]:
                return "Infrastructure Cloud Réseaux"
            elif num in [19, 43, 44, 45, 46, 49, 50, 77, 78, 149, 150, 151, 156, 173]:
                return "Sécurité Risque et Conformité"
            elif num in [47, 48, 131, 132, 133, 134, 157, 158]:
                return "Qualité & Audit"
            elif num in [109, 110, 111]:
                return "Produit & CX"
            elif num in [83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 99]:
                return "Marketing Growth & Ventes"
            elif num in [59, 60, 61, 62, 63, 64, 81, 82]:
                return "Stratégie & Gestion de projet"
            elif num in [65, 66, 67, 68, 69, 70, 71, 72, 73, 112, 113, 114, 143]:
                return "Opérations & Logistique"
            elif num in [123, 124, 127, 128, 129, 130, 140]:
                return "Finance et Comptabilité"
            elif num in [125, 126]:
                return "Partenariats"
            elif num in [135, 136, 137, 138, 139]:
                return "Juridique"
            elif num in [159, 160, 161, 162, 163, 164, 165, 167, 168, 169, 170, 171]:
                return "RH"
            elif num in [51, 52, 105, 106]:
                return "Support Technique"
            elif num in [53, 54, 98, 115, 116]:
                return "Outils & Systèmes"
            elif num in [4, 5]:
                return "Outils Techniques & Terrain"
            elif num in [17, 18]:
                return "Recherche & Méthode"
            elif num in [20, 21, 55, 56, 57, 58, 79, 80, 100, 101, 102, 103, 104, 107, 108, 118, 119, 120, 121, 122, 144, 145, 146, 147, 148, 174, 175, 176, 177]:
                return "Soft Skills"
            return "Autres fonctions"
        except:
            return "Général"

    # Logique d'affichage dynamique
    if "top_skills" in res_data and res_data["top_skills"]:
        bloc_scores = {}
        for s in res_data["top_skills"]:
            b_name = get_bloc_name(s["skill_id"])
            if b_name not in bloc_scores:
                bloc_scores[b_name] = []
            bloc_scores[b_name].append(s["score"])
        
        # Calcul de la moyenne par bloc
        bloc_averages = {k: sum(v)/len(v) for k, v in bloc_scores.items()}
        
        # Identification du plus fort et du plus faible (parmi ceux présents dans le JSON)
        strongest_bloc = max(bloc_averages, key=bloc_averages.get)
        weakest_bloc = min(bloc_averages, key=bloc_averages.get)
        
        # Récupération du score global du Job n°1 (ex: 0.25 -> 25.0%)
        best_job_score = res_data["top_jobs"][0].get("score", 0) * 100 if res_data.get("top_jobs") else 0

        gap_col1, gap_col2, gap_col3 = st.columns(3)

        with gap_col1:
            # Score moyen du bloc le plus fort
            score_fort = bloc_averages[strongest_bloc] * 100
            st.metric(
                label="Bloc le plus fort", 
                value=strongest_bloc, 
                delta=f"{score_fort:.1f}% couverture", 
                delta_color="normal"
            )

        with gap_col2:
            # Score moyen du bloc le plus faible
            score_faible = bloc_averages[weakest_bloc] * 100
            # On calcule le gap par rapport à une cible de 80% (0.8)
            gap_vs_target = score_faible - 80.0
            st.metric(
                label="Bloc prioritaire (Gap)", 
                value=weakest_bloc, 
                delta=f"{gap_vs_target:.1f}% vs cible", 
                delta_color="inverse"
            )

        with gap_col3:
            # Adéquation Globale (Top Job)
            st.metric(
                label="Adéquation Globale", 
                value=f"{best_job_score:.1f}%", 
                delta="Top Match", 
                delta_color="off"
            )
    else:
        st.info("Complétez l'analyse pour afficher les statistiques détaillées.")

# ══════════════════════════════════════════════════════════════
# ONGLET 3 — AGENT GENAI
# ══════════════════════════════════════════════════════════════
with tab3:
    # 1. Sécurité : Vérification de la soumission du formulaire
    if not st.session_state.get("form_submitted", False):
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>Formulaire non soumis.</strong><br>
            Complétez et soumettez le questionnaire de l'onglet <strong>Évaluation</strong>
            pour débloquer l'agent génératif.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # 2. Sécurité : Vérification de la présence des résultats NLP
    if not os.path.exists("data/match_results.json"):
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>Données SBERT manquantes.</strong><br>
            Les résultats de l'analyse sémantique sont introuvables. Veuillez relancer l'analyse dans l'onglet 1.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    st.markdown('<div class="step-badge">⬤ ÉTAPE 3 / 3 · Génération IA</div>', unsafe_allow_html=True)
    st.markdown("### Agent GenAI — Synthèse personnalisée")

    # Panneau d'intro
    st.markdown("""
    <div class="genai-panel">
        <div class="genai-title">✨ Ce que l'agent va générer pour vous</div>
        <ul style="color:#374151;font-size:0.9rem;line-height:2;margin:0.5rem 0 0 0;padding-left:1.2rem;">
            <li><strong>Biographie professionnelle</strong> — Executive Summary personnalisé à partir de vos forces et de votre compétence phare.</li>
            <li><strong>Plan de progression sur 12 mois</strong> — Roadmap actionnable ciblée spécifiquement sur vos lacunes (Skill Gap).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # 3. Les deux boutons d'action côte à côte
    col_btn1, col_btn2 = st.columns(2, gap="medium")
    
    with col_btn1:
        if st.button("✨ 1. Générer ma Biographie", type="primary", use_container_width=True):
            with st.spinner("Rédaction de la biographie en cours..."):
                # Appel à la fonction du backend (qui gère le cache et l'API Gemini)
                bio = generer_bio()
                st.session_state["genai_bio"] = bio
                st.rerun()

    with col_btn2:
        if st.button("🚀 2. Générer mon Plan de Progression", type="primary", use_container_width=True):
            with st.spinner("Analyse du Skill Gap et création du plan..."):
                # Appel à la fonction du backend
                plan = generer_plan()
                st.session_state["genai_plan"] = plan
                st.rerun()

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # 4. Affichage conditionnel des résultats
    col_bio, col_plan = st.columns([1, 1], gap="large")

    with col_bio:
        if "genai_bio" in st.session_state and st.session_state["genai_bio"]:
            st.markdown("#### 👤 Biographie professionnelle")
            st.caption("Executive Summary · Généré par Gemini 2.5 Flash")
            st.markdown(f"""
            <div class="genai-panel">
                <div class="genai-title">Executive Summary</div>
                <div class="genai-content">{st.session_state["genai_bio"]}</div>
            </div>
            """, unsafe_allow_html=True)
            st.download_button(
                label="⬇️  Télécharger la biographie",
                data=st.session_state["genai_bio"],
                file_name="aisca_biographie.txt",
                mime="text/plain",
                use_container_width=True,
            )

    with col_plan:
        if "genai_plan" in st.session_state and st.session_state["genai_plan"]:
            st.markdown("#### 🗺️ Plan de progression — 12 mois")
            st.caption("Étapes d'acquisition (Skill Gap) · Généré par Gemini 2.5 Flash")
            st.markdown(f"""
            <div class="genai-panel">
                <div class="genai-title">Roadmap personnalisée</div>
                <div class="genai-content">{st.session_state["genai_plan"]}</div>
            </div>
            """, unsafe_allow_html=True)
            st.download_button(
                label="⬇️  Télécharger le plan",
                data=st.session_state["genai_plan"],
                file_name="aisca_plan_progression.txt",
                mime="text/plain",
                use_container_width=True,
            )

    # 5. Bouton de réinitialisation si au moins un des éléments est généré
    if st.session_state.get("genai_bio") or st.session_state.get("genai_plan"):
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        col_regen, _ = st.columns([2, 3])
        with col_regen:
            if st.button("🔄  Effacer l'affichage", use_container_width=True):
                # On vide le state, ce qui va masquer les panneaux (mais ça reste dans votre cache JSON !)
                if "genai_bio" in st.session_state:
                    del st.session_state["genai_bio"]
                if "genai_plan" in st.session_state:
                    del st.session_state["genai_plan"]
                st.rerun()
# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#9CA3AF;font-size:0.78rem;'>"
    "AISCA · Projet Master Data Engineering & AI · EFREI Paris · "
    "<span style='font-family:Space Mono,monospace;'>v0.1-prototype</span>"
    "</p>",
    unsafe_allow_html=True,
)