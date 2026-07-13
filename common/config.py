"""
CohortX Task 1 — Shared Constants
All model names, query strings, regex patterns, and metric weights live here.
Import from any module to avoid scattered magic strings.
"""

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

BIOMEDBERT_NAME         = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
DISTILBERT_NAME         = "distilbert-base-uncased"
MINILM_NAME             = "all-MiniLM-L6-v2"
SCIFIVE_NAME            = "razent/SciFive-base-Pubmed"
SCIFIVE_CONDITIONS_NAME = "razent/SciFive-base-Pubmed"
BIOBERT_QA_NAME         = "deepset/deberta-v3-base-squad2"
DISEASE_NER_NAME        = "pruas/BENT-PubMedBERT-NER-Disease"
BIOBERT_EVAL_NAME       = "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"

# ---------------------------------------------------------------------------
# Generation / context limits
# ---------------------------------------------------------------------------

STAGE1_TOP_K       = 3
SCIFIVE_MAX_INPUT  = 512
SCIFIVE_MAX_OUTPUT = 256
AGE_MAX_CONTEXT    = 400

# ---------------------------------------------------------------------------
# Stage-1 eligibility query
# ---------------------------------------------------------------------------

ELIGIBILITY_QUERY = (
    "inclusion exclusion eligibility criteria study participants enrollment "
    "were eligible were excluded met criteria patients enrolled "
    "age sex gender minimum maximum demographic restrictions "
    "aged years old male female adult pediatric "
    "required diagnosis clinical conditions prior history comorbidities "
    "contraindications disqualifying factors "
    "study design randomized controlled trial observational prospective "
    "retrospective interventional cohort participants recruited "
    "disease condition disorder diagnosis primary secondary treated patients"
)

# ---------------------------------------------------------------------------
# Age QA questions
# ---------------------------------------------------------------------------

AGE_QUESTIONS = {
    "minimum": "What is the minimum age requirement for study participants?",
    "maximum": "What is the maximum age limit for study participants?",
}

# ---------------------------------------------------------------------------
# Evaluation metric constants
# ---------------------------------------------------------------------------

LAM   = 0.6   # FM3S lambda
ALPHA = 0.6   # CWO alpha

WEIGHTS = {
    "conditions":           0.15,
    "study_type":           0.10,
    "sex":                  0.05,
    "minimum_age":          0.10,
    "maximum_age":          0.10,
    "eligibility_criteria": 0.50,
}

STOP_VERBS = {
    "be", "have", "say", "do", "go", "make", "know", "get", "see",
    "come", "take", "think", "look", "want", "give", "use", "find",
    "tell", "ask", "seem", "feel", "try", "leave", "call",
}
