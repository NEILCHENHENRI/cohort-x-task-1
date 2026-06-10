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
# Regex patterns for sex and age
# ---------------------------------------------------------------------------

SEX_PATTERNS = [
    (r"\b(male and female|both sexes|all sexes|males and females)\b", "ALL"),
    (r"\b(men and women|women and men)\b",                            "ALL"),
    (r"\bmale[s]?\b(?!\s+and\s+female)",                              "MALE"),
    (r"\bfemale[s]?\b(?!\s+and\s+male)",                              "FEMALE"),
    (r"\bwomen\b",                                                     "FEMALE"),
    (r"\bmen\b(?!\s+and\s+women)",                                    "MALE"),
]

AGE_PATTERNS = {
    "minimum": [
        r"(?:aged?|age[d]?\s+(?:between|from|of|≥|>=|>|at\s+least))[\s:]*(\d+(?:\.\d+)?)\s*(?:to|-|–)?\s*(?:years?(?:\s+old)?|yrs?)",
        r"(?:≥|>=|>|at\s+least|minimum\s+age[\s:of]*)\s*(\d+(?:\.\d+)?)\s*(?:years?(?:\s+old)?|yrs?)",
        r"(\d+(?:\.\d+)?)\s*(?:years?(?:\s+old)?|yrs?)\s*(?:or\s+(?:older|above|over)|and\s+(?:older|above|over))",
        r"minimum\s+age[\s:of]*(\d+(?:\.\d+)?)",
        r"aged?\s+(\d+)\s*[-–]\s*\d+",
    ],
    "maximum": [
        r"(?:aged?\s+(?:between|from)\s+\d+\s*(?:to|-|–)\s*)(\d+(?:\.\d+)?)\s*(?:years?(?:\s+old)?|yrs?)",
        r"(?:≤|<=|<|up\s+to|at\s+most|maximum\s+age[\s:of]*)\s*(\d+(?:\.\d+)?)\s*(?:years?(?:\s+old)?|yrs?)",
        r"(\d+(?:\.\d+)?)\s*(?:years?(?:\s+old)?|yrs?)\s*(?:or\s+(?:younger|below|under)|and\s+(?:younger|below|under))",
        r"maximum\s+age[\s:of]*(\d+(?:\.\d+)?)",
        r"aged?\s+\d+\s*[-–]\s*(\d+)",
    ],
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
