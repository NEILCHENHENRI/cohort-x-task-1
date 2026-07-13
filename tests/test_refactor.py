"""
Tests for refactored CohortX codebase.
Run from the repo root: python -m pytest tests/  (or python -m tests.test_refactor)

Heavy-dependency tests (torch, sentence-transformers) are skipped automatically
when those packages are not installed — they pass in the full training environment.
"""

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

# Detect optional deps once
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


# ---------------------------------------------------------------------------
# Test 1 — imports
# ---------------------------------------------------------------------------

class TestImports(unittest.TestCase):

    def test_config_imports(self):
        from common import config
        self.assertIsInstance(config.ELIGIBILITY_QUERY, str)
        self.assertIsInstance(config.WEIGHTS, dict)
        self.assertIsInstance(config.AGE_QUESTIONS, dict)
        self.assertIsInstance(config.STOP_VERBS, set)

    def test_parser_imports(self):
        from common.parser import NXMLParser
        self.assertTrue(callable(NXMLParser))

    def test_evaluate_imports(self):
        from common.evaluate import (
            extract_numbers, number_similarity,
            score_row, evaluate_fast, evaluate,
        )
        for fn in [extract_numbers, number_similarity, score_row,
                   evaluate_fast, evaluate]:
            self.assertTrue(callable(fn))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
    def test_models_imports(self):
        from finetuned_models.models import (
            NXMLParser, BiomedBERTEmbedder, Stage1Ranker,
            SciFiveGenerator, EligibilityExtractor,
            BERTClassifier, ConditionsExtractor,
            AgeExtractor, CohortXPipeline,
        )
        for cls in [NXMLParser, BiomedBERTEmbedder, Stage1Ranker,
                    SciFiveGenerator, EligibilityExtractor,
                    BERTClassifier, ConditionsExtractor,
                    AgeExtractor, CohortXPipeline]:
            self.assertTrue(callable(cls))

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
    def test_train_cli_importable(self):
        import importlib.util
        spec   = importlib.util.spec_from_file_location("train_cli", "finetuned_models/train.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(callable(module.main))


# ---------------------------------------------------------------------------
# Test 2 — config consistency
# ---------------------------------------------------------------------------

class TestConfig(unittest.TestCase):

    def test_weights_sum_to_one(self):
        from common.config import WEIGHTS
        self.assertAlmostEqual(sum(WEIGHTS.values()), 1.0, places=5)

    def test_all_fields_present_in_weights(self):
        from common.config import WEIGHTS
        expected = {"conditions", "study_type", "sex",
                    "minimum_age", "maximum_age", "eligibility_criteria"}
        self.assertEqual(set(WEIGHTS.keys()), expected)

    def test_eligibility_has_highest_weight(self):
        from common.config import WEIGHTS
        self.assertEqual(max(WEIGHTS, key=WEIGHTS.get), "eligibility_criteria")

    def test_age_questions_have_both_bounds(self):
        from common.config import AGE_QUESTIONS
        self.assertIn("minimum", AGE_QUESTIONS)
        self.assertIn("maximum", AGE_QUESTIONS)
        self.assertGreater(len(AGE_QUESTIONS["minimum"]), 0)
        self.assertGreater(len(AGE_QUESTIONS["maximum"]), 0)

    def test_eligibility_query_nonempty(self):
        from common.config import ELIGIBILITY_QUERY
        self.assertGreater(len(ELIGIBILITY_QUERY), 50)

    def test_lam_alpha_in_range(self):
        from common.config import LAM, ALPHA
        self.assertGreater(LAM, 0.0)
        self.assertLess(LAM, 1.0)
        self.assertGreater(ALPHA, 0.0)
        self.assertLess(ALPHA, 1.0)

    def test_stop_verbs_is_set(self):
        from common.config import STOP_VERBS
        self.assertIsInstance(STOP_VERBS, set)
        self.assertIn("be", STOP_VERBS)

    def test_model_name_strings(self):
        from common.config import (BIOMEDBERT_NAME, DISTILBERT_NAME, MINILM_NAME,
                             SCIFIVE_NAME, BIOBERT_QA_NAME, BIOBERT_EVAL_NAME)
        for name in [BIOMEDBERT_NAME, DISTILBERT_NAME, MINILM_NAME,
                     SCIFIVE_NAME, BIOBERT_QA_NAME, BIOBERT_EVAL_NAME]:
            self.assertIsInstance(name, str)
            self.assertGreater(len(name), 0)


# ---------------------------------------------------------------------------
# Test 3 — NXMLParser
# ---------------------------------------------------------------------------

MINIMAL_NXML = textwrap.dedent("""\
    <?xml version="1.0"?>
    <article>
      <front>
        <article-meta>
          <title-group>
            <article-title>Test Article Title</article-title>
          </title-group>
          <abstract>
            <p>This is the abstract text.</p>
          </abstract>
          <kwd-group>
            <kwd>diabetes</kwd>
            <kwd>randomized trial</kwd>
          </kwd-group>
        </article-meta>
      </front>
      <body>
        <sec>
          <title>Methods</title>
          <p>Patients aged 18 to 65 years were eligible.</p>
          <p>Exclusion criteria: prior cardiovascular disease.</p>
        </sec>
        <sec>
          <title>Results</title>
          <p>We enrolled 120 patients.</p>
        </sec>
      </body>
    </article>
""")


class TestNXMLParser(unittest.TestCase):

    def setUp(self):
        from common.parser import NXMLParser
        self.parser = NXMLParser()
        self.tmp    = tempfile.NamedTemporaryFile(
            suffix=".nxml", delete=False, mode="w", encoding="utf-8"
        )
        self.tmp.write(MINIMAL_NXML)
        self.tmp.close()
        self.parsed = self.parser.parse(self.tmp.name)

    def tearDown(self):
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_returns_dict(self):
        self.assertIsInstance(self.parsed, dict)

    def test_title_extracted(self):
        self.assertEqual(self.parsed["title"], "Test Article Title")

    def test_abstract_extracted(self):
        self.assertIn("abstract", self.parsed["abstract"].lower())

    def test_keywords_extracted(self):
        self.assertIn("diabetes", self.parsed["keywords"])
        self.assertEqual(len(self.parsed["keywords"]), 2)

    def test_sections_extracted(self):
        titles = [s["title"] for s in self.parsed["sections"]]
        self.assertIn("Methods", titles)
        self.assertIn("Results", titles)

    def test_sections_have_paragraphs_field(self):
        for sec in self.parsed["sections"]:
            self.assertIn("paragraphs", sec)
            self.assertIsInstance(sec["paragraphs"], list)

    def test_section_text_joins_paragraphs(self):
        methods = next(s for s in self.parsed["sections"] if s["title"] == "Methods")
        self.assertIn("18 to 65", methods["text"])

    def test_missing_file_returns_empty_dict(self):
        result = self.parser.parse("/nonexistent/path/PMC999.nxml")
        self.assertEqual(result, {})

    @unittest.skipUnless(TORCH_AVAILABLE, "torch not installed")
    def test_models_parser_is_same_class_as_parser_module(self):
        from finetuned_models.models import NXMLParser as ModelParser
        from common.parser import NXMLParser as ParserModule
        self.assertIs(ModelParser, ParserModule)


# ---------------------------------------------------------------------------
# Test 4 — number_similarity (no model loading)
# ---------------------------------------------------------------------------

class TestNumberSimilarity(unittest.TestCase):

    def setUp(self):
        from common.evaluate import number_similarity, extract_numbers
        self.num_sim = number_similarity
        self.extract = extract_numbers

    def test_both_empty_returns_one(self):
        self.assertEqual(self.num_sim("Not Specified", "Not Specified"), 1.0)

    def test_one_empty_returns_zero(self):
        self.assertEqual(self.num_sim("18 Years", "Not Specified"), 0.0)
        self.assertEqual(self.num_sim("Not Specified", "18 Years"), 0.0)

    def test_exact_match(self):
        self.assertEqual(self.num_sim("18 Years", "18 Years"), 1.0)

    def test_no_overlap(self):
        self.assertEqual(self.num_sim("18 Years", "65 Years"), 0.0)

    def test_shared_number(self):
        score = self.num_sim("18 to 65 years", "18 Years")
        self.assertGreater(score, 0.0)

    def test_extract_digit(self):
        self.assertIn(18.0, self.extract("18 Years"))

    def test_extract_word_number(self):
        self.assertIn(18.0, self.extract("eighteen years old"))

    def test_both_no_numbers_returns_one(self):
        self.assertEqual(self.num_sim("male", "female"), 1.0)


# ---------------------------------------------------------------------------
# Test 5 — score_row (no model loading)
# ---------------------------------------------------------------------------

class TestScoreRow(unittest.TestCase):

    def _empty_pred_gold(self):
        from common.config import WEIGHTS
        return (
            {f: "Not Specified" for f in WEIGHTS},
            {f: "Not Specified" for f in WEIGHTS},
        )

    def test_score_row_has_all_keys(self):
        from common.evaluate import score_row
        from common.config import WEIGHTS
        pred, gold = self._empty_pred_gold()
        result = score_row(pred, gold)
        for k in list(WEIGHTS.keys()) + ["overall"]:
            self.assertIn(k, result)

    def test_overall_in_unit_range(self):
        from common.evaluate import score_row
        pred, gold = self._empty_pred_gold()
        result = score_row(pred, gold)
        self.assertGreaterEqual(result["overall"], 0.0)
        self.assertLessEqual(result["overall"], 1.0)

    def test_age_exact_match_scores_one(self):
        from common.evaluate import score_row
        pred, gold = self._empty_pred_gold()
        pred["minimum_age"] = gold["minimum_age"] = "18 Years"
        self.assertEqual(score_row(pred, gold)["minimum_age"], 1.0)

    def test_age_mismatch_scores_zero(self):
        from common.evaluate import score_row
        pred, gold = self._empty_pred_gold()
        pred["minimum_age"] = "18 Years"
        gold["minimum_age"] = "65 Years"
        self.assertEqual(score_row(pred, gold)["minimum_age"], 0.0)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
