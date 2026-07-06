"""
CohortX Task 1 — NXML Parser
Parses PMC NXML (JATS) files into structured dicts.
"""

import logging

from lxml import etree

log = logging.getLogger(__name__)


class NXMLParser:
    def parse(self, filepath: str) -> dict:
        try:
            root = etree.parse(filepath).getroot()
        except Exception as e:
            log.warning(f"Parse failed {filepath}: {e}")
            return {}
        return {
            "title":    self._text(root.find(".//article-title")),
            "abstract": " ".join(self._text(a) for a in root.findall(".//abstract")),
            "keywords": [self._text(k) for k in root.findall(".//kwd")],
            "sections": self._get_sections(root),
        }

    def _text(self, el) -> str:
        return " ".join(el.itertext()).strip() if el is not None else ""

    def _get_sections(self, root) -> list:
        sections = []
        for sec in root.findall(".//sec"):
            title_el   = sec.find("title")
            title      = self._text(title_el).strip() if title_el is not None else ""
            paragraphs = [self._text(p) for p in sec.findall("p") if self._text(p)]
            text       = " ".join(paragraphs).strip()
            if title or text:
                sections.append({
                    "title":      title,
                    "text":       text,
                    "paragraphs": paragraphs,   # kept for Stage1Ranker paragraph-level scoring
                })
        return sections
