"""
CohortX Task 1 — NXML Parser
Parses PMC NXML (JATS) files into structured dicts.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

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


@dataclass
class BlockNode:
    block_id: str
    block_type: str
    text: str
    parent_section_id: str
    section_path: list[str]
    order: int

    # links within same section only
    prev_sibling_id: Optional[str] = None
    next_sibling_id: Optional[str] = None

    # optional full document order
    prev_global_id: Optional[str] = None
    next_global_id: Optional[str] = None


@dataclass
class SectionNode:
    section_id: str
    title: str
    path: list[str]
    level: int
    parent_id: Optional[str]
    order: int

    child_section_ids: list[str] = field(default_factory=list)
    block_ids: list[str] = field(default_factory=list)

    # section siblings under same parent
    prev_sibling_id: Optional[str] = None
    next_sibling_id: Optional[str] = None


class HierarchicalNXMLParser:
    """
    Parses NXML into:
    1. Section tree
    2. Block nodes under each section
    3. Sibling links within sections
    4. Optional global block order
    """

    def __init__(self):
        self.doc_id = ""
        self.sections: dict[str, SectionNode] = {}
        self.blocks: dict[str, BlockNode] = {}
        self.block_order: list[str] = []
        self._sec_counter = 0
        self._block_counter = 0

    def parse_file(self, filepath: str | Path) -> dict:
        filepath = Path(filepath)
        root = etree.parse(str(filepath)).getroot()
        return self._parse_root(root, doc_id=filepath.stem)

    def parse_string(self, xml: str, doc_id: str = "doc") -> dict:
        root = etree.fromstring(xml.encode("utf-8"))
        return self._parse_root(root, doc_id=doc_id)

    def _parse_root(self, root, doc_id: str) -> dict:
        self.doc_id = doc_id
        self.sections = {}
        self.blocks = {}
        self.block_order = []
        self._sec_counter = 0
        self._block_counter = 0

        body = root.find(".//body")
        if body is not None:
            for sec in self._direct_children(body, "sec"):
                self._parse_section(sec, parent_id=None, parent_path=[])

        self._link_global_blocks()

        return {
            "doc_id": doc_id,
            "title": self._clean_text(root.find(".//article-title")),
            "abstract": " ".join(
                self._clean_text(a) for a in root.findall(".//abstract")
                if self._clean_text(a)
            ),
            "keywords": [
                self._clean_text(k) for k in root.findall(".//kwd")
                if self._clean_text(k)
            ],
            "sections": {sid: asdict(sec) for sid, sec in self.sections.items()},
            "blocks": {bid: asdict(block) for bid, block in self.blocks.items()},
            "block_order": self.block_order,
            "flat_blocks": [asdict(self.blocks[bid]) for bid in self.block_order],
        }

    def _parse_section(self, sec, parent_id: Optional[str], parent_path: list[str]) -> str:
        title = self._direct_title(sec) or "Untitled Section"
        path = parent_path + [title]

        section_id = f"{self.doc_id}::sec_{self._sec_counter}"
        self._sec_counter += 1

        section = SectionNode(
            section_id=section_id,
            title=title,
            path=path,
            level=len(path),
            parent_id=parent_id,
            order=self._sec_counter - 1,
        )

        self.sections[section_id] = section

        if parent_id is not None:
            parent = self.sections[parent_id]
            if parent.child_section_ids:
                prev_id = parent.child_section_ids[-1]
                section.prev_sibling_id = prev_id
                self.sections[prev_id].next_sibling_id = section_id
            parent.child_section_ids.append(section_id)

        # direct children only: prevents subsection duplication
        for child in sec:
            tag = self._tag(child)

            if tag in {"title", "sec"}:
                continue

            if tag == "p":
                self._add_block("paragraph", child, section)
            elif tag == "list":
                self._add_block("list", child, section)
            elif tag == "table-wrap":
                self._add_block("table", child, section)
            elif tag == "fig":
                caption = self._first_descendant(child, "caption")
                if caption is not None:
                    self._add_block("figure_caption", caption, section)

        self._link_section_blocks(section)

        # recurse into child sections
        for child_sec in self._direct_children(sec, "sec"):
            self._parse_section(child_sec, parent_id=section_id, parent_path=path)

        return section_id

    def _add_block(self, block_type: str, el, section: SectionNode) -> Optional[str]:
        text = self._clean_text(el)
        if not text:
            return None

        block_id = f"{self.doc_id}::block_{self._block_counter}"
        self._block_counter += 1

        block = BlockNode(
            block_id=block_id,
            block_type=block_type,
            text=text,
            parent_section_id=section.section_id,
            section_path=section.path.copy(),
            order=len(self.block_order),
        )

        self.blocks[block_id] = block
        section.block_ids.append(block_id)
        self.block_order.append(block_id)

        return block_id

    def _link_section_blocks(self, section: SectionNode) -> None:
        ids = section.block_ids
        for i, bid in enumerate(ids):
            block = self.blocks[bid]
            if i > 0:
                block.prev_sibling_id = ids[i - 1]
            if i + 1 < len(ids):
                block.next_sibling_id = ids[i + 1]

    def _link_global_blocks(self) -> None:
        for i, bid in enumerate(self.block_order):
            block = self.blocks[bid]
            if i > 0:
                block.prev_global_id = self.block_order[i - 1]
            if i + 1 < len(self.block_order):
                block.next_global_id = self.block_order[i + 1]

    def _direct_children(self, el, tag: str):
        return [child for child in el if self._tag(child) == tag]

    def _direct_title(self, sec) -> str:
        for child in sec:
            if self._tag(child) == "title":
                return self._clean_text(child)
        return ""

    def _first_descendant(self, el, tag: str):
        for desc in el.iter():
            if desc is not el and self._tag(desc) == tag:
                return desc
        return None

    def _tag(self, el) -> str:
        return str(el.tag).split("}", 1)[-1]

    def _clean_text(self, el) -> str:
        if el is None:
            return ""
        return re.sub(r"\s+", " ", " ".join(el.itertext())).strip()
