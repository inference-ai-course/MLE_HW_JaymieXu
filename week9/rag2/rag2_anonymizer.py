# rag2_anonymizer.py
"""
Minimal Presidio-based anonymizer for RAG preprocessing.

Removes (anonymizes) the following PII:
- PERSON names        -> [NAME]
- EMAIL_ADDRESS       -> [EMAIL]
- PHONE_NUMBER        -> [PHONE]

Usage:
    from rag2_anonymizer import anonymize_text
    clean = anonymize_text("Email me at jane.doe@example.com or call (555) 123-4567. - Jane")

Notes:
- Requires: presidio-analyzer, presidio-anonymizer, spacy, and an English model.
  Recommended:
      pip install presidio-analyzer presidio-anonymizer spacy spacy-transformers
      python -m spacy download en_core_web_trf
"""

from __future__ import annotations

from typing import Iterable, Optional, Dict
import logging

from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer.entities import OperatorConfig

import re

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# -----------------------------
# Build NLP + Presidio engines
# -----------------------------

def _register_relaxed_url_recognizer(analyzer: AnalyzerEngine) -> None:
    """
    Catch messy/broken URLs from PDFs:
      - spaced scheme: h t t p s : / /
      - truncated: https://di
      - www. and bare domains with TLD
      - DOI identifiers
    """
    patterns = [
        # e.g., "https://di", "http : // example . com/path" (allow spaces around scheme)
        Pattern(
            name="SCHEME_RELAXED",
            regex=r"(?i)h\s*t\s*t\s*p\s*s?\s*:\s*/\s*/[^\s)\]}>]+",
            score=0.85,
        ),
        # www. domain (+ optional path)
        Pattern(
            name="WWW_DOMAIN",
            regex=r"\bwww\.[A-Za-z0-9\-_.]+\.[A-Za-z]{2,}(?:/[^\s)\]}>]*)?",
            score=0.80,
        ),
        # bare domain with TLD (+ optional path)
        Pattern(
            name="BARE_DOMAIN",
            regex=r"\b(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,}(?:/[^\s)\]}>]*)?",
            score=0.70,
        ),
        # DOI-style identifiers (often pasted as links); case-insensitive via (?i)
        Pattern(
            name="DOI_ID",
            regex=r"(?i)\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+",
            score=0.75,
        ),
    ]

    recognizer = PatternRecognizer(
        supported_entity="URL",
        patterns=patterns,
        supported_language="en",
    )
    analyzer.registry.add_recognizer(recognizer)

def _register_initials_person_recognizer(analyzer: AnalyzerEngine) -> None:
    """
    Catch author-style initials and 'Lastname, I.' cases that spaCy misses.
    Examples: 'T., &', 'P. B.,', 'M., A.,', 'Zhao, P.', 'FUKOSHI, T.'
    """
    patterns = [
        # 1–3 initials like 'P.' or 'P. B.' optionally followed by comma/ampersand
        Pattern(
            name="INITIALS_SEQUENCE",
            regex=r"\b(?:[A-Z]\.\s?){1,3}(?=(?:\s*[,&])|\s*$)",
            score=0.68,
        ),
        # 'Lastname, I.' with normal casing
        Pattern(
            name="LASTNAME_COMMA_INITIALS",
            regex=r"\b[A-Z][A-Za-z'’-]{2,}\s*,\s*(?:[A-Z]\.\s?){1,3}",
            score=0.72,
        ),
        # ALL-CAPS surname version: 'FUKOSHI, T.' or 'MING, Z.'
        Pattern(
            name="ALLCAPS_SURNAME_COMMA_INITIALS",
            regex=r"\b[A-Z]{2,}\s*,\s*(?:[A-Z]\.\s?){1,3}",
            score=0.72,
        ),
    ]
    recognizer = PatternRecognizer(
        supported_entity="PERSON",
        patterns=patterns,
        context=["&", "and"],  # tiny boost around author lists
        supported_language="en",
    )
    analyzer.registry.add_recognizer(recognizer)


def _build_analyzer(preferred_model: str = "en_core_web_trf") -> AnalyzerEngine:
    """
    Create an AnalyzerEngine with spaCy NLP.
    Tries transformer English model first, then falls back to small.
    """
    nlp_conf = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": preferred_model},
        ],
    }

    try:
        provider = NlpEngineProvider(nlp_configuration=nlp_conf)
        nlp_engine = provider.create_engine()
    except Exception as e:
        logger.warning(f"Could not load '{preferred_model}' ({e}). Falling back to en_core_web_sm.")
        fallback_conf = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
        }
        provider = NlpEngineProvider(nlp_configuration=fallback_conf)
        nlp_engine = provider.create_engine()

    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    _register_initials_person_recognizer(analyzer)
    _register_relaxed_url_recognizer(analyzer)
    
    return analyzer


# Cache engines at import so they're loaded once
_ANALYZER = _build_analyzer()
_ANONYMIZER = AnonymizerEngine()

# Default operators for targeted entities
_DEFAULT_OPERATORS: Dict[str, OperatorConfig] = {
    "PERSON":        OperatorConfig("redact"),
    "EMAIL_ADDRESS": OperatorConfig("redact"),
    "PHONE_NUMBER":  OperatorConfig("redact"),
    "URL":           OperatorConfig("redact"),
}


# -----------------------------
# Public API
# -----------------------------

def anonymize_text(
    text: str,
    entities: Optional[Iterable[str]] = None,
    operators: Optional[Dict[str, OperatorConfig]] = None,
    language: str = "en",
) -> str:
    """
    Anonymize PII in `text` using Microsoft Presidio.
    """
    if not text:
        return text

    target_entities = list(entities) if entities else list(_DEFAULT_OPERATORS.keys())
    ops = operators if operators is not None else _DEFAULT_OPERATORS

    results = _ANALYZER.analyze(text=text, entities=target_entities, language=language)
    anonymized = _ANONYMIZER.anonymize(text=text, analyzer_results=results, operators=ops)
    return anonymized.text


# -----------------------------
# Quick manual test
# -----------------------------
if __name__ == "__main__":
    sample = (
        "Contact Jane Doe at jane.doe@example.com or +1 (415) 555-2671. "
        "Thanks, Jane. https://example.com"
    )
    print("[INPUT ]", sample)
    print("[OUTPUT]", anonymize_text(sample))