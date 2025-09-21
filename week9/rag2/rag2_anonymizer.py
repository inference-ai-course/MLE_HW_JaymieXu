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
      pip install presidio-analyzer presidio-anonymizer spacy
      python -m spacy download en_core_web_lg  # fallback to en_core_web_sm if lg unavailable
"""

from __future__ import annotations

from typing import Iterable, Optional, Dict
import logging

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer.entities import OperatorConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

# -----------------------------
# Build NLP + Presidio engines
# -----------------------------

def _build_analyzer(preferred_model: str = "en_core_web_lg") -> AnalyzerEngine:
    """
    Create an AnalyzerEngine with spaCy NLP.
    Tries large English model first, then falls back to small.
    """
    nlp_conf = {
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": preferred_model},
            {"lang_code": "en", "model_name": "xx_ent_wiki_sm"},
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

    return AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])


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
    operators: Optional[Dict[str, dict]] = None,
    language: str = "en",
) -> str:
    """
    Anonymize PII in `text` using Microsoft Presidio.

    Parameters
    ----------
    text : str
        Input text to anonymize.
    entities : Iterable[str] | None
        Which entity types to target. Defaults to PERSON, EMAIL_ADDRESS, PHONE_NUMBER.
        Examples of other entity types Presidio can detect: US_SSN, CREDIT_CARD, IBAN, etc.
    operators : Dict[str, dict] | None
        Mapping of entity type -> anonymizer operator config.
        If None, uses _DEFAULT_OPERATORS (replace with tags).
        See: https://microsoft.github.io/presidio/anonymizer/docs/anonymizers/
    language : str
        Language code for the analyzer (default "en").

    Returns
    -------
    str
        The anonymized text.
    """
    if not text:
        return text

    target_entities = list(entities) if entities else list(_DEFAULT_OPERATORS.keys())
    ops = operators if operators is not None else _DEFAULT_OPERATORS

    # Analyze to find PII spans
    results = _ANALYZER.analyze(text=text, entities=target_entities, language=language)

    # Apply anonymization
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