"""Generate a synthetic Wikipedia-style corpus, save it as JSONL, and index it.

Uses only the standard library plus the local ``search_engine`` package, so no
network access or external datasets are required. Output is deterministic for
a given seed.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NamedTuple



BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCUMENTS_PATH = DATA_DIR / "documents.jsonl"
INDEX_PATH = DATA_DIR / "index.json"

DEFAULT_NUM_DOCS = 2000
DEFAULT_SEED = 42

# --------------------------------------------------------------------------- #
# Vocabulary pools
# --------------------------------------------------------------------------- #
FIRST_NAMES: tuple[str, ...] = (
    "Anna", "Marcus", "Elena", "Tobias", "Priya", "Liam", "Sofia", "Kenji",
    "Amara", "Viktor", "Clara", "Rafael", "Ingrid", "Samir", "Helena", "Oscar",
)
LAST_NAMES: tuple[str, ...] = (
    "Hartmann", "Okafor", "Lindqvist", "Moreau", "Tanaka", "Castillo",
    "Novak", "Fitzgerald", "Banerjee", "Kowalski", "Almeida", "Sorensen",
    "Whitfield", "Duarte", "Petrov", "Nakamura",
)
PLACE_PREFIXES: tuple[str, ...] = (
    "Al", "Bren", "Cor", "Dal", "Eld", "Far", "Gal", "Hal", "Isk", "Jor",
    "Kel", "Lor", "Mar", "Nor", "Ors", "Pel", "Quin", "Ros", "Sel", "Tor",
    "Ul", "Val", "Wen", "Yar", "Zan",
)
PLACE_SUFFIXES: tuple[str, ...] = (
    "dor", "mere", "vik", "ton", "gard", "bury", "land", "stein", "dale",
    "wick", "ford", "haven", "mont", "port", "wyn",
)
ORG_KINDS: tuple[str, ...] = ("Institute", "Systems Lab", "Research Group", "University")

DECADES: tuple[str, ...] = ("1920", "1950", "1970", "1990", "2010")
CENTURIES: tuple[str, ...] = ("eighteenth", "nineteenth", "twentieth")
INSTRUMENTS: tuple[str, ...] = (
    "spectroscopy", "satellite imaging", "electron microscopy",
    "computer simulation", "particle detection", "deep-sea sensing",
)
APPLICATIONS: tuple[str, ...] = (
    "healthcare", "finance", "logistics", "education", "telecommunications",
    "manufacturing", "agriculture", "environmental policy",
)
FIELDS: tuple[str, ...] = (
    "medicine", "engineering", "chemistry", "ecology", "economics", "astronomy",
)
LANGUAGES: tuple[str, ...] = ("Python", "Rust", "Java", "Go", "C++")
FACTORS: tuple[str, ...] = (
    "data quality", "hardware cost", "network latency", "memory usage",
    "security requirements",
)
CLIMATES: tuple[str, ...] = ("temperate", "arid", "humid", "subpolar", "tropical")
INDUSTRIES: tuple[str, ...] = ("fishing", "agriculture", "mining", "tourism", "trade")
FEATURES: tuple[str, ...] = ("wildlife", "climate", "geology", "scenery")
INTERESTS: tuple[str, ...] = ("hiking", "birdwatching", "local history", "photography")

SCIENCE_SUBJECTS: tuple[tuple[str, str], ...] = (
    ("photosynthesis", "biology"),
    ("plate tectonics", "geology"),
    ("black hole", "astrophysics"),
    ("quantum entanglement", "physics"),
    ("natural selection", "evolutionary biology"),
    ("dark matter", "cosmology"),
    ("cell division", "cell biology"),
    ("nuclear fusion", "nuclear physics"),
    ("genetic mutation", "genetics"),
    ("ocean acidification", "marine science"),
    ("superconductivity", "condensed matter physics"),
    ("the water cycle", "hydrology"),
    ("protein folding", "biochemistry"),
    ("gravitational waves", "astrophysics"),
    ("antibiotic resistance", "microbiology"),
    ("the greenhouse effect", "climate science"),
)
TECH_SUBJECTS: tuple[tuple[str, str], ...] = (
    ("machine learning", "computer science"),
    ("neural network", "artificial intelligence"),
    ("public key cryptography", "computer security"),
    ("relational database", "data management"),
    ("cloud computing", "information technology"),
    ("operating system", "computer science"),
    ("search engine", "information retrieval"),
    ("computer vision", "artificial intelligence"),
    ("natural language processing", "artificial intelligence"),
    ("distributed system", "computer science"),
    ("quantum computing", "computer engineering"),
    ("version control", "software engineering"),
)
HISTORY_PATTERNS: tuple[str, ...] = (
    "The Battle of {place}",
    "The Treaty of {place}",
    "The {place} Rebellion",
    "The {place} Dynasty",
    "The Siege of {place}",
    "The {place} Expedition",
)
GEOGRAPHY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("River {place}", "river"),
    ("Mount {place}", "mountain"),
    ("Lake {place}", "lake"),
    ("{place} Archipelago", "archipelago"),
    ("{place} Desert", "desert"),
    ("{place}", "city"),
)


class Templates(NamedTuple):
    """Sentence templates for one document category."""

    intro: tuple[str, ...]
    details: tuple[str, ...]
    closing: tuple[str, ...]


SCIENCE_TEMPLATES = Templates(
    intro=(
        "Research on {subject} is a major focus of {field}.",
        "The study of {subject} has long been central to {field}.",
        "In {field}, {subject} is examined through both theory and experiment.",
        "Scholars of {field} have investigated {subject} since the {century} century.",
    ),
    details=(
        "Early work by {person} established the basic framework for understanding {subject}.",
        "In the {decade}s, {person} published influential measurements related to {subject}.",
        "Advances in {instrument} have allowed researchers to observe {subject} with greater precision.",
        "Models of {subject} often combine mathematical analysis with large observational datasets.",
        "Debate continues over the mechanisms underlying {subject}.",
        "Laboratories in {place} have contributed significant data on {subject}.",
    ),
    closing=(
        "Findings about {subject} have influenced work in {related}.",
        "Understanding {subject} remains important for {application}.",
        "Today the topic is taught in universities and appears in introductory courses on {field}.",
    ),
)

TECH_TEMPLATES = Templates(
    intro=(
        "Research on {subject} is a major focus of {field}.",
        "{Subject} refers to a family of techniques and systems within {field}.",
        "The development of {subject} has transformed {field} over recent decades.",
        "In {field}, {subject} is studied alongside related methods and tools.",
    ),
    details=(
        "Early prototypes were built at the {org} during the {decade}s.",
        "Open-source implementations are commonly written in {language} and maintained by volunteer communities.",
        "Engineers at the {org} popularized {subject} through large-scale deployments.",
        "Performance depends heavily on {factor}.",
        "Common challenges include scalability, reliability, and {factor}.",
        "Tutorials by {person} helped introduce {subject} to a wider audience.",
    ),
    closing=(
        "Applications of {subject} now span {application} and {related}.",
        "Standards for {subject} are developed by international working groups.",
        "Many organizations in {application} now rely on {subject}.",
    ),
)

HISTORY_TEMPLATES = Templates(
    intro=(
        "{title} is a significant subject in the history of the {place2} region.",
        "{title} dates to the {century} century and is associated with {person} and {person2}.",
        "{title} was a pivotal episode in the history of the {place2} region.",
    ),
    details=(
        "Contemporary accounts describe the involvement of {person} and {person2}.",
        "The events reshaped trade routes and political alliances across the {place3} region.",
        "Historians such as {person} have debated its causes, pointing to famine, taxation, and shifting alliances.",
        "Surviving records are preserved in the archives of {place}.",
        "Archaeological surveys in the {decade}s uncovered artifacts linked to the period.",
    ),
    closing=(
        "Its legacy is commemorated each year in {place}.",
        "Later chroniclers regarded it as a turning point for the {place2} region.",
        "Modern scholarship continues to revise the traditional narrative.",
    ),
)

GEOGRAPHY_TEMPLATES = Templates(
    intro=(
        "{title} is a notable {kind} located in the {place2} region.",
        "{title} is a well-known {kind} in the {place2} region.",
    ),
    details=(
        "The {kind} is known for its distinctive {feature}.",
        "It lies roughly {distance} kilometres from the nearest coast.",
        "The surrounding area has a {climate} climate, and local communities rely on {industry}.",
        "Researchers from {place3} have surveyed the area since the {decade}s.",
        "Seasonal changes strongly affect the {feature} of the area.",
    ),
    closing=(
        "It is a popular destination for visitors interested in {interest}.",
        "Conservation efforts in the {place2} region aim to protect its {feature}.",
    ),
)


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def _make_place(rng: random.Random) -> str:
    """Create a fictional place name.

    Args:
        rng: Seeded random generator.

    Returns:
        A name such as ``"Brenmere"``.
    """
    return rng.choice(PLACE_PREFIXES) + rng.choice(PLACE_SUFFIXES)


def _make_person(rng: random.Random) -> str:
    """Create a fictional person name.

    Args:
        rng: Seeded random generator.

    Returns:
        A name such as ``"Elena Novak"``.
    """
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def _base_context(rng: random.Random) -> dict[str, str]:
    """Build the placeholder values shared by all templates.

    Args:
        rng: Seeded random generator.

    Returns:
        A mapping of placeholder name to value.
    """
    place = _make_place(rng)
    return {
        "person": _make_person(rng),
        "person2": _make_person(rng),
        "place": place,
        "place2": _make_place(rng),
        "place3": _make_place(rng),
        "decade": rng.choice(DECADES),
        "century": rng.choice(CENTURIES),
        "instrument": rng.choice(INSTRUMENTS),
        "application": rng.choice(APPLICATIONS),
        "related": rng.choice(FIELDS),
        "language": rng.choice(LANGUAGES),
        "factor": rng.choice(FACTORS),
        "org": f"{place} {rng.choice(ORG_KINDS)}",
        "distance": str(rng.randint(12, 900)),
        "climate": rng.choice(CLIMATES),
        "industry": rng.choice(INDUSTRIES),
        "feature": rng.choice(FEATURES),
        "interest": rng.choice(INTERESTS),
    }


def _capitalize(text: str) -> str:
    """Uppercase only the first character.

    Args:
        text: Input string.

    Returns:
        ``text`` with its first character uppercased.
    """
    return text[:1].upper() + text[1:]


def _compose(
    rng: random.Random, templates: Templates, context: dict[str, str]
) -> str:
    """Assemble a four-sentence abstract from templates.

    Args:
        rng: Seeded random generator.
        templates: Sentence pools for the category.
        context: Placeholder values.

    Returns:
        The abstract text.
    """
    sentences = [
        rng.choice(templates.intro),
        *rng.sample(templates.details, 2),
        rng.choice(templates.closing),
    ]
    return " ".join(sentence.format(**context) for sentence in sentences)


def _slugify(text: str) -> str:
    """Convert text to a lowercase URL-style slug.

    Args:
        text: Input string.

    Returns:
        The slug.
    """
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _make_document(rng: random.Random, index: int) -> dict[str, str]:
    """Generate one synthetic document.

    Args:
        rng: Seeded random generator.
        index: Zero-based document number; selects the category cyclically.

    Returns:
        A ``{"doc_id": ..., "text": ...}`` mapping.
    """
    context = _base_context(rng)
    category = ("science", "tech", "history", "geography")[index % 4]

    if category in {"science", "tech"}:
        subjects = SCIENCE_SUBJECTS if category == "science" else TECH_SUBJECTS
        templates = SCIENCE_TEMPLATES if category == "science" else TECH_TEMPLATES
        subject, field = rng.choice(subjects)
        title = _capitalize(subject)
        context.update(
            subject=subject, Subject=_capitalize(subject), field=field, title=title
        )
    elif category == "history":
        templates = HISTORY_TEMPLATES
        title = rng.choice(HISTORY_PATTERNS).format(place=context["place"])
        context.update(title=title)
    else:
        templates = GEOGRAPHY_TEMPLATES
        pattern, kind = rng.choice(GEOGRAPHY_PATTERNS)
        title = pattern.format(place=context["place"])
        context.update(title=title, kind=kind)

    return {
        "doc_id": f"{category}-{_slugify(title)}-{index + 1:04d}",
        "text": _compose(rng, templates, context),
    }


def generate_documents(num_docs: int, seed: int) -> Iterator[dict[str, str]]:
    """Yield synthetic Wikipedia-style abstracts.

    Args:
        num_docs: Number of documents to generate.
        seed: Seed for reproducible output.

    Yields:
        Document mappings with ``doc_id`` and ``text`` keys.
    """
    rng = random.Random(seed)
    for index in range(num_docs):
        yield _make_document(rng, index)


# --------------------------------------------------------------------------- #
# JSONL I/O
# --------------------------------------------------------------------------- #
def write_jsonl(path: Path, documents: Iterator[dict[str, str]]) -> int:
    """Write documents to a JSONL file, one JSON object per line.

    Args:
        path: Destination file (parent directories are created).
        documents: Documents to write.

    Returns:
        The number of documents written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[dict[str, str]]:
    """Read and validate documents from a JSONL file.

    Args:
        path: Source file.

    Yields:
        Document mappings with ``doc_id`` and ``text`` keys.

    Raises:
        ValueError: If a line is not valid JSON or lacks string fields.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("doc_id"), str)
                or not isinstance(record.get("text"), str)
            ):
                raise ValueError(
                    f"{path}:{line_number}: expected string 'doc_id' and 'text'"
                )
            yield {"doc_id": record["doc_id"], "text": record["text"]}


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        The parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-docs", type=int, default=DEFAULT_NUM_DOCS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main() -> None:
    """Generate the corpus JSONL. Indexing is done by the ingest_* scripts."""
    args = parse_args()

    started = time.perf_counter()
    written = write_jsonl(DOCUMENTS_PATH, generate_documents(args.num_docs, args.seed))
    elapsed = time.perf_counter() - started
    print(f"Wrote {written} documents to {DOCUMENTS_PATH} in {elapsed:.2f}s")
    print("Next: run `python ingest_tantivy.py` then `python ingest_vectors.py`.")

if __name__ == "__main__":
    main()