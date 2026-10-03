import pytest

from search_engine.suggester import SuggestionEngine


@pytest.fixture
def suggester():
    eng = SuggestionEngine()
    eng.build_from_documents(
        ["a", "b"], ["Machine learning basics for beginners", "Marine biology of coral reefs"]
    )
    yield eng
    eng.close()


def test_minimum_prefix_length(suggester):
    assert suggester.suggest("m") == []
    assert suggester.suggest("") == []
    assert suggester.suggest("  m ") == []


def test_prefix_returns_matches(suggester):
    titles = [s["title"] for s in suggester.suggest("ma")]
    assert any(t.startswith("Machine") for t in titles)


def test_every_word_must_match(suggester):
    titles = [s["title"] for s in suggester.suggest("mach lear")]
    assert titles == ["Machine learning basics for beginners"]


def test_zero_limit_returns_nothing(suggester):
    assert suggester.suggest("ma", limit=0) == []


def test_negative_limit_returns_nothing(suggester):
    assert suggester.suggest("ma", limit=-1) == []


def test_punctuation_in_prefix_acts_like_a_space(suggester):
    assert suggester.suggest("ma-ch") == suggester.suggest("ma ch")


def test_punctuation_prefix_still_matches_titles(suggester):
    titles = [s["title"] for s in suggester.suggest("ma-le")]
    assert titles == ["Machine learning basics for beginners"]
    assert suggester.suggest("ma-le") == suggester.suggest("ma le")


def test_prefix_is_case_insensitive(suggester):
    upper = suggester.suggest("MA")
    assert upper == suggester.suggest("ma")
    assert {s["title"] for s in upper} == {
        "Machine learning basics for beginners",
        "Marine biology of coral reefs",
    }