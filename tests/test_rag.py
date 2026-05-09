import pytest
from quickserve.nlp.rag import retrieve


@pytest.mark.skipif(
    not __import__("os").path.exists("data/faq_kb.json"),
    reason="FAQ KB not found"
)
def test_retrieve_returns_results():
    results = retrieve("how long does delivery take", top_k=3)
    assert isinstance(results, list)
    assert len(results) <= 3


def test_retrieve_result_structure():
    results = retrieve("shipping cost", top_k=1)
    if results:
        assert "question" in results[0]
        assert "answer" in results[0]
        assert "score" in results[0]


def test_retrieve_top_k_respected():
    results = retrieve("payment", top_k=2)
    assert len(results) <= 2
