from quickserve.nlp.preprocess import clean_text, extract_order_ref


def test_clean_text_lowercases():
    assert clean_text("Hello WORLD") == "hello world"


def test_clean_text_removes_punctuation():
    assert clean_text("order now!") == "order now"


def test_clean_text_collapses_whitespace():
    assert clean_text("  too   many   spaces  ") == "too many spaces"


def test_clean_text_strips_accents():
    result = clean_text("café")
    assert result == "cafe"


def test_extract_order_ref_found():
    assert extract_order_ref("cancel order ORD123") == "ORD123"


def test_extract_order_ref_lowercase():
    assert extract_order_ref("track ord456 please") == "ORD456"


def test_extract_order_ref_not_found():
    assert extract_order_ref("where is my package") is None


def test_extract_order_ref_multiple_picks_first():
    result = extract_order_ref("ORD100 and ORD200")
    assert result == "ORD100"
