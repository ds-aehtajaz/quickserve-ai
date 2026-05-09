from quickserve.nlp.ner import extract_entities


def test_order_id_regex_fallback():
    # Even without a trained model the regex fallback should extract ORDER_ID
    result = extract_entities("cancel order ORD123")
    assert result["ORDER_ID"] == "ORD123"


def test_order_id_lowercase():
    result = extract_entities("track ord456")
    assert result["ORDER_ID"] == "ORD456"


def test_no_order_id():
    result = extract_entities("I want to buy a pizza")
    assert result["ORDER_ID"] is None


def test_quantity_regex_fallback():
    result = extract_entities("I want 3 burgers")
    assert result["QUANTITY"] == 3


def test_entities_keys_present():
    result = extract_entities("hello")
    assert "ITEM" in result
    assert "QUANTITY" in result
    assert "ORDER_ID" in result
