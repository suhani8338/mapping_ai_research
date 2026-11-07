from src.inference.name_parser import parse_name


def test_parses_hyphenated_given_name_and_family_name():
    parsed = parse_name("Dr. Mary-Jane Smith")
    assert parsed.given_name == "Mary-Jane"
    assert parsed.family_name == "Smith"


def test_initial_only_is_not_used_as_given_name():
    parsed = parse_name("J. Smith")
    assert parsed.given_name is None
    assert parsed.family_name == "Smith"


def test_mononym_has_no_family_name():
    parsed = parse_name("Cher")
    assert parsed.given_name is None
    assert parsed.family_name is None


def test_empty_name_returns_empty_components():
    parsed = parse_name("  ")
    assert parsed.given_name is None
    assert parsed.family_name is None
