from src.inference.pronoun_extractor import extract_pronoun_label

PRONOUNS = {
    "female": ["she", "her", "hers"],
    "male": ["he", "him", "his"],
    "non_binary": ["they", "them", "their"],
}


def test_finds_pronouns_without_matching_substrings():
    assert extract_pronoun_label("She is a researcher.", PRONOUNS) == ("female", 1)


def test_ties_and_no_evidence_are_unknown():
    assert extract_pronoun_label("She thanked him.", PRONOUNS) == ("unknown", 1)
    assert extract_pronoun_label("A researcher.", PRONOUNS) == ("unknown", 0)
