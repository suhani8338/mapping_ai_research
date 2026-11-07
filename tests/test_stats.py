import pytest

from src.analysis.statistical_tests import chi_square_2x2, fisher_exact_2x2
from src.validation.evaluation_metrics import binary_metrics
from src.validation.audit_sampler import stratified_sample


def test_binary_metrics():
    result = binary_metrics(
        ["female", "female", "male", "male"],
        ["female", "male", "female", "male"],
        positive_label="female",
    )
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["false_positive_rate"] == 0.5


def test_chi_square_for_equal_proportions():
    statistic, p_value = chi_square_2x2([[10, 10], [10, 10]])
    assert statistic == pytest.approx(0.0)
    assert p_value == pytest.approx(1.0)


def test_fisher_exact_known_table():
    odds_ratio, p_value = fisher_exact_2x2([[1, 9], [11, 3]])
    assert odds_ratio == pytest.approx(1 / 33)
    assert p_value == pytest.approx(0.002759456, rel=1e-5)


def test_stratified_sample_is_deterministic_and_keeps_each_group():
    records = [{"group": "a", "id": i} for i in range(10)]
    records += [{"group": "b", "id": i} for i in range(10)]
    sample = stratified_sample(
        records, stratum=lambda record: record["group"], sample_rate=0.1, seed=7
    )
    assert len(sample) == 2
    assert {record["group"] for record in sample} == {"a", "b"}


def test_invalid_contingency_table_is_rejected():
    with pytest.raises(ValueError):
        chi_square_2x2([[1, -1], [2, 3]])
