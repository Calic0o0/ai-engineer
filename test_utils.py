

import pytest

from utils import calculate_total


@pytest.mark.parametrize(
    "price, quantity, expected",
    [
        (12.5, 4, 50),
        (10, 3, 30),
        (0, 5, 0),
        (12.5, 0, 0),
        (0, 0, 0),
    ],
)
def test_calculate_total(price, quantity, expected):
    assert calculate_total(price, quantity) == expected


def test_negative_price():
    with pytest.raises(ValueError):
        calculate_total(-1, 3)


def test_negative_quantity():
    with pytest.raises(ValueError):
        calculate_total(10, -2)


def test_negative_price():
    with pytest.raises(ValueError):
        calculate_total(-1, 3)


def test_negative_quantity():
    with pytest.raises(ValueError):
        calculate_total(10, -2)


@pytest.mark.parametrize(
    "price, quantity",
    [
        ("12", 3),
        (10, "3"),
        (True, 2),
        (10, False),
    ],
)
def test_invalid_types(price, quantity):
    with pytest.raises(TypeError):
        calculate_total(price, quantity)


@pytest.mark.parametrize(
    "price, quantity",
    [
        (float("inf"), 2),
        (10, float("nan")),
    ],
)
def test_non_finite_numbers(price, quantity):
    with pytest.raises(ValueError):
        calculate_total(price, quantity)
