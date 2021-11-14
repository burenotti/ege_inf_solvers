import pytest
from main import solve

# Тест монотонных функций


def test_1():
    solution = solve(
        # (¬x ∧ y ∧ z) ∨ (¬x ∧ ¬y ∧ z) ∨ (¬x ∧ ¬y ∧ ¬z)
        function=lambda x, y, z: (not x and y and z) or (not x and not y and z) or (not x and not y and not z),
        model=[
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 1]
        ],
        expected_values=[True] * 3,
    )
    assert solution == {('z', 'x', 'y')}


def test_2():
    solution = solve(
        # (¬x ∧ z) ∨ (¬x ∧ ¬y ∧ ¬z)
        function=lambda x, y, z: (not x and z) or (not x and not y and not z),
        model=[
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1]
        ],
        expected_values=[True] * 3,
    )
    assert solution == {('x', 'y', 'z')}


# Тест немонотонных функций

def test_3():
    solution = solve(
        # (¬x ∧ z) ∨ (¬x ∧ ¬y ∧ ¬z)
        function=lambda x, y, z: (not z) and x or x and y,
        model=[
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        expected_values=[False, True, False, True, False, False, False, True],
    )
    assert solution == {('z', 'y', 'x')}


# Тест функций с пропущеннымми значениями

def test_4():
    _ = None
    solution = solve(
        # (x ∨ y) → (z ≡ x)
        function=lambda x, y, z: (x or y) <= (z == x),
        model=[
            [_, 0, 0],
            [_, 0, _],
        ],
        expected_values=[False, False],
    )
    assert solution == {('x', 'z', 'y')}


def test_5():
    _ = None
    solution = solve(
        # (x ≡ z ) ∨ (x → (y ∧ z))
        function=lambda x, y, z: (x == z) or (x <= (y and z)),
        model=[
            [0, 0, _],
            [1, _, _],
        ],
        expected_values=[False, False],
    )
    assert solution == {('y', 'z', 'x')}


def test_6():
    _ = None
    solution = solve(
        # ((x → y ) ∧ (y → w)) ∨ (z ≡ ( x ∨ y))
        function=lambda x, y, z, w: ((x <= y) and (y <= w)) or (z == (x or y)),
        model=[
            [1, _, _, 1],
            [1, _, _, _],
            [_, 1, _, 1]
        ],
        expected_values=[False] * 3,
    )
    assert solution == {('y', 'w', 'z', 'x')}
