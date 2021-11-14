from copy import deepcopy
from itertools import product, permutations
from typing import Callable, Union

ModelTemplate = list[list[int | None]]
Model = Union[list[list[int]], tuple[list[int]]]


def rotate(matrix: Model):
    """
    Swaps rows and columns of given matrix
    rotate([
        [1, 2, 3],
        [4, 5, 6]
    ]) == [
        [1, 4],
        [2, 5],
        [3, 6]
    ]
    :param matrix:
    :return: new matrix
    """
    return [
        [matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))
    ]


def count_none(model: ModelTemplate) -> int:
    """
    Returns quantity of None values in given model.
    :param model:
    :return: quantity of None values
    """
    summa = 0
    for row in model:
        summa += sum(1 for value in row if value is None)

    return summa


def fill_model(
        model: ModelTemplate,
        filler: tuple[int, ...],
) -> Model:
    """
    Replaces None in model with given filler list
    fill_model([[1, None, 3], [None, 5, None ]], [2, 4, 6]) == [[1, 2, 3], [4, 5, 6]]
    :param model:
    :param filler:
    :return:
    """

    if len(model) == 0:
        raise ValueError("Model must be not empty")

    try:
        filled_model = deepcopy(model)
        filler = iter(filler)
        for i in range(len(model)):
            for j in range(len(model[0])):
                if model[i][j] is None:
                    filled_model[i][j] = next(filler)

        return filled_model
    except StopIteration:
        raise ValueError(
            "Filler list doesn't fully cover skipped values"
        )


def generate_model_perms(
        model: Model,
        var_names: tuple[str, ...],
) -> tuple[tuple[str], Model]:

    var_perms = permutations(var_names)
    perms_by_columns = permutations(rotate(model))

    for var_set, model in zip(var_perms, perms_by_columns):
        yield var_set, rotate(model)


def check_model(
        model: Model,
        expected_values: list[int],
        checker: Callable[[int, ...], bool],
        only_unique_rows: bool = True,
) -> bool:
    rows_unique = all(map(lambda c: model.count(c) == 1, model))
    model_correct = all([checker(*row) == expected_values[index] for index, row in enumerate(model)])

    return model_correct and (not only_unique_rows or rows_unique)


def solve(
        variables: tuple[str, ...],
        function: Callable[[int, ...], bool],
        model: ModelTemplate,
        expected_values: list[int],
):
    result = set()

    none_count = count_none(model)
    none_fillers = product((0, 1), repeat=none_count)

    for filler in none_fillers:
        filled_model = fill_model(model, filler)
        perms = generate_model_perms(filled_model, variables)

        for var_names, model_variant in perms:
            print(*model_variant, sep='\n', end='\n\n')
            if check_model(model_variant, expected_values, function):
                print(*model_variant, sep='\n', end='\n\n')
                result.add(var_names)
    return result


if __name__ == '__main__':
    _ = None

    solution = solve(
        variables=('x', 'y', 'z'),
        function=lambda x, y, z: (not x and y and z) or (not x and not y and z) or (not x and not y and not z),
        model=[
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
        ],
        expected_values=[True] * 3
    )
    print("Найдены следующие решения:", *map(' '.join, solution), sep='\n', end='\n\n')
    print("Итого:", len(solution))
