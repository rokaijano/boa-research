from __future__ import annotations

import math

from ..schema import OPERATION_TYPES, PATCH_CATEGORIES, PatchDescriptor, TrialSummary


def successful_trials(memory: list[TrialSummary]) -> list[TrialSummary]:
    return [trial for trial in memory if trial.canonical_score is not None]


def descriptor_trials(memory: list[TrialSummary]) -> list[TrialSummary]:
    return [trial for trial in successful_trials(memory) if trial.descriptor is not None]


def one_hot(value: str, choices: list[str]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def numeric_signature(knobs: dict[str, float]) -> list[float]:
    if not knobs:
        return [0.0, 0.0, 0.0]
    values = [float(value) for value in knobs.values()]
    magnitudes = [math.log10(abs(value) + 1.0) for value in values]
    return [
        float(len(values)),
        sum(magnitudes) / len(magnitudes),
        max(magnitudes),
    ]


def descriptor_features(descriptor: PatchDescriptor) -> list[float]:
    features: list[float] = [1.0]
    features.extend(one_hot(descriptor.patch_category, sorted(PATCH_CATEGORIES)))
    features.extend(one_hot(descriptor.operation_type, sorted(OPERATION_TYPES)))
    features.extend(
        [
            float(descriptor.estimated_risk),
            math.log1p(len(descriptor.touched_files)),
            math.log1p(len(descriptor.touched_symbols)),
            1.0 if descriptor.parent_trial_id else 0.0,
        ]
    )
    features.extend(numeric_signature(descriptor.numeric_knobs))
    return features


def dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def gaussian_similarity(left: PatchDescriptor, right: PatchDescriptor) -> float:
    left_features = descriptor_features(left)
    right_features = descriptor_features(right)
    squared_distance = sum((a - b) ** 2 for a, b in zip(left_features, right_features))
    file_overlap = len(set(left.touched_files) & set(right.touched_files))
    symbol_overlap = len(set(left.touched_symbols) & set(right.touched_symbols))
    locality_bonus = 0.05 * min(file_overlap + symbol_overlap, 4)
    return math.exp(-0.5 * squared_distance) + locality_bonus


def solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(matrix)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for pivot in range(size):
        best_row = max(range(pivot, size), key=lambda index: abs(augmented[index][pivot]))
        augmented[pivot], augmented[best_row] = augmented[best_row], augmented[pivot]
        pivot_value = augmented[pivot][pivot]
        if abs(pivot_value) < 1e-9:
            raise ValueError("Singular matrix")
        inverse = 1.0 / pivot_value
        for column in range(pivot, size + 1):
            augmented[pivot][column] *= inverse
        for row_index in range(size):
            if row_index == pivot:
                continue
            factor = augmented[row_index][pivot]
            if factor == 0.0:
                continue
            for column in range(pivot, size + 1):
                augmented[row_index][column] -= factor * augmented[pivot][column]
    return [augmented[index][size] for index in range(size)]


def gp_posterior(
    descriptors: list[PatchDescriptor],
    scores: list[float],
    candidate: PatchDescriptor,
    *,
    noise: float,
) -> tuple[float, float]:
    size = len(descriptors)
    kernel = [[0.0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for column in range(size):
            kernel[row][column] = gaussian_similarity(descriptors[row], descriptors[column])
        kernel[row][row] += noise
    k_star = [gaussian_similarity(observed, candidate) for observed in descriptors]
    alpha = solve_linear_system(kernel, scores)
    beta = solve_linear_system(kernel, k_star)
    mean = dot(k_star, alpha)
    variance = max(0.0, gaussian_similarity(candidate, candidate) + noise - dot(k_star, beta))
    return mean, math.sqrt(variance)
