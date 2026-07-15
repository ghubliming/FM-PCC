import numpy as np


PILLAR_CENTERS = np.array(
    [
        [0.5, -0.1],
        [0.425, 0.08],
        [0.575, 0.08],
        [0.35, 0.26],
        [0.5, 0.26],
        [0.65, 0.26],
    ],
    dtype=float,
)

PILLAR_RADII = np.array([0.03, 0.025, 0.025, 0.025, 0.025, 0.025], dtype=float)

NOVEL_OBSTACLE_CENTER = np.array([0.5, -0.1], dtype=float)
NOVEL_OBSTACLE_RADIUS = 0.08

NOVEL_BOUNDARY_SEGMENTS = (
    (np.array([0.8, -0.3], dtype=float), np.array([0.575, 0.5], dtype=float), "below"),
    (np.array([0.2, -0.3], dtype=float), np.array([0.425, 0.5], dtype=float), "below"),
)

NOVEL_BOUNDARY_LINES = tuple(
    (
        float((p2[1] - p1[1]) / (p2[0] - p1[0])),
        float(p2[1] - ((p2[1] - p1[1]) / (p2[0] - p1[0])) * p2[0]),
        direction,
    )
    for p1, p2, direction in NOVEL_BOUNDARY_SEGMENTS
)

NOVEL_CONSTRAINTS = frozenset({"novel", "novel_hard"})

# Clockwise ordering makes the signed edge distance positive outside the polygon.
NOVEL_HARD_QUADRILATERAL_VERTICES = np.array(
    [
        [0.5, -0.25],
        [0.425, 0.08],
        [0.5, 0.35],
        [0.575, 0.08],
    ],
    dtype=float,
)

NOVEL_HARD_QUADRILATERAL_EDGES = tuple(
    (
        vertex,
        NOVEL_HARD_QUADRILATERAL_VERTICES[
            (idx + 1) % len(NOVEL_HARD_QUADRILATERAL_VERTICES)
        ],
    )
    for idx, vertex in enumerate(NOVEL_HARD_QUADRILATERAL_VERTICES)
)


def is_novel_constraint(constraint):
    return constraint in NOVEL_CONSTRAINTS


def uses_hard_quadrilateral(constraint):
    return constraint == "novel_hard"


def polygon_signed_distances_numpy(x, y, vertices=NOVEL_HARD_QUADRILATERAL_VERTICES):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    signed_distances = []
    if vertices is NOVEL_HARD_QUADRILATERAL_VERTICES:
        edge_pairs = NOVEL_HARD_QUADRILATERAL_EDGES
    else:
        edge_pairs = tuple(
            (vertex, vertices[(idx + 1) % len(vertices)])
            for idx, vertex in enumerate(vertices)
        )

    for vertex, next_vertex in edge_pairs:
        edge = next_vertex - vertex
        edge_norm = np.linalg.norm(edge)
        signed_distance = (
            edge[0] * (y_arr - vertex[1]) - edge[1] * (x_arr - vertex[0])
        ) / edge_norm
        signed_distances.append(signed_distance)

    return np.stack(signed_distances, axis=0)


def quadrilateral_constraint_value_numpy(
    x, y, vertices=NOVEL_HARD_QUADRILATERAL_VERTICES, margin=0.0
):
    signed_distances = polygon_signed_distances_numpy(x, y, vertices=vertices)
    return np.max(signed_distances, axis=0) - margin


def expand_convex_polygon(vertices, margin):
    if margin <= 0:
        return np.array(vertices, dtype=float)

    vertices = np.asarray(vertices, dtype=float)
    normals = []
    offsets = []

    for idx, vertex in enumerate(vertices):
        next_vertex = vertices[(idx + 1) % len(vertices)]
        edge = next_vertex - vertex
        edge_norm = np.linalg.norm(edge)
        normal = np.array([-edge[1], edge[0]], dtype=float) / edge_norm
        normals.append(normal)
        offsets.append(np.dot(normal, vertex) + margin)

    expanded_vertices = []
    for idx in range(len(vertices)):
        prev_normal = normals[idx - 1]
        curr_normal = normals[idx]
        system = np.vstack([prev_normal, curr_normal])
        rhs = np.array([offsets[idx - 1], offsets[idx]], dtype=float)
        expanded_vertices.append(np.linalg.solve(system, rhs))

    return np.array(expanded_vertices, dtype=float)
