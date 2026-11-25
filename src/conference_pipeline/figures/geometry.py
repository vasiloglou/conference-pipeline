"""Geometric primitives and deterministic placement helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Box:
    """An axis-aligned rectangle in canvas units."""

    x: float
    y: float
    width: float
    height: float

    @property
    def x1(self) -> float:
        return self.x + self.width

    @property
    def y1(self) -> float:
        return self.y + self.height

    @property
    def cx(self) -> float:
        return self.x + self.width / 2.0

    @property
    def cy(self) -> float:
        return self.y + self.height / 2.0

    def inset(self, dx: float, dy: float | None = None) -> "Box":
        dy = dx if dy is None else dy
        return Box(self.x + dx, self.y + dy, max(0.0, self.width - 2 * dx), max(0.0, self.height - 2 * dy))

    def expand(self, amount: float) -> "Box":
        return Box(self.x - amount, self.y - amount, self.width + 2 * amount, self.height + 2 * amount)

    def moved(self, dx: float, dy: float) -> "Box":
        return Box(self.x + dx, self.y + dy, self.width, self.height)

    def contains(self, other: "Box") -> bool:
        return (
            other.x >= self.x - 1e-6
            and other.y >= self.y - 1e-6
            and other.x1 <= self.x1 + 1e-6
            and other.y1 <= self.y1 + 1e-6
        )

    def contains_point(self, x: float, y: float) -> bool:
        """Is (x, y) inside this box? Used to keep a route on the canvas."""

        return (self.x - 1e-6 <= x <= self.x1 + 1e-6
                and self.y - 1e-6 <= y <= self.y1 + 1e-6)

    def overlaps(self, other: "Box", *, tolerance: float = 0.0) -> bool:
        return not (
            other.x >= self.x1 - tolerance
            or other.x1 <= self.x + tolerance
            or other.y >= self.y1 - tolerance
            or other.y1 <= self.y + tolerance
        )

    def overlap_area(self, other: "Box") -> float:
        dx = min(self.x1, other.x1) - max(self.x, other.x)
        dy = min(self.y1, other.y1) - max(self.y, other.y)
        return dx * dy if dx > 0 and dy > 0 else 0.0

    def split_rows(self, count: int, gap: float = 0.0) -> list["Box"]:
        if count <= 0:
            return []
        height = (self.height - gap * (count - 1)) / count
        return [Box(self.x, self.y + i * (height + gap), self.width, height) for i in range(count)]

    def split_cols(self, count: int, gap: float = 0.0) -> list["Box"]:
        if count <= 0:
            return []
        width = (self.width - gap * (count - 1)) / count
        return [Box(self.x + i * (width + gap), self.y, width, self.height) for i in range(count)]

    def split_cols_weighted(self, weights: list[float], gap: float = 0.0) -> list["Box"]:
        total = sum(weights) or 1.0
        available = self.width - gap * (len(weights) - 1)
        boxes: list[Box] = []
        cursor = self.x
        for weight in weights:
            width = available * (weight / total)
            boxes.append(Box(cursor, self.y, width, self.height))
            cursor += width + gap
        return boxes


def segment_intersects_box(p0: tuple[float, float], p1: tuple[float, float], box: Box) -> bool:
    """Liang-Barsky clip test: does the segment touch the rectangle?"""

    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - box.x), (dx, box.x1 - x0), (-dy, y0 - box.y), (dy, box.y1 - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return False
            t0 = max(t0, t)
        else:
            if t < t0:
                return False
            t1 = min(t1, t)
    return t0 <= t1


def polyline_intersects_box(points: list[tuple[float, float]], box: Box) -> bool:
    return any(
        segment_intersects_box(points[i], points[i + 1], box) for i in range(len(points) - 1)
    )


def box_gap(first: Box, second: Box) -> float:
    """Separation between two rectangles; negative when they overlap.

    Per axis, then the LARGER of the two: boxes that clear on either axis are
    clear. Taking the smaller (or mixing axes) reports two boxes that are far
    apart diagonally as nearly touching, which is how a node in one corner
    vetoed a label in another.
    """

    gap_x = max(first.x - second.x1, second.x - first.x1)
    gap_y = max(first.y - second.y1, second.y - first.y1)
    return max(gap_x, gap_y)


def border_point(box: Box, towards: tuple[float, float]) -> tuple[float, float]:
    """Where the ray from ``box``'s centre to ``towards`` leaves the box."""

    cx, cy = box.cx, box.cy
    dx, dy = towards[0] - cx, towards[1] - cy
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return cx, box.y1
    scale_x = (box.width / 2.0) / abs(dx) if abs(dx) > 1e-9 else math.inf
    scale_y = (box.height / 2.0) / abs(dy) if abs(dy) > 1e-9 else math.inf
    scale = min(scale_x, scale_y)
    return cx + dx * scale, cy + dy * scale


LABEL_OFFSETS: tuple[tuple[float, float], ...] = (
    (0.0, -1.0),   # above
    (0.0, 1.0),    # below
    (1.0, 0.0),    # right
    (-1.0, 0.0),   # left
    (1.0, -1.0),
    (-1.0, -1.0),
    (1.0, 1.0),
    (-1.0, 1.0),
)


def place_label(
    anchor: tuple[float, float],
    width: float,
    height: float,
    *,
    gap: float,
    occupied: list[Box],
    bounds: Box,
) -> tuple[Box, str] | None:
    """First collision-free label box around ``anchor``, tried in a fixed order.

    Deterministic by construction: the candidate order is fixed and the first
    box that clears every occupied rectangle and stays inside ``bounds`` wins.
    Returns the box and the text anchor ("start" | "middle" | "end").
    """

    ax, ay = anchor
    for ox, oy in LABEL_OFFSETS:
        if ox == 0:
            x = ax - width / 2.0
            anchor_name = "middle"
        elif ox > 0:
            x = ax + gap
            anchor_name = "start"
        else:
            x = ax - gap - width
            anchor_name = "end"
        if oy == 0:
            y = ay - height / 2.0
        elif oy > 0:
            y = ay + gap
        else:
            y = ay - gap - height
        candidate = Box(x, y, width, height)
        if not bounds.contains(candidate):
            continue
        if any(candidate.overlaps(box, tolerance=-1.0) for box in occupied):
            continue
        return candidate, anchor_name
    return None


def catmull_rom(points: list[tuple[float, float]], *, samples: int = 16) -> list[tuple[float, float]]:
    """Sampled Catmull-Rom spline through ``points`` (deterministic, no easing)."""

    if len(points) < 3:
        return list(points)
    padded = [points[0], *points, points[-1]]
    out: list[tuple[float, float]] = [points[0]]
    for i in range(len(padded) - 3):
        p0, p1, p2, p3 = padded[i], padded[i + 1], padded[i + 2], padded[i + 3]
        for step in range(1, samples + 1):
            t = step / samples
            t2, t3 = t * t, t * t * t
            x = 0.5 * (
                2 * p1[0]
                + (-p0[0] + p2[0]) * t
                + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                2 * p1[1]
                + (-p0[1] + p2[1]) * t
                + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
            )
            out.append((x, y))
    return out
