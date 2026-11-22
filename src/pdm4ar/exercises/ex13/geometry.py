"""
Geometry helpers for the spaceship planner.

* The spaceship footprint is over-approximated by a few circles placed on its body x-axis.
* Obstacles are represented analytically:
    - circles (planets, and asteroids moving with constant world velocity),
    - capsules / segments (the three bars of the docking station),
    - half-planes (edges of the convex map boundary).
* ``ObstacleSet.evaluate`` returns signed clearances g >= 0 (safe) and their gradients w.r.t.
  (x, y, psi) and the final-time parameter p, used to linearise the constraints in SCvx.
* ``ExactCollisionChecker`` checks the exact shapely footprint against the true obstacle shapes, used
  to verify candidate trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shapely
from numpy.typing import NDArray
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry

from dg_commons.sim.models.spaceship_structures import SpaceshipGeometry


def wrap_angle(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------------------------- ship model
@dataclass(frozen=True)
class ShipCircles:
    offsets: NDArray  # (n_c,) positions of the circle centres along the body x axis
    radius: float

    @property
    def n(self) -> int:
        return len(self.offsets)

    @property
    def bounding_radius(self) -> float:
        return float(np.max(np.abs(self.offsets)) + self.radius)


def ship_circles(sg: SpaceshipGeometry, inflation: float = 1.25) -> ShipCircles:
    """
    Cover the spaceship polygon (rectangle [-l_r, l_f] x [-w, w] plus nose triangle up to l_f + l_c) with
    circles of radius r = inflation * w along the body axis.
    """
    w = sg.w_half
    r = inflation * w
    h = np.sqrt(r**2 - w**2)  # half chord at |y| = w
    offsets = []
    c = -sg.l_r + h
    # cover the full-width rectangle
    while True:
        offsets.append(c)
        if c + h >= sg.l_f:
            break
        c += 2 * h
    # nose: the triangle narrows linearly to the tip at l_f + l_c
    tip = sg.l_f + sg.l_c
    c_nose = tip - r
    if c_nose > offsets[-1]:
        offsets.append(c_nose)
    return ShipCircles(offsets=np.array(offsets), radius=float(r))


def ship_polygon_coords(sg: SpaceshipGeometry) -> NDArray:
    return np.array(sg.outline)[:-1]  # (n_pts, 2) body-frame outline without repeated point


def ship_footprints(sg: SpaceshipGeometry, states: NDArray):
    """Vectorised shapely polygons for states (N, >=3) [x, y, psi, ...]."""
    body = ship_polygon_coords(sg)
    c, s = np.cos(states[:, 2]), np.sin(states[:, 2])
    xs = states[:, 0:1] + c[:, None] * body[None, :, 0] - s[:, None] * body[None, :, 1]
    ys = states[:, 1:2] + s[:, None] * body[None, :, 0] + c[:, None] * body[None, :, 1]
    coords = np.stack([xs, ys], axis=-1)
    coords = np.concatenate([coords, coords[:, :1]], axis=1)
    return shapely.polygons(coords)


# --------------------------------------------------------------------------------------------- obstacles
@dataclass
class ObstacleSet:
    circ_c0: NDArray = field(default_factory=lambda: np.zeros((0, 2)))  # centres at t = 0
    circ_v: NDArray = field(default_factory=lambda: np.zeros((0, 2)))  # world velocities
    circ_r: NDArray = field(default_factory=lambda: np.zeros(0))  # radii (incl. margins)
    seg_a: NDArray = field(default_factory=lambda: np.zeros((0, 2)))
    seg_b: NDArray = field(default_factory=lambda: np.zeros((0, 2)))
    seg_r: NDArray = field(default_factory=lambda: np.zeros(0))
    hp_n: NDArray = field(default_factory=lambda: np.zeros((0, 2)))  # outward unit normals
    hp_w: NDArray = field(default_factory=lambda: np.zeros(0))  # n . q <= w - r is the free side
    hp_r: NDArray = field(default_factory=lambda: np.zeros(0))

    @property
    def n_obs(self) -> int:
        return len(self.circ_r) + len(self.seg_r) + len(self.hp_w)

    def evaluate(self, X: NDArray, p: float, t0: float, tau: NDArray, ship: ShipCircles):
        """
        :param X: (>=3, K) states; rows x, y, psi are used.
        :param p: trajectory duration, node k is at absolute time t0 + p * tau[k]
        :return: g (M, K), dg/d[x, y, psi] (M, K, 3), dg/dp (M, K) with M = n_obs * n_circles,
                 ordering obstacle-major (all ship circles of obstacle 0, then obstacle 1, ...).
        """
        K = X.shape[1]
        x, y, psi = X[0], X[1], X[2]
        c, s = np.cos(psi), np.sin(psi)
        b = ship.offsets[:, None]  # (n_c, 1)
        q = np.stack([x[None] + b * c[None], y[None] + b * s[None]], axis=-1)  # (n_c, K, 2)
        dq_dpsi = np.stack([-b * s[None], b * c[None]], axis=-1)  # (n_c, K, 2)
        t = t0 + p * tau  # (K,)
        rs = ship.radius

        gs, grads, gps = [], [], []

        def _pack(nrm, g, gp):
            # nrm (n_o, n_c, K, 2) gradient of g wrt q
            gpsi = np.sum(nrm * dq_dpsi[None], axis=-1)
            grad = np.concatenate([nrm, gpsi[..., None]], axis=-1)
            gs.append(g.reshape(-1, K))
            grads.append(grad.reshape(-1, K, 3))
            gps.append(gp.reshape(-1, K))

        if len(self.circ_r):
            oc = self.circ_c0[:, None, :] + self.circ_v[:, None, :] * t[None, :, None]  # (n_o, K, 2)
            diff = q[None] - oc[:, None]  # (n_o, n_c, K, 2)
            dist = np.maximum(np.linalg.norm(diff, axis=-1), 1e-9)
            nrm = diff / dist[..., None]
            g = dist - (self.circ_r[:, None, None] + rs)
            gp = -np.sum(nrm * self.circ_v[:, None, None, :], axis=-1) * tau[None, None, :]
            _pack(nrm, g, gp)
        if len(self.seg_r):
            a = self.seg_a[:, None, None, :]
            ab = (self.seg_b - self.seg_a)[:, None, None, :]
            L2 = np.maximum(np.sum(ab**2, axis=-1), 1e-12)
            lam = np.clip(np.sum((q[None] - a) * ab, axis=-1) / L2, 0.0, 1.0)
            cp = a + lam[..., None] * ab
            diff = q[None] - cp
            dist = np.maximum(np.linalg.norm(diff, axis=-1), 1e-9)
            nrm = diff / dist[..., None]
            g = dist - (self.seg_r[:, None, None] + rs)
            _pack(nrm, g, np.zeros_like(g))
        if len(self.hp_w):
            proj = np.einsum("od,ckd->ock", self.hp_n, q)
            g = (self.hp_w - self.hp_r - rs)[:, None, None] - proj
            nrm = -np.broadcast_to(self.hp_n[:, None, None, :], proj.shape + (2,))
            _pack(nrm, g, np.zeros_like(g))

        if not gs:
            return np.zeros((0, K)), np.zeros((0, K, 3)), np.zeros((0, K))
        return np.concatenate(gs), np.concatenate(grads), np.concatenate(gps)


def boundary_halfplanes(corners: NDArray) -> tuple[NDArray, NDArray]:
    """Outward normals n and offsets w (free side n.q <= w) of a convex polygon given by corners."""
    pts = np.asarray(corners, dtype=float)
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    centroid = pts.mean(axis=0)
    normals, offsets = [], []
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        e = b - a
        if np.linalg.norm(e) < 1e-9:
            continue
        n = np.array([e[1], -e[0]]) / np.linalg.norm(e)
        if n @ (centroid - a) > 0:
            n = -n
        normals.append(n)
        offsets.append(n @ a)
    return np.array(normals), np.array(offsets)


def dock_segments(goal) -> tuple[NDArray, NDArray]:
    """Segments (base, two arms) of a DockingTarget, matching its plottable geometry."""
    A, B, C, A1, A2, _ = goal.get_landing_constraint_points()
    a = np.array([A1, A1, A2])
    b = np.array([A2, B, C])
    return a, b


# --------------------------------------------------------------------------------------------- exact check
@dataclass
class ExactCollisionChecker:
    sg: SpaceshipGeometry
    static_geoms: list[BaseGeometry]
    asteroid_c0: NDArray
    asteroid_v: NDArray
    asteroid_r: NDArray

    def clearance(self, states: NDArray, times: NDArray) -> NDArray:
        """Minimum distance between the exact spaceship footprint and any obstacle, per sample."""
        polys = ship_footprints(self.sg, states)
        dmin = np.full(len(states), np.inf)
        for geom in self.static_geoms:
            dmin = np.minimum(dmin, shapely.distance(polys, geom))
        for c0, v, r in zip(self.asteroid_c0, self.asteroid_v, self.asteroid_r):
            centres = c0[None] + v[None] * np.asarray(times)[:, None]
            pts = shapely.points(centres)
            dmin = np.minimum(dmin, shapely.distance(polys, pts) - r)
        return dmin


def static_geometries(planets: dict, boundary_corners: NDArray | None, dock_geom: BaseGeometry | None):
    geoms: list[BaseGeometry] = []
    for pl in planets.values():
        geoms.append(Point(pl.center).buffer(pl.radius))
    if boundary_corners is not None:
        geoms.append(LineString(boundary_corners))
    if dock_geom is not None:
        geoms.append(dock_geom)
    return geoms


def polygon_contains_ship(circles: ShipCircles, sg: SpaceshipGeometry) -> bool:
    """True if the union of the covering circles contains the spaceship polygon (sanity check)."""
    union = shapely.union_all([Point(o, 0.0).buffer(circles.radius, quad_segs=64) for o in circles.offsets])
    return union.buffer(1e-4).contains(Polygon(sg.outline))
