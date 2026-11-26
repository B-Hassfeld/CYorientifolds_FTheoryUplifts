# =============================================================================
# This file is part of CYTools.
#
# CYTools is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# CYTools is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# CYTools. If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  Calabi-Yau complete intersections of arbitrary codimension in
#               simplicial toric varieties.
#
#               Example (the quintic threefold in P^4):
#                   from CYorientifolds_FTheoryUplifts.CICY import CICY
#                   rays=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1],[-1,-1,-1,-1]]
#                   Y=CICY(rays,[[5,0,0,0,0]])    # shifted to [1,1,1,1,1]
#                   Y.hodge_numbers_stringy()     # no fan needed
#                   Y.hodge_numbers()             # builds the fan
#                   Y.toric_chern_class(2,in_glsm_basis=True)
#                   Y.mori_cone()                 # in the GLSM basis of Y.glsm()
#                   Y.compute_gv(max_deg=3)
# -----------------------------------------------------------------------------
from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from itertools import combinations, permutations, product
import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import qr
from flint import fmpz_mat

import cygv
from cytools import Polytope, h_polytope, Cone
from cytools.vector_config import VectorConfiguration
from cytools.vector_config.fan import Fan

from . import Uplift_functions as UF
from . import Hodge_Number_computations as HN

__all__ = ["CICY"]


class CICY:
    """
    **Description:**
    A Calabi-Yau complete intersection Y = D^(1) ∩ ... ∩ D^(r) of generic sections of r toric divisors in the
    simplicial toric variety X of a fan, of any codimension r. This is the general case of which the anticanonical
    hypersurface (CYTools' `CalabiYau`, r = 1) and the uplift fourfolds of `FT_CY.F_Theory_Uplift` (r = 2) are special
    cases. As for `FT_CY.CY_orientifold` the toric data can be given as a fan, a polytope, a vector configuration or
    the rays themselves; a fan is only constructed when a method needs one (`fan`). All toric data are expressed in
    the prime toric divisors D_1, ..., D_N of the rays, labelled 1, ..., N in the given order; tensors can also be
    expressed in a GLSM basis (see `glsm`, `set_glsm` and the `in_glsm_basis` arguments).
    The divisors are shifted by principal divisors on initialization: first (`UF.sums_to_anticanonical`) so that they
    sum to the anticanonical divisor (1, ..., 1), and then, if possible, so that all coefficients are 0 or 1, i.e. the
    divisors form a partition of the rays (`is_partition`).
    **Arguments:**
    - `fan_polytope_or_points (Polytope | VectorConfiguration | Fan | np.ndarray | list[list[int]])`: The toric data:
      a `Polytope` (its lattice points except the origin are the rays), a `VectorConfiguration`, a `Fan` (whose ray
      labels must be 1, ..., N, all used, e.g. `VectorConfiguration(rays).triangulate()`), or the rays themselves, one
      per row.
    - `divisors (Sequence[ArrayLike])`: The integer coefficient vectors of D^(1), ..., D^(r) in the prime toric
      divisors, one per divisor, ordered as the rays.
    **Raises:**
    - `TypeError`: Raised if the toric data have an unsupported type.
    - `ValueError`: Raised if the rays do not span the lattice, a fan is not simplicial or has other labels, the
      divisors are not integral vectors of the right shape, their number is not between 1 and dim X - 1, or they do
      not sum to the anticanonical class up to linear equivalence.
    """
    def __init__(self, fan_polytope_or_points: Polytope | VectorConfiguration | Fan | np.ndarray | list[list[int]], divisors: Sequence[ArrayLike]|None=None,keep_all=False) -> None:
        self.__fan = None
        self.__polytope=None
        match fan_polytope_or_points:
            case Polytope():
                pts = np.asarray(fan_polytope_or_points.points(), dtype=int)
                pts = pts[pts.any(axis=1)]
                self.__polytope=fan_polytope_or_points
            case VectorConfiguration():
                pts = np.asarray(fan_polytope_or_points.vectors(), dtype=int)
            case Fan():
                pts = np.asarray(fan_polytope_or_points.vectors(), dtype=int)
            case np.ndarray() | list():
                pts = np.asarray(fan_polytope_or_points)
                if pts.ndim != 2 or not np.array_equal(pts, np.rint(pts)):
                    raise ValueError("The rays must be an integer array with one ray per row.")
                pts = np.rint(pts).astype(int)
            case object():
                raise TypeError("Unsupported input type.")
        N, n = pts.shape
        if np.linalg.matrix_rank(pts) != n:
            raise ValueError("The rays must span the lattice.")
        if divisors is None:
            divisors=[np.ones(pts.shape[0],dtype=int)]
        D = np.asarray(divisors)
        if D.ndim != 2 or D.shape[1] != N:
            raise ValueError(f"`divisors` must be a list of coefficient vectors of length {N}, got shape {D.shape}.")
        if not np.array_equal(D, np.rint(D)):
            raise ValueError("The divisor coefficients must be integers.")
        D = np.rint(D).astype(int)
        if not 1 <= len(D) < n:
            raise ValueError(f"The number of divisors must be between 1 and {n - 1}, got {len(D)}.")
        self.__V = pts
        # the Newton polytopes of the given divisors on all given rays, if they are constructed below, and whether they are those of the final rays and divisors
        polytopes_given = None
        polytopes_kept = False
        if isinstance(fan_polytope_or_points, Fan):
            self.__fan = self._checked_fan(fan_polytope_or_points)
        else:
            if not keep_all:
                if pts.any(axis=1).all():
                    # constructed once: UF.intersecting_points uses them here, and `newton_polytopes` afterwards if the reduction of the rays does not change them
                    polytopes_given=[UF.Newton_Polytope(pts,d) for d in D]
                    D_given=D
                pts_for_fan=UF.intersecting_points(pts,D,polytopes=polytopes_given)
                if self.polytope().is_reflexive():
                    vert_diff=UF.row_difference(self.polytope().vertices(),pts_for_fan)
                    if len(vert_diff)>0:
                        pts_for_fan=np.vstack([pts_for_fan,vert_diff])
                # the position in pts of every row of pts_for_fan, in the order of pts_for_fan (UF.get_indices would sort them by pts)
                inds=np.array([UF.get_index(pts,v)[0] for v in pts_for_fan])
                D=[div[inds] for div in D]
                self.__V=pts_for_fan
                if polytopes_given is not None:
                    # the kept rays define the same polytopes as all rays if no ray was dropped (the same inequalities), or for the divisor (1, ..., 1) if they contain the vertices of the polytope of the rays, whose inequalities imply those of all other rays
                    polytopes_kept=len(pts_for_fan)==len(pts) or (bool((D_given==1).all()) and len(UF.row_difference(np.asarray(self.polytope().vertices(),dtype=int),pts_for_fan))==0)
        N, n = self.__V.shape
        # self.__V=pts_for_fan
        self.__N, self.__n, self.__r = N, n, len(D)
        # shift the divisors to sum to the anticanonical divisor, then to a partition if possible
        sums, m = UF.sums_to_anticanonical(self.__V, D)
        if not sums:
            raise ValueError("The divisors do not sum to the anticanonical class up to linear equivalence.")
        D[-1] += self.__V @ m
        D = np.asarray(D, dtype=int)
        if np.isin(D, (0, 1)).all():
            # the divisors now sum to (1, ..., 1), so with coefficients 0 and 1 they already form a partition:
            # they are kept and no search (and no choice of independent rays) is needed
            partition = D
        else:
            partition = _partition_representatives(self.__V, D, self._independent_rows())
        self.__is_partition = partition is not None
        self.__divisors = partition if self.__is_partition else D
        self.__basis = None
        self.__glsm = None
        self.__glsm_default = None
        # the polytopes constructed above are reused if, in addition, the divisors were not changed by the shift to the anticanonical divisor or to a partition
        self.__newton_polytopes = polytopes_given if polytopes_kept and np.array_equal(self.__divisors, D_given[:, inds]) else None
        self.__newton_polytopes_sum = None
        self.__newton_polytopes_conv = None
        self.__is_nef_decomposition = None
        self.__intersection_numbers = None
        self.__intersection_numbers_in_glsm_basis = None
        self.__intnums_sparse = None
        self.__chern = {}
        self.__chern_in_glsm = {}
        self.__mori_cone = None
        self.__mori_cone_not_in_basis = None
        self.__e_polynomial = None
        self.__Cayley_Polytope = None
        self.__hodge_numbers_nonstringy = None
        self.__hodge_numbers_stringy = None
        self.__mirror = None

    def __repr__(self) -> str:
        return (f"A Calabi-Yau complete intersection of codimension {self.__r} and dimension {self.dim()} in a "
                f"{self.__n}-dimensional toric variety with {self.__N} rays")

    # basic data
    # ----------
    def vectors_ambient(self) -> np.ndarray:
        """
        **Description:**
        Returns the ray generators, one per row, ordered by label 1, ..., N.
        **Returns:**
        - `np.ndarray`: The (N, dim X) integer array of rays.
        """
        return self.__V.copy()

    def divisors(self) -> np.ndarray:
        """
        **Description:**
        Returns the defining divisors as coefficient vectors in the prime toric divisors, after the shifts of the
        initialization (they sum to (1, ..., 1), with coefficients 0 and 1 if `is_partition`).
        **Returns:**
        - `np.ndarray`: The (r, N) integer array, one divisor per row.
        """
        return self.__divisors.copy()

    def ambient_dim(self) -> int:
        """
        **Description:**
        Returns the dimension of the ambient toric variety.
        **Returns:**
        - `int`: dim X.
        """
        return self.__n

    def polytope(self) -> Polytope:
        if self.__polytope is None:
            self.__polytope=Polytope(self.__V)
        return self.__polytope

    def codim(self) -> int:
        """
        **Description:**
        Returns the codimension r of Y in X, i.e. the number of defining divisors.
        **Returns:**
        - `int`: The codimension.
        """
        return self.__r

    def dim(self) -> int:
        """
        **Description:**
        Returns the dimension of Y.
        **Returns:**
        - `int`: dim Y = dim X - r.
        """
        return self.__n - self.__r

    # fan
    # ---
    def _checked_fan(self, fan: Fan) -> Fan:
        """
        **Description:**
        Verifies that a fan is simplicial, that its ray labels are 1, ..., N and all used, and that its rays are the
        rays of Y in the same order.
        **Arguments:**
        - `fan (Fan)`: The fan.
        **Raises:**
        - `ValueError`: Raised if one of the tests fails.
        **Returns:**
        - `Fan`: The fan.
        """
        labels = tuple(int(l) for l in fan.labels)
        if labels != tuple(int(l) for l in fan.used_labels) or labels != tuple(range(1, len(labels) + 1)):
            raise ValueError("The ray labels of the fan must be 1,...,N and every ray must be used by a cone "
                             "(e.g. build the fan from VectorConfiguration(rays) or use UF.reorder_fan).")
        if not fan.is_triangulation():
            raise ValueError("The fan must be simplicial.")
        if not np.array_equal(np.asarray(fan.vectors(), dtype=int), self.__V):
            raise ValueError("The rays of the fan differ from the rays of Y.")
        return fan

    def fan(self) -> Fan:
        """
        **Description:**
        The simplicial toric fan of the ambient variety. If none was given, it is constructed on the first call (as in
        `FT_CY.CY_orientifold.orbifold_toric_fan`): if the Newton polytopes of the divisors have a full-dimensional
        Minkowski sum whose normal fan (`UF.normal_fan`) has all its rays among the rays of Y, that normal fan is made
        simplicial (`UF.make_simplicial`) and refined by stellar subdivisions at the remaining rays (`UF.refine_fan`),
        so that the divisors are nef on it if they are nef on any fan over these rays; otherwise a regular
        triangulation of the rays (`VectorConfiguration.triangulate`) is used, with a warning, and the divisors need
        not be nef on it.
        **Raises:**
        - `ValueError`: Raised if a defining divisor has no global sections or the constructed fan does not pass
          `_checked_fan`.
        **Returns:**
        - `Fan`: The fan, with ray labels 1, ..., N in the order of `vectors_ambient`. A different fan can be set
          with `set_fan`.
        """
        if self.__fan is None:
            if self.polytope().is_reflexive():
                self.__fan=UF.reorder_fan(UF.refine_fan(UF.face_fan(self.polytope(),simplicial=True),self.__V),self.__V)
            else:
                self.__fan=VectorConfiguration(self.__V).triangulate()
            # polytopes = self.newton_polytopes()
            # if any(len(P.points()) == 0 for P in polytopes):
            #     raise ValueError("A defining divisor has no global sections.")
            # fan = None
            # if self.newton_polytopes_sum().dim() == self.__n:
            #     normal_fan = UF.normal_fan(polytopes)[0]
            #     if UF.contains_rows(self.__V, np.asarray(normal_fan.vectors(), dtype=int)):
            #         fan = UF.reorder_fan(UF.refine_fan(UF.make_simplicial(normal_fan), self.__V), self.__V)
            # if fan is None:
            #     warnings.warn("The rays do not support a fan refining the normal fan of the Newton polytopes; a regular "
            #                   "triangulation is used, on which the divisors need not be nef.")
            #     fan = VectorConfiguration(self.__V).triangulate()
            # self.__fan = self._checked_fan(fan)
        return self.__fan

    def set_fan(self, fan: Fan) -> None:
        """
        **Description:**
        Sets the simplicial toric fan of the ambient variety returned by `fan`, replacing a given or constructed one,
        and resets all quantities computed on the fan: `is_nef_decomposition`, the intersection numbers, the Chern
        tensors, the Mori cone, the E-polynomial and `hodge_numbers_nonstringy`. The fan must be over the rays of Y
        (`vectors_ambient`); if its rays are ordered or labelled differently, it is relabelled with `UF.reorder_fan`
        so that label i refers to the i-th row of `vectors_ambient`. The divisors, `basis`, `glsm`, the Newton
        polytopes and the stringy Hodge numbers do not depend on the fan and are kept.
        **Arguments:**
        - `fan (Fan)`: A simplicial fan over the rays of Y, using every ray.
        **Raises:**
        - `TypeError`: Raised if `fan` is not a `Fan`.
        - `ValueError`: Raised if the rays of the fan are not those of Y, or the fan is not simplicial or does not use
          every ray (`_checked_fan`).
        **Returns:**
        - `None`
        """
        if not isinstance(fan, Fan):
            raise TypeError(f"Expected a cytools.vector_config.fan.Fan, got {type(fan).__name__}.")
        V = np.asarray(fan.vectors(), dtype=int)
        if V.shape != self.__V.shape or not np.array_equal(np.unique(V, axis=0), np.unique(self.__V, axis=0)):
            raise ValueError("The rays of the fan differ from the rays of Y (`vectors_ambient`).")
        if not np.array_equal(V, self.__V) or tuple(int(l) for l in fan.labels) != tuple(range(1, self.__N + 1)):
            fan = UF.reorder_fan(fan, self.__V)
        self.__fan = self._checked_fan(fan)
        self.__is_nef_decomposition = None
        self.__intersection_numbers = None
        self.__intersection_numbers_in_glsm_basis = None
        self.__intnums_sparse = None
        self.__chern = {}
        self.__chern_in_glsm = {}
        self.__mori_cone = None
        self.__mori_cone_not_in_basis = None
        self.__e_polynomial = None
        self.__hodge_numbers_nonstringy = None

    def has_fan(self) -> bool:
        """
        **Description:**
        Whether the fan of the ambient variety is already known, i.e. was given at initialization, set with `set_fan`
        or constructed by an earlier call of `fan`. No fan is constructed.
        **Returns:**
        - `bool`: `True` if `fan` returns a fan without constructing one.
        """
        return self.__fan is not None

    # divisor properties
    # ------------------
    def is_partition(self) -> bool:
        """
        **Description:**
        Whether the divisors (after the shifts of the initialization) have coefficients 0 and 1 only, i.e. whether
        they are the sums of the prime toric divisors of a partition of the rays. As `FT_CY.F_Theory_Uplift.is_partition`.
        **Returns:**
        - `bool`: Whether the divisors form a partition.
        """
        return self.__is_partition

    def is_nef_decomposition(self) -> bool:
        """
        **Description:**
        Whether every defining divisor is Cartier and nef, i.e. whether the divisors form a nef decomposition of the
        anticanonical class. The result is cached and reset by `set_fan`.
        If a fan is known (`has_fan`), this is tested on it (`UF.is_Cartier`, `UF.is_nef`), as
        `FT_CY.F_Theory_Uplift.is_nef_decomposition` for an ambient triangulation.
        If no fan is known and the polytope Nabla of the rays (`polytope`) is reflexive, no fan is constructed: the
        divisors are Cartier and nef on the face fan of Nabla, and on every fan refining it such as the one `fan`
        constructs, iff the Minkowski sum of their Newton polytopes (`newton_polytopes_sum`) is the polar polytope
        Nabla^* (Batyrev-Borisov). Indeed the support functions psi_a of the Newton polytopes satisfy
        psi_a(v) >= -a^(a)_v on every ray v, the divisors sum to 1 and the support function of Nabla^* is -1 on every
        ray, so equality of the polytopes forces psi_a(v) = -a^(a)_v, and concave functions whose sum is linear on the
        cones of the face fan are linear there; conversely Newton polytopes of nef divisors add up to the one of their
        sum. This does not need coefficients 0 and 1.
        Otherwise (no fan and Nabla not reflexive) the fan is constructed and the divisors are tested on it.
        **Returns:**
        - `bool`: Whether the divisors form a nef decomposition of the anticanonical class.
        """
        if self.__is_nef_decomposition is None:
            if self.__fan is None and self.polytope().is_reflexive():
                try:
                    polar = np.asarray(self.polytope().dual().vertices(), dtype=int)
                    minkowski_sum = np.asarray(self.newton_polytopes_sum().vertices(), dtype=int)
                    self.__is_nef_decomposition = bool(np.array_equal(np.unique(polar, axis=0), np.unique(minkowski_sum, axis=0)))
                except ValueError:
                    # a divisor without global sections is not nef
                    self.__is_nef_decomposition = False
            else:
                fan = self.fan()
                self.__is_nef_decomposition = all(bool(UF.is_Cartier(fan, d)[0]) and bool(UF.is_nef(fan, d)) for d in self.__divisors)
        return self.__is_nef_decomposition

    def is_nef_partition(self) -> bool:
        """
        **Description:**
        Whether the divisors form a partition (`is_partition`) and a nef decomposition (`is_nef_decomposition`), as
        `FT_CY.F_Theory_Uplift.is_nef_partition`.
        **Returns:**
        - `bool`: Whether the divisors form a nef partition.
        """
        return self.is_partition() and self.is_nef_decomposition()

    def is_nef(self) -> list[bool]:
        """
        **Description:**
        Whether each defining divisor is nef on the fan `fan` (`UF.is_nef`), which on a simplicial toric variety means
        base-point free, so that the generic complete intersection is quasi-smooth and the Hodge numbers of
        `hodge_numbers` apply. The fan is constructed if needed.
        **Returns:**
        - `list[bool]`: One entry per defining divisor.
        """
        fan = self.fan()
        return [bool(UF.is_nef(fan, d)) for d in self.__divisors]

    def is_favorable(self, check: bool = True) -> bool:
        """
        **Description:**
        Whether no prime toric divisor splits on Y, i.e. whether every D_i ∩ Y has at most one component, with
        `UF.is_favorable`. The components are counted on the open orbit of D_i from the faces of the Newton polytopes of
        the defining divisors cut out by the ray of D_i, without intersection numbers. If the divisors form a nef
        decomposition on the fan `fan` (`is_nef_decomposition`, which needs no fan for a reflexive polytope), they are base-point
        free, so no section vanishes identically on a torus orbit and the base locus check of `UF.is_favorable` always
        passes: it is then run on the rays alone, without the check. Otherwise it is run on the fan with `check`. For a
        complete intersection non-splitting is necessary but not sufficient for favorability, since Pic(Y) can contain
        classes not supported on the toric boundary. `UF.is_favorable` also returns the splitting divisors and their
        numbers of components.
        **Arguments:**
        - `check (bool)`: Whether to verify, if the divisors do not form a nef decomposition, that Y has no components or
          divisors in the base locus of the defining divisors, which the face criterion requires (see
          `UF.is_favorable`). Defaults to `True`.
        **Raises:**
        - `ValueError`: Raised if a defining divisor vanishes identically on some D_i, or if `check` fails.
        **Returns:**
        - `bool`: `True` if no prime toric divisor splits on Y.
        """
        if self.is_nef_decomposition():
            return UF.is_favorable(self.__V, list(self.__divisors), check=False)
        return UF.is_favorable(self.fan(), list(self.__divisors), check=check)

    def newton_polytopes(self) -> list[h_polytope.HPolytope]:
        """
        **Description:**
        The Newton polytopes {m : <m, v_j> >= -a^(i)_j for all j} of the defining divisors (`UF.Newton_Polytope`),
        cached.
        **Returns:**
        - `list[h_polytope.HPolytope]`: One polytope per defining divisor.
        """
        if self.__newton_polytopes is None:
            self.__newton_polytopes = [UF.Newton_Polytope(self.__V, d) for d in self.__divisors]
        return list(self.__newton_polytopes)

    def newton_polytopes_sum(self) -> Polytope:
        """
        **Description:**
        The Minkowski sum of the Newton polytopes of the defining divisors (`Polytope.minkowski_sum`), i.e. the Newton
        polytope of the anticanonical divisor as a sum, cached. As `FT_CY.F_Theory_Uplift.pol_M_sum`.
        **Raises:**
        - `ValueError`: Raised if a defining divisor has no global sections.
        **Returns:**
        - `Polytope`: The Minkowski sum.
        """
        if self.__newton_polytopes_sum is None:
            polytopes = self.newton_polytopes()
            if any(len(P.points()) == 0 for P in polytopes):
                raise ValueError("A defining divisor has no global sections.")
            self.__newton_polytopes_sum = reduce(lambda P, Q: P.minkowski_sum(Q), polytopes)
        return self.__newton_polytopes_sum

    def newton_polytopes_conv(self) -> Polytope:
        """
        **Description:**
        The convex hull of the union of the Newton polytopes of the defining divisors, cached. As
        `FT_CY.F_Theory_Uplift.pol_M_conv`.
        **Raises:**
        - `ValueError`: Raised if a defining divisor has no global sections.
        **Returns:**
        - `Polytope`: The convex hull.
        """
        if self.__newton_polytopes_conv is None:
            polytopes = self.newton_polytopes()
            if any(len(P.points()) == 0 for P in polytopes):
                raise ValueError("A defining divisor has no global sections.")
            vertices = np.concatenate([np.asarray(P.vertices(), dtype=int) for P in polytopes], axis=0)
            self.__newton_polytopes_conv = Polytope(np.unique(vertices, axis=0))
        return self.__newton_polytopes_conv

    def dual(self) -> CICY:
        """
        **Description:**
        The Batyrev-Borisov mirror of the complete intersection. It requires a nef partition of a reflexive polytope:
        Nabla = conv(rays) (`polytope`) reflexive, the divisors a partition (`is_partition`) of the rays into
        E_1, ..., E_r, and the Minkowski sum of their Newton polytopes Delta_1 + ... + Delta_r (`newton_polytopes_sum`)
        equal to the polar polytope Nabla^* = {m : <m, v> >= -1 for all rays v}. The last condition does not need a fan:
        it holds iff every divisor is Cartier and nef on the face fan of Nabla (and then on every refinement of it by
        rays of Nabla), so no fan is constructed, unlike `is_nef_partition`.
        The mirror lives in the toric variety of the reflexive polytope conv(Delta_1 ∪ ... ∪ Delta_r)
        (`newton_polytopes_conv`), which is dual to Nabla_1 + ... + Nabla_r with Nabla_a = conv(0, E_a), and is cut out
        by the dual nef partition: a nonzero lattice point m has the coefficient w_a(m) = -min_{n in Nabla_a} <m, n> in
        the a-th divisor, so that the Newton polytopes of the mirror divisors are the Nabla_a. By the duality the w_a(m)
        are 0 or 1 and sum to 1 for every m, i.e. they form a partition (a nonzero lattice point m lies in exactly one
        Delta_a, the one with w_a(m) = 1). The mirror is initialized like any CICY from these points and divisors, so no
        fan is set, and the constructor keeps this partition.
        **Raises:**
        - `ValueError`: Raised if the divisors do not form a nef partition of a reflexive polytope.
        - `RuntimeError`: Raised if the dual divisors do not form a partition, which the duality excludes.
        **Returns:**
        - `CICY`: The mirror complete intersection.
        """
        if not self.__is_partition:
            raise ValueError("The divisors do not form a partition, so they are not a nef partition.")
        if not self.polytope().is_reflexive():
            raise ValueError("The polytope of the rays is not reflexive, so there is no Batyrev-Borisov mirror.")
        # polar = np.asarray(UF.Newton_Polytope(self.__V, np.ones(self.__N, dtype=int)).vertices(), dtype=int)
        # minkowski_sum = np.asarray(self.newton_polytopes_sum().vertices(), dtype=int)
        # if not np.array_equal(np.unique(polar, axis=0), np.unique(minkowski_sum, axis=0)):
        #     raise ValueError("The Minkowski sum of the Newton polytopes is not the polar polytope, so the divisors are not "
        #                      "nef and do not form a nef partition.")
        parts = [self.__V[d == 1] for d in self.__divisors]
        dual_partition = lambda P: np.array([-(P @ E.T).min(axis=1, initial=0) for E in parts], dtype=int)
        points = np.asarray(self.newton_polytopes_conv().points(), dtype=int)
        points = points[points.any(axis=1)]
        divisors = dual_partition(points)
        if not (np.isin(divisors, (0, 1)).all() and (divisors.sum(axis=0) == 1).all()):
            raise RuntimeError("The dual divisors do not form a partition.")
        dual_cicy = CICY(points, divisors)
        return dual_cicy

    # bases
    # -----
    def _independent_rows(self) -> np.ndarray:
        """
        **Description:**
        Zero-based indices of dim X linearly independent rays, chosen by a QR decomposition with column pivoting of
        the rays (as in `UF.divisor_in_zero_one_rep`), which costs O(N dim^2) and, unlike a search through subsets of
        rays, does not depend on the order of the rays. They need not form a lattice basis, which the rays may not
        contain (e.g. for mirror polytopes); this suffices for `_partition_representatives`. Their independence is
        confirmed exactly by an integer determinant.
        **Raises:**
        - `RuntimeError`: Raised if the chosen rays are not linearly independent (excluded, since the rays span).
        **Returns:**
        - `np.ndarray`: The sorted indices.
        """
        _, _, pivots = qr(self.__V.T.astype(float), mode="economic", pivoting=True)
        rows = np.sort(pivots[:self.__n])
        if int(fmpz_mat(self.__V[rows].tolist()).det()) == 0:
            raise RuntimeError("The rays chosen by the pivoted QR decomposition are not linearly independent.")
        return rows

    def basis(self) -> np.ndarray:
        """
        **Description:**
        One-indexed labels of the prime toric divisors forming a Z-basis of the class group of X (by default
        `UF.basis_homology(rays)`, or the labels given to `set_basis`). Tensors "in basis" are first restricted to these
        divisors (positions in this array, one-indexed) and then transformed to the GLSM basis of `glsm`.
        **Returns:**
        - `np.ndarray`: The N - dim X basis labels.
        """
        if self.__basis is None:
            self.__basis = np.asarray(UF.basis_homology(self.__V), dtype=int)
        return self.__basis.copy()

    def set_basis(self, labels: ArrayLike) -> None:
        """
        **Description:**
        Sets the prime toric divisor basis used by `basis`. The rays not in the basis must form a lattice basis, so
        that the basis divisors generate the class group (the criterion of `UF.basis_homology`). If a GLSM charge matrix
        was set with `set_glsm`, it must be invertible on the new basis columns; the default GLSM charge matrix is
        recomputed.
        **Arguments:**
        - `labels (ArrayLike)`: N - dim X distinct one-indexed labels. Their order fixes the basis positions.
        **Raises:**
        - `ValueError`: Raised if the labels are not valid or the complementary rays are not a lattice basis.
        **Returns:**
        - `None`
        """
        labels = np.asarray(labels, dtype=int).reshape(-1)
        N, n = self.__N, self.__n
        if len(labels) != N - n or len(set(labels.tolist())) != N - n or (labels < 1).any() or (labels > N).any():
            raise ValueError(f"The basis must consist of {N - n} distinct labels in 1,...,{N}.")
        complement = np.array(sorted(set(range(1, N + 1)) - set(labels.tolist())), dtype=int)
        if abs(int(fmpz_mat(self.__V[complement - 1].tolist()).det())) != 1:
            raise ValueError("The rays not in the basis must form a lattice basis.")
        if self.__glsm is not None and np.linalg.matrix_rank(self.__glsm[:, labels - 1].astype(float)) < N - n:
            raise ValueError("The GLSM charge matrix set with `set_glsm` is singular on the new basis columns.")
        self.__basis = labels.copy()
        self.__glsm_default = None
        self.__intersection_numbers_in_glsm_basis = None
        self.__intnums_sparse = None
        self.__chern_in_glsm = {}
        self.__mori_cone = None

    def glsm(self) -> np.ndarray:
        """
        **Description:**
        The GLSM charge matrix Q (rows: U(1) charges, columns: prime toric divisors, Q @ rays = 0) whose rows define
        the divisor basis J_a used for tensors "in basis": D_j = sum_a Q_aj J_a. By default it is the charge matrix in
        the prime toric basis of `basis`, i.e. Q restricted to the basis columns is the identity, so that the GLSM
        basis coincides with the prime toric basis. A different one can be set with `set_glsm`.
        **Returns:**
        - `np.ndarray`: The (N - dim X, N) charge matrix.
        """
        if self.__glsm is not None:
            return self.__glsm.copy()
        if self.__glsm_default is None:
            b = self.basis() - 1
            c = np.array(sorted(set(range(self.__N)) - set(b.tolist())), dtype=int)
            Q = np.zeros((self.__N - self.__n, self.__N), dtype=int)
            Q[:, b] = np.eye(len(b), dtype=int)
            Qc = -self.__V[b] @ np.linalg.inv(self.__V[c].astype(float))
            Q[:, c] = np.rint(Qc).astype(int)
            if not np.allclose(Qc, Q[:, c], atol=1e-8, rtol=0) or (Q @ self.__V).any():
                raise RuntimeError("Could not construct the GLSM charge matrix in the prime toric basis.")
            self.__glsm_default = Q
        return self.__glsm_default.copy()

    def set_glsm(self, Q: ArrayLike) -> None:
        """
        **Description:**
        Sets the GLSM charge matrix used by `glsm` (e.g. a saturated or symmetric-adapted charge matrix as `Qsat` of
        `FT_CY.F_Theory_Uplift`). Tensors "in basis" are transformed to the basis of its rows by
        `UF.tensor_symm_in_basis`, which inverts Q on the columns of `basis`.
        **Arguments:**
        - `Q (ArrayLike)`: A (N - dim X, N) matrix whose rows are linear relations among the rays and which is
          invertible on the columns of `basis`.
        **Raises:**
        - `ValueError`: Raised if Q has the wrong shape, its rows are not relations among the rays, or it is singular on
          the basis columns.
        **Returns:**
        - `None`
        """
        Q = np.asarray(Q)
        if Q.shape != (self.__N - self.__n, self.__N):
            raise ValueError(f"Q must have shape {(self.__N - self.__n, self.__N)}, got {Q.shape}.")
        if not np.allclose(Q.astype(float) @ self.__V, 0, atol=1e-8, rtol=0):
            raise ValueError("The rows of Q must be linear relations among the rays (Q @ rays = 0).")
        if np.linalg.matrix_rank(Q[:, self.basis() - 1].astype(float)) < self.__N - self.__n:
            raise ValueError("Q must be invertible on the columns of `basis`.")
        self.__glsm = Q.copy()
        self.__intersection_numbers_in_glsm_basis = None
        self.__intnums_sparse = None
        self.__chern_in_glsm = {}
        self.__mori_cone = None

    # intersection theory
    # -------------------
    def intersection_numbers(self, in_glsm_basis: bool = False, in_prime_toric_basis: bool=False, tol: float = 1e-5, decimals: int = 10) -> dict[tuple[int, ...], int | float]:
        """
        **Description:**
        Intersection numbers of the restricted prime toric divisors on Y, i.e. the ambient intersection products
        D_{a_1} ... D_{a_dim Y} D^(1) ... D^(r) on the fan `fan` (`UF.subvariety_intersection_numbers`), either in terms
        of all prime toric divisors or in the GLSM basis of `glsm` (via the prime toric basis `basis`,
        `UF.tensor_symm_in_basis`). The values `tol` and `decimals` only matter on the first call, which caches the
        prime toric result.
        **Arguments:**
        - `in_glsm_basis (bool)`: Whether to express the result in the GLSM basis. Defaults to `False`.
        - `in_prime_toric_basis (bool)`: Whether to restrict the result to the prime toric basis `basis` instead
          (ignored if `in_glsm_basis`). Defaults to `False`.
        - `tol (float)`: Rounding tolerance, see `UF.subvariety_intersection_numbers`. Defaults to `1e-5`.
        - `decimals (int)`: Number of decimals to round to. Defaults to `10`.
        **Returns:**
        - `dict[tuple[int, ...], int | float]`: The intersection numbers keyed by sorted one-indexed label tuples
          (prime toric labels, or basis positions with `in_glsm_basis` or `in_prime_toric_basis`); absent keys
          are zero.
        """
        if self.__intersection_numbers is None:
            self.__intersection_numbers = UF.subvariety_intersection_numbers(self.fan(), list(self.__divisors), tol=tol, decimals=decimals)
        # K = dict(self.__intersection_numbers)
        if in_glsm_basis:
            if self.__intersection_numbers_in_glsm_basis is None:
                self.__intersection_numbers_in_glsm_basis=UF.tensor_symm_in_basis(self.__intersection_numbers, self.basis(), self.glsm(), tol=tol, in_prime_basis=False)
            return self.__intersection_numbers_in_glsm_basis
        if in_prime_toric_basis:
            return UF.tensor_symm_in_basis(self.__intersection_numbers,self.basis())
        return self.__intersection_numbers

    def toric_chern_class(self, n: int, in_glsm_basis: bool = False) -> int | dict[tuple[int, ...], int] | dict[int, int]:
        """
        **Description:**
        The Chern tensor of c_n(Y) by torus localization on the fan `fan` (`UF.toric_chern_class`): from
        c(TY) = c(TX) / ((1 + D^(1)) ... (1 + D^(r))) the numbers T_{a_1..a_m} = int_Y c_n(Y) D_{a_1} ... D_{a_m} with
        m = dim Y - n, in terms of the prime toric divisors or in the GLSM basis of `glsm`. For n = dim Y this is the
        localization value of int_Y c_top(Y), which differs from the topological Euler characteristic `chi` if Y has
        orbifold singularities.
        **Arguments:**
        - `n (int)`: The degree of the Chern class, between 0 and dim Y.
        - `in_glsm_basis (bool)`: Whether to express the tensor in the GLSM basis (ignored for n = dim Y). Defaults
          to `False`.
        **Raises:**
        - `ValueError`: Raised if n is not between 0 and dim Y.
        **Returns:**
        - `int | dict[tuple[int, ...], int] | dict[int, int]`: A number if n = dim Y, {i: value} if n = dim Y - 1 and
          {sorted index tuple: value} otherwise, with one-indexed prime toric labels or GLSM indices; vanishing entries
          are omitted.
        """
        if n not in self.__chern:
            self.__chern[n] = UF.toric_chern_class(self.fan(), self.__divisors, n)
        c = self.__chern[n]
        if not isinstance(c, dict):
            return c
        c = dict(c)
        if in_glsm_basis:
            if self.__chern_in_glsm.get(n) is None:
                self.__chern_in_glsm[n] = UF.tensor_symm_in_basis(self.__chern[n], self.basis(), self.glsm(), in_prime_basis=False)
            return self.__chern_in_glsm.get(n)
        return self.__chern.get(n)

    # Mori cone and Gopakumar-Vafa invariants
    # ---------------------------------------
    def mori_cone(self, in_basis: bool = True) -> Cone:
        """
        **Description:**
        The Mori cone of the ambient toric phase `fan`, as `ToricVariety.mori_cone` of CYTools. It is generated by the
        wall curves of the fan (the secondary-cone hyperplane normals, i.e. the linear relations among the rays across
        each wall). In basis they are written in the GLSM basis of `glsm`, i.e. in the basis of the rows of Q
        (`UF.mori_cone_in_glsm_basis`), as `mori_cone_face_d7` of `FT_CY.F_Theory_Uplift` without the restriction to
        the D7 rows; the dual of this cone is the Kahler cone of X. Not in basis they are the relations themselves,
        i.e. a curve C has the coordinates C . D_j with all prime toric divisors, and the cone is not full-dimensional.
        The generators of both cones are in the same order, those not in basis being `mori_cone().rays() @ glsm()`.
        Cached; the cone in basis is reset by `set_basis` and `set_glsm`.
        **Arguments:**
        - `in_basis (bool)`: Whether to express the cone in the GLSM basis of `glsm` instead of all prime toric
          divisors. Defaults to `True`.
        **Returns:**
        - `Cone`: The Mori cone in R^(N - dim X), with rays in the coordinates dual to the rows of `glsm`, or the
          (N - dim X)-dimensional Mori cone in R^N if `in_basis` is `False`.
        """
        if not in_basis:
            if self.__mori_cone_not_in_basis is None:
                self.__mori_cone_not_in_basis = Cone(rays=np.asarray(self.fan().secondary_cone_hyperplanes()))
            return self.__mori_cone_not_in_basis
        if self.__mori_cone is None:
            self.__mori_cone = UF.mori_cone_in_glsm_basis(self.glsm(), self.fan())
        return self.__mori_cone

    def compute_gv(self, hilbert_basis: bool = False, min_points: int = 600, max_deg: int = 2, target_points: ArrayLike | None = None,grading_vector=None) -> list:
        """
        **Description:**
        Genus-zero Gopakumar-Vafa invariants of Y by the HKTY procedure of `cygv.compute_gv`, as `compute_d7_gv` of
        `FT_CY.F_Theory_Uplift`: the curve classes are the lattice points of the Mori cone `mori_cone` in the GLSM
        basis of `glsm`, graded by `Cone.find_grading_vector`, and the intersection numbers are those of
        `intersection_numbers(in_glsm_basis=True)`, converted to zero-based triples if dim Y = 3 and to the surface
        form `UF.intersection_number_surfaces` (D_a D_b paired with two basis divisors) if dim Y = 4. For codimension
        r > 1 the divisors must form a partition (`is_partition`), which is passed as the nef partition; the result is
        only meaningful if it is a nef partition (`is_nef_partition`).
        **Arguments:**
        - `hilbert_basis (bool)`: Whether to use the Hilbert basis of the Mori cone instead of its rays as generators.
          Defaults to `False`.
        - `min_points (int)`: Minimal number of curve classes to compute, see `cygv.compute_gv`. Defaults to `600`.
        - `max_deg (int)`: Maximal degree with respect to the grading vector. Defaults to `2`.
        - `target_points (ArrayLike | None)`: Curve classes to compute the invariants for, see `cygv.compute_gv`.
          Defaults to `None`.
        **Raises:**
        - `NotImplementedError`: Raised if dim Y is not 3 or 4.
        - `ValueError`: Raised if r > 1 and the divisors do not form a partition.
        **Returns:**
        - `list`: The output of `cygv.compute_gv`: pairs `(curve class, invariant)` if dim Y = 3, and
          `((curve class, surface index), invariant)` if dim Y = 4, with the surface indices of
          `UF.intersection_number_surfaces`.
        """
        if self.dim() not in (3, 4):
            raise NotImplementedError("Gopakumar-Vafa invariants are only implemented for threefolds and fourfolds.")
        if self.__r > 1 and not self.is_partition():
            raise ValueError("The divisors must form a (nef) partition to compute Gopakumar-Vafa invariants.")
        mori = self.mori_cone()
        gens = mori.hilbert_basis() if hilbert_basis else mori.rays()
        Q = self.glsm()
        IN = self.intersection_numbers(in_glsm_basis=True)
        # IN_GV={}
        # for k,v in IN.items():
        #     t=tuple(np.array(k)-1)
        #     IN_GV[t]=v
        if self.dim() == 3:
            intnums = {tuple(int(i) - 1 for i in key): val for key, val in IN.items()}
        else:
            intnums = UF.intersection_number_surfaces(IN, Q.shape[0])[0]
        nefpart = None if self.__r == 1 else [np.flatnonzero(d) for d in self.__divisors]
        if grading_vector is None:
            grading_vector=mori.find_grading_vector()
        return UF.compute_gv_h11(generators=gens, grading_vector=grading_vector, q=Q, intnums=intnums,max_deg=max_deg, min_points=min_points, target_points=target_points, nefpart=nefpart)
        # return cygv.compute_gv(generators=gens, grading_vector=mori.find_grading_vector(), q=Q, intnums=intnums,
                            #    max_deg=max_deg, min_points=min_points, target_points=target_points, nefpart=nefpart)

    def chi(self) -> int:
        """
        **Description:**
        The topological Euler characteristic of Y, the alternating sum of the Hodge diamond `hodge_numbers`.
        **Returns:**
        - `int`: chi(Y) = sum_{p,q} (-1)^(p+q) h^{p,q}.
        """
        h = self.hodge_numbers()
        return int((h * (-1) ** np.add.outer(np.arange(len(h)), np.arange(len(h)))).sum())

    # volumes and Kahler metric
    # -------------------------
    def _volume_derivatives(self, tloc: ArrayLike, order: int) -> float | np.ndarray:
        """
        **Description:**
        The derivatives of the volume V(t) = kappa_{a_1...a_d} t^{a_1} ... t^{a_d} / d! of Y (d = dim Y) with respect
        to the Kahler parameters, at the Kahler class J = sum_a t^a J_a in the GLSM basis J_a of `glsm`: V itself
        (order 0), dV/dt^a = kappa_{a b...} t^b ... / (d-1)! (order 1) and d^2V/dt^a dt^b = kappa_{a b c...} t^c ... / (d-2)!
        (order 2). The intersection numbers `intersection_numbers(in_glsm_basis=True)` are contracted as the sparse
        dictionary they are stored in, so the dense h^d tensor is never formed: a sorted key a_1 <= ... <= a_d with value
        kappa stands for d!/|Stab| equal entries of the symmetric tensor (`UF._stabilizer_order`), so it contributes the
        monomial kappa t^{a_1} ... t^{a_d} / |Stab| to V, and the monomials are differentiated slot by slot. The keys
        and the weights kappa/|Stab| are cached, and reset by `set_fan`, `set_basis` and `set_glsm`.
        **Arguments:**
        - `tloc (ArrayLike)`: The Kahler parameters t^a, one per row of `glsm`.
        - `order (int)`: The order of the derivative, 0, 1 or 2.
        **Raises:**
        - `ValueError`: Raised if `tloc` is not a vector of length N - dim X.
        **Returns:**
        - `float | np.ndarray`: The volume, its gradient of shape (N - dim X,) or its Hessian of shape
          (N - dim X, N - dim X).
        """
        t = np.asarray(tloc, dtype=float)
        h, d = self.__N - self.__n, self.dim()
        if t.shape != (h,):
            raise ValueError(f"`tloc` must be a vector of length {h}, one Kahler parameter per row of `glsm`, got shape {t.shape}.")
        if self.__intnums_sparse is None:
            IN = self.intersection_numbers(in_glsm_basis=True)
            keys = np.sort(np.array(list(IN.keys()), dtype=int).reshape(-1, d) - 1, axis=1)
            weights = np.array(list(IN.values()), dtype=float) / UF._stabilizer_order(keys)
            self.__intnums_sparse = (keys, weights)
        keys, weights = self.__intnums_sparse
        tk = t[keys]
        if order == 0:
            return float(weights @ tk.prod(axis=1))
        if order == 1:
            gradient = np.zeros(h)
            for p in range(d):
                gradient += np.bincount(keys[:, p], weights=weights * np.delete(tk, p, axis=1).prod(axis=1), minlength=h)
            return gradient
        hessian = np.zeros((h, h))
        for p, q in permutations(range(d), 2):
            np.add.at(hessian, (keys[:, p], keys[:, q]), weights * np.delete(tk, [p, q], axis=1).prod(axis=1))
        return hessian

    def compute_cy_volume(self, tloc: ArrayLike) -> float:
        """
        **Description:**
        Computes the volume of Y at a location in the Kahler cone, as `CalabiYau.compute_cy_volume` of CYTools: for the
        Kahler class J = sum_a t^a J_a in the GLSM basis of `glsm`, V = int_Y J^d / d! = kappa_{a_1...a_d} t^{a_1} ...
        t^{a_d} / d! with d = dim Y and the intersection numbers `intersection_numbers(in_glsm_basis=True)` on the fan
        `fan`, i.e. kappa_{abc} t^a t^b t^c / 6 for a threefold.
        **Arguments:**
        - `tloc (ArrayLike)`: A vector specifying a location in the Kahler cone, in the coordinates t^a dual to the rows
          of `glsm` (those of `mori_cone`).
        **Raises:**
        - `ValueError`: Raised if `tloc` is not a vector of length N - dim X.
        **Returns:**
        - `float`: The volume of Y at the specified location.
        """
        return self._volume_derivatives(tloc, 0)

    def compute_divisor_volumes(self, tloc: ArrayLike, in_basis: bool = False) -> np.ndarray:
        """
        **Description:**
        Computes the volumes of the divisors of Y at a location in the Kahler cone, as
        `CalabiYau.compute_divisor_volumes` of CYTools: the volume of the basis divisor J_a restricted to Y is
        tau_a = int_Y J_a J^(d-1) / (d-1)! = kappa_{a b_2...b_d} t^{b_2} ... t^{b_d} / (d-1)! with d = dim Y, i.e.
        kappa_{abc} t^b t^c / 2 for a threefold, and that of the prime toric divisor D_j = sum_a Q_aj J_a restricted to Y
        is sum_a Q_aj tau_a with the charge matrix Q of `glsm`.
        **Arguments:**
        - `tloc (ArrayLike)`: A vector specifying a location in the Kahler cone, in the coordinates t^a dual to the rows
          of `glsm`.
        - `in_basis (bool)`: Whether to return the volumes of the divisors of the GLSM basis instead of those of all
          prime toric divisors. Defaults to `False`.
        **Raises:**
        - `ValueError`: Raised if `tloc` is not a vector of length N - dim X.
        **Returns:**
        - `np.ndarray`: The volumes of the N prime toric divisors, ordered as the rays, or of the N - dim X basis
          divisors, at the specified location.
        """
        tau_basis = self._volume_derivatives(tloc, 1)
        if in_basis:
            return tau_basis
        return self.glsm().T @ tau_basis

    def compute_curve_volumes(self, tloc: ArrayLike, only_extremal: bool = False) -> np.ndarray:
        """
        **Description:**
        Computes the volumes of the curves corresponding to the (not necessarily minimal) generators of the Mori cone
        `mori_cone` of the ambient toric phase, as `CalabiYau.compute_curve_volumes` of CYTools: a generator with
        coordinates c_a = C . J_a has the volume int_C J = sum_a c_a t^a.
        **Arguments:**
        - `tloc (ArrayLike)`: A vector specifying a location in the Kahler cone, in the coordinates t^a dual to the rows
          of `glsm`.
        - `only_extremal (bool)`: Whether to use only the extremal rays of the Mori cone. Defaults to `False`.
        **Returns:**
        - `np.ndarray`: The volumes of the curves, ordered as the rays (or extremal rays) of `mori_cone`.
        """
        c = self.mori_cone()
        if only_extremal:
            return c.extremal_rays().dot(tloc)
        return c.rays().dot(tloc)

    def compute_kappa_matrix(self, tloc: ArrayLike) -> np.ndarray:
        """
        **Description:**
        Computes the matrix kappa_{abc} t^c at a location in the Kahler cone, with the intersection numbers
        `intersection_numbers(in_glsm_basis=True)`, as `CalabiYau.compute_kappa_matrix` of CYTools. Only for threefolds.
        Also available as `compute_AA`.
        **Arguments:**
        - `tloc (ArrayLike)`: A vector specifying a location in the Kahler cone, in the coordinates t^a dual to the rows
          of `glsm`.
        **Raises:**
        - `NotImplementedError`: Raised if dim Y is not 3.
        - `ValueError`: Raised if `tloc` is not a vector of length N - dim X.
        **Returns:**
        - `np.ndarray`: The (N - dim X, N - dim X) matrix kappa_{abc} t^c at the specified location.
        """
        if self.dim() != 3:
            raise NotImplementedError("This function only supports Calabi-Yau 3-folds.")
        return self._volume_derivatives(tloc, 2)

    # aliases
    compute_AA = compute_kappa_matrix

    def compute_kappa_vector(self, tloc: ArrayLike) -> np.ndarray:
        """
        **Description:**
        Computes the vector kappa_{abc} t^b t^c at a location in the Kahler cone, as `CalabiYau.compute_kappa_vector`
        of CYTools, i.e. twice the basis divisor volumes of `compute_divisor_volumes`. Only for threefolds.
        **Arguments:**
        - `tloc (ArrayLike)`: A vector specifying a location in the Kahler cone, in the coordinates t^a dual to the rows
          of `glsm`.
        **Raises:**
        - `NotImplementedError`: Raised if dim Y is not 3.
        - `ValueError`: Raised if `tloc` is not a vector of length N - dim X.
        **Returns:**
        - `np.ndarray`: The vector kappa_{abc} t^b t^c at the specified location.
        """
        AA = self.compute_kappa_matrix(tloc)
        return AA.dot(tloc)

    def compute_inverse_kahler_metric(self, tloc: ArrayLike) -> np.ndarray:
        """
        **Description:**
        Computes the inverse Kahler metric at a location in the Kahler cone, as
        `CalabiYau.compute_inverse_kahler_metric` of CYTools: K^{ab} = 4 (tau_a tau_b - V kappa_{abc} t^c) with the
        volume V of `compute_cy_volume`, the basis divisor volumes tau_a of `compute_divisor_volumes` and the matrix of
        `compute_kappa_matrix`. This is the inverse of the metric d^2 K / dT_a d(conj T_b) of the Kahler potential
        K = -2 log V in the coordinates T_a = tau_a + i theta_a. Only for threefolds.
        **Arguments:**
        - `tloc (ArrayLike)`: A vector specifying a location in the Kahler cone, in the coordinates t^a dual to the rows
          of `glsm`.
        **Raises:**
        - `NotImplementedError`: Raised if dim Y is not 3.
        - `ValueError`: Raised if `tloc` is not a vector of length N - dim X.
        **Returns:**
        - `np.ndarray`: The (N - dim X, N - dim X) inverse Kahler metric at the specified location.
        """
        if self.dim() != 3:
            raise NotImplementedError("This function only supports Calabi-Yau 3-folds.")
        xvol = self.compute_cy_volume(tloc)
        Tau = self.compute_divisor_volumes(tloc, in_basis=True)
        AA = self.compute_AA(tloc)
        Kinv = 4 * (np.outer(Tau, Tau) - AA * xvol)
        return Kinv

    def compute_kahler_metric(self, tloc: ArrayLike) -> np.ndarray:
        """
        **Description:**
        Computes the Kahler metric at a location in the Kahler cone, the matrix inverse of
        `compute_inverse_kahler_metric`, as `CalabiYau.compute_kahler_metric` of CYTools. Only for threefolds. The
        inverse Kahler metric is singular, and has no inverse, if the basis divisors J_a do not restrict to linearly
        independent classes on Y (e.g. if a prime toric divisor does not meet Y, which `keep_all` allows).
        **Arguments:**
        - `tloc (ArrayLike)`: A vector specifying a location in the Kahler cone, in the coordinates t^a dual to the rows
          of `glsm`.
        **Raises:**
        - `NotImplementedError`: Raised if dim Y is not 3.
        - `ValueError`: Raised if `tloc` is not a vector of length N - dim X.
        - `np.linalg.LinAlgError`: Raised if the inverse Kahler metric is singular.
        **Returns:**
        - `np.ndarray`: The (N - dim X, N - dim X) Kahler metric at the specified location.
        """
        return np.linalg.inv(self.compute_inverse_kahler_metric(tloc))

    # Hodge numbers
    # -------------
    def e_polynomial(self, backend: str = "auto") -> np.ndarray:
        """
        **Description:**
        Hodge-Deligne polynomial E(Y;u,v) = sum_{p,q} e^{p,q} u^p v^q of the generic complete intersection Y, computed as
        in `HN.e_polynomial_generic_hypersurface` by summing over the torus orbits O(s) of the fan `fan`: on
        O(s) = (C*)^k the section of D^(i) restricts to a Laurent polynomial with Newton polytope the face F^(i)_s of the
        Newton polytope of D^(i) on which <m, v_j> = -a^(i)_j for all rays v_j of s. The sections with empty F^(i)_s
        vanish identically on O(s) and are dropped; if a remaining F^(i)_s is a point, Y misses O(s); if none remain,
        O(s) lies in Y and contributes (uv-1)^k. Otherwise Y ∩ O(s) is a nondegenerate complete intersection of t
        Laurent polynomials, whose E-polynomial follows from the Cayley trick: the hypersurface
        Y~ = {sum_i lambda_i f_i = 0} in (C*)^k x P^(t-1) fibres over (C*)^k with fibre P^(t-1) over Y ∩ O(s) and
        P^(t-2) elsewhere, so
            E(Y ∩ O(s)) = ( E(Y~) - (uv-1)^k [P^(t-2)] ) / (uv)^(t-1),
        and E(Y~) is the sum over the non-empty subsets I of the t polynomials of the E-polynomials of the nondegenerate
        hypersurfaces {sum_{i in I} lambda_i f_i = 0} in (C*)^k x (C*)^(|I|-1), whose Newton polytopes are the faces
        Cay(F^(i)_s : i in I) of the Cayley polytope of the Newton polytopes (`HN._face_e_polynomials`). For r = 1 this
        is exactly the hypersurface computation. The result is cached.
        **Arguments:**
        - `backend (str)`: How the h*-polynomials of the faces are obtained, "faces" (face by face, without enumerating
          dilations), "normaliz", "cytools" or "auto" ("faces" if PyNormaliz is installed, see `HN._resolve_backend`). Only used
          on the first call. Defaults to `"auto"`.
        **Raises:**
        - `ValueError`: Raised if a defining divisor has no global sections.
        **Returns:**
        - `np.ndarray`: e[p,q] for 0 <= p,q <= dim Y, as an object array of Python integers.
        """
        if self.__e_polynomial is None:
            backend = HN._resolve_backend(backend)
            fan = self.fan()
            piece, on_cay, face_e = _cayley_face_data(fan, self.__divisors, backend)
            n, r = self.__n, self.__r
            size = n + r + 1
            e = np.zeros((size, size), dtype=object)
            cones_by_dim = [np.zeros((1, 0), dtype=int)] + UF._cones_by_dimension(fan)
            for cones in cones_by_dim:
                k = n - cones.shape[1]
                masks = on_cay[cones].all(axis=1)
                for mask, count in zip(*np.unique(masks, axis=0, return_counts=True)):
                    e += int(count) * _stratum_e_polynomial(mask, k, piece, r, face_e, size)
            d = n - r
            if e[d + 1:, :].any() or e[:, d + 1:].any():
                raise RuntimeError(f"The E-polynomial has terms of degree > dim Y:\n{e}")
            self.__e_polynomial = e[:d + 1, :d + 1]
        return self.__e_polynomial.copy()

    def hodge_numbers_nonstringy(self, backend: str = "auto", check: bool = True) -> np.ndarray:
        """
        **Description:**
        Hodge diamond h[p,q] = h^{p,q}(Y) of the generic complete intersection, h^{p,q} = (-1)^(p+q) e^{p,q} from
        `e_polynomial`. This requires the cohomology of Y to be pure, i.e. Y compact with at most quotient
        singularities, which holds for a generic complete intersection of base-point-free (`is_nef`) divisors in the
        toric variety of a complete simplicial fan. This is the method of `HN.hodge_numbers_generic_hypersurface`
        extended to any codimension; the result is cached, so `backend` and `check` only matter on the first call.
        **Arguments:**
        - `backend (str)`: See `e_polynomial`. Defaults to `"auto"`.
        - `check (bool)`: Whether to verify that the fan is complete and that the Hodge numbers are non-negative and
          satisfy h^{p,q} = h^{q,p} and h^{p,q} = h^{d-p,d-q} (necessary for purity, not sufficient). Defaults to `True`.
        **Raises:**
        - `ValueError`: Raised (with `check`) if the fan is not complete or the E-polynomial is not that of a pure Hodge
          structure.
        **Returns:**
        - `np.ndarray`: The integer Hodge diamond, shape (dim Y + 1, dim Y + 1).
        """
        if self.__hodge_numbers_nonstringy is None:
            if check:
                HN._check_complete(self.fan())
            self.__hodge_numbers_nonstringy = HN._pure_hodge_numbers(self.e_polynomial(backend), check)
        return self.__hodge_numbers_nonstringy.copy()

    def hodge_numbers(self, backend: str = "auto") -> np.ndarray:
        """
        **Description:**
        Hodge diamond from the stringy E-function of the Cayley polytope of the Newton polytopes of the defining
        divisors (Batyrev-Borisov), as `FT_CY.F_Theory_Uplift.hodge_numbers_CY`: `UF.hodge_diamond_cy4` for a
        two-part nef partition in six dimensions and `UF.hodge_diamond_stringy` otherwise. This requires the Cayley
        polytope to be a reflexive Gorenstein support, e.g. the Newton polytopes to form a nef partition, and then gives
        the Hodge numbers of the (crepant resolution of the) complete intersection. No fan is needed. The result is
        cached. If the mirror complete intersection is known (`_link_mirror`) and its Hodge diamond has already been
        computed, the diamond is read off from it, h^{p,q}(Y) = h^{d-p,q}(mirror), after checking that the two Cayley
        cones are dual to each other.
        **Arguments:**
        - `backend (str)`: How the h*-polynomials of the faces of the Cayley polytope are obtained, "auto", "faces",
          "normaliz", "cytools" or "latte" (see `UF.hodge_diamond_stringy`): "faces" computes them face by face without
          enumerating dilations of the Cayley polytope, and is what "auto" uses if PyNormaliz is installed, except for
          a two-part nef partition in six dimensions, where "auto" uses the closed formulas of `UF.hodge_diamond_cy4`.
          Only used on the first call. Defaults to `"auto"`.
        **Raises:**
        - `ValueError`: Raised if the Cayley polytope is not a reflexive Gorenstein support.
        **Returns:**
        - `np.ndarray`: The integer Hodge diamond, shape (dim Y + 1, dim Y + 1).
        """
        if self.__hodge_numbers_stringy is None:
            self.__hodge_numbers_stringy = self.__hodge_numbers_from_mirror()
        if self.__hodge_numbers_stringy is None:
            cayley = self.Cayley_polytope()
            vertices = np.asarray(cayley.vertices(), dtype=int)
            diamond = UF.hodge_diamond_cy4 if (self.__n == 6 and self.__r == 2 and backend != "faces") else UF.hodge_diamond_stringy
            self.__hodge_numbers_stringy = diamond(vertices, backend)
        return self.__hodge_numbers_stringy.copy()

    def _link_mirror(self, mirror: CICY) -> None:
        """
        **Description:**
        Registers `mirror` as the mirror complete intersection of Y (the dual nef partition) and Y as the mirror of
        `mirror`, so that the stringy Hodge diamond is computed only once for the pair (see `hodge_numbers`). Nothing is
        assumed about `mirror` here: the duality of the two Cayley cones is checked before a diamond is taken over.
        **Arguments:**
        - `mirror (CICY)`: The mirror complete intersection.
        """
        self.__mirror = mirror
        mirror.__mirror = self

    def __hodge_numbers_from_mirror(self) -> np.ndarray | None:
        """
        **Description:**
        The stringy Hodge diamond of Y from that of the linked mirror, h^{p,q}(Y) = h^{d-p,q}(mirror), if the mirror
        has one cached and the vertices of the Cayley polytope of Y are exactly the primitive generators of the dual of
        the Cayley cone of the mirror (in the dual coordinates, with the natural pairing of M x Z^r and N x Z^r). The
        stringy E-function of `UF.hodge_diamond_stringy` is symmetric under exchanging a Gorenstein cone with its dual
        up to this reflection, so no further assumption enters.
        **Returns:**
        - `np.ndarray | None`: The Hodge diamond, or `None` if it cannot be taken from a mirror.
        """
        mirror = self.__mirror
        if mirror is None or mirror.__hodge_numbers_stringy is None:
            return None
        try:
            own = np.asarray(self.Cayley_polytope().vertices(), dtype=int)
            _, dual, _, _ = UF._hd_setup(np.asarray(mirror.Cayley_polytope().vertices(), dtype=int), "cytools")
        except ValueError:
            return None
        if own.shape[1] != dual.shape[1] or set(map(tuple, own.tolist())) != set(map(tuple, dual.tolist())):
            return None
        return mirror.__hodge_numbers_stringy[::-1, :].copy()

    def Cayley_polytope(self) -> Polytope:
        if self.__Cayley_Polytope is None:
            self.__Cayley_Polytope = UF.cayley_polytope([P.vertices() for P in self.newton_polytopes()])
        return self.__Cayley_Polytope

def _partition_representatives(V: np.ndarray, divisors: np.ndarray, lattice_basis: np.ndarray) -> np.ndarray | None:
    """
    **Description:**
    Shifts divisors a^(1), ..., a^(r) with a^(1) + ... + a^(r) = (1, ..., 1) by principal divisors to a partition, i.e.
    representatives b^(i) = a^(i) + V m_i with coefficients 0 and 1 and b^(1) + ... + b^(r) = (1, ..., 1), if possible.
    A shift is determined by the values of b^(i) on dim X linearly independent rays, so each divisor has at most
    2^dim X representatives with coefficients 0 and 1 (as in `UF.divisor_in_zero_one_rep`); the rays need not form a
    lattice basis, since for a representative with integral shift m its values give back m exactly, and a rounded
    non-integral shift is only kept if it gives coefficients 0 and 1. Among them a combination
    with disjoint supports covering all rays is searched. The given divisors are tried first, so divisors that already
    form a partition are returned unchanged (as `UF.divisor_in_zero_one_rep` returns a divisor with coefficients 0 and 1
    unchanged), and otherwise those that already have coefficients 0 and 1 are kept whenever possible.
    **Arguments:**
    - `V (np.ndarray)`: The rays, one per row.
    - `divisors (np.ndarray)`: The divisor coefficient vectors, one per row, summing to (1, ..., 1).
    - `lattice_basis (np.ndarray)`: Zero-based indices of dim X linearly independent rays.
    **Returns:**
    - `np.ndarray | None`: The (r, N) array of partition representatives, or `None` if there is none.
    """
    n = V.shape[1]
    inverse = np.linalg.inv(V[lattice_basis].astype(float))
    candidates = []
    for a in divisors:
        representatives = []
        for values in product((0, 1), repeat=n):
            m = np.rint(inverse @ (np.array(values) - a[lattice_basis])).astype(int)
            b = a + V @ m
            if np.all((b == 0) | (b == 1)):
                representatives.append(b)
        if not representatives:
            return None
        # the given divisor first (if it has coefficients 0 and 1), so that a given partition is returned unchanged
        representatives.sort(key=lambda b: not np.array_equal(b, a))
        candidates.append(representatives)

    def search(i: int, total: np.ndarray) -> list[np.ndarray] | None:
        if i == len(candidates):
            return [] if np.all(total == 1) else None
        for b in candidates[i]:
            if np.all(total + b <= 1):
                rest = search(i + 1, total + b)
                if rest is not None:
                    return [b] + rest
        return None

    solution = search(0, np.zeros(V.shape[0], dtype=int))
    return None if solution is None else np.array(solution, dtype=int)


def _cayley_face_data(fan: Fan, divisors: np.ndarray, backend: str) -> tuple[np.ndarray, np.ndarray, dict[frozenset[int], tuple[int, np.ndarray]]]:
    """
    **Description:**
    The Cayley polytope Cay = conv(Delta_1 x {e_1}, ..., Delta_r x {e_r}) of the Newton polytopes Delta_i of the
    divisors, the ray-vertex incidences of its vertices and the E-polynomials of the nondegenerate torus hypersurfaces
    of all its faces (`HN._face_e_polynomials`), keyed by the vertex sets of the faces. The vertices of Cay are the
    vertices of the Delta_i, and for a cone s the vertices of Cay on the face Cay(F^(i)_s : i in I) are the vertices of
    the Delta_i, i in I, that lie on F^(i)_s.
    **Arguments:**
    - `fan (Fan)`: The fan; its rays are `fan.vectors()`.
    - `divisors (np.ndarray)`: The divisor coefficient vectors, one per row.
    - `backend (str)`: "faces", "normaliz" or "cytools".
    **Raises:**
    - `ValueError`: Raised if a divisor has no global sections.
    **Returns:**
    - `tuple[np.ndarray, np.ndarray, dict]`: `(piece, on_cay, face_e)`: piece[v] = i for the vertices v of Cay coming
      from Delta_i, on_cay[j, v] = (vertex v lies on the face of Delta_i selected by ray j), and the per-face
      E-polynomials.
    """
    vertex_sets, incidences = [], []
    for a in divisors:
        _, W, on_face = HN._newton_polytope_incidence(fan, a)
        vertex_sets.append(W)
        incidences.append(on_face)
    r = len(vertex_sets)
    E = np.eye(r, dtype=int)
    W_cay = np.vstack([np.column_stack([W, np.tile(E[i], (len(W), 1))]) for i, W in enumerate(vertex_sets)])
    piece = np.concatenate([np.full(len(W), i, dtype=int) for i, W in enumerate(vertex_sets)])
    on_cay = np.concatenate(incidences, axis=1)
    return piece, on_cay, HN._face_e_polynomials(W_cay, backend)


def _stratum_e_polynomial(mask: np.ndarray, k: int, piece: np.ndarray, r: int, face_e: dict[frozenset[int], tuple[int, np.ndarray]], size: int) -> np.ndarray:
    """
    **Description:**
    E-polynomial of Y ∩ O(s) for a torus orbit O(s) of dimension k, see `CICY.e_polynomial`: with the Cayley vertices
    on the faces F^(i)_s given by `mask`, the sections with empty face are dropped, a point face gives 0, no remaining
    section gives (uv-1)^k, and otherwise the Cayley trick is applied to the t remaining sections.
    **Arguments:**
    - `mask (np.ndarray)`: Boolean array over the vertices of the Cayley polytope, True for the vertices on the faces
      F^(i)_s.
    - `k (int)`: The dimension of the orbit.
    - `piece (np.ndarray)`: piece[v] = i for the vertices v coming from Delta_i.
    - `r (int)`: The number of divisors.
    - `face_e (dict)`: The per-face E-polynomials of the Cayley polytope, see `_cayley_face_data`.
    - `size (int)`: The size of the coefficient arrays.
    **Raises:**
    - `RuntimeError`: Raised if a required face of the Cayley polytope is missing or the division by (uv)^(t-1) fails.
    **Returns:**
    - `np.ndarray`: The (size, size) coefficient array of E(Y ∩ O(s)).
    """
    zero = np.zeros((size, size), dtype=object)
    counts = np.bincount(piece[mask], minlength=r)
    alive = np.flatnonzero(counts).tolist()
    if not alive:
        return HN._uv_minus_one_power(k, size)
    if (counts[alive] == 1).any():
        return zero
    t = len(alive)
    e = zero.copy()
    for s in range(1, t + 1):
        for I in combinations(alive, s):
            key = frozenset(np.flatnonzero(mask & np.isin(piece, I)).tolist())
            if key not in face_e:
                raise RuntimeError(f"The vertex set {sorted(key)} is not a face of the Cayley polytope.")
            dim_G, E_G = face_e[key]
            torus_dim = k + s - 1
            if dim_G > torus_dim:
                raise RuntimeError(f"A face of dimension {dim_G} in a torus of dimension {torus_dim}.")
            padded = zero.copy()
            padded[:E_G.shape[0], :E_G.shape[1]] = E_G
            e += HN._poly_mul(padded, HN._uv_minus_one_power(torus_dim - dim_G, size))
    projective = zero.copy()
    for j in range(t - 1):
        projective[j, j] = 1
    e -= HN._poly_mul(HN._uv_minus_one_power(k, size), projective)
    shift = t - 1
    if shift:
        if e[:shift, :].any() or e[:, :shift].any():
            raise RuntimeError(f"The Cayley-trick E-polynomial is not divisible by (uv)^{shift}:\n{e}")
        divided = zero.copy()
        divided[:size - shift, :size - shift] = e[shift:, shift:]
        e = divided
    return e

    