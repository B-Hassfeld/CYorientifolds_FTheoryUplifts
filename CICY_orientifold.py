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
# Description:  This module contains tools designed for orientifolds of
#               Calabi-Yau complete intersections of any codimension in toric
#               varieties, generalizing `FT_CY.CY_orientifold` (codimension 1),
#               and for their F-theory uplifts, generalizing
#               `FT_CY.F_Theory_Uplift`.
# -----------------------------------------------------------------------------


from __future__ import annotations

# 'standard' imports
import warnings
from functools import reduce

# 3rd party imports
import numpy as np

# CYTools imports
from cytools import Polytope
from cytools.vector_config import VectorConfiguration
from cytools.vector_config.fan import Fan
from . import CICY_orientifold_functions as COF
from .CICY import CICY
from cygv import compute_gv
from numpy.typing import ArrayLike

__all__ = [
    "CICY_orientifold",
    "CICY_F_Theory_Uplift",
]


class CICY_orientifold():
    """
    This class describes an O3/O7 orientifold of a Calabi-Yau complete
    intersection in a toric ambient variety.
    """


    def __init__(self, cicy_or_points: CICY | Polytope | VectorConfiguration | Fan | np.ndarray | list[list[int]], xi: ArrayLike | None = None, resolve_A1_singularities: bool = True, construct_nef_decomposition: bool = True, parities: ArrayLike | None = None, divisors: ArrayLike | None = None, keep_all: bool = False) -> None:
        """
        **Description:**

        Constructs a `CICY_orientifold` object describing an O3/O7 orientifold of a Calabi-Yau complete intersection X = D^(1) ∩ ... ∩ D^(r) in a toric ambient variety, given as a `CICY` or by the inputs of a `CICY`, from which the `CICY` is constructed first. This generalizes `FT_CY.CY_orientifold`, which is the case r = 1. Unless the CICY has a fan, the rays are all nonzero lattice points of the polytope of the CICY (`CICY.polytope`), as for a `Polytope` given to `CY_orientifold`: the rays of the CICY (`CICY.vectors_ambient`) first, with their labels, followed by the remaining points, on which D^(1), ..., D^(r-1) get the support functions of their Newton polytopes and D^(r) the coefficient that makes the sum (1, ..., 1). The rays are orbifolded by the `Z_2` action `xi`, and the orientifold divisors L_1, ..., L_r on the orbifold, whose generic sections cut out B = X/sigma, are determined with `COF.O3O7_line_bundles`. Optionally, the construction resolves A1 singularities and attempts to refine the toric data so that L_1, ..., L_r and -6K_B = 6(1 - L_1 - ... - L_r) become Cartier and nef.

        **Arguments:**

        - `cicy_or_points (CICY | Polytope | VectorConfiguration | Fan | np.ndarray | list[list[int]])`: The Calabi-Yau complete intersection as a `CICY`, or the toric data of one (a `Polytope`, `VectorConfiguration`, `Fan`, or the rays as an array or list), from which `CICY(cicy_or_points, divisors, keep_all)` is constructed. If the CICY has a fan (`CICY.has_fan`, e.g. if a `Fan` is given), this fan is used as the ambient triangulation, as a `Fan` given to `CY_orientifold`, and only its rays are used.
        - `xi (ArrayLike | None)`: The integer vector `xi` of the orientifold action (the `Z_2` action is defined by `xi/2`). If the polytope of the CICY (`CICY.polytope`) is trilayer, it may be `None` and is then inferred automatically. Defaults to `None`.
        - `resolve_A1_singularities (bool)`: Whether to resolve A1 singularities in the toric orbifold (only done if the CICY has a fan and no nef decomposition was constructed). Defaults to `True`.
        - `construct_nef_decomposition (bool)`: Whether to attempt to construct a Cartier and nef decomposition by refining the orbifold data. Defaults to `True`.
        - `parities (ArrayLike | None)`: The parities eps_a in {0, 1} under the orientifold action of the monomials of the r defining equations, with odd sum (see `COF.O3O7_line_bundles`). For r = 1 the only choice is (1). For r > 1 different choices give different orientifolds. If `None`, the first regular choice is used, with a warning if there are several. Defaults to `None`.

        - `divisors (ArrayLike | None)`: The defining divisors of the CICY, one coefficient vector per row, if toric data are given instead of a `CICY`; `None` then gives the anticanonical hypersurface (see `CICY`). Must be `None` if a `CICY` is given. Defaults to `None`.
        - `keep_all (bool)`: Whether the CICY keeps all given rays instead of only those whose divisors intersect it (see `CICY`). Only used if toric data are given instead of a `CICY`. It only affects `cicy()` and the order of the rays, since the orientifold uses all nonzero points of the polytope anyway. Defaults to `False`.

        **Raises:**

        - `ValueError`: Raised if `xi` is required but not provided, if `parities` is invalid, if `divisors` are given together with a `CICY`, or by `CICY` for invalid toric data.
        - `TypeError`: Raised by `CICY` if the toric data have an unsupported type.

        **Returns:**

        - `None`

        """
        self.__p = None
        self.__cicy = None
        self.__toric_fan = None
        self.__xi = None
        self._original_verts=None
        self.__orbifold_toric_fan = None
        self.__n_separating_rays = 0
        self.__orbifold_divisors = None
        self.__parities = None
        self.__yields_nef_decomposition = None
        self.__yields_partition = None
        self.__regular = None
        self.__orbifold_pts = None
        self.__CY_ambient_pts=None
        self.__intersection_numbers_orbifold = None
        self.__normal_fan = None
        self.__Newton_Polytopes = None
        self.__Newton_Polytopes_untranslated = None
        self.__NHC_labels = None
        self.ambient_triangulation = False
        self._multiplier = 6
        self.__o_planes = None
        self.__fixed_locus = {}
        self.__hodge_numbers_weighted=None
        self.__h_prime_is_lattice=None
        self.__is_trilayer_orientifold=None
        match cicy_or_points:
            case CICY():
                if divisors is not None:
                    raise ValueError("Divisors can only be given together with toric data, not with a CICY, which already has divisors")
                cicy = cicy_or_points
            case object():
                cicy = CICY(cicy_or_points, divisors, keep_all=keep_all)
        if xi is None:
            if cicy.polytope().is_trilayer():
                xi = -COF.find_trilayer_vertex(cicy.polytope())
            else:
                raise ValueError("If the polytope of the CICY is not trilayer, xi has to be defined")
        self.__cicy = cicy
        self.__p = cicy.polytope()
        self._original_verts=self.__p.vertices()
        pts = cicy.vectors_ambient()
        divisors = cicy.divisors()
        # the Newton polytopes of `divisors` on `pts` that the CICY already has (`None` for those it does not have): they are passed to `COF.O3O7_line_bundles`, which constructs the others
        polytopes_CY = None
        if cicy.has_fan():
            self.__toric_fan = cicy.fan()
            self.ambient_triangulation = True
            polytopes_CY = cicy.newton_polytopes()
        else:
            # as `CY_orientifold` for a `Polytope`, the orientifold is constructed from all nonzero points of the polytope of the CICY: rays the CICY dropped miss X, but their constraints on the Newton polytope of -6K_B on the orbifold are not implied by the others. The rays of the CICY come first, so their labels are kept
            all_pts = np.asarray(self.__p.points(), dtype=int)
            extra = COF.row_difference(all_pts[all_pts.any(axis=1)], pts)
            # on the new rays D^(1), ..., D^(r-1) get the support functions of their Newton polytopes, so that their sections do not change, and D^(r) is fixed by the sum (1, ..., 1); for a nef partition of a reflexive polytope this is its support function as well, and for r = 1 it is 1
            if len(extra) == 0:
                polytopes_CY = cicy.newton_polytopes()
            else:
                # the new rays do not change the Newton polytopes of D^(1), ..., D^(r-1), on which they have the support functions, but in general that of D^(r). For r = 1, D^(1) = (1, ..., 1) and the new rays do not change its Newton polytope if the rays of the CICY contain the vertices of its polytope, whose inequalities imply those of all its other points
                last_unchanged = len(divisors) == 1 and bool((divisors == 1).all()) and len(COF.row_difference(np.asarray(self._original_verts, dtype=int), pts)) == 0
                polytopes_cicy = cicy.newton_polytopes() if len(divisors) > 1 or last_unchanged else None
                monomials = [np.asarray(P.points(), dtype=int).reshape(-1, pts.shape[1]) for P in polytopes_cicy[:-1]] if polytopes_cicy is not None else []
                if all(len(m) > 0 for m in monomials):
                    extra_divisors = np.array([np.max(-(extra @ m.T), axis=1) for m in monomials], dtype=int).reshape(len(monomials), len(extra))
                    extra_divisors = np.vstack([extra_divisors, 1 - extra_divisors.sum(axis=0)])
                    pts = np.vstack([pts, extra])
                    divisors = np.hstack([divisors, extra_divisors])
                    if polytopes_cicy is not None:
                        polytopes_CY = polytopes_cicy[:-1] + [polytopes_cicy[-1] if last_unchanged else None]
                else:
                    polytopes_CY = polytopes_cicy
        self.__CY_ambient_pts = pts
        self.__dim = np.linalg.matrix_rank(pts,tol=1e-6)
        self.__xi = xi

        orbifold_pts, rescalings, Lambda = COF.toric_orbifold(pts_CY_ambient=pts, q=xi)
        # the regularity test of the chosen divisors and their Newton polytopes on the orbifold rays are those `COF.O3O7_line_bundles` has computed to choose them
        orbifold_divisors, self.__parities, degenerate, NPs_orbifold = COF.O3O7_line_bundles(pts_CY_ambient=pts, divisors=divisors, q=xi, rescalings=rescalings, orbifold_points=orbifold_pts, parities=parities, polytopes=polytopes_CY, return_polytopes=True)
        self.__regular = orbifold_divisors is not None and not degenerate
        orbifold_pts_unrefined, orbifold_divisors_unrefined = orbifold_pts, orbifold_divisors
        partition_shifts = None

        if self.__regular:
            if construct_nef_decomposition:
                self.__yields_nef_decomposition=False
                KB_multiplier = COF.Newton_Polytope(orbifold_pts, self._multiplier * (np.ones(len(orbifold_pts), dtype=int) - orbifold_divisors.sum(axis=0)))
                if len(KB_multiplier.points()) > 0:
                    if NPs_orbifold is None:
                        NPs_orbifold = [COF.Newton_Polytope(orbifold_pts, L) for L in orbifold_divisors]
                    self.__Newton_Polytopes = NPs_orbifold
                    # the normal fan of the Minkowski sum refines the normal fans of all Newton polytopes, and its support function is the sum of theirs, so COF.normal_fan treats the sum like the single Newton polytope of a hypersurface
                    NP_sum = reduce(lambda P, Q: P.minkowski_sum(Q), NPs_orbifold)
                    if NP_sum.minkowski_sum(KB_multiplier).dim() == self.__dim:
                        orbifold_pts_refined, refined_weights, normal_fan = COF.normal_fan([NP_sum, KB_multiplier], np.array([self._multiplier, 1, self._multiplier]), maximal_refinement=True, triangulate_refinement=False, return_unrefined_fan=True)
                        if orbifold_pts_refined is not None:
                            self.__yields_nef_decomposition = True
                            self.__normal_fan = normal_fan
                            # the weights of the individual divisors on the refined rays are the support functions of their Newton polytopes
                            refined_divisors = np.array([COF.linebundle_weights_from_Newton_Polytope(orbifold_pts_refined, P) for P in NPs_orbifold])
                            if (refined_divisors.sum(axis=0) != refined_weights[:, 0]).any():
                                raise RuntimeError("The support functions of the Newton polytopes do not add up to that of their Minkowski sum.")
                            # the rays of the unrefined normal fan (the first rows) alone cut out all Newton polytopes, so all other non-intersecting rays can be dropped
                            inds=np.union1d(COF.intersecting_points(orbifold_pts_refined,polytopes=NPs_orbifold,as_indices=True),np.arange(len(normal_fan.vectors())))
                            # except the blowups of D_v cap D_v' for NHC rays v,v': they miss B as well, but are kept (appended at the end) so that NHC divisors do not meet in the ambient
                            nhc_refined=np.flatnonzero(self._multiplier*(1-refined_weights[:,0])-refined_weights[:,1]==3)
                            dropped=np.setdiff1d(np.arange(len(orbifold_pts_refined)),np.union1d(inds,nhc_refined))
                            separating=COF.NHC_separating_rays(orbifold_pts_refined,nhc_refined,dropped)
                            self.__n_separating_rays=len(separating)
                            inds=np.concatenate([inds,separating])
                            orbifold_pts_refined=orbifold_pts_refined[inds]
                            refined_weights=refined_weights[inds]
                            Lambda_inv=np.linalg.inv(Lambda)
                            self.__CY_ambient_pts=COF.make_rows_integer((Lambda_inv@orbifold_pts_refined.T).T)
                            self.__p=None
                            self.__toric_fan=None
                            orbifold_pts = orbifold_pts_refined
                            orbifold_divisors = refined_divisors[:, inds]
                            self.ambient_triangulation=False
                            self.__NHC_labels = np.where(self._multiplier*(1-orbifold_divisors.sum(axis=0))-refined_weights[:, 1]==3)[0]+1
            # L_1, ..., L_r and -K_B = 1 - L_1 - ... - L_r form a partition of the anticanonical divisor if possible
            partition, shifts = COF.divisors_in_partition_rep(orbifold_pts, np.vstack([orbifold_divisors, 1-orbifold_divisors.sum(axis=0)]))
            if partition is not None:
                orbifold_divisors = partition[:-1]
                partition_shifts = shifts[:-1]
                if self.__Newton_Polytopes is not None:
                    self.__Newton_Polytopes = [Polytope(P.vertices()-m) for P, m in zip(self.__Newton_Polytopes, shifts[:-1])]
                self.__yields_partition=True
            else:
                self.__yields_partition=False
        else:
            self.__yields_nef_decomposition=False
            self.__yields_partition=False

        self.__orbifold_pts = orbifold_pts
        self.__orbifold_divisors = orbifold_divisors
        if resolve_A1_singularities:
            if self.ambient_triangulation:
                self.__resolve_A1_singularities()
        # if no Newton polytopes were stored above and the orbifold rays have not changed (refinement, A1 blowups), those `COF.O3O7_line_bundles` has constructed are the ones `Newton_Polytopes` would construct, or, if the divisors were replaced by the partition, their translates by the shifts, which `Newton_Polytopes` then computes when first called
        if self.__Newton_Polytopes is None and NPs_orbifold is not None and self.__orbifold_pts is orbifold_pts_unrefined:
            if np.array_equal(self.__orbifold_divisors, orbifold_divisors_unrefined):
                self.__Newton_Polytopes = NPs_orbifold
            elif partition_shifts is not None:
                self.__Newton_Polytopes_untranslated = (NPs_orbifold, partition_shifts)

    def __repr__(self) -> str:
        """
        **Description:**

        Produces a short human-readable representation of the object.

        **Returns:**

        - `str`: A description of the object.

        """
        return "A Calabi-Yau complete intersection orientifold"

    def __resolve_A1_singularities(self) -> None:

        """
        **Description:**

        Finds codimension-two `Z_2` fixed loci in the Calabi-Yau ambient toric fan and refines the orbifold fan by adding the corresponding blowup rays. The orbifold divisors are updated by adding the weights of the proper transforms on the exceptional divisors: at a fixed 2-cone (i, j) the exponents of x_i x_j in all sections of L_a have the parity of L_a,i + L_a,j, so a generic section vanishes to order (L_a,i + L_a,j) mod 2 there and its proper transform has the weight floor((L_a,i + L_a,j)/2) (for r = 1 always (L_i + L_j - 1)/2, as in `CY_orientifold`).

        **Returns:**

        - `None`

        """

        singular_two_cones = COF.Z2_fixed_locus(self.CY_ambient_toric_fan(),self.xi(),cone_dimension=2,denominator=2)
        orbifold_blowups=[]
        if len(singular_two_cones)>0:
            for c in singular_two_cones:
                vec=np.sum(self.vectors_orbifold(c),axis=0)
                vec_add=np.rint(vec/np.gcd.reduce(vec)).astype(int)
                orbifold_blowups.append(vec_add.tolist())
            orbifold_blowups=np.array(orbifold_blowups)
            fan=COF.refine_fan(self.orbifold_toric_fan(),orbifold_blowups)
            self.__orbifold_pts=fan.vectors()
            if self.divisors() is not None:
                blow_up_weights = np.array([self.divisors()[:, np.array(c)-1].sum(axis=1)//2 for c in singular_two_cones]).T
                self.__orbifold_divisors=np.concatenate((self.divisors(),blow_up_weights),axis=1)
            self.__orbifold_toric_fan=fan
        return None

    def orbifold_toric_fan(self) -> Fan:
        """
        **Description:**

        Constructs the toric fan associated with the orbifold vector configuration if it has not already been cached. If a nef decomposition was constructed, the fan is obtained from the corresponding normal fan refinement: the rays are inserted by stellar subdivisions, and the rays separating NHC divisors (the last rows of `vectors_orbifold()`, see `COF.NHC_separating_rays`) are inserted last, so that no two NHC divisors meet and the cones meeting the orientifold base are not changed by them. Otherwise, it uses the cone structure inherited from the Calabi-Yau ambient toric fan.

        **Raises:**

        - `ValueError`: Raised if two NHC divisors still meet in the refined normal fan.

        **Returns:**

        - `Fan`: The toric fan of the orbifold ambient variety.

        """
        if self.__orbifold_toric_fan is None:
            if self.ambient_triangulation:
                self.__orbifold_toric_fan=Fan(vc=VectorConfiguration(self.vectors_orbifold()),cones=self.CY_ambient_toric_fan().cones())
            else:
                if self.yields_nef_decomposition():
                    nf=COF.make_simplicial(self.normal_fan())
                    VC=VectorConfiguration(self.vectors_orbifold())
                    inds=VC.vectors_to_labels(nf.vectors())
                    new_cones=set()
                    for c in nf.cones():
                        nc=tuple(sorted([inds[x-1] for x in c]))
                        new_cones.add(nc)
                    V=self.vectors_orbifold()
                    n=len(V)-self.__n_separating_rays
                    fan=COF.refine_fan(Fan(vc=VectorConfiguration(V[:n]),cones=new_cones))
                    if n<len(V):
                        fan=COF.reorder_fan(COF.refine_fan(fan,V[n:]),V)
                    NHC=set(int(l) for l in self.NHC(as_labels=True))
                    touching=[c for c in COF.get_lower_dimensional_cones(fan,2) if all(int(l) in NHC for l in c)]
                    if touching:
                        raise ValueError(f"The NHC divisors of the cones {touching[:5]} still meet in the refined normal fan.")
                    self.__orbifold_toric_fan = fan
                else:
                    self.__orbifold_toric_fan = Fan(VectorConfiguration(self.vectors_orbifold()),cones=self.CY_ambient_toric_fan().cones())
        return self.__orbifold_toric_fan

    def CY_ambient_toric_fan(self) -> Fan:
        """
        **Description:**

        Returns the toric fan before orbifolding. If the CICY had no fan at initialization, one is constructed: from the cones of the orbifold toric fan if the construction yields a nef decomposition, and otherwise by triangulating the ambient vector configuration.

        **Returns:**

        - `Fan`: The toric fan of the original Calabi-Yau ambient variety.

        """
        if self.__toric_fan is None:
            if self.yields_nef_decomposition():
                self.__toric_fan = Fan(vc=VectorConfiguration(self.vectors_CY_ambient()),cones=self.orbifold_toric_fan().cones())
            else:
                self.__toric_fan = VectorConfiguration(self.vectors_CY_ambient()).triangulate()
        return self.__toric_fan

    def hodge_numbers_weighted(self, backend: str = "auto", method: str = "auto") -> dict[str, int]:
        """
        **Description:**

        Computes and caches the orientifold-even and -odd Hodge numbers `h^{1,1}_+`, `h^{1,1}_-`, `h^{2,1}_+`, `h^{2,1}_-` of the Calabi-Yau threefold complete intersection via `COF.hodge_numbers_weighted`: h^{p,q}_+ = h^{p,q}(B) and h^{p,q}_- = h^{p,q}(X) - h^{p,q}(B), both from the orbit stratification of the orbifold. Its checks compare h^{p,q}(X) with the stringy Hodge numbers of the CICY (`CICY.hodge_numbers`), which need a reflexive Gorenstein Cayley polytope. For a hypersurface in a reflexive polytope of dimension 4 the closed formulas of `COF.hodge_numbers_weighted_hypersurface` are used by default instead (see `method`).

        **Arguments:**

        - `backend (str)`: How the h*-polynomials of the faces of the Cayley polytope are obtained, "faces" (face by face, without enumerating dilations), "normaliz", "cytools" or "auto" ("faces" if PyNormaliz is installed, see `COF._resolve_backend`). Only used on the first call, and not by the method "hypersurface". Defaults to `"auto"`.
        - `method (str)`: "hypersurface" uses the closed formulas for hypersurfaces of `COF.hodge_numbers_weighted_hypersurface`, as `FT_CY.CY_orientifold` (needs a single defining equation and a reflexive `polytope()` of dimension 4), "general" the E-polynomials of B and X, and "auto" the method "hypersurface" if it applies and "general" otherwise. Only used on the first call. Defaults to `"auto"`.

        **Raises:**

        - `ValueError`: Raised by `COF.hodge_numbers_weighted`, e.g. if the orientifold is not regular, a check fails, or the method "hypersurface" does not apply.

        **Returns:**

        - `dict[str, int]`: The Hodge numbers with keys `"h11+"`, `"h11-"`, `"h21+"`, `"h21-"`.

        """
        if self.__hodge_numbers_weighted is None:
            self.__hodge_numbers_weighted = COF.hodge_numbers_weighted(self, backend, method=method)
        return self.__hodge_numbers_weighted

    def cicy(self) -> CICY:
        """
        **Description:**

        Returns the `CICY` used to initialize the orientifold, or constructed from the toric data given at initialization. Its rays (`CICY.vectors_ambient`) are the first rows of the rays the orientifold was constructed from, which also contain the other nonzero points of `CICY.polytope` if the CICY has no fan. If the toric data were refined by a nef decomposition, its rays differ from `vectors_CY_ambient()`.

        **Returns:**

        - `CICY`: The Calabi-Yau complete intersection.

        """
        return self.__cicy

    def vectors_orbifold(self, c: ArrayLike | None = None) -> np.ndarray:
        """
        **Description:**

        Returns all orbifold rays, or only those with labels specified by `c`. Labels are one-indexed, following the CYTools convention for toric rays.

        **Arguments:**

        - `c (ArrayLike | None)`: Optional collection of one-indexed ray labels. Defaults to `None` (all rays).

        **Returns:**

        - `np.ndarray`: The selected orbifold vectors, one per row.

        """
        if c is not None:
            return self.__orbifold_pts[np.array(c)-1]
        else:
            return self.__orbifold_pts

    def vectors_CY_ambient(self, c: ArrayLike | None = None) -> np.ndarray:
        """
        **Description:**

        Returns all rays of the toric ambient space before orbifolding, or only those with labels specified by `c`. Labels are one-indexed.

        **Arguments:**

        - `c (ArrayLike | None)`: Optional collection of one-indexed ray labels. Defaults to `None` (all rays).

        **Returns:**

        - `np.ndarray`: The selected ambient-space vectors, one per row.

        """
        if c is not None:
            return self.__CY_ambient_pts[np.array(c)-1]
        else:
            return self.__CY_ambient_pts

    def xi(self) -> ArrayLike:
        """
        **Description:**

        Returns the vector `xi` specifying the toric `Z_2` orientifold action.

        **Returns:**

        - `ArrayLike`: The orientifold action vector, as passed at initialization (an `np.ndarray` if it was inferred from a trilayer polytope).

        """
        return self.__xi

    def parities(self) -> np.ndarray | None:
        """
        **Description:**

        Returns the parities eps_a under the orientifold action of the monomials of the defining equations (see `COF.O3O7_line_bundles`), as passed at initialization or chosen there.

        **Returns:**

        - `np.ndarray | None`: The parities, one per defining divisor, or `None` if there is no projected-in choice.

        """
        return self.__parities

    def is_regular(self) -> bool:
        """
        **Description:**

        Returns whether the generic invariant Calabi-Yau complete intersection descends regularly to the toric orbifold, i.e. whether the generic complete intersection of the orbifold divisors is not degenerate (`COF.generic_sections_degenerate`): every toric coordinate appears in one of the equations and no equation has a fixed component. Unlike for a hypersurface, a single equation need not involve all coordinates.

        **Returns:**

        - `bool`: `True` if the orientifold is regular, otherwise `False`.

        """
        return self.__regular

    def is_trilayer_orientifold(self) -> bool:
        """
        **Description:**

        Returns whether the orientifold is a trilayer orientifold, i.e. whether `polytope()` is trilayer and the orientifold action is the one defined by its trilayer vertex `v` (see `COF.find_trilayer_vertex`). Since the `Z_2` action defined by `xi/2` only depends on `xi` modulo 2, `xi` and `v` are compared modulo 2 (in particular, the default `xi=-v` inferred for trilayer polytopes is accepted).

        **Returns:**

        - `bool`: `True` if the orientifold is a trilayer orientifold, otherwise `False`.

        """
        if self.__is_trilayer_orientifold is None:
            p = self.polytope()
            self.__is_trilayer_orientifold = bool(p.is_trilayer()) and bool(np.all((np.asarray(self.xi(), dtype=int) - COF.find_trilayer_vertex(p)) % 2 == 0))
        return self.__is_trilayer_orientifold


    def divisors(self) -> np.ndarray | None:
        """
        **Description:**

        Returns the toric divisor coefficients of the O3/O7 divisors L_1, ..., L_r on the orbifold ambient space, whose generic sections cut out B = X/sigma. If no projected-in choice exists (`COF.O3O7_line_bundles` returns `None`), the orientifold is not regular and `None` is returned.

        **Returns:**

        - `np.ndarray | None`: The (r, N) orbifold divisor coefficients, one divisor per row, or `None` if the orientifold is not regular.

        """
        return self.__orbifold_divisors

    def dim(self) -> np.integer:
        """
        **Description:**

        Computes the rank of the ambient vector configuration used to define the Calabi-Yau orientifold.

        **Returns:**

        - `np.integer`: The dimension of the toric ambient space (as returned by `np.linalg.matrix_rank`).

        """
        return self.__dim

    def normal_fan(self) -> Fan:
        """
        **Description:**

        Computes and caches the normal fan associated with the relevant Newton polytopes (those of L_1, ..., L_r and of -6K_B) if it has not already been constructed. This fan is used to test or implement the Cartier/nef refinement of the orientifold data.

        **Returns:**

        - `Fan`: The normal fan associated with the Newton-polytope data.

        """
        if self.__normal_fan is None:
            self.__normal_fan = COF.normal_fan([*self.Newton_Polytopes(), COF.Newton_Polytope(self.vectors_orbifold(), self._multiplier*(1-self.divisors().sum(axis=0)))])[0]
        return self.__normal_fan

    def ambient_dim(self) -> int:
        """
        **Description:**

        Returns the number of coordinates of the orbifold vectors.

        **Returns:**

        - `int`: The ambient lattice dimension.

        """
        return len(self.vectors_orbifold()[0])

    def Newton_Polytopes(self) -> list[Polytope]:
        """
        **Description:**

        Computes and caches the Newton polytopes associated with the orbifold vector configuration and the orientifold divisors. The polytopes constructed at initialization for the regularity test are reused: as they are, or translated by the shifts of the divisors to the partition.

        **Returns:**

        - `list[Polytope]`: The Newton polytopes of L_1, ..., L_r (`HPolytope`s, a subclass of `Polytope`, if they were computed from the divisors directly and not translated).

        """
        if self.__Newton_Polytopes is None:
            if self.__Newton_Polytopes_untranslated is not None:
                polytopes, shifts = self.__Newton_Polytopes_untranslated
                self.__Newton_Polytopes = [Polytope(P.vertices()-m) for P, m in zip(polytopes, shifts)]
                self.__Newton_Polytopes_untranslated = None
            else:
                self.__Newton_Polytopes = [COF.Newton_Polytope(self.vectors_orbifold(), L) for L in self.divisors()]
        return self.__Newton_Polytopes

    def yields_nef_decomposition(self) -> bool | np.bool_:
        """
        **Description:**

        Determines whether the relevant divisors (L_1, ..., L_r and -6K_B minus three times the NHC divisors) define a Cartier and nef decomposition. If the CICY had a fan, this is tested directly on the orbifold toric fan. Otherwise, it is checked through the normal-fan refinement data.

        **Returns:**

        - `bool | np.bool_`: `True` if the construction yields a nef decomposition, otherwise `False`.

        """
        if self.__yields_nef_decomposition is None:
            if self.ambient_triangulation:
                L=self.divisors()
                NHC_divisor=np.zeros(L.shape[1])
                NHC_inds=self.NHC(as_labels=True)-1
                NHC_divisor[NHC_inds]=1
                divisor_KB=self._multiplier*(1-L.sum(axis=0))-np.rint((self._multiplier/2)*NHC_divisor).astype(int)
                fan=self.orbifold_toric_fan()
                self.__yields_nef_decomposition = COF.is_Cartier(fan,divisor_KB)[0] and COF.is_nef(fan,divisor_KB) and all(COF.is_Cartier(fan,l)[0] and COF.is_nef(fan,l) for l in L)
            else:
                self.__yields_nef_decomposition = COF.contains_rows(self.vectors_orbifold(),self.normal_fan().vectors())
        return self.__yields_nef_decomposition

    def yields_partition(self) -> bool:
        """
        **Description:**

        Returns `True` if L_1, ..., L_r and -K_B = 1 - L_1 - ... - L_r form a partition of the anticanonical divisor of the orbifold (coefficients 0 and 1, disjoint supports). Otherwise, returns `False`.

        **Returns:**

        - `bool`: `True` if the orientifold yields a partition, otherwise `False`.

        """
        return self.__yields_partition

    def yields_nef_partition(self) -> bool | np.bool_:
        """
        **Description:**

        Returns `True` if the orientifold yields a partition and a Cartier and nef decomposition. Otherwise, returns `False`.

        **Returns:**

        - `bool | np.bool_`: `True` if the orientifold yields a nef partition, otherwise `False`.

        """
        return self.yields_nef_decomposition() and self.yields_partition()

    def h_prime_is_lattice(self) -> bool:
        """
        **Description:**

        Returns whether the Newton polytope of -2K_B = 2(1 - L_1 - ... - L_r) is a lattice polytope (`COF.is_lattice_newton_polytope`).

        **Returns:**

        - `bool`: `True` if it is a non-empty lattice polytope, otherwise `False`.

        """
        if self.__h_prime_is_lattice is None:
            self.__h_prime_is_lattice = COF.is_lattice_newton_polytope(self.vectors_orbifold(),2*(1-self.divisors().sum(axis=0)))
        return self.__h_prime_is_lattice

    def polytope(self) -> Polytope:
        """
        **Description:**

        Returns the polytope of the CICY used to initialize the orientifold (`CICY.polytope`). If the toric data were refined by a nef decomposition, the polytope is constructed as the convex hull of the Calabi-Yau ambient vectors and cached.

        **Returns:**

        - `Polytope`: The polytope of the CICY, or the convex hull of the ambient vectors.

        """
        if self.__p is None:
            self.__p=Polytope(self.__CY_ambient_pts)
        return self.__p

    def intersection_numbers_orbifold(self) -> dict[tuple[int, ...], float]:
        """
        **Description:**

        Computes and caches the intersection numbers of the orbifold toric fan.

        **Returns:**

        - `dict[tuple[int, ...], float]`: The intersection numbers of the orbifold ambient toric variety, keyed by sorted tuples of one-indexed divisor labels.

        """
        if self.__intersection_numbers_orbifold is None:
            self.__intersection_numbers_orbifold = self.orbifold_toric_fan().intersection_numbers()
        return self.__intersection_numbers_orbifold

    def NHC(self, as_labels: bool = False) -> np.ndarray:
        """
        **Description:**

        Determines the toric divisors supporting non-Higgsable clusters by checking the sections of -2K_B = 2(1 - L_1 - ... - L_r). The result may be returned either as one-indexed ray labels or as lattice vectors.

        **Arguments:**

        - `as_labels (bool)`: If `True`, return one-indexed ray labels. If `False`, return the corresponding lattice vectors. Defaults to `False`.

        **Returns:**

        - `np.ndarray`: The NHC labels or corresponding toric rays (an empty array if there are none).

        """
        if self.__NHC_labels is None:
            sections_NP2 = COF.sections(self.vectors_orbifold(), 2 * (1 - self.divisors().sum(axis=0)))
            if len(sections_NP2) > 0:
                self.__NHC_labels = np.where(np.min(sections_NP2, axis=1) == 1)[0] + 1
            else:
                self.__NHC_labels = np.array([], dtype=int)
        if as_labels:
            return self.__NHC_labels
        else:
            if len(self.__NHC_labels)==0:
                return np.array([])
            return self.vectors_orbifold(self.__NHC_labels)

    def o_planes(self) -> list[tuple[int, ...]]:
        """
        **Description:**

        Computes and caches the O-planes of the orientifold: the cones of the orbifold toric fan whose torus orbits carry the O7-planes (O7 rays and the seed cones of non-divisorial O7-planes) and the O3-planes (see `fixed_locus`).

        **Returns:**

        - `list[tuple[int, ...]]`: The O7 cones followed by the O3 cones, as tuples of one-indexed labels of `vectors_orbifold()`.

        """
        if self.__o_planes is None:
            fixed = self.fixed_locus()
            self.__o_planes = [(int(r),) for r in fixed["O7_rays"]] + fixed["O7_cones"] + list(fixed["O3_cones"])
        return self.__o_planes

    def fixed_locus(self, check: bool = True, return_chi: bool = False, method: str = "auto") -> dict:
        """
        **Description:**

        Computes and caches the fixed locus of the orientifold involution with `COF.orientifold_fixed_locus`: the O7-planes with their Euler characteristics and the O3-planes.

        **Arguments:**

        - `check (bool)`: Whether to verify that the orientifold-invariant Calabi-Yau is smooth. Defaults to `True`.
        - `return_chi (bool)`: Whether to also compute chi(B) and chi(X). Defaults to `False`.
        - `method (str)`: "hypersurface" uses the computation for hypersurfaces of `COF.orientifold_fixed_locus_hypersurface`, as `FT_CY.CY_orientifold` (needs a single defining equation), "general" the one for any codimension, and "auto" the method "hypersurface" if it applies and "general" otherwise. Only used on the first call with these `check` and `return_chi`. Defaults to `"auto"`.

        **Raises:**

        - `ValueError`: Raised by `COF.orientifold_fixed_locus`, e.g. if the orientifold-invariant Calabi-Yau is singular or the method "hypersurface" does not apply.

        **Returns:**

        - `dict`: The output of `COF.orientifold_fixed_locus`.

        """
        key = (check, return_chi)
        if key not in self.__fixed_locus:
            cached = self.__fixed_locus.get((check, True))
            self.__fixed_locus[key] = cached if cached is not None else COF.orientifold_fixed_locus(self, check=check, return_chi=return_chi, method=method)
        return self.__fixed_locus[key]

    def chi_fixed_locus(self):
        return self.fixed_locus().get("chi_I")

    def chi_O7(self):
        return self.fixed_locus().get("chi_O7")

    def chi_O3(self):
        return self.fixed_locus().get("n_O3")

    def O7(self):
        return (self.fixed_locus().get("O7_rays"),self.fixed_locus().get("O7_cones"))

    def O3(self):
        return self.fixed_locus().get("O3_cones")

    def p_g_O7(self, check: bool = True, full: bool = False, method: str = "auto") -> dict:
        """
        **Description:**

        Computes the geometric genus p_g=h^{2,0} of every O7-plane with `COF.O7_holomorphic_data`. For a hypersurface (the default for a single defining equation) as `FT_CY.CY_orientifold`: K_S^2 from adjunction in the orbifold toric variety, chi(O_S) from Noether's formula, and the number of components and q from the face of the Newton polytope on the O7-plane. In general: the Hodge numbers of the O7-plane from the E-polynomial of its toric strata, K_S^2 from adjunction, checked with Noether's formula. Both check the holomorphic Lefschetz formula chi_I=16+8*sum_S chi(O_S).

        **Arguments:**

        - `check (bool)`: Whether to run the consistency checks of `COF.O7_holomorphic_data`. Defaults to `True`.
        - `full (bool)`: Whether to return all data (K_S^2, chi(O_S), components, q, totals) instead of p_g only. Defaults to `False`.
        - `method (str)`: "hypersurface" uses `COF.O7_holomorphic_data_hypersurface` (needs a single defining equation), "general" the E-polynomials, and "auto" the method "hypersurface" if it applies and "general" otherwise. Defaults to `"auto"`.

        **Raises:**

        - `ValueError`: Raised by `COF.O7_holomorphic_data` if a check fails or the method "hypersurface" does not apply.

        **Returns:**

        - `dict`: O7-plane (keyed as in `fixed_locus()["chi_O7_components"]`) -> p_g, or with `full` the output of `COF.O7_holomorphic_data`.

        """
        data = COF.O7_holomorphic_data(self, check=check, method=method)
        return data if full else data["p_g"]


class CICY_F_Theory_Uplift():
    """
    **Description:**

    Constructs and stores the toric data associated with an F-theory uplift of a `CICY_orientifold`, generalizing `FT_CY.F_Theory_Uplift` to orientifolds of Calabi-Yau complete intersections. The base B = X/sigma is the complete intersection of the orbifold divisors L_1, ..., L_r, and the uplift is the complete intersection of their pull-backs and of the Weierstrass divisor in the `P^2_[2,3,1]` bundle over the orbifold that is twisted by -K_B = 1 - L_1 - ... - L_r, with optional blowups over non-Higgsable-cluster loci. The uplift is stored as a `CICY` of codimension r + 1 in ambient dimension orbifold dimension + 2 (`cicy`), which provides the Newton polytopes, Hodge numbers, nef tests, intersection numbers and Chern classes. The mirror is stored as a second `CICY` on the M-side rays (`dual_cicy`), which carries the M-side fan and the saturated GLSM charge matrix.

    **Arguments:**

    - `orientifold_or_points`: Either a `CICY_orientifold` object, or a `CICY` or the toric data of one, from which a `CICY_orientifold` should first be constructed.
    - `xi`: The orientifold action/vector. Required if no `CICY_orientifold` is provided.
    - `resolve_A1_singularities (bool)`: Whether to resolve A1 singularities in the underlying orientifold. Defaults to `False`.
    - `construct_nef_decomposition (bool)`: Whether to attempt to construct a Cartier and nef decomposition for the underlying orientifold. Defaults to `True`.
    - `parities`: The parities of the defining equations under the orientifold action, see `CICY_orientifold`. Defaults to `None`.
    - `divisors`: The defining divisors of the CICY, if toric data are provided. Defaults to `None`.
    - `keep_all (bool)`: Whether the CICY keeps all given rays, if toric data are provided. Defaults to `False`.

    **Returns:**

    Nothing.

    """

    def __init__(self, orientifold_or_points: CICY_orientifold | CICY | Polytope | VectorConfiguration | Fan | np.ndarray | list[list[int]] | None = None, xi: ArrayLike | None = None, resolve_A1_singularities: bool = False, construct_nef_decomposition: bool = True, parities: ArrayLike | None = None, divisors: ArrayLike | None = None, keep_all: bool = False) -> None:
        """
        **Description:**
    
        Constructs a `CICY_F_Theory_Uplift` object from an existing `CICY_orientifold`, from a `CICY`, or from the toric data of a `CICY`; in the last two cases the `CICY_orientifold` (and with it the `CICY`) is constructed first. The object stores the underlying orientifold and initializes caches for the singular uplift, the resolved uplift and the uplift `CICY`, all of which are constructed when first needed.
    
        **Arguments:**
    
        - `orientifold_or_points (CICY_orientifold | CICY | Polytope | VectorConfiguration | Fan | np.ndarray | list[list[int]] | None)`: Either a `CICY_orientifold` object, or a `CICY` or toric data accepted by `CICY_orientifold`.
        - `xi (ArrayLike | None)`: The orientifold action/vector. Required unless a `CICY_orientifold` is provided or the polytope of the CICY is trilayer. Defaults to `None`.
        - `resolve_A1_singularities (bool)`: Whether to resolve A1 singularities in the underlying orientifold. Defaults to `False`.
        - `construct_nef_decomposition (bool)`: Whether to attempt to construct a Cartier and nef decomposition for the underlying orientifold. Defaults to `True`.
        - `parities (ArrayLike | None)`: The parities of the defining equations under the orientifold action, see `CICY_orientifold`. Defaults to `None`.
        - `divisors (ArrayLike | None)`: The defining divisors of the CICY, if toric data are provided (see `CICY_orientifold`). Must be `None` if a `CICY_orientifold` or a `CICY` is provided. Defaults to `None`.
        - `keep_all (bool)`: Whether the CICY keeps all given rays, only used if toric data are provided (see `CICY_orientifold`). Defaults to `False`.
    
        **Raises:**
    
        - `ValueError`: Raised if `divisors` are given together with a `CICY_orientifold`, or by `CICY_orientifold`, e.g. if a required `xi` is missing or `divisors` are given together with a `CICY`.
        - `TypeError`: Raised by `CICY` if the toric data have an unsupported type.
    
        **Returns:**
    
        - `None`
    
        """
        self.__blowups = None
        self.__cicy = None
        self.__cicy_has_fan = False
        self.__Cayley_N = None
        self.__pol_N_sum = None
        self.__pol_N_conv = None
        self.__intersecting_labels_B_N = None
        self.__intersecting_labels_W_N = None
        self.__intersecting_labels_N=None
        self.__pol_W_N = None
        self.__pol_B_N = None
        self.__intersection_numbers_smooth_uplift = None
        self.__basis_homology_N = None
        self.__singular_uplift_toric_fan = None
        self.__pts_singular_uplift=None
        self.__pts_smooth_uplift=None
        self.__smooth_uplift_toric_fan = None
        self.__chi = None
        self.__is_nef_decomposition = None
        self.__CICY_orientifold = None
        self.__intersection_numbers_N=None
        self.__intersecting_labels_M=None
        self.__intersecting_labels_B_M = None
        self.__intersecting_labels_W_M = None
        self.__vectors_M_conv_toric_fan = None
        self.__dual_cicy = None
        self.__dual_cicy_has_fan = False
        self.__dual_cicy_has_glsm = False
        self.__M_conv_toric_fan = None
        self.__blow_down_fan = None
        self.__basis_homology_M = None
        self.__intersection_numbers_M=None
        self.__Qd7=None
        self.__Qsat=None
        self.__block_sizes=None
        # the D7 faces of the Mori cone, keyed by `mori_cone_cap` (see `mori_cone_face_d7`)
        self.__MC_face={}
        self.__intersection_numbers_surfaces_sat=None
        self.__vertical_intersection_form = None

        match orientifold_or_points:
            case CICY_orientifold():
                if divisors is not None:
                    raise ValueError("Divisors can only be given together with toric data, not with a CICY_orientifold")
                self.__CICY_orientifold = orientifold_or_points
            case object():
                self.__CICY_orientifold = CICY_orientifold(orientifold_or_points, xi=xi, resolve_A1_singularities=resolve_A1_singularities, construct_nef_decomposition=construct_nef_decomposition, parities=parities, divisors=divisors, keep_all=keep_all)
        self.x=np.zeros(self.ambient_dim_base()+2,dtype=int)
        self.y=self.x.copy()
        self.z=self.x.copy()
        self.x[-2:]=[3,1]
        self.y[-2:]=[-2,-1]
        self.z[-2:]=[0,1]
        self.u=self.x+self.y

    def __uplift_cicy(self, with_fan: bool = False) -> CICY:
        """
        **Description:**
    
        Constructs and caches the uplift as a `CICY` on the rays `vectors_smooth_uplift_ambient()` with the divisors (pull-backs of L_1, ..., L_r, Weierstrass divisor), as `F_Theory_Uplift.__set_divisor_representations`: if the orientifold yields a partition, the pull-backs vanish on the fiber and blowup rays and the Weierstrass divisor is their complement; otherwise the Weierstrass divisor is 3 D_x, the pull-backs have the coefficients L_a, 2 L_a on the two blowup rays of an NHC ray, and the `CICY` shifts the Weierstrass divisor (the last one) so that all divisors sum to the anticanonical divisor. No fan is needed for this; with `with_fan` the resolved uplift fan is constructed and set on the `CICY`, once.
    
        **Arguments:**
    
        - `with_fan (bool)`: Whether the `CICY` must carry the fan `smooth_uplift_ambient_toric_fan()`. Defaults to `False`.
    
        **Raises:**
    
        - `ValueError`: Raised if the orientifold is not regular, or by `CICY` if the divisors do not sum to the anticanonical class.
    
        **Returns:**
    
        - `CICY`: The uplift.
    
        """
        if self.__cicy is None:
            L = self.divisors_orbifold()
            if L is None:
                raise ValueError("The orientifold is not regular, so its uplift is not defined.")
            V = self.vectors_smooth_uplift_ambient()
            n_orb = L.shape[1]
            divisors_base = np.zeros((len(L), len(V)), dtype=int)
            divisors_base[:, :n_orb] = L
            if self.orientifold().yields_partition():
                divisor_weierstrass = 1 - divisors_base.sum(axis=0)
            else:
                divisor_weierstrass = np.zeros(len(V), dtype=int)
                divisor_weierstrass[n_orb] = 3
                for n, nhc in enumerate(self.NHC(as_labels=True) - 1):
                    divisors_base[:, n_orb + 3 + 2*n] = L[:, nhc]
                    divisors_base[:, n_orb + 3 + 2*n + 1] = 2 * L[:, nhc]
            self.__cicy = CICY(V, np.vstack([divisors_base, divisor_weierstrass]), keep_all=True)
            if self.__dual_cicy is not None:
                # the Hodge diamond is then computed only once for the mirror pair, see `CICY.hodge_numbers`
                self.__cicy._link_mirror(self.__dual_cicy)
        if with_fan and not self.__cicy_has_fan:
            self.__cicy.set_fan(self.smooth_uplift_ambient_toric_fan())
            self.__cicy_has_fan = True
        return self.__cicy

    def __repr__(self) -> str:

        """
        **Description:**
    
        Produces a short human-readable representation of the object.
    
        **Returns:**
    
        - `str`: A description of the object.
    
        """
        
        return "An F-theory uplift of a Calabi-Yau complete intersection orientifold"
        
    def orientifold(self) -> CICY_orientifold:

        """
        **Description:**
    
        Returns the `CICY_orientifold` object from which the F-theory uplift is constructed.
    
        **Returns:**
    
        - `CICY_orientifold`: The underlying orientifold.
    
        """
        
        return self.__CICY_orientifold

    def cicy(self) -> CICY:
        """
        **Description:**
    
        Returns the uplift as a `CICY`: the complete intersection of the pull-backs of L_1, ..., L_r and of the Weierstrass divisor (the last divisor) on the rays `vectors_smooth_uplift_ambient()`, carrying the fan `smooth_uplift_ambient_toric_fan()`. Its divisors are those of `divisors_base_N()` and `divisor_weierstrass_N()`.
    
        **Returns:**
    
        - `CICY`: The uplift, of codimension r + 1 in ambient dimension `ambient_dim_base()` + 2.
    
        """
        return self.__uplift_cicy(with_fan=True)
    
    def hodge_numbers_orientifold(self) -> dict[str, int]:
        """
        **Description:**
    
        Returns the orientifold-even and -odd Hodge numbers of the underlying Calabi-Yau orientifold, computed and cached by `CICY_orientifold.hodge_numbers_weighted`.
    
        **Returns:**
    
        - `dict[str, int]`: The Hodge numbers with keys `"h11+"`, `"h11-"`, `"h21+"`, `"h21-"`.
    
        """

        return self.orientifold().hodge_numbers_weighted()
    
    def hodge_numbers_CY(self, backend: str = 'auto') -> np.ndarray:
        """
        **Description:**
    
        Computes the Hodge diamond of the uplift from the stringy E-function of the Cayley polytope `Cayley_M()` with `CICY.hodge_numbers`, which caches it. Works for nef-decompositions as well as nef-partitions. No fan is needed.
    
        **Arguments:**
    
        - `backend (str)`: How the h*-polynomials of the faces of the Cayley polytope are obtained, "auto", "faces", "normaliz", "cytools" or "latte" (see `CICY.hodge_numbers` and `COF.hodge_diamond_stringy`); "auto" computes them face by face, without enumerating dilations of the Cayley polytope. Only used on the first call. Defaults to `'auto'`.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef decomposition.
    
        **Returns:**
    
        - `np.ndarray`: The Hodge diamond, `h[p, q] = h^{p,q}`.
    
        """
        if not self.is_nef_decomposition():
            raise ValueError("Hodge Numbers can only be computed for nef-partitions and nef-decompositions")
        return self.__uplift_cicy().hodge_numbers(backend)
    
    def is_trilayer_orientifold(self) -> bool:
        """
        **Description:**
    
        Returns whether the underlying orientifold is a trilayer orientifold (see `CICY_orientifold.is_trilayer_orientifold`).
    
        **Returns:**
    
        - `bool`: `True` if the orientifold is a trilayer orientifold, otherwise `False`.
    
        """
        return self.orientifold().is_trilayer_orientifold()

    def is_nef_partition(self) -> bool | np.bool_:
        """
        **Description:**
    
        Returns whether the resolved F-theory uplift defines both a valid partition and a nef decomposition.
    
        **Returns:**
    
        - `bool | np.bool_`: `True` if the uplift defines a nef partition, otherwise `False`.
    
        """
        return self.is_nef_decomposition() and self.is_partition()
        
    def is_partition(self) -> bool:
        """
        **Description:**
    
        Returns whether the divisors of the uplift (`divisors_base_N()`, `divisor_weierstrass_N()`) form a partition of the rays of the resolved uplift ambient space (`CICY.is_partition`). This is the case if the underlying orientifold yields a partition (`CICY_orientifold.yields_partition`).
    
        **Returns:**
    
        - `bool`: `True` if the uplift defines a partition, otherwise `False`.
    
        """
        return self.__uplift_cicy().is_partition()

    def smooth_uplift_ambient_toric_fan(self) -> Fan:
        """
        **Description:**
    
        Constructs and caches the toric fan of the resolved F-theory uplift ambient space. Starting from the singular uplift fan, the method inserts the blowup rays associated with non-Higgsable-cluster divisors and performs the corresponding local cone refinements.
    
        **Returns:**
    
        - `Fan`: The toric fan of the resolved uplift ambient space.
    
        """
        
        if self.__smooth_uplift_toric_fan is None:
            n_sing_uplift = len(self.vectors_singular_uplift_ambient())
            xlabel = n_sing_uplift - 2
            ylabel = n_sing_uplift - 1
            ct = 0
            vc_smooth = VectorConfiguration(self.vectors_smooth_uplift_ambient())
            new_fan = Fan(vc=vc_smooth, cones=self.singular_uplift_ambient_toric_fan().cones())
        
            for xxx in self.NHC_singular_uplift(as_labels=True):
                all_cones = set(new_fan.cones())
                for link in new_fan.link((xxx, xlabel, ylabel)):
                    all_cones.discard(tuple(sorted((xxx, xlabel, ylabel) + link)))
                    all_cones.add(tuple(sorted(link + (xxx, n_sing_uplift + 2*ct + 1, n_sing_uplift + 2*ct + 2))))
                    all_cones.add(tuple(sorted(link + (xxx, xlabel, n_sing_uplift + 2*ct + 2))))
                    all_cones.add(tuple(sorted(link + (xxx, ylabel, n_sing_uplift + 2*ct + 1))))
                    all_cones.add(tuple(sorted(link + (xlabel, ylabel, n_sing_uplift + 2*ct + 1))))
                    all_cones.add(tuple(sorted(link + (xlabel, n_sing_uplift + 2*ct + 1, n_sing_uplift + 2*ct + 2))))
                ct += 1
                new_fan = Fan(vc=vc_smooth, cones=all_cones)
            self.__smooth_uplift_toric_fan = new_fan
        return self.__smooth_uplift_toric_fan

    def singular_uplift_ambient_toric_fan(self) -> Fan:
        """
        **Description:**
    
        Constructs and caches the toric fan obtained by adding the toric `P^2_[2,3,1]` fiber rays to the orbifold base fan, before resolving non-Higgsable-cluster singularities.
    
        **Returns:**
    
        - `Fan`: The toric fan of the singular uplift ambient space.
    
        """
        if self.__singular_uplift_toric_fan is None:
            cones_uplift = set()
            x_pts = len(self.vectors_orbifold()) + 1
            y_pts = x_pts + 1
            z_pts = x_pts + 2
            
            for c in self.orbifold_toric_fan().cones():
                cones_uplift.add(c + (x_pts, y_pts))
                cones_uplift.add(c + (x_pts, z_pts))
                cones_uplift.add(c + (y_pts, z_pts))
            self.__singular_uplift_toric_fan = Fan(VectorConfiguration(self.vectors_singular_uplift_ambient()),cones=cones_uplift)
        return self.__singular_uplift_toric_fan
        
    def Cayley_M(self) -> Polytope:
        """
        **Description:**
    
        Returns the Cayley polytope of the Newton polytopes `pol_B_M()` and `pol_W_M()` of the uplift divisors in the M-lattice (`CICY.Cayley_polytope`). If the uplift is not a nef partition, a warning is raised.
    
        **Returns:**
    
        - `Polytope`: The M-lattice Cayley polytope.
    
        """
        if not self.is_nef_partition():
            warnings.warn("Careful, uplift is not a nef-partition")
        return self.__uplift_cicy().Cayley_polytope()
        
    def Cayley_N(self) -> Polytope:
        """
        **Description:**
    
        Constructs and caches the Cayley polytope associated with the dual N-lattice nef partition, i.e. of `pol_B_N()` and `pol_W_N()`. This is only defined when the uplift gives a valid partition. If the uplift is not a nef partition, a warning is raised.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a partition.
    
        **Returns:**
    
        - `Polytope`: The N-lattice Cayley polytope.
    
        """
        
        if self.is_partition():
            if self.__Cayley_N is None:
                self.__Cayley_N = COF.cayley_polytope([*self.pol_B_N(), self.pol_W_N()])
        else:
            raise ValueError("Uplift is not a partition")
        if not self.is_nef_partition():
            warnings.warn("Careful, uplift is not a nef-partition")
        return self.__Cayley_N

    def pol_W_M(self) -> Polytope:
        """
        **Description:**
    
        Returns the Newton polytope associated with the Weierstrass divisor on the resolved uplift ambient space (`CICY.newton_polytopes`).
    
        **Returns:**
    
        - `Polytope`: The Newton polytope of the Weierstrass divisor.
    
        """
        return self.__uplift_cicy().newton_polytopes()[-1]

    def pol_B_M(self) -> list[Polytope]:
        """
        **Description:**
    
        Returns the Newton polytopes associated with the base divisors, the pull-backs of L_1, ..., L_r, on the resolved uplift ambient space (`CICY.newton_polytopes`).
    
        **Returns:**
    
        - `list[Polytope]`: The Newton polytopes of the r base divisors.
    
        """
        return self.__uplift_cicy().newton_polytopes()[:-1]

    def pol_M_conv(self) -> Polytope:
        """
        **Description:**
    
        Returns the convex hull of the Newton polytopes `pol_B_M()` and `pol_W_M()` (`CICY.newton_polytopes_conv`).
    
        **Returns:**
    
        - `Polytope`: The convex hull.
    
        """
        return self.__uplift_cicy().newton_polytopes_conv()

    def pol_M_sum(self) -> Polytope:
        """
        **Description:**
    
        Returns the Minkowski sum of the Newton polytopes `pol_B_M()` and `pol_W_M()` (`CICY.newton_polytopes_sum`).
    
        **Returns:**
    
        - `Polytope`: The Minkowski sum.
    
        """
        return self.__uplift_cicy().newton_polytopes_sum()

    def pol_W_N(self) -> Polytope:
        """
        **Description:**
    
        Constructs and caches the N-lattice polytope of the Weierstrass part of the partition: the convex hull of the origin and the rays of the resolved uplift ambient space on which the Weierstrass divisor has coefficient 1. If the uplift is not a nef partition, a warning is raised.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a partition.
    
        **Returns:**
    
        - `Polytope`: The N-lattice polytope of the Weierstrass divisor.
    
        """
        if self.is_partition():
            if self.__pol_W_N is None:
                self.__pol_W_N = Polytope(np.concatenate((np.zeros((1, self.ambient_dim_base()+2), dtype=int), self.vectors_smooth_uplift_ambient()[np.where(self.divisor_weierstrass_N()==1)[0]]), axis=0))
        else:
            raise ValueError("Uplift is not a partition")
        if not self.is_nef_partition():
            warnings.warn("Careful, uplift is not a nef-partition")
        return self.__pol_W_N

    def pol_B_N(self) -> list[Polytope]:
        """
        **Description:**
    
        Constructs and caches the N-lattice polytopes of the base parts of the partition: for each base divisor the convex hull of the origin and the rays of the resolved uplift ambient space on which it has coefficient 1. If the uplift is not a nef partition, a warning is raised.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a partition.
    
        **Returns:**
    
        - `list[Polytope]`: The N-lattice polytopes of the r base divisors.
    
        """
        if self.is_partition():
            if self.__pol_B_N is None:
                self.__pol_B_N = [Polytope(np.concatenate((np.zeros((1, self.ambient_dim_base()+2), dtype=int), self.vectors_smooth_uplift_ambient()[np.where(D==1)[0]]), axis=0)) for D in self.divisors_base_N()]
        else:
            raise ValueError("Uplift does not yield a partition")
        if not self.is_nef_partition():
            warnings.warn("Careful, uplift is not a nef-partition")
        return self.__pol_B_N

    def pol_N_conv(self) -> Polytope:
        """
        **Description:**
    
        Constructs and caches the convex hull of the rays of the resolved uplift ambient space.
    
        **Returns:**
    
        - `Polytope`: The convex hull.
    
        """
        if self.__pol_N_conv is None:
            self.__pol_N_conv = Polytope(self.vectors_smooth_uplift_ambient())
        return self.__pol_N_conv

    def pol_N_sum(self) -> Polytope:
        """
        **Description:**
    
        Constructs and caches the Minkowski sum of `pol_B_N()` and `pol_W_N()`. This is only defined when the uplift gives a valid partition. If the uplift is not a nef partition, a warning is raised.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a partition.
    
        **Returns:**
    
        - `Polytope`: The Minkowski sum.
    
        """
        if self.is_partition():
            if self.__pol_N_sum is None:
                self.__pol_N_sum = reduce(lambda P, Q: P.minkowski_sum(Q), [*self.pol_B_N(), self.pol_W_N()])
        else:
            raise ValueError("Uplift is not a partition")
        if not self.is_nef_partition():
            warnings.warn("Careful, uplift is not a nef-partition")
        return self.__pol_N_sum

    def vectors_singular_uplift_ambient(self, labels: ArrayLike | None = None) -> np.ndarray:
        """
        **Description:**
    
        Constructs and caches the rays of the singular uplift ambient space: the orbifold rays v, lifted to (v, 0, 1 - L_1(v) - ... - L_r(v)), i.e. by the coefficient of -K_B, followed by the three `P^2_[2,3,1]` fiber rays x, y, z. Returns all of them, or those with the given one-indexed labels.
    
        **Arguments:**
    
        - `labels (ArrayLike | None)`: Optional collection of one-indexed ray labels. Defaults to `None` (all rays).
    
        **Returns:**
    
        - `np.ndarray`: The selected rays, one per row.
    
        """
        if self.__pts_singular_uplift is None:
            coordinates231 = (1 - self.divisors_orbifold().sum(axis=0))[:, None] @ self.z[-2:][None, :]
            pts_uplift = np.concatenate((self.vectors_orbifold(), coordinates231), axis=1)
            pts_uplift = np.concatenate((pts_uplift, np.array([self.x, self.y, self.z])), axis=0)
            self.__pts_singular_uplift=pts_uplift
        if labels is None:
            return self.__pts_singular_uplift
        return self.__pts_singular_uplift[np.array(labels)-1]

    def blowups(self, as_labels: bool = False) -> list[np.ndarray] | np.ndarray:
        """
        **Description:**
    
        Constructs and caches the blowup rays of the resolved uplift ambient space: for every NHC ray v of the singular uplift the two rays v + x + 2y and 2v + 2x + 3y.
    
        **Arguments:**
    
        - `as_labels (bool)`: If `True`, return the one-indexed labels of the blowup rays in `vectors_smooth_uplift_ambient()`. Defaults to `False`.
    
        **Returns:**
    
        - `list[np.ndarray] | np.ndarray`: One array with the two blowup rays per NHC ray, or the labels.
    
        """
        if self.__blowups is None:
            bus = []
            for xxx in self.NHC_singular_uplift(as_labels=True):
                bus.append(np.array([
                    self.vectors_singular_uplift_ambient()[xxx-1] + self.x + 2*self.y,
                    2*self.vectors_singular_uplift_ambient()[xxx-1] + 2*self.x + 3*self.y
                ]))
            self.__blowups = bus
        if as_labels:
            return np.arange(len(self.vectors_singular_uplift_ambient())+1,len(self.vectors_smooth_uplift_ambient())+1)
        return self.__blowups

    def intersecting_divisors_B_N(self, as_labels: bool = False) -> list[np.ndarray]:
        """
        **Description:**
    
        For every base part of the nef partition, determines the points of `pol_B_N()` whose toric divisors intersect the uplift.
    
        **Arguments:**
    
        - `as_labels (bool)`: If `True`, return the indices of the points in the corresponding polytope of `pol_B_N()`. Defaults to `False`.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef partition.
    
        **Returns:**
    
        - `list[np.ndarray]`: The intersecting points, or their indices, one array per base divisor.
    
        """
        if not self.is_nef_partition():
            raise ValueError("Intersecting divisors can only be computed for nef partitions")
        if self.__intersecting_labels_B_N is None:
            self.__intersecting_labels_B_N = [COF.get_indices(P.points(),self.intersecting_divisors_N()) for P in self.pol_B_N()]
        if as_labels:
            return self.__intersecting_labels_B_N
        return [P.points()[l] for P, l in zip(self.pol_B_N(), self.__intersecting_labels_B_N)]

    def intersecting_divisors_W_N(self, as_labels: bool = False) -> np.ndarray:
        """
        **Description:**
    
        Determines the points of `pol_W_N()` whose toric divisors intersect the uplift.
    
        **Arguments:**
    
        - `as_labels (bool)`: If `True`, return the indices of the points in `pol_W_N()`. Defaults to `False`.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef partition.
    
        **Returns:**
    
        - `np.ndarray`: The intersecting points, or their indices.
    
        """
        if not self.is_nef_partition():
            raise ValueError("Intersecting divisors can only be computed for nef partitions")
        if self.__intersecting_labels_W_N is None:
            self.__intersecting_labels_W_N = COF.get_indices(self.pol_W_N().points(),self.intersecting_divisors_N())
        if as_labels:
            return self.__intersecting_labels_W_N
        return self.pol_W_N().points()[self.__intersecting_labels_W_N]

    def intersecting_divisors_N(self, as_labels: bool = False) -> np.ndarray:
        """
        **Description:**
    
        Determines the rays of the resolved uplift ambient space whose toric divisors intersect the uplift (`COF.intersecting_points`).
    
        **Arguments:**
    
        - `as_labels (bool)`: If `True`, return one-indexed labels of `vectors_smooth_uplift_ambient()`. Defaults to `False`.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef decomposition.
    
        **Returns:**
    
        - `np.ndarray`: The intersecting rays, or their labels.
    
        """
        if not self.is_nef_decomposition():
            raise ValueError("Intersecting divisors can only be computed for nef partitions")
        if self.__intersecting_labels_N is None:
            self.__intersecting_labels_N = COF.intersecting_points(self.vectors_smooth_uplift_ambient(),divisors=list(self.__uplift_cicy().divisors()),polytopes=self.__uplift_cicy().newton_polytopes(),as_indices=True)+1
        if as_labels:
            return self.__intersecting_labels_N
        else:
            return self.vectors_smooth_uplift_ambient()[self.__intersecting_labels_N-1]

    def polytope(self) -> Polytope:
        """
        **Description:**
    
        Returns the polytope of the underlying orientifold (`CICY_orientifold.polytope`).
    
        **Returns:**
    
        - `Polytope`: The polytope.
    
        """
        return self.orientifold().polytope()

    def intersection_numbers_smooth_uplift_ambient(self) -> dict[tuple[int, ...], float]:
        """
        **Description:**
    
        Computes and caches the intersection numbers of the resolved uplift ambient toric fan.
    
        **Returns:**
    
        - `dict[tuple[int, ...], float]`: The intersection numbers, keyed by sorted tuples of one-indexed divisor labels.
    
        """
        if self.__intersection_numbers_smooth_uplift is None:
            self.__intersection_numbers_smooth_uplift = self.smooth_uplift_ambient_toric_fan().intersection_numbers()
        return self.__intersection_numbers_smooth_uplift

    def h11(self) -> int:
        """
        **Description:**
    
        Returns the Hodge number h^{1,1} of the uplift (see `hodge_numbers_CY`).
    
        **Returns:**
    
        - `int`: h^{1,1}.
    
        """
        return int(self.hodge_numbers_CY()[1, 1])

    def h31(self) -> int:
        """
        **Description:**
    
        Returns the Hodge number h^{3,1} of the uplift fourfold (see `hodge_numbers_CY`).
    
        **Returns:**
    
        - `int`: h^{3,1}.
    
        """
        return int(self.hodge_numbers_CY()[3, 1])

    def h21(self) -> int:
        """
        **Description:**
    
        Returns the Hodge number h^{2,1} of the uplift (see `hodge_numbers_CY`).
    
        **Returns:**
    
        - `int`: h^{2,1}.
    
        """
        return int(self.hodge_numbers_CY()[2, 1])

    def h22(self) -> int:
        """
        **Description:**
    
        Returns the Hodge number h^{2,2} of the uplift (see `hodge_numbers_CY`).
    
        **Returns:**
    
        - `int`: h^{2,2}.
    
        """
        return int(self.hodge_numbers_CY()[2, 2])

    def chi(self) -> int:
        """
        **Description:**
    
        Computes and caches the Euler characteristic of the uplift as the alternating sum of the Hodge diamond `hodge_numbers_CY`.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef decomposition.
    
        **Returns:**
    
        - `int`: The Euler characteristic.
    
        """
        if not self.is_nef_decomposition():
            raise ValueError("Euler number can only be computed for a nef-decomposition")
        if self.__chi is None:
            h = self.hodge_numbers_CY()
            self.__chi = int((h * (-1) ** np.add.outer(np.arange(len(h)), np.arange(len(h)))).sum())
        return self.__chi

    def divisor_weierstrass_N(self, only_intersecting: bool = False) -> np.ndarray:
        """
        **Description:**
    
        Returns the toric divisor coefficients of the Weierstrass divisor on the resolved uplift ambient space, the last divisor of `cicy`.
    
        **Arguments:**
    
        - `only_intersecting (bool)`: Whether to restrict to the rays of `intersecting_divisors_N()`. Defaults to `False`.
    
        **Returns:**
    
        - `np.ndarray`: The coefficients, ordered as `vectors_smooth_uplift_ambient()`.
    
        """
        divisor_weierstrass = self.__uplift_cicy().divisors()[-1]
        if only_intersecting:
            return divisor_weierstrass[self.intersecting_divisors_N(as_labels=True)-1]
        return divisor_weierstrass

    def divisors_base_N(self, only_intersecting: bool = False) -> np.ndarray:
        """
        **Description:**
    
        Returns the toric divisor coefficients of the base divisors, the pull-backs of L_1, ..., L_r, on the resolved uplift ambient space, the first r divisors of `cicy`.
    
        **Arguments:**
    
        - `only_intersecting (bool)`: Whether to restrict to the rays of `intersecting_divisors_N()`. Defaults to `False`.
    
        **Returns:**
    
        - `np.ndarray`: The (r, N) coefficients, one divisor per row, ordered as `vectors_smooth_uplift_ambient()`.
    
        """
        divisors_base = self.__uplift_cicy().divisors()[:-1]
        if only_intersecting:
            return divisors_base[:, self.intersecting_divisors_N(as_labels=True)-1]
        return divisors_base

    def divisors_orbifold(self) -> np.ndarray | None:
        """
        **Description:**
    
        Returns the orbifold divisors L_1, ..., L_r of the underlying orientifold (`CICY_orientifold.divisors`).
    
        **Returns:**
    
        - `np.ndarray | None`: The (r, N) coefficients, or `None` if the orientifold is not regular.
    
        """
        return self.orientifold().divisors()

    def NHC(self, as_labels: bool = False) -> np.ndarray:
        """
        **Description:**
    
        Returns the toric divisors of the orbifold supporting non-Higgsable clusters (`CICY_orientifold.NHC`).
    
        **Arguments:**
    
        - `as_labels (bool)`: If `True`, return one-indexed ray labels. Defaults to `False`.
    
        **Returns:**
    
        - `np.ndarray`: The NHC labels or the corresponding orbifold rays.
    
        """
        return self.orientifold().NHC(as_labels)

    def NHC_singular_uplift(self, as_labels: bool = False) -> np.ndarray | None:
        """
        **Description:**
    
        Returns the NHC rays in the singular uplift ambient space, as labels (the same as in the orbifold) or as rays.
    
        **Arguments:**
    
        - `as_labels (bool)`: If `True`, return one-indexed ray labels. Defaults to `False`.
    
        **Returns:**
    
        - `np.ndarray | None`: The NHC labels, or the corresponding rays (`None` if there are none).
    
        """
        if as_labels:
            return self.NHC(as_labels=True)
        else:
            labels = self.NHC(as_labels=True)
            if len(labels) > 0:
                return self.vectors_singular_uplift_ambient(labels)

    def vectors_orbifold(self, labels: ArrayLike | None = None) -> np.ndarray:
        """
        **Description:**
    
        Returns the orbifold rays of the underlying orientifold (`CICY_orientifold.vectors_orbifold`).
    
        **Arguments:**
    
        - `labels (ArrayLike | None)`: Optional collection of one-indexed ray labels. Defaults to `None` (all rays).
    
        **Returns:**
    
        - `np.ndarray`: The selected orbifold vectors, one per row.
    
        """
        return self.orientifold().vectors_orbifold(labels)

    def vectors_CY_ambient(self, labels: ArrayLike | None = None) -> np.ndarray:
        """
        **Description:**
    
        Returns the rays of the ambient space of the Calabi-Yau before orbifolding (`CICY_orientifold.vectors_CY_ambient`).
    
        **Arguments:**
    
        - `labels (ArrayLike | None)`: Optional collection of one-indexed ray labels. Defaults to `None` (all rays).
    
        **Returns:**
    
        - `np.ndarray`: The selected ambient-space vectors, one per row.
    
        """
        return self.orientifold().vectors_CY_ambient(labels)

    def dim_base(self) -> np.integer:
        """
        **Description:**
    
        Returns the dimension of the toric ambient space of the orientifold (`CICY_orientifold.dim`).
    
        **Returns:**
    
        - `np.integer`: The dimension.
    
        """
        return self.orientifold().dim()

    def ambient_dim_base(self) -> int:
        """
        **Description:**
    
        Returns the number of coordinates of the orbifold vectors (`CICY_orientifold.ambient_dim`).
    
        **Returns:**
    
        - `int`: The ambient lattice dimension of the orbifold.
    
        """
        return self.orientifold().ambient_dim()

    def CY_ambient_toric_fan(self) -> Fan:
        """
        **Description:**
    
        Returns the toric fan of the ambient space of the Calabi-Yau before orbifolding (`CICY_orientifold.CY_ambient_toric_fan`).
    
        **Returns:**
    
        - `Fan`: The fan.
    
        """
        return self.orientifold().CY_ambient_toric_fan()

    def orbifold_toric_fan(self) -> Fan:
        """
        **Description:**
    
        Returns the toric fan of the orbifold ambient variety (`CICY_orientifold.orbifold_toric_fan`).
    
        **Returns:**
    
        - `Fan`: The fan.
    
        """
        return self.orientifold().orbifold_toric_fan()

    def vectors_smooth_uplift_ambient(self, labels: ArrayLike | None = None) -> np.ndarray:
        """
        **Description:**
    
        Constructs and caches the rays of the resolved uplift ambient space: the rays of the singular uplift followed by the blowup rays of the NHC rays (`blowups`). Returns all of them, or those with the given one-indexed labels. These are the rays of `cicy`.
    
        **Arguments:**
    
        - `labels (ArrayLike | None)`: Optional collection of one-indexed ray labels. Defaults to `None` (all rays).
    
        **Returns:**
    
        - `np.ndarray`: The selected rays, one per row.
    
        """
        if self.__pts_smooth_uplift is None:
            if len(self.NHC()) > 0:
                self.__pts_smooth_uplift = np.concatenate((self.vectors_singular_uplift_ambient(), np.vstack(self.blowups())), axis=0)
            else: 
                self.__pts_smooth_uplift = self.vectors_singular_uplift_ambient()
        if labels is None:
            return self.__pts_smooth_uplift
        return self.__pts_smooth_uplift[np.array(labels)-1]

    def xi(self) -> ArrayLike:
        """
        **Description:**
    
        Returns the vector `xi` specifying the toric `Z_2` orientifold action.
    
        **Returns:**
    
        - `ArrayLike`: The orientifold action vector.
    
        """
        return self.__CICY_orientifold.xi()

    def c2_N(self) -> dict[tuple[int, ...], int]:
        """
        **Description:**
    
        Computes the second Chern class of the uplift by torus localization on the resolved uplift ambient fan (`CICY.toric_chern_class`, which caches it).
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef decomposition.
    
        **Returns:**
    
        - `dict[tuple[int, ...], int]`: The tensor of c_2 in terms of the prime toric divisors.
    
        """
        if not self.is_nef_decomposition():
            raise ValueError("Second Chern class can only be computed for a nef-decomposition")
        return self.cicy().toric_chern_class(2)

    def c3_N(self) -> dict[tuple[int, ...], int] | dict[int, int]:
        """
        **Description:**
    
        Computes the third Chern class of the uplift by torus localization on the resolved uplift ambient fan (`CICY.toric_chern_class`, which caches it).
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef decomposition.
    
        **Returns:**
    
        - `dict[tuple[int, ...], int] | dict[int, int]`: The tensor of c_3 in terms of the prime toric divisors.
    
        """
        if not self.is_nef_decomposition():
            raise ValueError("Third Chern class can only be computed for a nef-decomposition")
        return self.cicy().toric_chern_class(3)

    def is_regular(self) -> bool:
        """
        **Description:**
    
        Returns whether the underlying orientifold is regular (`CICY_orientifold.is_regular`).
    
        **Returns:**
    
        - `bool`: `True` if the orientifold is regular, otherwise `False`.
    
        """
        return self.orientifold().is_regular()

    def normal_fan(self) -> Fan:
        """
        **Description:**
    
        Returns the normal fan of the Newton-polytope data of the underlying orientifold (`CICY_orientifold.normal_fan`).
    
        **Returns:**
    
        - `Fan`: The normal fan.
    
        """
        return self.orientifold().normal_fan()

    def h_prime_is_lattice(self) -> bool:
        """
        **Description:**
    
        Returns whether the Newton polytope of -2K_B is a lattice polytope (`CICY_orientifold.h_prime_is_lattice`).
    
        **Returns:**
    
        - `bool`: `True` if it is a lattice polytope, otherwise `False`.
    
        """
        return self.orientifold().h_prime_is_lattice()

    def basis_homology_N(self) -> np.ndarray:
        """
        **Description:**
    
        Computes and caches a basis of prime toric divisors of the resolved uplift ambient space (`COF.basis_homology` on its fan), and sets it as the basis of `cicy` (`CICY.set_basis`), so that the tensors of the `CICY` in a basis refer to the same divisors.
    
        **Returns:**
    
        - `np.ndarray`: The one-indexed labels of the basis divisors.
    
        """
        if self.__basis_homology_N is None:
            self.__basis_homology_N = COF.basis_homology(self.smooth_uplift_ambient_toric_fan())
            self.cicy().set_basis(self.__basis_homology_N)
        return self.__basis_homology_N

    def nef_partition_N(self,only_intersecting: bool = False) -> tuple[tuple[np.integer, ...], ...]:
        """
        **Description:**
    
        Returns the one-indexed labels of the parts of the N-lattice nef partition: the r base parts followed by the Weierstrass part. If the uplift is not a nef partition, returns empty tuples.
    
        **Arguments:**
    
        - `only_intersecting (bool)`: Whether to restrict to (and label within) the rays of `intersecting_divisors_N()`. Defaults to `False`.
    
        **Returns:**
    
        - `tuple[tuple[np.integer, ...], ...]`: The r + 1 parts.
    
        """
        if self.is_nef_partition():
            return tuple(tuple(np.where(D==1)[0]+1) for D in [*self.divisors_base_N(only_intersecting=only_intersecting),self.divisor_weierstrass_N(only_intersecting=only_intersecting)])
        else:
            return tuple(() for _ in range(len(self.divisors_orbifold())+1))

    def is_nef_decomposition(self) -> bool | np.bool_:
        """
        **Description:**
    
        Determines and caches whether the uplift divisors are all Cartier and nef. If the underlying orientifold uses an ambient triangulation, this is tested on the resolved uplift ambient fan (`CICY.is_nef_decomposition`). Otherwise, it follows from the nef decomposition of the orientifold (`CICY_orientifold.yields_nef_decomposition`).
    
        **Returns:**
    
        - `bool | np.bool_`: `True` if the uplift defines a nef decomposition, otherwise `False`.
    
        """
        if self.__is_nef_decomposition is None:
            if self.orientifold().ambient_triangulation:
                self.__is_nef_decomposition = self.cicy().is_nef_decomposition()
            else:
                self.__is_nef_decomposition = self.orientifold().yields_nef_decomposition()
        return self.__is_nef_decomposition

    def intersection_numbers_N(self) -> dict[tuple[int, ...], float]:
        """
        **Description:**
    
        Computes and caches the intersection numbers of the uplift in the basis `basis_homology_N()`: the intersection numbers of `cicy` in terms of all prime toric divisors (`CICY.intersection_numbers`), restricted to the basis with `COF.tensor_symm_in_basis`, as `COF.subvariety_intersection_numbers` with a basis.
    
        **Returns:**
    
        - `dict[tuple[int, ...], float]`: The intersection numbers, keyed by sorted tuples of one-indexed basis positions.
    
        """
        if self.__intersection_numbers_N is None:
            self.__intersection_numbers_N=COF.tensor_symm_in_basis(self.cicy().intersection_numbers(),self.basis_homology_N(),in_prime_basis=False)
        return self.__intersection_numbers_N

    def __dual_partition(self, points: np.ndarray) -> np.ndarray:
        """
        **Description:**
    
        Computes the divisors of the dual nef partition on lattice points of `pol_M_conv()`: a nonzero point m has the coefficient -min_{n in Nabla_a} <m, n> in the a-th divisor, with Nabla_a the convex hull of the origin and the rays of the a-th part of the N-lattice partition (`pol_B_N()`, `pol_W_N()`), as `CICY.dual`. This is 1 if m lies in the a-th Newton polytope (`pol_B_M()`, `pol_W_M()`) and 0 otherwise, i.e. `line_bundle_base_M` and `line_bundle_weierstrass_M` of `F_Theory_Uplift`.
    
        **Arguments:**
    
        - `points (np.ndarray)`: Nonzero lattice points of `pol_M_conv()`, one per row.
    
        **Raises:**
    
        - `RuntimeError`: Raised if the coefficients do not form a partition, which the duality of nef partitions excludes.
    
        **Returns:**
    
        - `np.ndarray`: The (r + 1, number of points) coefficients, the r base divisors followed by the Weierstrass divisor.
    
        """
        V = self.vectors_smooth_uplift_ambient()
        divisors = np.array([-(points @ V[D==1].T).min(axis=1, initial=0) for D in self.__uplift_cicy().divisors()], dtype=int)
        if not (np.isin(divisors, (0, 1)).all() and (divisors.sum(axis=0) == 1).all()):
            raise RuntimeError("The dual divisors do not form a partition.")
        return divisors

    def intersecting_divisors_M(self, as_labels: bool = False) -> np.ndarray:
        """
        **Description:**
    
        Determines the nonzero lattice points of `pol_M_conv()` whose toric divisors intersect the dual complete intersection, defined by the dual nef partition (`COF.intersecting_points` with the N-lattice polytopes `pol_B_N()` and `pol_W_N()` as Newton polytopes).
    
        **Arguments:**
    
        - `as_labels (bool)`: If `True`, return the indices of the points in `pol_M_conv().points()`. Defaults to `False`.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef partition.
    
        **Returns:**
    
        - `np.ndarray`: The intersecting points, or their indices.
    
        """
        if not self.is_nef_partition():
            raise ValueError("Intersecting divisors can only be computed for nef partitions")
        if self.__intersecting_labels_M is None:
            points = np.asarray(self.pol_M_conv().points(), dtype=int)
            nonzero = np.flatnonzero(points.any(axis=1))
            self.__intersecting_labels_M = nonzero[COF.intersecting_points(points[nonzero],divisors=list(self.__dual_partition(points[nonzero])),polytopes=[*self.pol_B_N(),self.pol_W_N()],as_indices=True)]
        if as_labels:
            return self.__intersecting_labels_M
        else:
            return self.pol_M_conv().points()[self.__intersecting_labels_M]

    def _missing_points(self) -> np.ndarray:
        """
        **Description:**
    
        Determines the lattice points of `pol_M_conv()` at height <m, u> = 0 that are needed for the fibration over P^1 but are not among the intersecting points, as `F_Theory_Uplift._missing_points`: the points between a vertex at height -1 or -2 and the intersecting point of positive height.
    
        **Returns:**
    
        - `np.ndarray`: The missing points, one per row (possibly none).
    
        """
        l1=self.pol_M_conv().vertices()[self.pol_M_conv().vertices()@self.u==-1]
        l2=self.pol_M_conv().vertices()[self.pol_M_conv().vertices()@self.u==-2]
        points0=self.intersecting_divisors_M()[self.intersecting_divisors_M()@self.u==0]
        pos_point=self.intersecting_divisors_M()[self.intersecting_divisors_M()@self.u>0][0]
        p1s=np.rint(l1/2+pos_point/2).astype(int)
        p2s=np.rint(l2/3+2*pos_point/3).astype(int)
        miss=np.unique(COF.row_difference(np.concatenate((p1s,p2s),axis=0),points0),axis=0)
        if len(miss)>0:
            zero_ind=COF.get_index(miss,np.zeros(miss.shape[1],dtype=int))
            if len(zero_ind)>0:
                miss=np.delete(miss,zero_ind[0],0)
        return miss

    def vectors_M_conv_toric_fan(self, labels: ArrayLike | None = None) -> np.ndarray:
        """
        **Description:**
    
        Constructs and caches the rays used on the M-side, as `F_Theory_Uplift.vectors_M_conv_toric_fan`: the intersecting points of the Newton polytopes of the r base divisors (`pol_B_M()`) and of the Weierstrass divisor (`pol_W_M()`), each in the order of the points of its polytope, followed by the vertices of `pol_M_conv()` that are not among them and, if h' is a lattice polytope, by the points of `_missing_points()`. These are the rays of `dual_cicy()`.
    
        **Arguments:**
    
        - `labels (ArrayLike | None)`: Optional collection of one-indexed ray labels. Defaults to `None` (all rays).
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef partition.
    
        **Returns:**
    
        - `np.ndarray`: The selected rays, one per row.
    
        """
        if not self.is_nef_partition():
            raise ValueError("Vectors M-conv can only be computed for nef partitions")
        if self.__vectors_M_conv_toric_fan is None:
            int_vecs=np.vstack([*self.intersecting_divisors_B_M(),self.intersecting_divisors_W_M()])
            vert_diff=COF.row_difference(self.pol_M_conv().vertices(),int_vecs)
            if self.h_prime_is_lattice():
                all_points=np.vstack([int_vecs,vert_diff,self._missing_points()])
            else:
                all_points=np.vstack([int_vecs,vert_diff])
            self.__vectors_M_conv_toric_fan=np.asarray(all_points,dtype=int)
        if labels is not None:
            return self.__vectors_M_conv_toric_fan[np.array(labels)-1]
        return self.__vectors_M_conv_toric_fan

    def __mirror_cicy(self, with_fan: bool = False, with_glsm: bool = False) -> CICY:
        """
        **Description:**
    
        Constructs and caches the mirror of the uplift as a `CICY`: the complete intersection of the dual nef partition on exactly the rays `vectors_M_conv_toric_fan()` (all of them are kept), with the r base divisors followed by the Weierstrass divisor, as on the N-side. No fan is needed for this. With `with_fan` the fan `M_conv_toric_fan()` is constructed and set on the `CICY`, once; with `with_glsm` also the basis `basis_homology_M()` and the charge matrix `Qsat()` are set, again whenever `Qsat` or the fan has changed.
    
        **Arguments:**
    
        - `with_fan (bool)`: Whether the `CICY` must carry the fan `M_conv_toric_fan()`. Defaults to `False`.
        - `with_glsm (bool)`: Whether the `CICY` must also carry the basis `basis_homology_M()` and the GLSM charge matrix `Qsat()`. Defaults to `False`.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef partition.
    
        **Returns:**
    
        - `CICY`: The dual complete intersection.
    
        """
        if self.__dual_cicy is None:
            V = self.vectors_M_conv_toric_fan()
            self.__dual_cicy = CICY(V, self.__dual_partition(V), keep_all=True)
            if self.__cicy is not None:
                # the Hodge diamond is then computed only once for the mirror pair, see `CICY.hodge_numbers`
                self.__dual_cicy._link_mirror(self.__cicy)
        if (with_fan or with_glsm) and not self.__dual_cicy_has_fan:
            self.__dual_cicy.set_fan(self.M_conv_toric_fan())
            self.__dual_cicy_has_fan = True
        if with_glsm and not self.__dual_cicy_has_glsm:
            self.__dual_cicy.set_basis(self.basis_homology_M())
            self.__dual_cicy.set_glsm(self.Qsat())
            self.__dual_cicy_has_glsm = True
        return self.__dual_cicy

    def dual_cicy(self,with_fan=True, with_glsm=True) -> CICY:
        """
        **Description:**
    
        Returns the mirror of the uplift as a `CICY`: the complete intersection of the dual nef partition on exactly the rays `vectors_M_conv_toric_fan()`, with the r base divisors followed by the Weierstrass divisor, as on the N-side. A ray has coefficient 1 in the divisor whose Newton polytope (`pol_B_M()`, `pol_W_M()`) contains it. It carries the fan `M_conv_toric_fan()`, which is constructed on the first call. Once `Qsat` is used by one of the methods in the GLSM basis (`intersection_numbers_Qsat`, `c2_sat`, `c3_sat`), its basis is `basis_homology_M()` and its GLSM charge matrix is `Qsat()`.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef partition.
    
        **Returns:**
    
        - `CICY`: The dual complete intersection, of codimension r + 1 in ambient dimension `ambient_dim_base()` + 2.
    
        """
        return self.__mirror_cicy(with_fan=with_fan,with_glsm=with_glsm)

    def intersecting_divisors_B_M(self, as_labels: bool = False) -> list[np.ndarray]:
        """
        **Description:**
    
        For every base divisor, determines the points of its Newton polytope (`pol_B_M()`) whose toric divisors intersect the dual complete intersection, in the order of the points of the polytope.
    
        **Arguments:**
    
        - `as_labels (bool)`: If `True`, return the indices of the points in the corresponding polytope of `pol_B_M()`. Defaults to `False`.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef partition.
    
        **Returns:**
    
        - `list[np.ndarray]`: The intersecting points, or their indices, one array per base divisor.
    
        """
        if not self.is_nef_partition():
            raise ValueError("Intersecting divisors can only be computed for nef partitions")
        if self.__intersecting_labels_B_M is None:
            self.__intersecting_labels_B_M = [COF.get_indices(P.points(),self.intersecting_divisors_M()) for P in self.pol_B_M()]
        if as_labels:
            return self.__intersecting_labels_B_M
        return [P.points()[l] for P, l in zip(self.pol_B_M(), self.__intersecting_labels_B_M)]

    def intersecting_divisors_W_M(self, as_labels: bool = False) -> np.ndarray:
        """
        **Description:**
    
        Determines the points of the Newton polytope of the Weierstrass divisor (`pol_W_M()`) whose toric divisors intersect the dual complete intersection, in the order of the points of the polytope.
    
        **Arguments:**
    
        - `as_labels (bool)`: If `True`, return the indices of the points in `pol_W_M()`. Defaults to `False`.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef partition.
    
        **Returns:**
    
        - `np.ndarray`: The intersecting points, or their indices.
    
        """
        if not self.is_nef_partition():
            raise ValueError("Intersecting divisors can only be computed for nef partitions")
        if self.__intersecting_labels_W_M is None:
            self.__intersecting_labels_W_M = COF.get_indices(self.pol_W_M().points(),self.intersecting_divisors_M())
        if as_labels:
            return self.__intersecting_labels_W_M
        return self.pol_W_M().points()[self.__intersecting_labels_W_M]

    def divisor_weierstrass_M(self) -> np.ndarray:
        """
        **Description:**
    
        Returns the Weierstrass part of the dual nef partition as a divisor on the rays `vectors_M_conv_toric_fan()`, the last divisor of `dual_cicy` (`line_bundle_weierstrass_M(toric_fan_adapted=True)` of `F_Theory_Uplift`).
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef partition.
    
        **Returns:**
    
        - `np.ndarray`: The coefficients (0 or 1), ordered as `vectors_M_conv_toric_fan()`.
    
        """
        return self.__mirror_cicy().divisors()[-1]

    def divisors_base_M(self) -> np.ndarray:
        """
        **Description:**
    
        Returns the base parts of the dual nef partition as divisors on the rays `vectors_M_conv_toric_fan()`, the first r divisors of `dual_cicy` (`line_bundle_base_M(toric_fan_adapted=True)` of `F_Theory_Uplift`).
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef partition.
    
        **Returns:**
    
        - `np.ndarray`: The (r, N) coefficients (0 or 1), one divisor per row, ordered as `vectors_M_conv_toric_fan()`.
    
        """
        return self.__mirror_cicy().divisors()[:-1]

    def nef_partition_M(self, zero_based: bool = False, as_divisors: bool = False) -> list[np.ndarray] | np.ndarray:
        """
        **Description:**
    
        Returns the M-lattice nef partition on the rays `vectors_M_conv_toric_fan()`: the r base parts followed by the Weierstrass part (`nef_partition_M(toric_fan_adapted=True)` of `F_Theory_Uplift`). If the uplift is not a nef partition, returns empty tuples.
    
        **Arguments:**
    
        - `zero_based (bool)`: If `True`, return zero-indexed instead of one-indexed labels. Defaults to `False`.
        - `as_divisors (bool)`: Whether to return the 0/1 indicator vectors on the rays instead of labels. Defaults to `False`.
    
        **Returns:**
    
        - `list[np.ndarray] | np.ndarray`: The r + 1 parts as arrays of labels, or the (r + 1, N) array of indicator vectors.
    
        """
        if self.is_nef_partition():
            divisors = self.__mirror_cicy().divisors()
            if as_divisors:
                return divisors
            return [np.where(D==1)[0] + (0 if zero_based else 1) for D in divisors]
        else:
            return [() for _ in range(len(self.divisors_orbifold())+1)]

    def blow_down_fan(self) -> Fan:
        """
        **Description:**
    
        Constructs and caches the blow-down fan {rho_+, rho_-} * facefan(nabla_fib), as `F_Theory_Uplift.blow_down_fan`: the simplicial face fan of the polytope of the fiber rays (the rays of `vectors_M_conv_toric_fan()` at height <m, u> = 0, projected along u), refined by all fiber rays, joined with rho_+ = (0; 1, -1) and rho_- = (0; -2, 5).
    
        **Raises:**
    
        - `ValueError`: Raised if h' is not a lattice polytope.
    
        **Returns:**
    
        - `Fan`: The blow-down fan; its last two rays are rho_+ and rho_-.
    
        """
        if not self.h_prime_is_lattice():
            raise ValueError("h' is not a lattice this fan may not yield a fibration over P1")
        if self.__blow_down_fan is None:
            fiber_points=self.vectors_M_conv_toric_fan()[self.vectors_M_conv_toric_fan()@self.u==0]
            fiber_points_projected=COF.project_points(fiber_points,self.u)
            f_fib=COF.face_fan(Polytope(fiber_points_projected),simplicial=True)
            f_fib=COF.reorder_fan(COF.refine_fan(f_fib,fiber_points_projected),fiber_points_projected)
            rhos=np.zeros((2,self.ambient_dim_base()+2),dtype=int)
            rhos[:,-2:]=np.array([[1,-1],[-2,5]])
            vc_bd=VectorConfiguration(np.concatenate((fiber_points,rhos),axis=0))
            indp=len(fiber_points)+1
            indm=indp+1
            cones_bd=set()
            for c in f_fib.cones():
                cones_bd.add(c+(indp,))
                cones_bd.add(c+(indm,))
            self.__blow_down_fan=Fan(vc_bd,cones=cones_bd)
        return self.__blow_down_fan

    def M_conv_toric_fan(self) -> Fan:
        """
        **Description:**
    
        Constructs and caches the simplicial toric fan over the rays `vectors_M_conv_toric_fan()`, as `F_Theory_Uplift.M_conv_toric_fan`: if h' is a lattice polytope, the common refinement of the blow-down fan and the face fan of `pol_M_conv()`, refined by all rays, which is compatible with the fibration over P^1; otherwise the simplicial face fan of `pol_M_conv()` refined by all rays, with a warning. This is the fan of `dual_cicy()`.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef partition.
    
        **Returns:**
    
        - `Fan`: The M-side fan, with rays labelled in the order of `vectors_M_conv_toric_fan()`.
    
        """
        if not self.is_nef_partition():
            raise ValueError("Uplift is not a nef-partition")
        if not self.h_prime_is_lattice():
            warnings.warn("h'' is not a lattice this fan may not yield a fibration over P1")
        if self.__M_conv_toric_fan is None:
            if self.h_prime_is_lattice():
                f_common_ref=COF.common_fan_refinement(self.blow_down_fan(),COF.face_fan(self.pol_M_conv()),simplicial=True,vectors_fine_to_add=self.vectors_M_conv_toric_fan())
                newf=COF.refine_fan(f_common_ref,self.vectors_M_conv_toric_fan())
            else:
                newf=COF.refine_fan(COF.face_fan(self.pol_M_conv(),simplicial=True),self.vectors_M_conv_toric_fan())
            self.__M_conv_toric_fan=COF.reorder_fan(newf,self.vectors_M_conv_toric_fan())
        return self.__M_conv_toric_fan

    def set_M_conv_toric_fan(self,fan: Fan) -> None:
        """
        **Description:**
    
        Sets the M-side fan, relabelled to the order of `vectors_M_conv_toric_fan()` (`COF.reorder_fan`), sets it as the fan of `dual_cicy()` (`CICY.set_fan`, which resets its intersection numbers, Chern classes and Mori cone) and resets all quantities computed on the fan here. `Qd7` and `Qsat` are kept, as they are built from the blow-down fan.
    
        **Arguments:**
    
        - `fan (Fan)`: A simplicial fan over the rays `vectors_M_conv_toric_fan()`.
    
        **Returns:**
    
        - `None`
    
        """
        self.__M_conv_toric_fan = COF.reorder_fan(fan,self.vectors_M_conv_toric_fan())
        self.__mirror_cicy().set_fan(self.__M_conv_toric_fan)
        self.__dual_cicy_has_fan = True
        self.__dual_cicy_has_glsm = False
        self.__basis_homology_M = None
        self.__intersection_numbers_M = None
        self.__MC_face = {}
        self.__intersection_numbers_surfaces_sat = None
        self.__vertical_intersection_form = None

    def basis_homology_M(self) -> np.ndarray:
        """
        **Description:**
    
        Computes and caches a basis of prime toric divisors of the M-side ambient space (`COF.basis_homology` on `M_conv_toric_fan()`).
    
        **Returns:**
    
        - `np.ndarray`: The one-indexed labels of the basis divisors.
    
        """
        if self.__basis_homology_M is None:
            self.__basis_homology_M = COF.basis_homology(self.M_conv_toric_fan())
        return self.__basis_homology_M

    def intersection_numbers_M(self,tol: float = 1e-5,decimals: int = 10) -> dict[tuple[int, ...], float]:
        """
        **Description:**
    
        Computes and caches the intersection numbers of the dual complete intersection in the basis `basis_homology_M()`: the intersection numbers of `dual_cicy()` in terms of all prime toric divisors (`CICY.intersection_numbers`), restricted to the basis with `COF.tensor_symm_in_basis`, as `COF.subvariety_intersection_numbers` with a basis.
    
        **Arguments:**
    
        - `tol (float)`: Rounding tolerance, only used on the first call. Defaults to `1e-5`.
        - `decimals (int)`: Number of decimals to round to, only used on the first call. Defaults to `10`.
    
        **Raises:**
    
        - `ValueError`: Raised if the uplift is not a nef partition.
    
        **Returns:**
    
        - `dict[tuple[int, ...], float]`: The intersection numbers, keyed by sorted tuples of one-indexed basis positions.
    
        """
        if not self.is_nef_partition():
            raise ValueError("Intersection numbers on the M side can only be computed for nef partitions")
        if self.__intersection_numbers_M is None:
            self.__intersection_numbers_M=COF.tensor_symm_in_basis(self.dual_cicy().intersection_numbers(tol=tol,decimals=decimals),self.basis_homology_M(),in_prime_basis=False)
        return self.__intersection_numbers_M

    def c2_M(self) -> dict[tuple[int, ...], int]:
        """
        **Description:**
    
        Computes the second Chern class of the dual complete intersection by torus localization on `M_conv_toric_fan()` (`CICY.toric_chern_class` of `dual_cicy()`, which caches it).
    
        **Returns:**
    
        - `dict[tuple[int, ...], int]`: The tensor of c_2 in terms of the prime toric divisors.
    
        """
        return self.dual_cicy().toric_chern_class(2)

    def c3_M(self) -> dict[tuple[int, ...], int] | dict[int, int]:
        """
        **Description:**
    
        Computes the third Chern class of the dual complete intersection by torus localization on `M_conv_toric_fan()` (`CICY.toric_chern_class` of `dual_cicy()`, which caches it).
    
        **Returns:**
    
        - `dict[tuple[int, ...], int] | dict[int, int]`: The tensor of c_3 in terms of the prime toric divisors.
    
        """
        return self.dual_cicy().toric_chern_class(3)

    def fiber_vectors(self) -> np.ndarray:
        """
        **Description:**
    
        Returns the fiber rays: the rays of `vectors_M_conv_toric_fan()` at height <m, u> = 0, projected along u (`COF.project_points`).
    
        **Returns:**
    
        - `np.ndarray`: The projected fiber rays, one per row.
    
        """
        vecs=self.vectors_M_conv_toric_fan()[self.vectors_M_conv_toric_fan()@self.u==0]
        return COF.project_points(vecs,self.u)

    def Qd7(self,D7_only: bool = False) -> np.ndarray:
        """
        **Description:**
    
        Computes and caches the GLSM charge matrix of the M-side built from the blow-down fan and the D7 points (`COF.glsm_from_refinement`), whose columns are the rays `vectors_M_conv_toric_fan()` and whose rows are ordered as [intersecting fiber | intersecting D7 | tau | non-intersecting D7 | completing fiber] (see `block_sizes`).
    
        **Arguments:**
    
        - `D7_only (bool)`: Whether to return only the rows of the intersecting D7 points. Defaults to `False`.
    
        **Returns:**
    
        - `np.ndarray`: The charge matrix, or its D7 rows.
    
        """
        if self.__Qd7 is None:
            self.__Qd7,self.__block_sizes=COF.glsm_from_refinement(self)
        if D7_only:
            return self.__Qd7[self.block_sizes()[0]:self.block_sizes()[0]+self.block_sizes()[1]]
        return self.__Qd7

    def Qsat(self,D7_only: bool = False) -> np.ndarray:
        """
        **Description:**
    
        Computes and caches the saturation of `Qd7()` (`COF.saturate_glsm`), in which the D7 rows are only combined among themselves, or returns the matrix set with `set_Qsat`.
    
        **Arguments:**
    
        - `D7_only (bool)`: Whether to return only the rows of the intersecting D7 points. Defaults to `False`.
    
        **Returns:**
    
        - `np.ndarray`: The saturated charge matrix, or its D7 rows.
    
        """
        if self.__Qsat is None:
            Q=self.Qd7(); s=self.block_sizes()
            d7_rows=np.r_[s[0]:s[0]+s[1],s[0]+s[1]+1:s[0]+s[1]+1+s[3]]
            self.__Qsat=COF.saturate_glsm(Q,protected_rows=d7_rows)
        if D7_only:
            return self.__Qsat[self.block_sizes()[0]:self.block_sizes()[0]+self.block_sizes()[1]]
        return self.__Qsat

    def set_symmetric_glsm(self):
        Q=self.Qsat().copy()
        Q[self.D7_indices()]=COF.symmetric_so8_rows(Q[self.D7_indices()],d7_cols=self.Qd7(D7_only=True).argmin(axis=1))
        self.set_Qsat(Q)
        return None

    def block_sizes(self) -> np.ndarray:
        """
        **Description:**
    
        Returns the numbers of rows of the five blocks of `Qd7()` and `Qsat()`, in the order of their rows: intersecting fiber, intersecting D7, tau, non-intersecting D7, completing fiber. They are determined by `COF.glsm_from_refinement` together with `Qd7()`, for any codimension.
    
        **Returns:**
    
        - `np.ndarray`: The five block sizes.
    
        """
        if self.__block_sizes is None:
            self.Qd7()
        return self.__block_sizes

    def D7_indices(self) -> np.ndarray:
        """
        **Description:**
    
        Returns the zero-based indices of the rows of the intersecting D7 points in `Qd7()` and `Qsat()`.
    
        **Returns:**
    
        - `np.ndarray`: The row indices.
    
        """
        return np.arange(self.block_sizes()[0],self.block_sizes()[0]+self.block_sizes()[1])

    def set_Qsat(self,Qsat_new: ArrayLike) -> None:
        """
        **Description:**
    
        Sets the GLSM charge matrix returned by `Qsat()` and resets all quantities expressed in its basis. It is set as the GLSM charge matrix of `dual_cicy()` when one of them is next computed.
    
        **Arguments:**
    
        - `Qsat_new (ArrayLike)`: The new charge matrix, with the columns ordered as `vectors_M_conv_toric_fan()` and the same row blocks as `Qd7()`.
    
        **Returns:**
    
        - `None`
    
        """
        self.__Qsat=Qsat_new
        self.__dual_cicy_has_glsm=False
        self.__intersection_numbers_surfaces_sat=None
        self.__MC_face={}
        self.__vertical_intersection_form=None

    def intersection_numbers_Qsat(self) -> dict[tuple[int, ...], int | float]:
        """
        **Description:**
    
        Returns the intersection numbers of the dual complete intersection in the GLSM basis of the rows of `Qsat()`: `CICY.intersection_numbers(in_glsm_basis=True)` of `dual_cicy()` with the basis `basis_homology_M()` and the charge matrix `Qsat()`, which caches them.
    
        **Returns:**
    
        - `dict[tuple[int, ...], int | float]`: The intersection numbers, keyed by sorted tuples of one-indexed GLSM indices.
    
        """
        return self.__mirror_cicy(with_glsm=True).intersection_numbers(in_glsm_basis=True)

    def intersection_numbers_surfaces_sat(self) -> tuple[dict[tuple[int, int, int], int | float], dict[int, tuple[int, int]]]:
        """
        **Description:**
    
        Computes and caches the intersection numbers of the surfaces J_a J_b with pairs of basis divisors in the GLSM basis of `Qsat()` (`COF.intersection_number_surfaces`), the form used by `compute_d7_gv`.
    
        **Returns:**
    
        - `tuple[dict, dict]`: The intersection numbers {(surface index, i, j): value} and the dictionary of the surfaces.
    
        """
        if self.__intersection_numbers_surfaces_sat is None:
            self.__intersection_numbers_surfaces_sat=COF.intersection_number_surfaces(self.intersection_numbers_Qsat(),self.Qsat().shape[0])
        return self.__intersection_numbers_surfaces_sat

    def vertical_intersection_form(self) -> tuple:
        """
        **Description:**
    
        Computes and caches the intersection form on the vertical part of H^{2,2} in the GLSM basis of `Qsat()` (`COF.intersection_form_H22_vertical`).
    
        **Returns:**
    
        - `tuple`: The output of `COF.intersection_form_H22_vertical`.
    
        """
        if self.__vertical_intersection_form is None:
            self.__vertical_intersection_form = COF.intersection_form_H22_vertical(self.intersection_numbers_Qsat(),return_gram=False)
        return self.__vertical_intersection_form

    def c2_sat(self) -> dict[tuple[int, ...], int]:
        """
        **Description:**
    
        Returns the second Chern class of the dual complete intersection in the GLSM basis of `Qsat()` (`CICY.toric_chern_class(2, in_glsm_basis=True)` of `dual_cicy()`, which caches it).
    
        **Returns:**
    
        - `dict[tuple[int, ...], int]`: The tensor of c_2 in the GLSM basis.
    
        """
        return self.__mirror_cicy(with_glsm=True).toric_chern_class(2,in_glsm_basis=True)

    def c3_sat(self) -> dict[tuple[int, ...], int] | dict[int, int]:
        """
        **Description:**
    
        Returns the third Chern class of the dual complete intersection in the GLSM basis of `Qsat()` (`CICY.toric_chern_class(3, in_glsm_basis=True)` of `dual_cicy()`, which caches it).
    
        **Returns:**
    
        - `dict[tuple[int, ...], int] | dict[int, int]`: The tensor of c_3 in the GLSM basis.
    
        """
        return self.__mirror_cicy(with_glsm=True).toric_chern_class(3,in_glsm_basis=True)

    def mori_cone_face_d7(self, check_face: bool | str = False, mori_cone_cap: bool = True):
        """
        **Description:**

        Computes and caches the face of the Mori cone cut out by the curve classes in the row span of the D7 rows of `Qsat()`, in the basis of all rows of `Qsat()`. With `mori_cone_cap` it is the face of the Mori cone cap of `dual_cicy()` (`CICY.mori_cone_face` with the fan `M_conv_toric_fan()` and the charge matrix `Qsat()`), which contains the curves of the dual complete intersection and is in general smaller than the toric Mori cone, and only depends on the cones of the fan meeting the complete intersection; the classes of degree 0 on the prime toric divisors missing it are kept (`restrict_to_Y`). Otherwise it is the face of the toric Mori cone of `M_conv_toric_fan()` (`COF.mori_cone_in_glsm_basis`). The faces of both cones are cached separately.

        **Arguments:**

        - `check_face (bool | str)`: The face check of `COF.mori_cone_in_glsm_basis`: `True` (strict: the intersection of the cone with the row span of the D7 rows is a face), `"face"` (the generators in the row span span a face) or `False`. Only used on the first call. Defaults to `False`.
        - `mori_cone_cap (bool)`: Whether to take the face of the Mori cone cap instead of the toric Mori cone. Defaults to `True`.

        **Returns:**

        - `Cone`: The D7 face of the Mori cone cap or of the toric Mori cone.

        """
        key=bool(mori_cone_cap)
        if key not in self.__MC_face:
            if key:
                self.__MC_face[key]=self.dual_cicy(with_fan=True,with_glsm=True).mori_cone_face(self.D7_indices(),mori_cone_cap=True,as_face_in_higher_dim_ambient=True,check_face=check_face)
            else:
                self.__MC_face[key]=COF.mori_cone_in_glsm_basis(self.Qsat(),self.M_conv_toric_fan(),sub_rows=self.D7_indices(),as_face_in_higher_dim_ambient=True,check_face=check_face)
        return self.__MC_face[key]

    def compute_d7_gv(self,hilbert_basis: bool = False,min_points: int = 600,max_deg: int = 2,target_points: ArrayLike | None = None,mori_cone_cap: bool = True) -> list:
        """
        **Description:**

        Computes the genus-zero Gopakumar-Vafa invariants of the dual complete intersection for the curve classes on the D7 face of the Mori cone (`mori_cone_face_d7`), with `cygv.compute_gv`, the charge matrix `Qsat()`, the surface intersection numbers `intersection_numbers_surfaces_sat()` and the dual nef partition with its r + 1 parts.

        **Arguments:**

        - `hilbert_basis (bool)`: Whether to use the Hilbert basis of the face instead of its rays as generators. Defaults to `False`.
        - `min_points (int)`: Minimal number of curve classes to compute. Defaults to `600`.
        - `max_deg (int)`: Maximal degree with respect to the grading vector. Defaults to `2`.
        - `target_points (ArrayLike | None)`: Curve classes to compute the invariants for. Defaults to `None`.
        - `mori_cone_cap (bool)`: Whether to use the D7 face of the Mori cone cap instead of that of the toric Mori cone (`mori_cone_face_d7`). Defaults to `True`.

        **Returns:**

        - `list`: The output of `cygv.compute_gv`, pairs `((curve class, surface index), invariant)`.

        """
        face=self.mori_cone_face_d7(mori_cone_cap=mori_cone_cap)
        if hilbert_basis:
            gens=face.hilbert_basis()
        else:
            gens=face.rays()
        gv=compute_gv(generators=gens,min_points=min_points,grading_vector=face.find_grading_vector(),q=self.Qsat(),intnums=self.intersection_numbers_surfaces_sat()[0],max_deg=max_deg,nefpart=self.nef_partition_M(zero_based=True),target_points=target_points)
        return gv

