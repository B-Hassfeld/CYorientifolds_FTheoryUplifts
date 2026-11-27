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
# Description:  Helper functions of the Calabi-Yau complete intersection classes
#               `CICY.CICY`, `CICY_orientifold.CICY_orientifold` and
#               `CICY_orientifold.CICY_F_Theory_Uplift`, i.e. the counterpart of
#               `Uplift_functions` for the classes of `FT_CY`. All helpers are
#               available from this single module:
#               1. the functions of `Utility_functions` (star imports, as in
#                  `Uplift_functions`), including `hodge_number_utilities`
#                  (stringy Hodge diamonds, E-polynomials and Hodge numbers of
#                  generic hypersurfaces, from `Uplift_functions` and
#                  `Hodge_Number_computations`),
#               2. copies of the other functions of `Uplift_functions` needed by
#                  the CICY classes, so that they do not depend on that module,
#               3. the orientifold functions for Calabi-Yau complete
#                  intersections of any codimension r. For r = 1 they agree
#                  with the hypersurface functions of `Uplift_functions`.
#               Copies whose names clash with the functions of 3. carry the
#               suffix `_hypersurface`.
# -----------------------------------------------------------------------------
from __future__ import annotations
# 'standard' imports
from itertools import combinations, combinations_with_replacement, permutations, product
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from functools import cache
from math import factorial
import warnings
from typing import Any, TYPE_CHECKING
import math
# 3rd party imports
import numpy as np
from flint import fmpz, fmpz_mat
from scipy.linalg import qr
import scipy.sparse as sps
from scipy.spatial import ConvexHull
# CYTools imports
from cytools import h_polytope, Polytope
from cytools.vector_config.fan import Fan
from cygv import compute_gv
from numpy.typing import ArrayLike
# helper modules (in the same order as in `Uplift_functions`, so that the copies below see the same names)
from .Utility_functions.matrix_utilities import *
from .Utility_functions.matrix_utilities import _row_counts
from .Utility_functions.fan_utilities import *
from .Utility_functions.toric_geometry_utilities import *
from .Utility_functions.hodge_number_utilities import *
# private helpers are not exported by `import *`
from .Utility_functions.hodge_number_utilities import (
    _check_complete, _cones_by_dimension, _even_sublattice_basis, _face_e_polynomials,
    _hd_setup, _newton_polytope_incidence, _poly_mul, _pure_hodge_numbers,
    _resolve_backend, _saturated_kernel, _uv_minus_one_power)
# `CICY` imports this module, so its helpers are looked up on the module at call time (`cicy._partition_representatives`)
from . import CICY as cicy
if TYPE_CHECKING:
    from .FT_CY import CY_orientifold, F_Theory_Uplift
    from .CICY_orientifold import CICY_orientifold


# =============================================================================
# 2. Functions copied from `Uplift_functions` (2026-11-27), in their original
#    order (its Hodge-diamond helpers are in `hodge_number_utilities`). Only
#    the names of these six were changed, to avoid clashes with the
#    complete-intersection versions in section 3:
#   orientifold_fixed_locus   -> orientifold_fixed_locus_hypersurface
#   _fixed_locus              -> _fixed_locus_hypersurface
#   _orbit_pieces             -> _orbit_pieces_hypersurface
#   _singular_orbits          -> _singular_orbits_hypersurface
#   hodge_numbers_weighted    -> hodge_numbers_weighted_hypersurface
#   O7_holomorphic_data       -> O7_holomorphic_data_hypersurface
# =============================================================================

def find_trilayer_vertex(p_or_verts: Polytope|ArrayLike,as_index: bool=False)->np.ndarray|np.intp:
    """
    **Description:**
    Uses the GLSM charge matrix of the vertices to identify the trilayer vertex, i.e. the vertex whose GLSM charges are half the anticanonical class.
    **Arguments:**
    - `p_or_verts (Polytope | ArrayLike)`: The trilayer polytope, or its vertices (one per row).
    - `as_index (bool)`: Whether to return the vertex index rather than the vertex coordinates. Defaults to `False`.
    **Raises:**
    - `ValueError`: Raised if no vertex corresponds to half the anticanonical class, i.e. the polytope is not trilayer.
    **Returns:**
    - `np.ndarray | np.intp`: The trilayer vertex, or its row index in the vertex array (`p.vertices()` for a `Polytope`) if `as_index=True`.
    """
    V=np.asarray(p_or_verts.vertices() if isinstance(p_or_verts,Polytope) else p_or_verts,dtype=int)
    glsm_vert=glsm_from_points(V)
    index=np.where(np.all(2*glsm_vert.T==np.sum(glsm_vert,axis=1),axis=1))[0]
    if len(index)==0:
        raise ValueError("Polytope is not trilayer.")
    return index[0] if as_index else V[index[0]]

def lattice_refinement(q: Sequence[int]|np.ndarray,denominator: int=2)->np.ndarray:
    """
    **Description:**
    Returns the smallest integral embedding of the unit lattice into a refined lattice in which `q/denominator` becomes integral.
    **Arguments:**
    - `q (Sequence[int] | np.ndarray)`: Integer vector defining the fractional lattice refinement.
    - `denominator (int)`: Denominator of the fractional vector. Defaults to `2`.
    **Returns:**
    - `np.ndarray`: The lattice refinement map, an integer square matrix.
    """
    lattice_basis=denominator*np.identity(len(q)).astype(int)
    lattice_basis=np.vstack([denominator*np.identity(len(q)).astype(int),[q]])
    A=fmpz_mat(lattice_basis.tolist())
    A_lll=A.lll()
    scaled_up_basis=np.array(A_lll.tolist()).astype(int)
    vanishing_pos=np.where(np.all(scaled_up_basis==0,axis=1))[0]
    scaled_up_basis=np.delete(scaled_up_basis,vanishing_pos,axis=0)
    Lambda=np.linalg.inv(scaled_up_basis.T)
    return np.rint(Lambda*denominator).astype(int)

def toric_orbifold(pts_CY_ambient: np.ndarray,q: Sequence[int]|np.ndarray,denominator: int=2)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    **Description:**
    Applies the lattice refinement defined by `q/denominator` to the ambient toric rays and returns the primitive orbifold rays together with the ray rescalings and the refinement map.
    **Arguments:**
    - `pts_CY_ambient (np.ndarray)`: Rays of the original Calabi-Yau ambient toric fan, one per row.
    - `q (Sequence[int] | np.ndarray)`: Integer vector defining the fractional lattice refinement.
    - `denominator (int)`: Denominator of the fractional vector. Defaults to `2`.
    **Returns:**
    - `tuple[np.ndarray, np.ndarray, np.ndarray]`: A triple `(orbifold_points, rescalings, Lambda)` consisting of the primitive orbifold rays, the corresponding edge rescalings and the lattice refinement map from `lattice_refinement`.
    """
    Lambda=lattice_refinement(q,denominator)
    orbifold_points=pts_CY_ambient@(Lambda.T)
    rescalings=np.array([math.gcd(*list(i)) for i in orbifold_points])
    orbifold_points=np.rint((orbifold_points.T/rescalings).T).astype(int)
    return (orbifold_points,rescalings,Lambda)

def Z2_fixed_locus(vc_triangulation: Fan,q: ArrayLike,cone_dimension: int|None=None,denominator: int=2)->list[tuple[int,...]]:
    """
    **Description:**
    Finds cones whose associated toric strata are fixed by the lattice refinement defined by `q/denominator`. Optionally restricts to cones of a specified dimension.
    **Arguments:**
    - `vc_triangulation (Fan)`: The toric fan of the ambient variety.
    - `q (ArrayLike)`: Integer vector defining the fractional lattice refinement.
    - `cone_dimension (int | None)`: If specified, only cones with this number of rays are considered. Defaults to `None`.
    - `denominator (int)`: Denominator of the fractional vector. Defaults to `2`.
    **Returns:**
    - `list[tuple[int, ...]]`: Fixed-locus cones, represented as tuples of one-indexed ray labels.
    """
    cones=vc_triangulation.cones()
    if cone_dimension is None:
        dims=range(1,max(len(c) for c in cones))
    else:
        dims=[cone_dimension]
    V=np.asarray(vc_triangulation.vectors())
    out=[]
    for k in dims:
        cs=np.array(get_lower_dimensional_cones(cones,k),dtype=int).reshape(-1,k)
        fixed=np.all(np.mod(V[cs-1].sum(axis=1)+q,denominator)==0,axis=1)
        out.extend(tuple(int(l) for l in c) for c in cs[fixed])
    return out

def NHC_separating_rays(points: ArrayLike,nhc: ArrayLike,candidates: ArrayLike)->np.ndarray:
    """
    **Description:**
    Selects the candidate rays that are the primitive vector of v+v' for two NHC rays v,v', i.e. the rays whose stellar
    subdivisions blow up the intersections D_v cap D_v' of NHC divisors (inserted into the 2-cone (v,v'), such a ray removes
    it). Used by `CY_orientifold` to keep these rays in the refined orbifold fan, so that no two NHC divisors meet in the
    ambient space. The pair sums are formed at once and looked up among the candidates by sorting (`_row_counts`), in
    O(len(nhc)^2+len(candidates)) up to logarithms, with exact integer arithmetic.
    **Arguments:**
    - `points (ArrayLike)`: The integer rays, one per row.
    - `nhc (ArrayLike)`: Zero-based row indices of the NHC rays.
    - `candidates (ArrayLike)`: Zero-based row indices of the rays to test.
    **Returns:**
    - `np.ndarray`: The zero-based indices of the selected candidates, in the order of `candidates`.
    """
    P=np.asarray(points,dtype=np.int64)
    nhc=np.asarray(nhc,dtype=int)
    candidates=np.asarray(candidates,dtype=int)
    if len(nhc)<2 or len(candidates)==0:
        return np.array([],dtype=int)
    a,b=np.triu_indices(len(nhc),1)
    S=P[nhc[a]]+P[nhc[b]]
    g=np.gcd.reduce(np.abs(S),axis=1)
    S=S[g>0]//g[g>0,None]
    return candidates[_row_counts(P[candidates],S)>0]

def hodge_numbers_weighted_hypersurface(O: CY_orientifold)->dict[str,int|np.integer]:
    """
    **Description:**
    Computes the orientifold-even and -odd Hodge numbers `h^{1,1}_+`, `h^{1,1}_-`, `h^{2,1}_+` and `h^{2,1}_-` of the Calabi-Yau threefold of a reflexive polytope under the involution of `O`. `h^{1,1}_-` is zero for favorable polytopes and is otherwise obtained from the non-favorable 2-faces whose dual faces are exchanged by the involution; `h^{2,1}_-` follows from the Euler characteristic of the fixed locus via `h^{2,1}_- = h^{1,1}_- + (chi_I(O) - chi(X))/4 - 1` (Lefschetz), where chi(X) of the double cover described by the toric data of `O` must agree with the polytope.
    **Arguments:**
    - `O (CY_orientifold)`: The Calabi-Yau orientifold.
    **Raises:**
    - `ValueError`: Raised if the polytope of `O` is not reflexive, if the orientifold-invariant Calabi-Yau is singular, or if the toric data of `O` describe a different Calabi-Yau than the polytope (e.g. after a refinement of the orbifold fan).
    **Returns:**
    - `dict[str,int|np.integer]`: Dictionary with keys `"h11+"`, `"h11-"`, `"h21+"` and `"h21-"`.
    """
    pol=O.polytope()
    if not pol.is_reflexive():
        raise ValueError("Weighted Hodge Numbers can only be computed if the original CY is defined by a reflexive polytope")
    if not pol.dim()==4:
        raise ValueError("Weighted Hodge Numbers can only be computed if the original CY is defined by a reflexive polytope of dimension 4")
    hodge_numbers={}
    if pol.is_favorable(lattice="N"):
        hodge_numbers["h11-"]=0
        hodge_numbers["h11+"]=pol.h11()
    else:
        h11m=0
        for f in pol.faces(2):
            cf=len(f.interior_points())
            if cf>0:
                fd=f.dual_face()
                cfd=len(fd.interior_points())
                if cfd>0:
                    if cfd%2==1:
                        fdverts=fd.vertices()
                        dq=np.round((fdverts[1]-fdverts[0])/(cfd+1)).astype(int)
                        if (dq@O.xi())%2==1:
                            h11m+=np.round(cf*(cfd+1)/2).astype(int)
        hodge_numbers["h11+"]=pol.h11()-h11m
        hodge_numbers["h11-"]=h11m
    fixed=_fixed_locus_hypersurface(O,return_chi=True)
    if fixed["chi_X"]!=pol.chi(lattice="N"):
        raise ValueError(f"The toric data of the orientifold describe a Calabi-Yau with chi={fixed['chi_X']}, not the one of the polytope (chi={pol.chi(lattice='N')}).")
    hodge_numbers["h21-"]=hodge_numbers.get("h11-")+(fixed["chi_I"]-fixed["chi_X"])//4-1
    hodge_numbers["h21+"]=pol.h21()-hodge_numbers.get("h21-")
    return hodge_numbers

def orientifold_mod2_character(xi: ArrayLike)->np.ndarray:
    """
    **Description:**
    Character mu of the orbifold lattice N'=N+Z*xi/2 (coordinates of `lattice_refinement(xi)`, i.e. of `vectors_orbifold()`)
    that detects the original lattice: N={n' in N' : <mu,n'> even}. A ray of N' is mu-odd iff it is not in N, i.e. iff
    its toric divisor is pointwise fixed by the orientifold involution.
    **Arguments:**
    - `xi (ArrayLike)`: The integer vector of the Z_2 action xi/2.
    **Raises:**
    - `ValueError`: Raised if `xi` is even, i.e. the action is trivial.
    **Returns:**
    - `np.ndarray`: mu with entries in {0,1}.
    """
    Lam=lattice_refinement(np.asarray(xi,dtype=int))
    d=len(Lam)
    for bits in range(1,2**d):
        mu=(bits>>np.arange(d))&1
        if not np.any((mu@Lam)%2):
            return mu
    raise ValueError("xi/2 does not define an index-2 refinement (xi = 0 mod 2?).")

def _normalized_volume(F: np.ndarray,K: np.ndarray)->int:
    """
    **Description:**
    Normalized volume of conv(F) in the affine lattice F[0]+Z<columns of K>, where K is a saturated lattice basis and
    conv(F) is full-dimensional in it.
    **Arguments:**
    - `F (np.ndarray)`: The points (at least the vertices), one per row.
    - `K (np.ndarray)`: The lattice basis, as columns.
    **Returns:**
    - `int`: k! times the volume of conv(F) in the coordinates of K.
    """
    k=K.shape[1]
    c=np.rint(np.linalg.lstsq(K.astype(float),(F-F[0]).T.astype(float),rcond=None)[0].T).astype(int)
    if k==1:
        return int(c.max()-c.min())
    hull=ConvexHull(c)
    if k==2:
        x,y=c[hull.vertices].T
        return abs(int(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1))))
    return int(round(hull.volume*math.factorial(k)))

def _orbit_pieces_hypersurface(V: np.ndarray,cones: np.ndarray,W: np.ndarray,E: np.ndarray,cache: dict)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    **Description:**
    Euler characteristics and dimensions of a generic hypersurface on the torus orbits O(cone) of equal-dimensional cones.
    The monomials surviving on O(cone) are the lattice points of the face of the Newton polytope cut out by the rays of
    the cone. With k the orbit dimension: no surviving monomial means O(cone) lies in the hypersurface (chi=[k=0]); one
    monomial means the orbit is missed; otherwise Kouchnirenko gives chi=(-1)^(k-1) Vol_k(face), normalized in the lattice
    cone^perp, which vanishes unless the face is k-dimensional. Then cone^perp is the direction of the face, so the volume
    only depends on the face and is cached per face.
    **Arguments:**
    - `V (np.ndarray)`: The rays, one per row.
    - `cones (np.ndarray)`: Cones of one dimension, as rows of zero-based ray indices (shape (C,0) for the zero cone).
    - `W (np.ndarray)`: The vertices of the Newton polytope, one per row.
    - `E (np.ndarray)`: Boolean matrix, E[i,j] if vertex j lies on the face cut out by ray i.
    - `cache (dict)`: Face cache shared between calls with the same `W`.
    **Returns:**
    - `tuple[np.ndarray,np.ndarray,np.ndarray]`: (chi, dim, number of face vertices) per cone, with dim=-1 for empty pieces.
    """
    n=V.shape[1]
    k=n-cones.shape[1]
    masks=E[cones].all(axis=1) if cones.shape[1] else np.ones((len(cones),len(W)),dtype=bool)
    n_vert=masks.sum(axis=1)
    chi=np.zeros(len(cones),dtype=np.int64)
    dim=np.where(n_vert==0,k,np.where(n_vert==1,-1,k-1))
    if k==0:
        chi[n_vert==0]=1
    for i in np.flatnonzero(n_vert>=2):
        key=masks[i].tobytes()
        if key not in cache:
            F=W[masks[i]]
            cache[key]=np.linalg.matrix_rank(F[1:]-F[0]),None,F
        rank,vol,F=cache[key]
        if rank<k:
            continue
        if vol is None:
            K=_saturated_kernel(V[cones[i]])
            vol=_normalized_volume(F,K)
            cache[key]=rank,vol,F
        chi[i]=(-1)**(k-1)*vol
    return chi,dim,n_vert

def _fixed_cones(V: np.ndarray,cones: np.ndarray,mu: np.ndarray,odd: np.ndarray)->np.ndarray:
    """
    **Description:**
    Tests which cones are pointwise fixed by the orientifold involution, i.e. whether mu is odd on the saturated lattice
    span(cone) cap N'. Cones with a mu-odd ray are fixed, maximal cones always are, and otherwise only cones whose rays span
    a sublattice of even index can be fixed (then the saturation is computed explicitly).
    **Arguments:**
    - `V (np.ndarray)`: The orbifold rays, one per row.
    - `cones (np.ndarray)`: The cones of one dimension, as rows of zero-based ray indices.
    - `mu (np.ndarray)`: The character from `orientifold_mod2_character`.
    - `odd (np.ndarray)`: Boolean array, True for mu-odd rays.
    **Returns:**
    - `np.ndarray`: Boolean array with one entry per cone.
    """
    n=V.shape[1]
    d=cones.shape[1]
    if d==n:
        return np.ones(len(cones),dtype=bool)
    fixed=odd[cones].any(axis=1)
    rest=np.flatnonzero(~fixed)
    if len(rest)==0:
        return fixed
    M=V[cones[rest]].astype(float)
    minors=np.stack([np.rint(np.linalg.det(M[:,:,list(c)])) for c in combinations(range(n),d)],axis=1).astype(np.int64)
    index=np.gcd.reduce(np.abs(minors),axis=1)
    for i in rest[index%2==0]:
        span=_saturated_kernel(_saturated_kernel(V[cones[i]]).T)
        fixed[i]=bool(np.any((mu@span)%2))
    return fixed

def _newton_data(V: np.ndarray,D: np.ndarray)->tuple[h_polytope.HPolytope,np.ndarray,np.ndarray]:
    """
    **Description:**
    Newton polytope of a toric divisor, its vertices and the vertex-face incidences of the rays.
    **Arguments:**
    - `V (np.ndarray)`: The rays, one per row.
    - `D (np.ndarray)`: Divisor coefficients in the basis of the rays.
    **Returns:**
    - `tuple[h_polytope.HPolytope,np.ndarray,np.ndarray]`: (Newton polytope, vertices W, E) with E[i,j] if <W_j,V_i>=-D_i.
    """
    NP=Newton_Polytope(V,D)
    W=np.asarray(NP.vertices(),dtype=int).reshape(-1,V.shape[1])
    return NP,W,(V@W.T+D[:,None])==0

def _singular_orbits_hypersurface(V: np.ndarray,V_up: np.ndarray,r: np.ndarray,L: np.ndarray,NP: h_polytope.HPolytope,E: np.ndarray,cones_by_dim: list[np.ndarray])->list[tuple[tuple[int,...],str]]:
    """
    **Description:**
    Finds the torus orbits along which the orientifold-invariant Calabi-Yau X is singular. X is the double cover of B,
    cut out upstairs by the pull-backs of the sections of B, whose exponent of x_i is r_i times the exponent on B
    (r_i=2 for mu-odd rays). On an orbit O(tau) that X meets (not exactly one surviving monomial), X is singular if the
    upstairs cone is not unimodular (X passes through a quotient singularity). If at least two monomials survive, the
    generic restriction is smooth. If none survives, O(tau) lies in X and X=sum_j x_j f_j+O(x^2) near it, so X is singular
    exactly at the common zeros on O(tau) of the first-order coefficients f_j, j in tau. Their supports T_j are the lattice
    points whose upstairs exponents along tau are the unit vector e_j; a generic system has a common zero iff
    dim(sum_{j in I} T_j)>=|I| for every set I of nonempty supports (Bernstein).
    **Arguments:**
    - `V (np.ndarray)`: The orbifold rays, one per row.
    - `V_up (np.ndarray)`: The corresponding primitive rays of the Calabi-Yau ambient lattice N, one per row.
    - `r (np.ndarray)`: Ray rescalings (2 for mu-odd rays, else 1).
    - `L (np.ndarray)`: The orientifold line bundle.
    - `NP (h_polytope.HPolytope)`: Its Newton polytope.
    - `E (np.ndarray)`: Vertex-face incidences, see `_newton_data`.
    - `cones_by_dim (list[np.ndarray])`: All cones of the fan, see `_cones_by_dimension`.
    **Returns:**
    - `list[tuple[tuple[int,...],str]]`: The singular orbits as (one-indexed cone, reason).
    """
    n=V.shape[1]
    points=None
    out=[]
    for cones in cones_by_dim:
        d=cones.shape[1]
        n_vert=E[cones].all(axis=1).sum(axis=1)
        meets=np.flatnonzero(n_vert!=1)
        if len(meets)==0:
            continue
        M=V_up[cones[meets]].astype(float)
        minors=np.stack([np.rint(np.abs(np.linalg.det(M[:,:,list(c)]))) for c in combinations(range(n),d)],axis=1).astype(np.int64)
        for i in meets[np.gcd.reduce(minors,axis=1)!=1]:
            out.append((tuple(int(x)+1 for x in cones[i]),"quotient singularity"))
        for i in np.flatnonzero(n_vert==0):
            c=cones[i]
            points=np.asarray(NP.points(),dtype=int) if points is None else points
            exps=r[c][:,None]*(V[c]@points.T+L[c][:,None])
            first=[points[(exps.sum(axis=0)==1)&(exps[j]==1)] for j in range(d)]
            first=[T[1:]-T[0] for T in first if len(T)]
            solvable=all(np.linalg.matrix_rank(np.vstack([first[j] for j in I]).reshape(-1,n))>=len(I) if any(len(first[j]) for j in I) else False for k in range(1,len(first)+1) for I in combinations(range(len(first)),k))
            if solvable:
                out.append((tuple(int(x)+1 for x in c),"first-order terms vanish simultaneously"))
    return out

def orientifold_fixed_locus_hypersurface(O: CY_orientifold|F_Theory_Uplift,check: bool=True,return_chi: bool=False)->dict:
    """
    **Description:**
    Fixed locus of the orientifold involution on the Calabi-Yau X, computed on B=X/sigma in the orbifold toric variety
    (`vectors_orbifold()`, `line_bundle()`, `orbifold_toric_fan()`). X is the double cover of B for the toric data of `O`,
    i.e. if the orbifold fan was refined (nef decomposition, A1 blow-ups) the corresponding modification of the original
    Calabi-Yau.
    The fixed locus is the disjoint union of the pieces B cap O(tau) over the pointwise fixed cones tau (all maximal cones,
    all cones with a mu-odd ray, and cones whose rays span an even-index sublattice on whose saturation mu is odd), since a
    torus element acts freely on all other orbits. On a fixed orbit of dimension k the surviving monomials are the lattice
    points S of a face of the Newton polytope of B: the piece is O(tau) itself if S is empty, empty if |S|=1, and otherwise
    a generic hypersurface in (C*)^k with Euler characteristic (-1)^(k-1) Vol_k(S) (Kouchnirenko, normalized in tau^perp).
    The O7-planes are the closures of the 2-dimensional pieces: mu-odd rays meeting B and fixed 2-cones whose orbit lies in
    B (non-divisorial O7-planes); the pieces on cones containing one of them add up to chi_O7. All remaining pieces must be
    points, the O3-planes: lattice lengths on fixed 3-cones and torus-fixed points lying on X.
    **Arguments:**
    - `O (CY_orientifold|F_Theory_Uplift)`: The Calabi-Yau orientifold or an F-theory uplift.
    - `check (bool)`: Whether to verify that X is smooth (see `_singular_orbits_hypersurface`) and that the O7-planes are disjoint, as
      the O-plane interpretation requires. Defaults to `True`.
    - `return_chi (bool)`: Whether to also compute chi(B) and chi(X)=2chi(B)-chi_I by orbit sums over all cones, and to
      verify the Lefschetz congruence chi_I = chi(X) mod 4. Defaults to `False`.
    **Raises:**
    - `ValueError`: Raised if the fixed locus contains a curve or a fixed divisor inside B, or (with `check`) if X is
      singular, or (with `return_chi`) if the Lefschetz congruence fails.
    **Returns:**
    - `dict`: `O7_rays` (one-indexed labels of `vectors_orbifold()`), `O7_two_cones` (non-divisorial O7-planes),
      `chi_O7_components` (O7-plane -> Euler characteristic), `O3_cones` (cone -> number of O3-planes on its orbit),
      `chi_O7`, `n_O3`, `chi_I` and, with `return_chi`, `chi_B` and `chi_X`.
    """
    O=O.orientifold() if hasattr(O,"orientifold") else O
    if O.line_bundle() is None:
        raise ValueError("The orientifold is not regular (no projected-in monomial), so B and its fixed locus are not defined.")
    V=np.asarray(O.vectors_orbifold(),dtype=int)
    if V.shape[1]!=4:
        raise ValueError("The fixed locus is only implemented for Calabi-Yau threefolds (four-dimensional ambient space).")
    L=np.asarray(O.line_bundle(),dtype=int)
    NP,W,E=_newton_data(V,L)
    mu=orientifold_mod2_character(O.xi())
    odd=(V@mu)%2==1
    r=1+odd.astype(int)
    cones_by_dim=_cones_by_dimension(O.orbifold_toric_fan())
    if check:
        Lam=lattice_refinement(np.asarray(O.xi(),dtype=int))
        V_up=np.rint(np.linalg.solve(Lam,(r[:,None]*V).T).T).astype(int)
        singular=_singular_orbits_hypersurface(V,V_up,r,L,NP,E,cones_by_dim)
        if singular:
            raise ValueError(f"The orientifold-invariant Calabi-Yau is singular along the orbits of {singular[:5]}{' ...' if len(singular)>5 else ''}.")
    cache={}
    fixed_pieces=[]
    for cones in cones_by_dim:
        fixed=cones[_fixed_cones(V,cones,mu,odd)]
        fixed_pieces.append((fixed,)+_orbit_pieces_hypersurface(V,fixed,W,E,cache))
    rays,_,dim,n_vert=fixed_pieces[0]
    if np.any(n_vert==0):
        raise ValueError(f"The fixed divisors {rays[n_vert==0,0]+1} are contained in B.")
    two,_,_,n_vert2=fixed_pieces[1]
    seed_rays=rays[dim==2,0]
    seeds=[(int(i),) for i in seed_rays]+[tuple(int(x) for x in c) for c in two[(n_vert2==0)&~np.isin(two,seed_rays).any(axis=1)]]
    chi_O7={s:0 for s in seeds}
    o3={}
    for cones,chi,dim,n_vert in fixed_pieces:
        owners=np.stack([np.isin(cones,s).sum(axis=1)==len(s) for s in seeds],axis=1) if seeds else np.zeros((len(cones),0),dtype=bool)
        for i in np.flatnonzero(dim>=0):
            label=tuple(int(x)+1 for x in cones[i])
            own=np.flatnonzero(owners[i])
            if len(own):
                if check and len(own)>1:
                    raise ValueError(f"The O7-planes {[tuple(x+1 for x in seeds[j]) for j in own]} meet on the orbit of cone {label}.")
                chi_O7[seeds[own[0]]]+=int(chi[i])
            elif dim[i]>0:
                raise ValueError(f"The fixed locus contains a {dim[i]}-dimensional piece on the orbit of cone {label} that is not part of an O7-plane.")
            elif chi[i]:
                o3[label]=int(chi[i])
    out={"O7_rays":np.array(sorted(s[0]+1 for s in seeds if len(s)==1),dtype=int),
         "O7_two_cones":sorted(tuple(x+1 for x in s) for s in seeds if len(s)==2),
         "chi_O7_components":{tuple(x+1 for x in s):v for s,v in chi_O7.items()},
         "O3_cones":o3,"chi_O7":int(sum(chi_O7.values())),"n_O3":int(sum(o3.values()))}
    out["chi_I"]=out["chi_O7"]+out["n_O3"]
    if return_chi:
        chi_B=int(sum(_orbit_pieces_hypersurface(V,cones,W,E,cache)[0].sum() for cones in [np.zeros((1,0),dtype=int)]+cones_by_dim))
        chi_X=2*chi_B-out["chi_I"]
        if chi_X%2 or (out["chi_I"]-chi_X)%4:
            raise ValueError(f"chi(X)={chi_X} and chi_I={out['chi_I']} violate the Lefschetz congruence chi_I = chi(X) = 0 mod 2, chi_I = chi(X) mod 4.")
        out["chi_B"]=chi_B
        out["chi_X"]=chi_X
    return out

def _fixed_locus_hypersurface(O: CY_orientifold|F_Theory_Uplift,return_chi: bool=False)->dict:
    """
    **Description:**
    Cached `orientifold_fixed_locus_hypersurface` of a Calabi-Yau orientifold or of the orientifold of an F-theory uplift.
    **Arguments:**
    - `O (CY_orientifold|F_Theory_Uplift)`: The Calabi-Yau orientifold or F-theory uplift.
    - `return_chi (bool)`: Whether chi(B) and chi(X) are needed. Defaults to `False`.
    **Returns:**
    - `dict`: The output of `orientifold_fixed_locus_hypersurface`.
    """
    O=O.orientifold() if hasattr(O,"orientifold") else O
    return O.fixed_locus(return_chi=return_chi) if hasattr(O,"fixed_locus") else orientifold_fixed_locus_hypersurface(O,return_chi=return_chi)

def O7_holomorphic_data_hypersurface(O: CY_orientifold|F_Theory_Uplift,check: bool=True)->dict:
    """
    **Description:**
    Holomorphic invariants of the O7-planes S of `orientifold_fixed_locus_hypersurface` (keyed as in `chi_O7_components`): K_S^2,
    chi(O_S), the number of connected components, the irregularity q and the geometric genus
    p_g=h^{2,0}=chi(O_S)-#components+q.
    For a divisorial O7-plane S=D_i cap B, adjunction in the orbifold toric variety gives K_S=(K_V+B+D_i)|_S, so
    K_S^2=(K_V+B+D_i)^2 D_i B with the intersection numbers of `orbifold_toric_fan()`, and Noether's formula gives
    chi(O_S)=(K_S^2+chi(S))/12 with chi(S) from `chi_O7_components`. The face F of the Newton polytope of B on D_i fixes
    the topology (Danilov-Khovanskii): dim F=1 of lattice length k gives k disjoint toric surfaces, dim F=2 a connected
    surface birationally ruled over a curve of genus q=#(interior lattice points of F), dim F=3 a connected surface
    with q=0. Non-divisorial O7-planes are toric surfaces: chi(O_S)=1, p_g=q=0, K_S^2=12-chi(S).
    **Arguments:**
    - `O (CY_orientifold|F_Theory_Uplift)`: The Calabi-Yau orientifold or F-theory uplift.
    - `check (bool)`: Whether to verify (i) K_S=D_i|_S/2, which follows from -2K_B ~ sum of the disjoint O7-planes,
      (ii) the integrality of K_S^2 and chi(O_S) and the values chi(O_S)=k (dim F=1) and 1-q (dim F=2), and (iii) the
      holomorphic Lefschetz formula chi_I=16+8*sum_S chi(O_S). Defaults to `True`.
    **Raises:**
    - `ValueError`: Raised (with `check`) if one of these tests fails.
    **Returns:**
    - `dict`: `p_g`, `chi_O`, `K2`, `components`, `q` (each a dict O7-plane -> int) and the totals `sum_p_g`,
      `sum_chi_O`.
    """
    O=O.orientifold() if hasattr(O,"orientifold") else O
    fl=_fixed_locus_hypersurface(O)
    V=np.asarray(O.vectors_orbifold(),dtype=int)
    L=np.asarray(O.line_bundle(),dtype=int)
    _,W,E=_newton_data(V,L)
    IN={k:v for k,v in O.intersection_numbers_orbifold().items() if min(k)>=1}
    K=np.array(list(IN),dtype=int)-1
    P=[K[:,s] for s in permutations(range(4))]
    w=np.array(list(IN.values()),dtype=float)/sum((p==K).all(axis=1) for p in P)
    T=lambda x0,x1,x2,x3: float(sum((w*x0[p[:,0]]*x1[p[:,1]]*x2[p[:,2]]*x3[p[:,3]]).sum() for p in P))
    out={key:{} for key in ("p_g","chi_O","K2","components","q")}
    for S,chi_S in fl["chi_O7_components"].items():
        if len(S)==2:
            chi_O,K2,c,q=1,12-chi_S,1,0
        else:
            e=np.zeros(len(V)); e[S[0]-1]=1
            a=L+e-1
            K2=T(e,a,a,L)
            F=W[E[S[0]-1]]
            r=np.linalg.matrix_rank(F[1:]-F[0])
            if r==1:
                c,q=int(np.gcd.reduce(F[1]-F[0])),0
            elif r==2:
                span=_saturated_kernel(_saturated_kernel(F[1:]-F[0]).T)
                x=np.rint(np.linalg.lstsq(span.astype(float),(F-F[0]).T.astype(float),rcond=None)[0].T).astype(int)
                hull=x[ConvexHull(x).vertices]
                b=int(np.gcd.reduce(np.abs(np.roll(hull,-1,axis=0)-hull),axis=1).sum())
                c,q=1,(_normalized_volume(F,span)-b+2)//2
            else:
                c,q=1,0
            chi_O=(K2+chi_S)/12
            if check:
                if not np.isclose(K2,T(e,e,e,L)/4,atol=1e-6):
                    raise ValueError(f"O7-plane {S}: adjunction gives K^2={K2}, the orientifold formula D^3B/4={T(e,e,e,L)/4}.")
                if not (np.isclose(K2,round(K2),atol=1e-6) and np.isclose(chi_O,round(chi_O),atol=1e-6)):
                    raise ValueError(f"O7-plane {S}: non-integral K^2={K2} or chi(O)={chi_O}.")
            K2,chi_O=int(round(K2)),int(round(chi_O))
            if check and ((r==1 and chi_O!=c) or (r==2 and chi_O!=1-q)):
                raise ValueError(f"O7-plane {S}: chi(O)={chi_O} contradicts the topology (face dimension {r}, {c} components, q={q}).")
        out["p_g"][S],out["chi_O"][S],out["K2"][S],out["components"][S],out["q"][S]=chi_O-c+q,chi_O,K2,c,q
    out["sum_p_g"]=sum(out["p_g"].values())
    out["sum_chi_O"]=sum(out["chi_O"].values())
    if check and 16+8*out["sum_chi_O"]!=fl["chi_I"]:
        raise ValueError(f"Holomorphic Lefschetz fails: chi_I={fl['chi_I']}, but 16+8*sum chi(O_S)={16+8*out['sum_chi_O']}.")
    return out

def _integer_solve(B: ArrayLike,b: ArrayLike,atol: float=1e-8)->np.ndarray|None:
    """
    **Description:**
    Solve Bx = b numerically and return x if it is sufficiently
    close to an integer vector.
    **Arguments:**
    - `B (ArrayLike)`: Square coefficient matrix.
    - `b (ArrayLike)`: Right-hand side vector.
    - `atol (float)`: Absolute tolerance for integrality and for the residual. Defaults to `1e-8`.
    **Returns:**
    - `np.ndarray|None`: The integer solution vector, or `None` if B is singular or the solution is not integral.
    """
    B=np.asarray(B,dtype=float)
    b=np.asarray(b,dtype=float)
    try:
        x=np.linalg.solve(B,b)
    except np.linalg.LinAlgError:
        return None
    x_int=np.rint(x).astype(int)
    if not np.allclose(x,x_int,atol=atol,rtol=0):
        return None
    if not np.allclose(B@x_int,b,atol=atol,rtol=0):
        return None
    return x_int

def sums_to_anticanonical(points: ArrayLike,divisors: ArrayLike,atol: float=1e-8)->tuple[bool,np.ndarray|None]:
    """
    **Description:**
    Determine whether L1 + ... + Lr is linearly equivalent to
    the anticanonical divisor.
    **Arguments:**
    - `points (ArrayLike)`: The ray generators of the toric fan, one per row.
    - `divisors (ArrayLike)`: The coefficient vectors of L1,...,Lr in the prime toric divisor basis, one per row.
    - `atol (float)`: Numerical tolerance for the integrality check. Defaults to `1e-8`.
    **Returns:**
    - `tuple[bool,np.ndarray|None]`: `(equivalent, m)`, where `m` is the integer vector with `points@m = 1 - (L1 + ... + Lr)` if it exists, otherwise `None`.
    """
    points=np.asarray(points,dtype=int)
    divisors=np.asarray(divisors,dtype=int)
    total=np.sum(divisors,axis=0)
    b=1-total
    if np.all(b==0):
        return True, np.zeros(points.shape[1],dtype=int)
    # the rays not in a Q-basis of divisors (`basis_homology` with integral=False) are linearly independent, so the square system has the only possible solution m; it need not be a lattice basis (the rays may contain none), since m is then integral iff a solution exists, which the checks below decide
    lattice_basis=np.setdiff1d(np.arange(len(points)),np.asarray(basis_homology(points,integral=False),dtype=int)-1)
    B=points[lattice_basis]
    bB=b[lattice_basis]
    m=_integer_solve(B,bB,atol=atol)
    if m is None:
        return False, None
    if np.array_equal(points@m,b):
        return True, m
    return False, None

def intersecting_points(points: np.ndarray,divisors: Sequence[np.ndarray]|None=None,polytopes: Sequence[Polytope|h_polytope.HPolytope]|None=None,as_indices: bool=False,return_not_intersecting: bool=False)->np.ndarray:
    """
    **Description:**
    Determines the nonzero lattice points whose toric divisors intersect the generic complete intersection defined by the divisors/Newton polytopes. A point is excluded if, for one of the divisors, exactly one of its monomial sections does not vanish on the corresponding toric divisor (so the toric divisor misses the complete intersection).
    **Arguments:**
    - `points (np.ndarray)`: The lattice points (rays), one per row; zero rows are removed first.
    - `divisors (Sequence[np.ndarray]|None)`: Coefficient vectors of the divisors in the basis of the nonzero points. If `None`, they are computed from `polytopes`. Defaults to `None`.
    - `polytopes (Sequence[Polytope|h_polytope.HPolytope]|None)`: The Newton polytopes of the divisors. If `None`, they are computed from `divisors`. Defaults to `None`.
    - `as_indices (bool)`: Whether to return row indices (into the nonzero points) instead of the points. Defaults to `False`.
    - `return_not_intersecting (bool)`: Whether to return the non-intersecting points instead. Defaults to `False`.
    **Raises:**
    - `ValueError`: Raised if neither `divisors` nor `polytopes` is given.
    **Returns:**
    - `np.ndarray`: The (non-)intersecting nonzero points, one per row, or their zero-based indices among the nonzero points if `as_indices` is `True`.
    """
    points=points[np.any(points!=0,axis=1)]
    if divisors is None:
        if polytopes is None:
            raise ValueError("At least divisors or polytopes must be defined")
        divisors=[linebundle_weights_from_Newton_Polytope(points,p) for p in polytopes]
    else:
        if polytopes is None:
            polytopes=[Newton_Polytope(points,d) for d in divisors]
    out=[]
    for i,p in enumerate(polytopes):
        secs=sections(points,divisors[i],p)
        out.extend(np.where(np.sum(secs==0,axis=1)==1)[0].tolist())
    out=np.unique(np.asarray(out,dtype=int))
    if not return_not_intersecting:
        mask=np.ones(len(points),dtype=bool)
        mask[out]=False
        out=np.flatnonzero(mask)
    if as_indices:
        return out
    else:
        return points[out]

def intersecting_cones(fan: Fan,cones: Iterable[Iterable[int]]|None=None,divisors: Sequence[np.ndarray]|None=None,polytopes: Sequence[Polytope|h_polytope.HPolytope]|None=None,closure: bool=True,return_not_intersecting: bool=False,cone_dim: int|None=None)->list[tuple[int,...]]:
    """
    **Description:**
    Determines which cones of the fan intersect the generic complete intersection X={f_1=...=f_k=0} defined by the divisors/Newton polytopes. This generalizes `intersecting_points` from rays to arbitrary cones.
    On the torus orbit O(σ) (all x_ρ=0 for ρ in σ, all other coordinates nonzero) only the monomials of f_i with vanishing exponent along every ray of σ survive, i.e. the face F_i(σ) of the i-th Newton polytope. A generic system with these Newton polytopes has a solution on the orbit torus iff dim(Σ_{i∈I} F_i(σ)) >= |I| for every subset I of the nonempty faces (Bernstein); empty faces mean f_i vanishes identically on O(σ) and impose no condition. For a hypersurface this reduces to: O(σ) meets X iff F(σ) does not consist of a single monomial.
    The closure V(σ) is the union of the orbits O(τ) for cones τ ⊇ σ of the fan, so with `closure=True` the cone counts as intersecting if any such orbit meets X. For nef divisors this agrees with the orbit test. If σ is not a cone of the fan, its orbit is empty and with `closure=True` the test answers whether the intersection of the toric divisors D_ρ, ρ in σ, meets X.
    **Arguments:**
    - `fan (Fan)`: The toric fan of the ambient variety.
    - `cones (Iterable[Iterable[int]]|None)`: The cones to test, as tuples of one-indexed ray labels of `fan`. If `None`, all cones of `fan` of dimension >= 1 are tested. Defaults to `None`.
    - `divisors (Sequence[np.ndarray]|None)`: Coefficient vectors of the divisors in the basis of `fan.vectors()`. If `None`, they are computed from `polytopes`. Defaults to `None`.
    - `polytopes (Sequence[Polytope|h_polytope.HPolytope]|None)`: The Newton polytopes of the divisors. If `None`, they are computed from `divisors`. Defaults to `None`.
    - `closure (bool)`: Whether to test the orbit closure V(σ) instead of the orbit O(σ). Defaults to `True`.
    - `return_not_intersecting (bool)`: Whether to return the non-intersecting cones instead. Defaults to `False`.
    **Raises:**
    - `ValueError`: Raised if neither `divisors` nor `polytopes` is given.
    **Returns:**
    - `list[tuple[int,...]]`: The (non-)intersecting cones, as tuples of one-indexed ray labels.
    """
    V=np.asarray(fan.vectors())
    if divisors is None:
        if polytopes is None:
            raise ValueError("At least divisors or polytopes must be defined")
        divisors=[linebundle_weights_from_Newton_Polytope(V,p) for p in polytopes]
    elif polytopes is None:
        polytopes=[Newton_Polytope(V,d) for d in divisors]
    secs=[np.asarray(sections(V,d,p)).reshape(len(V),-1) for d,p in zip(divisors,polytopes)]
    row={l:i for i,l in enumerate(fan.used_labels)}
    fan_cones={frozenset(c) for d in range(1,np.linalg.matrix_rank(V)+1) for c in fan.cones(dim=d)}
    if cones is None:
        if cone_dim is None:
            cones=sorted((tuple(sorted(c)) for c in fan_cones),key=lambda c:(len(c),c))
        else:
            cones=get_lower_dimensional_cones(fan_cones,cone_dim)

    @cache
    def orbit_meets(c: frozenset[int])->bool:
        if c not in fan_cones:
            return False
        r=[row[l] for l in c]
        faces=[S[:,np.all(S[r]==0,axis=0)] for S in secs]
        diffs=[F[:,1:]-F[:,:1] for F in faces if F.shape[1]>0]
        for k in range(1,len(diffs)+1):
            for I in combinations(diffs,k):
                M=np.hstack(I)
                if (np.linalg.matrix_rank(M) if M.size else 0)<k:
                    return False
        return True
    out=[]
    for c in cones:
        s=frozenset(c)
        meets=orbit_meets(s) or (closure and any(orbit_meets(t) for t in fan_cones if t>s))
        if meets!=return_not_intersecting:
            out.append(tuple(c))
    return out

def _bernstein_count(supports: Sequence[np.ndarray],K: np.ndarray)->int:
    """
    **Description:**
    Number of solutions of a generic system of r Laurent polynomials with the given supports in the r-dimensional torus
    whose character lattice has the saturated basis K (Bernstein), i.e. the normalized mixed volume
    sum_S (-1)^(r-|S|) Vol(sum_{j in S} P_j), with Vol measured in the coordinates of K. For r=1 this is the lattice length,
    for r=2 it is Area(P+Q)-Area(P)-Area(Q).
    **Arguments:**
    - `supports (Sequence[np.ndarray])`: r point sets (one point per row), each containing the origin and lying in span(K).
    - `K (np.ndarray)`: Saturated lattice basis, as columns, of shape (n,r).
    **Raises:**
    - `RuntimeError`: Raised if the computed mixed volume is not an integer.
    **Returns:**
    - `int`: The number of solutions.
    """
    r=K.shape[1]
    coords=[]
    for F in supports:
        c=np.unique(np.rint(np.linalg.lstsq(K.astype(float),F.T.astype(float),rcond=None)[0].T).astype(int),axis=0)
        if r==1:
            c=np.array([c.min(axis=0),c.max(axis=0)])
        elif len(c)>r+1 and np.linalg.matrix_rank(c[1:]-c[0])==r:
            c=c[ConvexHull(c).vertices]
        coords.append(c)
    total=0
    for s in range(1,r+1):
        for S in combinations(range(r),s):
            pts=coords[S[0]]
            for j in S[1:]:
                pts=np.unique((pts[:,None,:]+coords[j][None,:,:]).reshape(-1,r),axis=0)
            if len(pts)>r and np.linalg.matrix_rank(pts[1:]-pts[0])==r:
                total+=(-1)**(r-s)*_normalized_volume(pts,np.eye(r,dtype=int))
    if total%math.factorial(r):
        raise RuntimeError("The mixed volume is not an integer.")
    return total//math.factorial(r)

def _generic_ci_components(supports: Sequence[np.ndarray])->int:
    """
    **Description:**
    Number of irreducible components of the zero set, in the torus with character lattice Z^n, of generic Laurent
    polynomials f_1,...,f_k with the given nonempty supports (Khovanskii; Zhizhin, arXiv:2409.00188, Thm. 4.4). With the
    defect d(J)=dim(sum_{j in J} A_j)-|J|: 0 if d(J)<0 for some J (Bernstein), 1 if d(J)>0 for all J, and otherwise the
    normalized mixed volume (`_bernstein_count`) of the supports in the largest zero-defect subset J_0, measured in the
    saturated lattice they span. For k=1 this gives 0 (point), the lattice length (segment) or 1; for k=2 it reproduces the
    face table (0 / MV(P,Q) if dim(P+Q)=2 / lattice length of a segment face / 1).
    **Arguments:**
    - `supports (Sequence[np.ndarray])`: The exponent vectors of each f_j, one per row.
    **Returns:**
    - `int`: The number of irreducible components (0 if the zero set is empty).
    """
    supports=[np.asarray(F,dtype=int)-np.asarray(F,dtype=int)[0] for F in supports]
    k=len(supports)
    if k==0:
        return 1
    defect={J:np.linalg.matrix_rank(np.vstack([supports[j] for j in J]))-len(J) for s in range(1,k+1) for J in combinations(range(k),s)}
    if min(defect.values())<0:
        return 0
    J0=tuple(sorted(set().union(*(J for J,d in defect.items() if d==0))))
    if not J0:
        return 1
    D=np.vstack([supports[j] for j in J0])
    return _bernstein_count([supports[j] for j in J0],_saturated_kernel(_saturated_kernel(D).T))

def _toric_divisor_faces(fan_or_vectors: Fan|ArrayLike,divisors_or_polytopes: Sequence[ArrayLike|Polytope],check: bool=True)->tuple[np.ndarray,list,list[np.ndarray],list[np.ndarray],list[list[np.ndarray]]]:
    """
    **Description:**
    Common set-up of `is_favorable` and `has_polynomial_deformations`. For every ray rho it returns the faces
    Theta_{i,rho}={m in Delta_i: <m,u_rho>=-a_{i,rho}} of the Newton polytopes Delta_i of the defining divisors, i.e. the
    exponents of the monomials of s_i that survive on the open orbit of D_rho. The faces only depend on the rays, so a fan
    is only needed for `check`, which tests its cones.
    **Arguments:**
    - `fan_or_vectors (Fan|ArrayLike)`: The simplicial toric fan of the ambient space, or only its ray generators, one per row (labelled 1,...,N).
    - `divisors_or_polytopes (Sequence[ArrayLike|Polytope])`: The defining divisors, as coefficient vectors in the order of the rays, or their Newton polytopes.
    - `check (bool)`: Whether to verify that no torus orbit O(tau) meeting Y lies in the base locus of at least max(dim(tau)-1,1) of the defining divisors. This needs the cones, i.e. a fan. Defaults to `True`.
    **Raises:**
    - `ValueError`: Raised if a defining divisor vanishes identically on some D_rho, if `check` fails, or if `check` is requested without a fan.
    **Returns:**
    - `tuple[np.ndarray,list,list[np.ndarray],list[np.ndarray],list[list[np.ndarray]]]`: The ray generators, the one-indexed ray labels, the divisor coefficient vectors, the lattice points of the Newton polytopes, and for each ray the list of the k faces Theta_{i,rho} (as point arrays).
    """
    if isinstance(fan_or_vectors,Fan):
        fan=fan_or_vectors
        V=np.asarray(fan.vectors(),dtype=int)
        labels=list(fan.used_labels)
    else:
        if check:
            raise ValueError("The base locus check needs the cones of a fan: pass a Fan, or pass check=False (e.g. for divisors that are nef on the intended fan, for which the check always passes).")
        fan=None
        V=np.asarray(fan_or_vectors,dtype=int)
        if V.ndim!=2:
            raise ValueError(f"The ray generators must be given one per row, got shape {V.shape}.")
        labels=list(range(1,len(V)+1))
    divisors,polytopes=[],[]
    for x in divisors_or_polytopes:
        if isinstance(x,Polytope):
            polytopes.append(x)
            divisors.append(linebundle_weights_from_Newton_Polytope(V,x))
        else:
            divisors.append(np.asarray(x,dtype=int))
            polytopes.append(Newton_Polytope(V,divisors[-1]))
    points=[np.asarray(p.points(),dtype=int) for p in polytopes]
    Z=[V@P.T+d[:,None]==0 for P,d in zip(points,divisors)]
    if check:
        row={l:i for i,l in enumerate(labels)}
        Zp=[np.packbits(z,axis=1) for z in Z]
        suspect=[]
        for dim in range(1,min(len(Z)+1,V.shape[1])+1):
            cones=get_lower_dimensional_cones(fan,dim)
            C=np.array([[row[l] for l in c] for c in cones],dtype=int).reshape(-1,dim)
            empty=sum(~np.bitwise_and.reduce(z[C],axis=1).any(axis=1) for z in Zp)
            suspect+=[cones[i] for i in np.flatnonzero(empty>=max(dim-1,1))]
        bad=intersecting_cones(fan,cones=suspect,divisors=divisors,polytopes=polytopes,closure=False) if suspect else []
        if bad:
            raise ValueError(f"Y has components or divisors in the base locus of the defining divisors (orbits of the cones {bad[:5]}), so the face criterion does not apply.")
    faces=[]
    for i,l in enumerate(labels):
        F=[P[z[i]] for P,z in zip(points,Z)]
        if any(len(f)==0 for f in F):
            raise ValueError(f"A defining divisor vanishes identically on D_{l}.")
        faces.append(F)
    return V,labels,divisors,points,faces

def is_favorable(fan_or_vectors: Fan|ArrayLike,divisors_or_polytopes: Sequence[ArrayLike|Polytope],check: bool=True)->tuple[bool,dict[int,int]]:
    """
    **Description:**
    Determines which toric prime divisors D_rho split on the generic complete intersection Y={s_1=...=s_k=0} of the given
    divisors (any codimension k), without computing intersection numbers. On the open orbit of D_rho only the monomials of the
    faces Theta_{i,rho}={m in Delta_i: <m,u_rho>=-a_{i,rho}} survive, and the components of D_rho cap Y are the closures of the
    components of this generic torus complete intersection, counted by `_generic_ci_components`. This requires that Y has no
    components or divisors inside the toric boundary strata on which some s_i vanishes identically, which holds for nef
    divisors and is verified with `check`. The count only uses the rays, so instead of a fan its ray generators suffice,
    e.g. for divisors known to be nef on the intended fan (as for a Batyrev-Borisov nef partition); `check` tests the cones
    of the fan and is then not possible. Note that for complete intersections non-splitting is necessary but not
    sufficient for favorability: Pic(Y) can contain classes that are not supported on the toric boundary.
    **Arguments:**
    - `fan_or_vectors (Fan|ArrayLike)`: The simplicial toric fan of the ambient space, or only its ray generators, one per row (labelled 1,...,N).
    - `divisors_or_polytopes (Sequence[ArrayLike|Polytope])`: The defining divisors, as coefficient vectors in the order of the rays, or their Newton polytopes.
    - `check (bool)`: Whether to verify that no torus orbit O(tau) meeting Y lies in the base locus of at least max(dim(tau)-1,1) of the defining divisors. This needs a fan; with ray generators only it must be `False`. Defaults to `True`.
    **Raises:**
    - `ValueError`: Raised if a defining divisor vanishes identically on some D_rho, if `check` fails, or if `check` is requested without a fan.
    **Returns:**
    - `tuple[bool,dict[int,int]]`: Whether no toric divisor splits, and {one-indexed ray label: number of components} for the splitting divisors.
    """
    _,labels,_,_,faces=_toric_divisor_faces(fan_or_vectors,divisors_or_polytopes,check)
    splitting={}
    for l,F in zip(labels,faces):
        k=_generic_ci_components(F)
        if k>1:
            splitting[int(l)]=k
    return not splitting,splitting

def intersection_number_surfaces(IN: Mapping[tuple[int,...],int|float],basis_len: int|None=None)->tuple[dict[tuple[int,int,int],int|float],dict[int,tuple[int,int]]]:
    """
    **Description:**
    Calculates intersection numbers of a CICY with ambient toric divisors.
    The quadruple intersection numbers are rewritten as intersection numbers of the surfaces D_a*D_b (a <= b) with pairs of basis divisors.
    Runs in O(len(IN)) by splitting each sorted 4-tuple into its (at most six) surface/divisor-pair decompositions.
    **Arguments:**
    - `IN (Mapping[tuple[int,...],int|float])`: Dictionary of intersection numbers of the CICY with ambient divisors, keyed by sorted one-indexed 4-tuples of basis labels.
    - `basis_len (int|None)`: Length of the basis of ambient divisors. Defaults to `None`, in which case the largest label in `IN` is used.
    **Returns:**
    - `tuple[dict[tuple[int,int,int],int|float],dict[int,tuple[int,int]]]`: A tuple containing:
        - new_IN: Dictionary of intersection numbers `{(surface index, i, j): value}` with zero-based basis indices i <= j.
        - surface_dict: Dictionary mapping surface indices to the one-indexed pairs (a, b) of basis divisors.
    """
    n=max(map(max,IN),default=0) if basis_len is None else int(basis_len)
    surface_dict=dict(enumerate(combinations_with_replacement(range(1,n+1),2)))
    offset=[a*n-a*(a+1)//2 for a in range(n)]
    new_IN={}
    for key,val in IN.items():
        if len(key)!=4:
            continue
        i,j,k,l=(int(x)-1 for x in key)
        if not 0<=i<=j<=k<=l<n:
            continue
        new_IN[(offset[i]+j,k,l)]=val
        new_IN[(offset[i]+k,j,l)]=val
        new_IN[(offset[i]+l,j,k)]=val
        new_IN[(offset[j]+k,i,l)]=val
        new_IN[(offset[j]+l,i,k)]=val
        new_IN[(offset[k]+l,i,j)]=val
    return dict(sorted(new_IN.items())),surface_dict

def tensor_symm_in_basis(tensor: Mapping[int,Any]|Mapping[tuple[int,...],Any],basis: ArrayLike,Q: ArrayLike|None=None,tol: float=1e-5,in_prime_basis: bool|None=None)->dict[int,Any]|dict[tuple[int,...],Any]:
    """
    **Description:**
    Transform a symmetric divisor tensor from prime-toric divisors
    to a chosen prime-toric or GLSM divisor basis.
    Prime-toric indices are 1-based.
    If the tensor is already given in the prime toric basis (keyed by one-indexed basis positions), the restriction
    to the basis is skipped and only the GLSM transformation is applied (if `Q` is given).
    By default this is detected automatically: the tensor counts as being in the prime toric basis iff all of its
    indices are at most `len(basis)`. For `basis = [1,...,len(basis)]` both readings agree.
    **Arguments:**
    - `tensor (Mapping[int,Any]|Mapping[tuple[int,...],Any])`: The symmetric tensor in terms of all prime toric divisors or of the prime toric basis, keyed by labels (rank 1) or sorted label tuples.
    - `basis (ArrayLike)`: The one-indexed labels of the prime toric divisors forming the basis.
    - `Q (ArrayLike|None)`: GLSM charge matrix; if given, the result is further transformed to the GLSM basis given by the rows of Q. Defaults to `None`.
    - `tol (float)`: Tolerance for dropping and rounding entries in the GLSM transformation. Defaults to `1e-5`.
    - `in_prime_basis (bool|None)`: Whether `tensor` is already in the prime toric basis; `None` detects it automatically. Defaults to `None`.
    **Returns:**
    - `dict[int,Any]|dict[tuple[int,...],Any]`: The tensor in the new basis, keyed by one-indexed basis positions (rank 1) or sorted tuples of them.
    """
    basis=np.asarray(basis)
    if np.ndim(basis)!=1 or (Q is not None and np.ndim(Q)!=2):
        raise ValueError(f"Expected a 1D `basis` and a 2D `Q`, got shapes {np.shape(basis)} and {np.shape(Q)}; note the argument order (tensor,basis,Q).")
    if not tensor:
        return {}
    if in_prime_basis is None:
        h=len(basis)
        if isinstance(next(iter(tensor)),(int,np.integer)):
            in_prime_basis=all(i<=h for i in tensor)
        else:
            in_prime_basis=all(i<=h for key in tensor for i in key)
    out=dict(tensor) if in_prime_basis else _tensor_symm_in_prime_basis(tensor,basis)
    if Q is not None:
        out=_tensor_symm_in_glsm_basis(out,Q,basis,tol=tol)
    return out

def _tensor_symm_in_prime_basis(tensor: Mapping[int,Any]|Mapping[tuple[int,...],Any],basis: ArrayLike)->dict[int,Any]|dict[tuple[int,...],Any]:
    """
    **Description:**
    Transforms a symmetric tensor of all prime torics into the basis of a subset of prime torics.
    Entries involving a divisor outside `basis` are dropped, and the labels are replaced by their one-indexed position in `basis`.
    **Arguments:**
    - `tensor (Mapping[int,Any]|Mapping[tuple[int,...],Any])`: The symmetric tensor keyed by one-indexed prime toric labels (rank 1) or sorted label tuples.
    - `basis (ArrayLike)`: The one-indexed labels of the prime toric divisors forming the basis.
    **Returns:**
    - `dict[int,Any]|dict[tuple[int,...],Any]`: The restricted tensor, keyed by one-indexed basis positions (rank 1) or sorted tuples of them.
    """
    basis_map={int(i):a+1 for a,i in enumerate(basis)}
    if isinstance(next(iter(tensor)),(int,np.integer)):
        return {basis_map[i]:v for i,v in tensor.items() if i in basis_map}
    result={}
    for key,value in tensor.items():
        if all(i in basis_map for i in key):
            new=tuple(sorted(basis_map[i] for i in key))
            result[new]=value
    return result

def _sorted_index_tuples(h: int,rank: int)->np.ndarray:
    """
    **Description:**
    Vectorized `np.array(list(combinations_with_replacement(range(h),rank)))`.
    **Arguments:**
    - `h (int)`: Number of indices.
    - `rank (int)`: Length of the tuples (at least 1).
    **Returns:**
    - `np.ndarray`: All non-decreasing index tuples, one per row, in lexicographic order.
    """
    idx=np.arange(h)[:,None]
    for _ in range(rank-1):
        last=idx[:,-1]
        counts=h-last
        starts=np.cumsum(counts)-counts
        new_col=np.repeat(last,counts)+np.arange(counts.sum())-np.repeat(starts,counts)
        idx=np.column_stack([np.repeat(idx,counts,axis=0),new_col])
    return idx

def _reduce_hnf(v,M):
    """
    **Description:**
    Reduces the integer vector `v` (or every row of a matrix `v`) modulo the rows of a row Hermite normal form `M`: for each
    row of `M` in turn, the multiple that makes the entry at that row's pivot column lie in [0, pivot) is subtracted. The
    result is zero exactly when `v` lies in the lattice spanned by `M`.
    **Arguments:**
    - `v (ArrayLike)`: Integer vector or matrix (one vector per row) to reduce.
    - `M (ArrayLike)`: Integer matrix in row Hermite normal form (first nonzero entry of each row positive, pivots strictly increasing).
    **Returns:**
    - `np.ndarray`: The reduced vector(s), of the same shape as `v`.
    """
    v=np.array(v,dtype=int)
    for row in M:
        c=(row!=0).argmax(); v-=np.multiply.outer(v[...,c]//row[c],row)
    return v

def kernel_hnf_N(V,cols):
    """
    **Description:**
    Row Hermite normal form, pivoting in the order `cols`, of the saturated relation lattice {q in Z^n : q @ V = 0, supp(q) in cols}
    of the points `V`. The output coincides with `hnf_N(rel, cols)` for a basis `rel` of that lattice, but is computed without
    first building a kernel basis: with N the greedy basis of V[cols] chosen from the end of `cols` and P the remaining (pivot)
    columns, the projection of the lattice to Z^P is the kernel of Z^P -> G = Z<V[cols]> / Z<V[N]>. Its Hermite form is computed
    inside the finite group G (a pivot larger than 1 appears exactly where G grows), and the entries in the N columns follow
    from the coordinates of V[P] in the basis V[N], which are exact rational numbers with denominator det(V[N]).
    **Arguments:**
    - `V (ArrayLike)`: Integer point configuration, one point per row.
    - `cols (Iterable[int])`: Column (point) indices allowed in the support of the relations, in the order used for the pivots.
    **Returns:**
    - `tuple[np.ndarray, np.ndarray]`: The pair `(H, pivots)`: the relations as rows of an integer matrix with `len(V)` columns
      (zero outside `cols`), and the pivot column of each row (the columns P, in the order of `cols`).
    """
    V=np.asarray(V); cols=np.asarray(list(cols),int); W=V[cols]; n=len(cols)
    Hr=hnf(W[::-1].T); r=np.count_nonzero(Hr.any(axis=1))
    N=np.sort(n-1-(Hr[:r]!=0).argmax(axis=1)); P=np.setdiff1d(np.arange(n),N)
    A=W[N]; s=(hnf(A)[:r]!=0).argmax(axis=1); As=A[:,s]
    D=abs(round(np.linalg.det(As))); Gam=W[P][:,s]@np.rint(np.linalg.inv(As)*D).astype(int)
    assert D>0 and np.array_equal(Gam@A,D*W[P]), "inexact coordinates in kernel_hnf_N"
    K=np.eye(len(P),dtype=int); g=Gam%D
    if g.any():
        Msub=D*np.eye(r,dtype=int); Mall=hnf(np.vstack([Msub,g]))[:r]; T=[]; Mafter={}
        for k in range(len(P)-1,-1,-1):
            if np.array_equal(Msub,Mall): break
            t=1
            while _reduce_hnf(t*g[k],Msub).any(): t+=1
            if t>1: K[k,k]=t; T.append(k); Mafter[k]=Msub; Msub=hnf(np.vstack([Msub,g[k]]))[:r]
        res=K.diagonal()[:,None]*g
        for j in T[::-1]:
            dig=np.full(j,-1)
            for a in range(K[j,j]):
                dig[(dig<0)&~_reduce_hnf(res[:j]+a*g[j],Mafter[j]).any(axis=1)]=a
            K[:j,j]=dig; res[:j]+=dig[:,None]*g[j]
    KN=-(K@Gam); assert not (KN%D).any()
    H=np.zeros((len(P),len(V)),int); H[:,cols[P]]=K; H[:,cols[N]]=KN//D
    return H,cols[P]

def gram_matrix(IN_in_basis: Mapping[tuple[int,...],int|float])->tuple[np.ndarray,dict[tuple[int,int],int],dict[int,tuple[int,int]]]:
    """
    **Description:**
    Compute the Gram matrix for the intersection numbers in the basis of the CY.
    This is done by transforming the intersection numbers from the toric basis to the basis of the CY.
    **Arguments:**
    - `IN_in_basis (Mapping[tuple[int,...],int|float])`: Intersection numbers in the basis of the CY.
    **Returns:**
    - `tuple[np.ndarray,dict[tuple[int,int],int],dict[int,tuple[int,int]]]`: (kappa_mat, k_dict, inv_dict)
        kappa_mat : Gram matrix (float) for the intersection numbers in the basis of the CY.
        k_dict : Dictionary mapping pairs of basis vectors to indices in the Gram matrix.
        inv_dict : Dictionary mapping indices in the Gram matrix to pairs of basis vectors.
    """
    n=max(map(max,IN_in_basis),default=0)
    l=n*(n+1)//2
    k_dict={a: i for i,a in enumerate(combinations_with_replacement(range(1,n+1),2))}
    inv_dict={i: a for a,i in k_dict.items()}
    kappa_mat=np.zeros((l,l),dtype=float)
    items=[(k,v) for k,v in IN_in_basis.items() if len(k)==4 and min(k)>=1]
    if not items:
        return kappa_mat, k_dict, inv_dict
    K=np.array([k for k,_ in items],dtype=np.int64)-1
    vals=np.array([v for _,v in items],dtype=float)
    pair=lambda a, b: a*n-a*(a-1)//2+(b-a)
    for (i,j), (k,m) in [((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]:
        p, q=pair(K[:,i],K[:,j]), pair(K[:,k],K[:,m])
        kappa_mat[p,q]=vals
        kappa_mat[q,p]=vals
    rounded=np.rint(kappa_mat)
    if np.allclose(kappa_mat,rounded,atol=1e-8,rtol=0):
        kappa_mat=rounded.astype(int)
    return kappa_mat, k_dict, inv_dict

def _unimodular_column_subset(P: fmpz_mat,max_nodes: int=10**3)->list[int]|None:
    """
    **Description:**
    Searches a subset of the columns of P that is a Z-basis of Z^r.
    First, starting from the lexicographically first Q-basis among the columns, single columns are exchanged as long as
    this decreases |det|: exchanging the b-th basis column for column j multiplies the determinant by the b-th
    coordinate of column j in the current basis. If this gets stuck at |det|>1, a depth-first search follows. A set of
    columns extends to a Z-basis iff it spans a saturated sublattice. This is tracked by unimodular row operations moving
    the k chosen columns to e_1,...,e_k: column j can then be added iff its last r-k entries are coprime. The last r-k
    rows are LLL-reduced and the sparsest columns are tried first. Branches whose remaining columns no longer span the
    last r-k coordinates are pruned, and columns whose branch has been searched are excluded from their siblings.
    **Arguments:**
    - `P (fmpz_mat)`: (r,N) integer matrix whose columns span Z^r.
    - `max_nodes (int)`: Maximal number of columns added during the depth-first search. Defaults to `10**3`.
    **Returns:**
    - `list[int]|None`: The increasing column indices of a Z-basis, or `None` if none was found.
    """
    r,N=P.nrows(),P.ncols()
    R=P.rref()[0]
    S=[next(j for j in range(N) if R[i,j]!=0) for i in range(r)]
    while True:
        A=fmpz_mat(r,r,[P[i,j] for i in range(r) for j in S])
        if abs(A.det())==1:
            return sorted(S)
        X=A.solve(P)
        rest=sorted(set(range(N))-set(S))
        step=min(((abs(X[b,j]),j,-S[b],b) for b in range(r) for j in rest if 0<abs(X[b,j])<1),default=None)
        if step is None:
            break
        S[step[3]]=step[1]
    nodes=0
    def search(B,S,excluded):
        nonlocal nodes
        m=B.nrows()
        if m==0:
            return sorted(S)
        cols=[j for j in range(N) if j not in S and j not in excluded]
        if len(cols)<m:
            return None
        # the columns cols of B span Z^m iff the Hermite normal form of their transpose starts with the identity
        H=fmpz_mat([[B[i,j] for i in range(m)] for j in cols]).hnf()
        if any(H[i,i]!=1 for i in range(m)):
            return None
        B=B.lll()
        rows=B.tolist()
        key=lambda j: (sum(row[j]!=0 for row in rows),sum(abs(row[j]) for row in rows),j)
        tried=set()
        for j in sorted((j for j in cols if math.gcd(*(row[j] for row in rows))==1),key=key):
            if nodes>=max_nodes:
                return None
            nodes+=1
            T=fmpz_mat(m,1,[row[j] for row in rows]).hnf(transform=True)[1]
            res=search(fmpz_mat(m-1,N,[x for row in (T*B).tolist()[1:] for x in row]),S+[j],excluded|tried)
            if res is not None:
                return res
            tried.add(j)
        return None
    return search(P,[],set())

def intersection_form_H22_vertical_from_gram_matrix(M: ArrayLike,max_nodes: int=10**3)->tuple[np.ndarray,np.ndarray]:
    """
    **Description:**
    Z-basis of the vertical cycles spanned by the N products and the intersection form on it. The vertical lattice is
    Z^N modulo the kernel of the Gram matrix, which v -> v M maps isomorphically onto the row lattice of M. If possible,
    the basis is a subset of the products themselves, found with `_unimodular_column_subset` (preferring products with
    small index) from the coordinates of the rows of M in the Hermite normal form basis of the row lattice. Otherwise a
    warning is raised and the basis is read off from the Hermite normal form U M = H: the first r rows of U.
    **Arguments:**
    - `M (ArrayLike)`: (N,N) integer Gram matrix on the products.
    - `max_nodes (int)`: Maximal number of depth-first search steps for the subset of products. Defaults to `10**3`.
    **Returns:**
    - `tuple[np.ndarray,np.ndarray]`: C (r,N) integer: row k = coefficients of the k-th Z-basis vector e_k
        in the N products (a unit vector if e_k is a single product), and G = C M C^T, the intersection form on that
        basis (both int64).
    **Raises:**
    - `OverflowError`: Raised if an entry of C or G does not fit into int64.
    """
    M=np.asarray(M,dtype=object)
    N=len(M)
    M=fmpz_mat(M.tolist())
    H=M.hnf()
    r=H.rank()
    P=fmpz_mat(0,N)
    if r:
        piv=[next(j for j in range(N) if H[k,j]!=0) for k in range(r)]
        X=fmpz_mat(r,r,[H[l,j] for j in piv for l in range(r)]).solve(fmpz_mat(r,N,[M[j,i] for j in piv for i in range(N)]))
        P=fmpz_mat(r,N,[X[k,i].p for k in range(r) for i in range(N)])
    S=_unimodular_column_subset(P,max_nodes)
    if S is None:
        warnings.warn(f"No subset of the products forming a Z-basis was found (depth-first search limited to {max_nodes} steps); the basis vectors are combinations of products.")
        U=M.hnf(transform=True)[1]
        C=fmpz_mat(r,N,[x for row in U.tolist()[:r] for x in row])
    else:
        C=fmpz_mat(r,N,[int(i==j) for j in S for i in range(N)])
    G=C*M*C.transpose()
    to_int64=lambda X,n,m: np.array([int(x) for row in X.tolist() for x in row],dtype=np.int64).reshape(n,m)
    return to_int64(C,r,N), to_int64(G,r,r)

def intersection_form_H22_vertical(IN_in_basis: Mapping[tuple[int,...],int|float],max_nodes: int=10**3,return_gram=False)->tuple[np.ndarray,np.ndarray,dict[int,tuple[int,int]],np.ndarray]:
    """
    **Description:**
    Compute the intersection form on the vertical cycles in H22(X,Z). If possible, the basis consists of products of
    two basis divisors, see `intersection_form_H22_vertical_from_gram_matrix`.
    **Arguments:**
    - `IN_in_basis (Mapping[tuple[int,...],int|float])`: Intersection numbers in the basis of the CY.
    - `max_nodes (int)`: Maximal number of depth-first search steps for a basis of products. Defaults to `10**3`.
    **Returns:**
    - `tuple[np.ndarray,np.ndarray,dict[int,tuple[int,int]],np.ndarray]`: (C, G, inv_dict, mat)
        C : Integer matrix whose rows are the basis vectors of the vertical cycles in H22(X,Z), in terms of the products.
        G : Intersection form on the vertical cycles in H22(X,Z).
        inv_dict : Dictionary mapping indices in the Gram matrix to pairs of basis vectors.
        mat : Gram matrix of the products.
    """
    mat,dict_pair_indices,inv_dict=gram_matrix(IN_in_basis)
    C,G=intersection_form_H22_vertical_from_gram_matrix(mat,max_nodes)
    if return_gram:
        return C, G, inv_dict,mat
    return C, G, inv_dict

def _stabilizer_order(idx: np.ndarray)->np.ndarray:
    """
    **Description:**
    Number of index permutations fixing each sorted index tuple, i.e. the product of the factorials of the multiplicities.
    **Arguments:**
    - `idx (np.ndarray)`: Non-decreasing index tuples, one per row.
    **Returns:**
    - `np.ndarray`: The stabilizer order of each row.
    """
    order=np.ones(len(idx),dtype=np.int64)
    run=np.ones(len(idx),dtype=np.int64)
    for j in range(1,idx.shape[1]):
        run=np.where(idx[:,j]==idx[:,j-1],run+1,1)
        order*=run
    return order

def _tensor_symm_in_glsm_basis(tensor: Mapping[int,int|float]|Mapping[tuple[int,...],int|float],Q: ArrayLike,basis: ArrayLike,tol: float=1e-5,block_bytes: int=2**27,max_sparse_entries: int=2*10**7)->dict[int,int|float]|dict[tuple[int,...],int|float]:
    """
    **Description:**
    Transform a symmetric tensor from a prime-toric basis to the GLSM basis.
    The tensor is contracted in every slot with the inverse A of the basis columns `Q[:,basis-1]`.
    If A is sparse enough, the sorted input entries (weighted by 1/|stabilizer|) are contracted slot by slot as a sparse
    tensor and symmetrized at the end. Otherwise the first slot is contracted directly from the sparse input, in blocks
    of output indices a; the remaining slots then only need the columns b>=a, so the dense h^rank tensor is never formed.
    **Arguments:**
    - `tensor (Mapping[int,int|float]|Mapping[tuple[int,...],int|float])`: The symmetric tensor in the prime toric basis, keyed by one-indexed basis positions (rank 1) or tuples of them.
    - `Q (ArrayLike)`: GLSM charge matrix (rows: GLSM charges, columns: prime toric divisors).
    - `basis (ArrayLike)`: The one-indexed labels of the prime toric basis divisors.
    - `tol (float)`: Entries with absolute value below `tol` are dropped and entries within `tol` of an integer are rounded. Defaults to `1e-5`.
    - `block_bytes (int)`: Approximate memory of one block of partially contracted slices in the dense method. Defaults to `2**27`.
    - `max_sparse_entries (int)`: The sparse method is used if the estimated number of intermediate entries is at most this. Defaults to `2*10**7`.
    **Returns:**
    - `dict[int,int|float]|dict[tuple[int,...],int|float]`: The tensor in the GLSM basis, keyed by one-indexed GLSM indices (rank 1) or sorted tuples of them; empty if `tensor` is empty.
    """
    if not tensor:
        return {}
    basis=np.asarray(basis,int)
    integer_keys=isinstance(next(iter(tensor)),(int,np.integer))
    if integer_keys:
        tensor={(k,):v for k,v in tensor.items()}
    else:
        tensor={tuple(sorted(k)):v for k,v in tensor.items()}
    rank=len(next(iter(tensor)))
    h=len(basis)
    A=np.linalg.inv(np.asarray(Q)[:,basis-1])
    old=np.array(list(tensor),dtype=np.int64).reshape(-1,rank)-1
    vals=np.array(list(tensor.values()),dtype=float)
    As=sps.csr_matrix(np.where(np.abs(A)>1e-12*np.abs(A).max(),A,0))
    if len(vals)*(As.nnz/h)**rank<=max_sparse_entries:
        idx,vals=old,vals/_stabilizer_order(old)
        for k in range(rank):
            cnt=np.diff(As.indptr)[idx[:,k]]
            pos=np.repeat(As.indptr[idx[:,k]]-np.cumsum(cnt)+cnt,cnt)+np.arange(cnt.sum())
            idx=np.repeat(idx,cnt,axis=0)
            idx[:,k]=As.indices[pos]
            vals=np.repeat(vals,cnt)*As.data[pos]
            if k==rank-1:
                idx=np.sort(idx,axis=1)
            lin,inv=np.unique(np.ravel_multi_index(idx.T,(h,)*rank),return_inverse=True)
            vals=np.bincount(inv,vals)
            idx=np.column_stack(np.unravel_index(lin,(h,)*rank))
        vals*=_stabilizer_order(idx)
        keep=np.abs(vals)>=tol
        keys,values=idx[keep]+1,vals[keep]
    else:
        lin=np.concatenate([np.ravel_multi_index(old[:,p].T,(h,)*rank) for p in permutations(range(rank))])
        lin,pos=np.unique(lin,return_index=True)
        vals=np.tile(vals,factorial(rank))[pos]
        first,rest=np.divmod(lin,h**(rank-1))
        T=sps.csr_matrix((vals,(first,rest)),shape=(h,h**(rank-1)))
        blk=max(1,block_bytes//(8*h**(rank-1)))
        keys,values=[],[]
        for a0 in range(0,h,blk):
            a1=min(a0+blk,h)
            S=np.asarray(A[:,a0:a1].T@T).reshape((a1-a0,)+(h,)*(rank-1))
            Aa=A[:,a0:]
            for _ in range(rank-1):
                S=np.tensordot(S,Aa,axes=([1],[0]))
            for a in range(a0,a1):
                idx=_sorted_index_tuples(h-a,rank-1)+a if rank>1 else np.empty((1,0),dtype=np.intp)
                v=np.atleast_1d(S[a-a0][tuple((idx-a0).T)])
                keep=np.abs(v)>=tol
                keys.append(np.column_stack([np.full(keep.sum(),a),idx[keep]]))
                values.append(v[keep])
        keys=np.concatenate(keys)+1
        values=np.concatenate(values)
    rounded=np.rint(values)
    is_int=np.abs(values-rounded)<tol
    out=values.astype(object)
    out[is_int]=rounded[is_int].astype(np.int64).astype(object)
    keys=keys[:,0].tolist() if integer_keys else zip(*keys.T.tolist())
    return dict(zip(keys,out.tolist()))

def glsm_from_refinement(F):
    """
    **Description:**
    GLSM charge matrix of an F-theory uplift built from its blow-down fan {rho_+, rho_-} * facefan(nabla_fib) and the D7 points
    (the points of negative height of the adapted toric fan, except the axis point rho_- = (0;-2,5)). The columns are the
    vectors of `F.vectors_M_conv_toric_fan()` in their order, so that `Q @ F.vectors_M_conv_toric_fan() = 0`; the rows are,
    in this order,
        [fiber relations among the intersecting fiber rays | D7 rows of the intersecting D7 points | tau |
         D7 rows of the non-intersecting D7 points | fiber relations completing the first block to a Z-basis of all fiber relations],
    the D7 rows of each block in the order of `F.vectors_M_conv_toric_fan()`.
    The first fiber block is `glsm_from_points` of the intersecting fiber rays; the completing block consists of the rows of
    `kernel_hnf_N` of all fiber rays whose pivot lies in a non-intersecting column (the remaining rows are then supported on the
    intersecting columns, which is verified). The D7 row of a point d is the relation m d = sum_j s_j v_j with v_j the rays of
    the cone of the blow-down fan containing d and m the smallest positive integer for which all s_j are integers
    (`minimal_cone_relations`), with entry -m in the column of d; usually m is 1 or 2. The tau row is 3 y^2 + 2 rho_+ + rho_-
    with y^2 = (0;0,-1).
    **Arguments:**
    - `F (F_Theory_Uplift)`: The F-theory uplift.
    **Raises:**
    - `ValueError`: Raised if rho_+ is not the only vector of positive height, if the blow-down fan does not end in rho_+, rho_-,
      if y^2 is not a fiber ray, if the completing fiber block does not yield a Z-basis, if the blow-down fan is not simplicial,
      if a D7 point lies in no cone of the blow-down fan, or if the result does not annihilate the vectors.
    **Returns:**
    - `tuple[np.ndarray, tuple[int,...]]`: `(Q, sizes)`: the charge matrix, whose columns correspond to the vectors of
      `F.vectors_M_conv_toric_fan()`, and the numbers of rows of the five blocks in the order of the rows of `Q`
      (intersecting fiber, intersecting D7, tau, non-intersecting D7, completing fiber).
    """
    V=np.asarray(F.vectors_M_conv_toric_fan(),dtype=int); h=V@F.u; n,d=V.shape
    col={tuple(map(int,v)):k for k,v in enumerate(V)}
    def c(v):
        t=tuple(map(int,v))
        if t not in col: raise ValueError(f"{t} is not a vector of vectors_M_conv_toric_fan.")
        return col[t]
    rhop=np.zeros(d,dtype=int); rhom=rhop.copy(); y=rhop.copy()
    rhop[-2:]=[1,-1]; rhom[-2:]=[-2,5]; y[-1]=-1
    if not np.array_equal(V[h>0],rhop[None]): raise ValueError("rho_+ = (0;1,-1) must be the only vector of positive height.")
    f_bd=F.blow_down_fan(); X=np.asarray(f_bd.vectors(),dtype=int)
    if not np.array_equal(X[-2:],np.array([rhop,rhom])): raise ValueError("The blow-down fan must end in rho_+, rho_-.")
    cX=np.array([c(v) for v in X]); r_fib=X[:-2]; cf=cX[:-2]
    inter_set={tuple(map(int,v)) for v in F.intersecting_divisors_M()}
    inter=np.array([tuple(map(int,v)) in inter_set for v in r_fib],dtype=bool)
    ii,ni=np.flatnonzero(inter),np.flatnonzero(~inter)
    Qi=glsm_from_points(r_fib[ii])
    Qb_intersecting=np.zeros((len(Qi),n),dtype=int); Qb_intersecting[:,cf[ii]]=Qi
    Hker,pv=kernel_hnf_N(r_fib,[*ni,*ii]); keep=np.isin(pv,ni)
    if Hker[~keep][:,ni].any(): raise ValueError("The intersecting fiber rays do not span the fiber relations; no Z-basis completion.")
    Qb_not_intersecting=np.zeros((keep.sum(),n),dtype=int); Qb_not_intersecting[:,cf]=Hker[keep]
    is_d7=(h<0)&~(V==rhom).all(axis=1)
    d7_inter=np.array([tuple(map(int,v)) in inter_set for v in V],dtype=bool)
    # def d7_row(k):
    #     d7=V[k]; c7=find_cone(d7,f_bd.cones(),X)
    #     if c7 is None: raise ValueError(f"The D7 point {tuple(d7)} lies in no cone of the blow-down fan.")
    #     c7=np.array(sorted(c7)); sol=solve_integer(X[c7-1].T,2*d7); row=np.zeros(n,dtype=int)
    #     if np.all(sol%2==0): sol=sol//2; row[k]=-1
    #     else: row[k]=-2
    #     row[cX[c7-1]]=sol
    #     return row
    # Qd7_intersecting=np.array([d7_row(k) for k in np.flatnonzero(is_d7&d7_inter)],dtype=int).reshape(-1,n)
    # Qd7_not_intersecting=np.array([d7_row(k) for k in np.flatnonzero(is_d7&~d7_inter)],dtype=int).reshape(-1,n)
    d7_idx=np.flatnonzero(is_d7)
    cones7,coeffs,mult=minimal_cone_relations(f_bd,V[d7_idx])
    Qd7=np.zeros((len(d7_idx),n),dtype=int)
    Qd7[np.arange(len(d7_idx)),d7_idx]=-mult
    Qd7[np.arange(len(d7_idx))[:,None],cX[cones7-1]]=coeffs
    Qd7_intersecting=Qd7[d7_inter[d7_idx]]
    Qd7_not_intersecting=Qd7[~d7_inter[d7_idx]]
    qfinal=np.zeros((1,n),dtype=int); qfinal[0,c(y)]=3; qfinal[0,c(rhop)]=2; qfinal[0,c(rhom)]=1
    Q=np.concatenate((Qb_intersecting,Qd7_intersecting,qfinal,Qd7_not_intersecting,Qb_not_intersecting),axis=0)
    if (Q@V).any(): raise ValueError("The charge matrix does not annihilate the vectors.")
    return (Q,np.array([len(Qb_intersecting),len(Qd7_intersecting),1,len(Qd7_not_intersecting),len(Qb_not_intersecting)]))

def _left_kernel_mod_p(M,p):
    """
    **Description:**
    Basis of the left kernel of an integer matrix over the prime field F_p, i.e. of the vectors y in {0,...,p-1}^r with
    y @ M = 0 mod p, by Gaussian elimination on the augmented matrix [M mod p | 1].
    **Arguments:**
    - `M (ArrayLike)`: Integer matrix of shape (r,n).
    - `p (int)`: A prime.
    **Returns:**
    - `np.ndarray`: Integer matrix with entries in {0,...,p-1} whose rows form a basis of the left kernel modulo p
      (shape (r - rank_p(M), r)).
    """
    A=np.asarray(M,dtype=np.int64)%p; r,n=A.shape; E=np.hstack([A,np.eye(r,dtype=np.int64)]); piv=0
    for c in range(n):
        rows=np.flatnonzero(E[piv:,c])+piv
        if not len(rows): continue
        E[[piv,rows[0]]]=E[[rows[0],piv]]
        E[piv]=(E[piv]*pow(int(E[piv,c]),-1,p))%p
        for i in np.flatnonzero(E[:,c]):
            if i!=piv: E[i]=(E[i]-E[i,c]*E[piv])%p
        piv+=1
        if piv==r: break
    return E[piv:,n:]

def saturate_glsm(Q,protected_rows=None,primes=None,return_combinations=False):
    """
    **Description:**
    Saturates the lattice spanned by the rows of an integer matrix with linearly independent rows, i.e. enlarges it to all
    integer vectors that are rational combinations of the rows, by replacing rows in place. For every prime p dividing an
    invariant factor of Q, passes are made until the rows are independent modulo p: in each pass the left kernel of Q modulo p
    (`_left_kernel_mod_p`) is brought to reduced row echelon form over F_p with the pivots chosen among the unprotected rows
    first (later rows first) and the protected rows last (again later rows first), and every pivot row is replaced by the
    p-th part of its kernel combination, which involves, besides the pivot row, only rows kept in that pass. Hence a
    combination involving an unprotected row replaces an unprotected row, and a protected row is replaced only by a
    combination of protected rows alone: the protected rows stay within themselves. In the combination replacing an
    unprotected row the protected rows are subtracted rather than added, i.e. their coefficients are taken in
    {-(p-1),...,0} instead of {0,...,p-1} (e.g. (q - d)/2 instead of (q + d)/2 for p = 2), which spans the same lattice. For the matrix of `glsm_from_refinement`
    with block sizes s, the D7 rows are `np.r_[s[0]:s[0]+s[1], s[0]+s[1]+1:s[0]+s[1]+1+s[3]]`.
    **Arguments:**
    - `Q (ArrayLike)`: Integer matrix with linearly independent rows, e.g. a GLSM charge matrix.
    - `protected_rows (Iterable[int], optional)`: Indices of the rows that may only be combined among themselves. Defaults to
      none, in which case the lattice is saturated with the later rows replaced first.
    - `primes (Iterable[int], optional)`: The primes at which to saturate. Defaults to all primes dividing the invariant factors
      of Q (computed with `snf`); passing e.g. `[2]` skips the Smith normal form and saturates at two only, which is complete
      iff no invariant factor has an odd prime factor.
    - `return_combinations (bool)`: Whether to also return how the changed rows are composed of the input rows. Defaults to
      `False`.
    **Raises:**
    - `ValueError`: Raised if the rows of Q are linearly dependent or a protected row index is out of range.
    - `OverflowError`: Raised if an entry of the result does not fit into a 64-bit integer.
    **Returns:**
    - `np.ndarray | tuple[np.ndarray, dict[int, dict[int, float]]]`: The saturated matrix, rows in the original order. If
      `return_combinations` is `True`, also a dictionary with one entry per changed row i, mapping i to the coefficients
      {j: c_j} of the input rows (floats, computed exactly and converted) such that the new row i equals
      sum_j c_j * (input row j); e.g. {5: {3: 0.5, 5: 0.5}} means that the new row 5 is half the sum of the input rows 3 and 5.
      Rows that are not keys are unchanged.
    """
    Q=np.array(Q,dtype=object); r,n=Q.shape
    if np.linalg.matrix_rank(Q.astype(float))<r: raise ValueError("The rows of Q must be linearly independent.")
    prot=np.zeros(r,dtype=bool); idx=np.array(list(protected_rows) if protected_rows is not None else [],dtype=int)
    if len(idx) and (idx.min()<0 or idx.max()>=r): raise ValueError("protected_rows out of range.")
    prot[idx]=True; pref=np.concatenate((np.flatnonzero(~prot)[::-1],np.flatnonzero(prot)[::-1]))
    if primes is None:
        D=snf(Q); primes=[int(q) for q,_ in fmpz(int(D[r-1,r-1])).factor()]   # every invariant factor divides the last one
    T=np.array([[Fraction(int(i==j)) for j in range(r)] for i in range(r)],dtype=object)
    for p in primes:
        while True:
            N=_left_kernel_mod_p((Q%p).astype(np.int64),p)
            if len(N)==0: break
            pivots=[]; i=0
            for j in pref:
                if i==len(N): break
                rows=np.flatnonzero(N[i:,j])+i
                if not len(rows): continue
                N[[i,rows[0]]]=N[[rows[0],i]]
                N[i]=(N[i]*pow(int(N[i,j]),-1,p))%p
                for k in np.flatnonzero(N[:,j]):
                    if k!=i: N[k]=(N[k]-N[k,j]*N[i])%p
                pivots.append(int(j)); i+=1
            M=N[:len(pivots)].copy()   # protected rows enter an unprotected pivot's combination with coefficients in {-(p-1),...,0}
            M[(~prot[pivots])[:,None]&prot[None,:]&(M!=0)]-=p
            new=M.astype(object)@Q
            if (new%p).any(): raise RuntimeError("inexact kernel combination in saturate_glsm")
            Q[pivots]=new//p; T[pivots]=(M.astype(object)@T)/p
    if max((abs(int(x)) for x in Q.flat),default=0)>np.iinfo(np.int64).max: raise OverflowError('Saturated entries exceed int64; use the exact object array.')
    if not return_combinations: return Q.astype(np.int64)
    combos={i:{int(j):float(T[i,j]) for j in np.flatnonzero(T[i]!=0)} for i in range(r) if any(T[i,j]!=int(i==j) for j in range(r))}
    return Q.astype(np.int64),combos

def project_points(points, u):
    """Project points along u into one dimension lower; shape (N, n-1).

    Each point p is sheared along u to p - (p_k / u_k) u, which lies in the
    hyperplane {x_k = 0}, and coordinate k is dropped.  The map is linear with
    kernel exactly R*u.  k is chosen with |u_k| = 1 if possible, so that for
    integer inputs the result is an integer array and the map is a lattice
    isomorphism Z^n / Z*u -> Z^(n-1).  If the inputs are integral but no such
    k exists, integer arithmetic is still used whenever every entry divides
    exactly; otherwise a float array is returned.
    """
    points = np.asarray(points)
    u = np.asarray(u).ravel()
    units = np.flatnonzero(np.abs(u) == 1)
    k = units[0] if units.size else np.argmax(np.abs(u))

    integral = np.issubdtype(points.dtype, np.integer) and np.issubdtype(u.dtype, np.integer)
    if integral:
        num = u[k] * points - np.outer(points[:, k], u)      # = u_k * (p - (p_k/u_k) u)
        if np.all(num % u[k] == 0):
            return np.delete(num // u[k], k, axis=1)
        points = points.astype(float)

    sheared = points - np.outer(points[:, k] / u[k], u)
    return np.delete(sheared, k, axis=1)

def compute_gv_h11(generators, grading_vector, q, intnums, **kwargs):
    """compute_gv, but with h11 taken from q.shape[0] instead of the largest index in intnums."""
    q = np.asarray(q)
    h11 = q.shape[0]
    intnums = {k: v for k, v in intnums.items() if v != 0}
    m = max(i for k in intnums for i in k)
    if m == h11 - 1:
        return compute_gv(generators, grading_vector, q, intnums, **kwargs)
    perm = np.arange(h11)
    perm[[m, h11 - 1]] = perm[[h11 - 1, m]]        # transposition, so perm is its own inverse
    intnums = {tuple(sorted(int(perm[i]) for i in k)): v for k, v in intnums.items()}
    if kwargs.get("target_points") is not None:
        kwargs["target_points"] = np.atleast_2d(kwargs["target_points"])[:, perm]
    res = compute_gv(np.asarray(generators)[:, perm], np.asarray(grading_vector)[perm], q[perm], intnums, **kwargs)
    back = lambda v: tuple(int(x) for x in np.asarray(v)[perm])
    if res and isinstance(res[0][0][0], tuple):     # non-threefold output: ((v, c), gv)
        return [((back(v), c), gv) for (v, c), gv in res]
    return [(back(v), gv) for v, gv in res]

def cayley_polytope(polytopes):
    """
    **Description:**
    Constructs the Cayley polytope of the Newton polytopes Delta_1, ..., Delta_r of an
    r-part nef partition,
        Cay(Delta_1, ..., Delta_r) = conv(Delta_1 x {e_1}, ..., Delta_r x {e_r}),
    where e_j is the j-th unit vector of Z^r.
    **Arguments:**
    - `polytopes (Sequence[Polytope | ArrayLike])`: The Newton polytopes Delta_j (or their
      lattice points, one per row), all in the same lattice M.
    **Raises:**
    - `ValueError`: Raised if the polytopes do not all live in the same lattice.
    **Returns:**
    - `Polytope`: The Cayley polytope in M x Z^r.
    """
    pts = [np.asarray(P.points() if hasattr(P, "points") else P, dtype=int) for P in polytopes]
    if len({p.shape[1] for p in pts}) != 1:
        raise ValueError("All Newton polytopes must live in the same lattice M.")
    r = len(pts)
    E = np.eye(r, dtype=int)
    cayley_pol = Polytope(np.concatenate(
        [np.column_stack([p, np.tile(E[j], (len(p), 1))]) for j, p in enumerate(pts)],
        axis=0))
    return cayley_pol

def _so8_chain(q: np.ndarray,d7_cols: Sequence[int])->np.ndarray:
    """
    **Description:**
    The symmetric rows of `symmetric_so8_rows` for a given choice of the D7 columns, see there.
    **Arguments:**
    - `q (np.ndarray)`: Integer matrix whose rows are a basis of the D7 lattice.
    - `d7_cols (Sequence[int])`: The columns of the D7 points, one per row of `q`.
    **Raises:**
    - `ValueError`: Raised, with the reason, if the rows are not a single SO(8) chain with respect to these columns.
    **Returns:**
    - `np.ndarray`: The symmetric rows, the row of the D7 point furthest from rho_- first.
    """
    r,n=q.shape; T=np.asarray(d7_cols,dtype=int).ravel(); rest=np.setdiff1d(np.arange(n),T)
    if len(T)!=r or len(set(T.tolist()))!=r: raise ValueError(f"{r} distinct D7 columns are needed, one per row.")
    B=q[:,T]; det=int(round(np.linalg.det(B)))
    if det==0: raise ValueError("The block of the rows on the D7 columns is singular.")
    adj=np.rint(det*np.linalg.inv(B)).astype(np.int64)
    if not np.array_equal(adj@B,det*np.eye(r,dtype=np.int64)): raise ValueError("Inexact adjugate of the D7 block.")
    # canonical rows: N[k] is the primitive relation in the span of q supported on the D7 column T[k] and on the other columns, -m_k at T[k]
    N=-np.sign(det)*(adj@q); N//=np.gcd.reduce(N,axis=1)[:,None]
    if (N[:,rest]<0).any(): raise ValueError("A canonical D7 row has a negative entry outside the D7 columns.")
    order=np.argsort(N[:,rest].sum(axis=1),kind="stable"); N=N[order]
    if (np.diff(N[:,rest],axis=0)<0).any(): raise ValueError("The canonical D7 rows are not totally ordered: not a single SO(8) chain.")
    S=np.vstack([N[:1],np.diff(N,axis=0)])[::-1]
    # S = A @ q with A = S[:,T] @ B^-1 must be unimodular
    A=S[:,T]@adj
    if (A%det).any() or abs(int(np.prod(N[np.arange(r),T[order]])))!=abs(det):
        raise ValueError("The symmetric rows and the input rows do not generate the same lattice.")
    return S

def symmetric_so8_rows(q: ArrayLike,d7_cols: Sequence[int]|None=None)->np.ndarray:
    """
    **Description:**
    Brings the D7 rows of one SO(8) stack (four rows) to the symmetric basis, in which the SO(8) locus is z^mu = 1/2 for all mu
    and the D7 curves with non-vanishing GV invariants are the unit vectors. The input may be any Z-basis of the D7 lattice:
    the D7 rows of `Qd7()`, of `Qsat()`, or rows that are already symmetric, which are returned unchanged.
    The construction only uses the lattice spanned by the rows and the D7 columns:
    1. For every D7 column d there is, up to scale, one relation in the span of the rows that vanishes on the other D7 columns.
       Its primitive multiple N_d with negative entry -m_d at d is the row of `glsm_from_refinement`, m_d d = sum_j s_j v_j with
       v_j the non-D7 rays (rho_- and fiber rays) and s_j >= 0. It does not depend on the basis the rows are given in.
    2. On an SO(8) chain the non-D7 parts s of these rows are totally ordered, N_1 < N_2 < ... componentwise: the D7 points move
       step by step from rho_- towards the O7 point (in the Tate form, N_p belongs to the coefficient of h^p of the D7 polynomial
       eta^2 - h chi in the O7 section h, and its roots, the D7 positions, are the ratios of consecutive coefficients).
    3. The symmetric rows are the consecutive differences N_p - N_(p-1) (N_0 = 0), returned with the last difference first:
       each has exactly one negative entry, -m_p at its own D7 column, the entry m_(p-1) at the D7 column of its predecessor
       (rho_- for the last row) and non-negative entries on the fiber rays.
    4. The result is verified to be a unimodular transformation of the input rows.
    For four D7 points with multiplicities (1,1,1,2) the D7 columns become (-2,0,0,0), (1,-1,0,0), (0,1,-1,0), (0,0,1,-1).
    **Arguments:**
    - `q (ArrayLike)`: Integer matrix whose rows are a Z-basis of the D7 lattice, e.g. `Q[F.D7_indices()]`. The columns may be
      all columns of the charge matrix or any subset containing the support of the rows.
    - `d7_cols (Sequence[int], optional)`: The columns of the D7 points, e.g. `F.Qd7(D7_only=True).argmin(axis=1)`. If not
      given, they are determined from the rows: the columns with a negative entry are tried first, then all choices among the
      non-zero columns; the D7 columns are the ones for which steps 1-4 go through.
    **Raises:**
    - `ValueError`: Raised if the rows are not a single SO(8) chain: a canonical row has a negative entry outside the D7
      columns, the canonical rows are not totally ordered, or their differences do not generate the lattice of the input rows.
      With `d7_cols` given the message states which of these failed.
    **Returns:**
    - `np.ndarray`: The symmetric rows. If the rows of `q` already are the symmetric rows, in any order, a copy of `q`.
    """
    q=np.asarray(q,dtype=np.int64); r=len(q)
    if d7_cols is not None:
        S=_so8_chain(q,d7_cols)
    else:
        supp=np.flatnonzero(q.any(axis=0)); neg=np.flatnonzero((q<0).any(axis=0))
        for T in ([tuple(neg)] if len(neg)==r else [])+list(combinations(supp,r)):
            try: S=_so8_chain(q,T); break
            except ValueError: continue
        else:
            raise ValueError("The rows are not an SO(8) chain for any choice of D7 columns; pass d7_cols to see the reason.")
    return q.copy() if same_rows(S,q) else S

# =============================================================================
# 3. Orientifold functions for Calabi-Yau complete intersections of any
#    codimension r.
# =============================================================================

def generic_sections_degenerate(points: np.ndarray,line_bundles: ArrayLike,polytopes: Sequence|None=None,return_polytopes: bool=False)->bool|tuple[bool,list|None]:
    """
    **Description:**
    Complete-intersection version of `UF.generic_section_factorizes`: whether the generic complete intersection of
    sections of the line bundles L_1,...,L_r on the fan with rays `points` is degenerate, i.e. whether
    - some L_a has no sections,
    - some toric coordinate x_rho appears in no monomial of any section of any L_a (then the generic complete intersection
      is invariant under the one-parameter subgroup of the ray rho; for r=1 this is the test of
      `UF.generic_section_factorizes`), or
    - some toric coordinate divides every section of some L_a (a fixed component, so the complete intersection is
      reducible).
    A single equation of a complete intersection need not involve all coordinates (e.g. for a partition), so a coordinate
    only has to appear in one of the equations; applying `UF.generic_section_factorizes` to every L_a separately would
    wrongly reject such complete intersections.
    **Arguments:**
    - `points (np.ndarray)`: Rays of the toric fan, one per row.
    - `line_bundles (ArrayLike)`: The (r,N) divisor coefficients of L_1,...,L_r in the toric prime divisor basis.
    - `polytopes (Sequence|None)`: The Newton polytopes of L_1,...,L_r on `points` (`Newton_Polytope`), if they are
      already known; otherwise they are constructed here. Defaults to `None`.
    - `return_polytopes (bool)`: Whether to also return the Newton polytopes, so that the caller does not have to
      construct them again. Defaults to `False`.
    **Returns:**
    - `bool|tuple[bool,list|None]`: `True` if the generic complete intersection is degenerate, otherwise `False`. With
      `return_polytopes` the pair of this and the list of the Newton polytopes of L_1,...,L_r, or `None` in its place if
      not all of them were constructed (the test stops at the first degenerate L_a).
    """
    V=np.asarray(points,dtype=int)
    line_bundles=np.asarray(line_bundles,dtype=int).reshape(-1,len(V))
    appears=np.zeros(len(V),dtype=bool)
    newton_polytopes=[]
    degenerate=False
    for a,L in enumerate(line_bundles):
        try:
            NP=Newton_Polytope(V,L) if polytopes is None else polytopes[a]
            P=np.asarray(NP.points(),dtype=int).reshape(-1,V.shape[1])
        except ValueError:
            degenerate=True
            break
        newton_polytopes.append(NP)
        exps=V@P.T+L[:,None]
        if len(P)==0 or (exps>0).all(axis=1).any():
            degenerate=True
            break
        appears|=(exps!=0).any(axis=1)
    else:
        degenerate=not appears.all()
    if return_polytopes:
        return degenerate,(newton_polytopes if len(newton_polytopes)==len(line_bundles) else None)
    return degenerate

def O3O7_line_bundles(pts_CY_ambient: np.ndarray,divisors: ArrayLike,q: ArrayLike,rescalings: ArrayLike,orbifold_points: np.ndarray,parities: ArrayLike|None=None,polytopes: Sequence|None=None,return_polytopes: bool=False)->tuple:
    """
    **Description:**
    Determines the orientifold line bundles L_1,...,L_r on the toric orbifold of a Calabi-Yau complete intersection
    X={P_1=...=P_r=0}, P_a a section of D^(a), with D^(1)+...+D^(r)=(1,...,1); the generalization of `UF.O3O7_line_bundle`.
    The Z_2 action q/2 multiplies the monomial of the lattice point m of the Newton polytope of D^(a) by (-1)^<m,q>, up to a
    factor common to all sections of D^(a). Since the D^(a) sum to the anticanonical divisor, these factors cancel against
    that of the measure mu in Omega=Res(mu/(P_1...P_r)), so if every P_a only contains monomials with <m,q>=eps_a mod 2,
    Omega is multiplied by (-1)^(eps_1+...+eps_r). Hence every choice of parities eps with odd sum (and monomials of these
    parities) defines an O3/O7 orientifold; for r=1 this is eps=1, the projected-in monomials of `UF.O3O7_line_bundle`,
    and for r>1 different choices give different orientifolds. For a monomial m_a of parity eps_a, B=X/sigma is cut out in
    the orbifold by the sections of L_a=(pts@m_a+D^(a))/rescalings (`toric_orbifold`). If pts@m_a+D^(a) is odd on a ray
    with rescaling 2, the fixed coordinate of that ray divides every monomial of parity eps_a, and the choice is discarded.
    A choice is regular if the generic complete intersection of the L_a is not degenerate (`generic_sections_degenerate`).
    **Arguments:**
    - `pts_CY_ambient (np.ndarray)`: Rays of the original Calabi-Yau ambient toric fan, one per row.
    - `divisors (ArrayLike)`: The (r,N) divisor coefficients of D^(1),...,D^(r), summing to (1,...,1).
    - `q (ArrayLike)`: Integer vector defining the `Z_2` action.
    - `rescalings (ArrayLike)`: Rescalings of toric divisor classes under the orbifold map.
    - `orbifold_points (np.ndarray)`: The orbifold rays, one per row, in the order of `pts_CY_ambient`.
    - `parities (ArrayLike|None)`: The parities eps_a in {0,1}, with odd sum. If `None`, the first regular choice in
      lexicographic order is used (with a warning if there are several), or the first non-regular one if none is regular.
      Defaults to `None`.
    **Raises:**
    - `ValueError`: Raised if the divisors do not sum to (1,...,1), or if `parities` does not consist of r entries in
      {0,1} with odd sum.
    - `polytopes (Sequence|None)`: The Newton polytopes of D^(1),...,D^(r) on `pts_CY_ambient` (`Newton_Polytope`), as
      far as they are already known: a sequence of r entries, with `None` for every polytope that has to be constructed
      here. Defaults to `None` (all are constructed here).
    - `return_polytopes (bool)`: Whether to also return the result of the regularity test of the chosen line bundles and
      their Newton polytopes on `orbifold_points`, which this function has constructed for that test, so that the caller
      does not have to repeat it. Defaults to `False`.
    **Returns:**
    - `tuple`: `(line_bundles, parities)`, the (r,N) integer orbifold line bundles and the parities used, or
      `(None, None)` if no choice exists. With `return_polytopes` the tuple `(line_bundles, parities, degenerate,
      newton_polytopes)`: `degenerate` is `generic_sections_degenerate` of the line bundles (`True` if no choice exists)
      and `newton_polytopes` the list of their Newton polytopes on `orbifold_points`, or `None` if not all of them were
      constructed (only possible for degenerate line bundles).
    """
    pts=np.asarray(pts_CY_ambient,dtype=int)
    D=np.asarray(divisors,dtype=int).reshape(-1,len(pts))
    q=np.asarray(q,dtype=int)
    rescalings=np.asarray(rescalings,dtype=int)
    if (D.sum(axis=0)!=1).any():
        raise ValueError("The divisors must sum to (1,...,1).")
    polytopes=[None]*len(D) if polytopes is None else list(polytopes)
    monomials=[np.asarray((Newton_Polytope(pts,d) if P is None else P).points(),dtype=int).reshape(-1,pts.shape[1]) for d,P in zip(D,polytopes,strict=True)]
    classes=[{p:m[(m@q)%2==p][0] for p in (0,1) if ((m@q)%2==p).any()} for m in monomials]
    if parities is None:
        choices=[e for e in product(*[sorted(c) for c in classes]) if sum(e)%2==1]
    else:
        e=tuple(int(x) for x in np.asarray(parities).reshape(-1))
        if len(e)!=len(D) or any(x not in (0,1) for x in e) or sum(e)%2==0:
            raise ValueError(f"`parities` must consist of {len(D)} entries in {{0,1}} with odd sum, got {list(e)}.")
        choices=[e] if all(p in c for p,c in zip(e,classes)) else []
    bundles=[]
    for e in choices:
        w=np.array([pts@classes[a][p]+D[a] for a,p in enumerate(e)])
        if not (w%rescalings).any():
            bundles.append((w//rescalings,np.array(e,dtype=int)))
    if not bundles:
        return (None,None,True,None) if return_polytopes else (None,None)
    # every candidate is tested once; the test constructs the Newton polytopes of its line bundles, which are returned with it
    tested=[b+generic_sections_degenerate(orbifold_points,b[0],return_polytopes=True) for b in bundles]
    regular=[t for t in tested if not t[2]]
    if len(regular)>1:
        warnings.warn(f"The parities {[t[1].tolist() for t in regular]} all give regular orientifolds, which differ in general; "
                      f"{regular[0][1].tolist()} is used. Pass `parities` to choose another one.")
    chosen=regular[0] if regular else tested[0]
    return chosen if return_polytopes else chosen[:2]

def divisors_in_partition_rep(rays: ArrayLike,divisors: ArrayLike)->tuple[np.ndarray,np.ndarray]|tuple[None,None]:
    """
    **Description:**
    Shifts divisors a^(1),...,a^(k) with a^(1)+...+a^(k)=(1,...,1) by principal divisors to a partition,
    b^(i)=a^(i)+rays@m_i with coefficients 0 and 1 and disjoint supports, using `_partition_representatives` of `CICY`;
    the generalization of `UF.divisor_in_zero_one_rep` (for k=2 and a^(2)=1-a^(1) both find the same shifts of a^(1)).
    As there, a pivoted QR decomposition chooses rays forming a basis of the vector space; they need not form a lattice
    basis, since the values of a 0/1 representative on them determine its integral shift exactly.
    **Arguments:**
    - `rays (ArrayLike)`: The ray generators of the toric fan, one per row.
    - `divisors (ArrayLike)`: The (k,N) integer divisor coefficients, summing to (1,...,1).
    **Raises:**
    - `ValueError`: Raised if the shapes are incompatible, the divisors do not sum to (1,...,1) or the rays do not span.
    **Returns:**
    - `tuple[np.ndarray,np.ndarray]|tuple[None,None]`: `(b, m)`, the (k,N) partition and the (k,dim) shifts, or
      `(None, None)` if there is no partition.
    """
    V=np.asarray(rays,dtype=int)
    A=np.asarray(divisors,dtype=int)
    if V.ndim!=2 or A.ndim!=2 or A.shape[1]!=V.shape[0]:
        raise ValueError("Incompatible shapes for divisors and rays.")
    if (A.sum(axis=0)!=1).any():
        raise ValueError("The divisors must sum to (1,...,1).")
    n=V.shape[1]
    _,R,piv=qr(V.T.astype(float),pivoting=True)
    if np.sum(np.abs(np.diag(R))>1e-9)<n:
        raise ValueError("Rays do not span: shift m is not unique.")
    S=piv[:n]
    B=cicy._partition_representatives(V,A,S)
    if B is None:
        return None,None
    M=np.rint(np.linalg.solve(V[S].astype(float),(B-A)[:,S].T).T).astype(int)
    if (A+M@V.T!=B).any():
        raise RuntimeError("The partition is not a shift of the divisors by principal divisors.")
    return B,M

def _bernstein_solvable(directions: list[np.ndarray])->bool:
    """
    **Description:**
    Whether a generic system of Laurent polynomials f_1,...,f_m with supports S_1,...,S_m (independent generic
    coefficients) has a common zero in the torus: by Bernstein's theorem iff dim(sum_{i in I} S_i)>=|I| for every non-empty
    subset I. The dimension of the Minkowski sum is the rank of the stacked direction vectors S_i[1:]-S_i[0], so a single
    monomial (no directions) never vanishes. The criterion of `_singular_orbits_hypersurface` and `intersecting_cones`.
    **Arguments:**
    - `directions (list[np.ndarray])`: For each support S_i the (|S_i|-1,d) array S_i[1:]-S_i[0], all with the same d.
    **Returns:**
    - `bool`: `True` if the generic system has a common zero (also if m=0).
    """
    m=len(directions)
    for s in range(1,m+1):
        for I in combinations(range(m),s):
            D=np.vstack([directions[i] for i in I])
            if (np.linalg.matrix_rank(D) if len(D) else 0)<s:
                return False
    return True

def _complete_intersection_chi(F: list[np.ndarray],K: np.ndarray)->int:
    """
    **Description:**
    Euler characteristic of a nondegenerate complete intersection Y={f_1=...=f_t=0} in a torus T of dimension k>=1 with
    character lattice spanned by the columns of K, f_a with Newton polytope conv(F[a]) (each F[a] in its own translate of
    that lattice). Cayley trick: Y~={sum_a lambda_a f_a=0} in T x P^(t-1) fibres over T with fibre P^(t-1) over Y and
    P^(t-2) elsewhere, so chi(Y)=chi(Y~) as chi(T)=0, and Y~ is the disjoint union over the non-empty subsets I of the
    nondegenerate hypersurfaces {sum_{a in I} lambda_a f_a=0} in T x (C*)^(|I|-1), whose Newton polytopes are the Cayley
    polytopes Cay(F_a : a in I). By Kouchnirenko chi=(-1)^(k+|I|) Vol(Cay), normalized in the lattice of K plus
    {y in Z^|I| : sum y=0}, and 0 if Cay is not full-dimensional (as in `_orbit_pieces_hypersurface`, the case t=1). Every F[a] is
    shifted by its first point, which is a lattice isomorphism of the Cayley polytope.
    **Arguments:**
    - `F (list[np.ndarray])`: The vertices of the Newton polytopes, one array (one vertex per row) per polynomial.
    - `K (np.ndarray)`: The (n,k) basis of the character lattice of T, as columns.
    **Returns:**
    - `int`: chi(Y).
    """
    k=K.shape[1]
    chi=0
    for s in range(1,len(F)+1):
        D=np.vstack([-np.ones((1,s-1),dtype=int),np.eye(s-1,dtype=int)])
        Kc=np.block([[K,np.zeros((K.shape[0],s-1),dtype=int)],[np.zeros((s,k),dtype=int),D]])
        for I in combinations(range(len(F)),s):
            P=np.vstack([np.column_stack([F[a]-F[a][0],np.tile(np.eye(s,dtype=int)[j],(len(F[a]),1))]) for j,a in enumerate(I)])
            if np.linalg.matrix_rank(P[1:]-P[0])<k+s-1:
                continue
            chi+=(-1)**(k+s)*_normalized_volume(P,Kc)
    return int(chi)

def _orbit_pieces(V: np.ndarray,cones: np.ndarray,W: list[np.ndarray],E: list[np.ndarray],cache: dict)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    **Description:**
    Euler characteristics and dimensions of a generic complete intersection on the torus orbits O(cone) of
    equal-dimensional cones; the generalization of `_orbit_pieces_hypersurface`. On O(cone) the sections of L_a restrict to Laurent
    polynomials whose Newton polytopes are the faces of the Newton polytopes cut out by the rays of the cone. With k the
    orbit dimension: sections with empty faces vanish identically and are dropped; if none is left, O(cone) lies in the
    complete intersection (chi=[k=0]); if a face is a single monomial, or the remaining system has no solution by
    Bernstein's theorem (`_bernstein_solvable`), the orbit is missed; otherwise the piece has dimension k minus the number
    of remaining sections and its Euler characteristic follows from `_complete_intersection_chi`. Since a full-dimensional
    Cayley polytope spans cone^perp, the result only depends on the faces and is cached per face.
    **Arguments:**
    - `V (np.ndarray)`: The rays, one per row.
    - `cones (np.ndarray)`: Cones of one dimension, as rows of zero-based ray indices (shape (C,0) for the zero cone).
    - `W (list[np.ndarray])`: The vertices of the Newton polytopes, one array per line bundle.
    - `E (list[np.ndarray])`: Boolean matrices, E[a][i,j] if vertex j of the a-th Newton polytope lies on the face cut out
      by ray i.
    - `cache (dict)`: Face cache shared between calls with the same `W`.
    **Returns:**
    - `tuple[np.ndarray,np.ndarray,np.ndarray]`: (chi, dim, number of face vertices) per cone, with dim=-1 for empty
      pieces and the face vertices counted per line bundle (shape (C,r)).
    """
    n=V.shape[1]
    k=n-cones.shape[1]
    masks=[e[cones].all(axis=1) if cones.shape[1] else np.ones((len(cones),len(w)),dtype=bool) for w,e in zip(W,E)]
    n_vert=np.stack([m.sum(axis=1) for m in masks],axis=1).reshape(len(cones),len(W))
    chi=np.zeros(len(cones),dtype=np.int64)
    dim=np.full(len(cones),-1)
    for i in range(len(cones)):
        alive=np.flatnonzero(n_vert[i])
        if len(alive)==0:
            dim[i]=k
            chi[i]=int(k==0)
            continue
        if (n_vert[i,alive]==1).any():
            continue
        key=(k,)+tuple(m[i].tobytes() for m in masks)
        if key not in cache:
            F=[W[a][masks[a][i]] for a in alive]
            if _bernstein_solvable([f[1:]-f[0] for f in F]):
                cache[key]=(k-len(alive),_complete_intersection_chi(F,_saturated_kernel(V[cones[i]])))
            else:
                cache[key]=(-1,0)
        dim[i],chi[i]=cache[key]
    return chi,dim,n_vert

def _singular_orbits(V: np.ndarray,V_up: np.ndarray,r: np.ndarray,L: np.ndarray,NP: list,W: list[np.ndarray],E: list[np.ndarray],cones_by_dim: list[np.ndarray])->list[tuple[tuple[int,...],str]]:
    """
    **Description:**
    Finds the torus orbits along which the orientifold-invariant complete intersection X is singular; the generalization of
    `_singular_orbits_hypersurface`. X is the double cover of B, cut out upstairs by the pull-backs of the sections of the L_a, whose
    exponent of x_i is r_i times the exponent on B. On an orbit O(tau) that X meets (the sections with non-empty faces have
    a common zero by Bernstein's theorem), X is singular if the upstairs cone is not unimodular (X passes through a
    quotient singularity). The restrictions of the sections with non-empty faces are generic, so X cap O(tau) is smooth and
    their differentials along O(tau) are independent. The sections a in the set D of sections vanishing identically on
    O(tau) are P_a=sum_{j in tau} x_j f_{a,j}+O(x^2) there, so X is singular exactly where the matrix (f_{a,j}) has rank
    <|D|, i.e. where sum_{a in J} lambda_a f_{a,j}=0 for all j in tau for some lambda in (C*)^J and non-empty J in D. For
    each J this is a generic system on O(tau) x (C*)^(|J|-1) together with the remaining sections (the supports of the
    f_{a,j} are the lattice points whose upstairs exponents along tau are the unit vector e_j, and each monomial lies in at
    most one of them), and it has a solution iff `_bernstein_solvable`. For r=1 this is the test of `_singular_orbits_hypersurface`.
    **Arguments:**
    - `V (np.ndarray)`: The orbifold rays, one per row.
    - `V_up (np.ndarray)`: The corresponding primitive rays of the Calabi-Yau ambient lattice N, one per row.
    - `r (np.ndarray)`: Ray rescalings (2 for mu-odd rays, else 1).
    - `L (np.ndarray)`: The (r,N) orientifold line bundles.
    - `NP (list)`: Their Newton polytopes.
    - `W (list[np.ndarray])`: Their vertices, one array per line bundle.
    - `E (list[np.ndarray])`: Vertex-face incidences, see `_newton_data`.
    - `cones_by_dim (list[np.ndarray])`: All cones of the fan, see `_cones_by_dimension`.
    **Returns:**
    - `list[tuple[tuple[int,...],str]]`: The singular orbits as (one-indexed cone, reason).
    """
    n=V.shape[1]
    points=[None]*len(L)
    meets_cache={}
    out=[]
    for cones in cones_by_dim:
        d=cones.shape[1]
        masks=[e[cones].all(axis=1) for e in E]
        n_vert=np.stack([m.sum(axis=1) for m in masks],axis=1).reshape(len(cones),len(L))
        faces=lambda i: [W[a][masks[a][i]] for a in np.flatnonzero(n_vert[i])]
        meets=np.zeros(len(cones),dtype=bool)
        for i in range(len(cones)):
            key=tuple(m[i].tobytes() for m in masks)
            if key not in meets_cache:
                meets_cache[key]=_bernstein_solvable([F[1:]-F[0] for F in faces(i)])
            meets[i]=meets_cache[key]
        meets=np.flatnonzero(meets)
        if len(meets)==0:
            continue
        M=V_up[cones[meets]].astype(float)
        minors=np.stack([np.rint(np.abs(np.linalg.det(M[:,:,list(c)]))) for c in combinations(range(n),d)],axis=1).astype(np.int64)
        for i in meets[np.gcd.reduce(minors,axis=1)!=1]:
            out.append((tuple(int(x)+1 for x in cones[i]),"quotient singularity"))
        for i in meets[(n_vert[meets]==0).any(axis=1)]:
            c=cones[i]
            dead=np.flatnonzero(n_vert[i]==0)
            first={}
            for a in dead:
                points[a]=np.asarray(NP[a].points(),dtype=int).reshape(-1,n) if points[a] is None else points[a]
                exps=r[c][:,None]*(V[c]@points[a].T+L[a][c][:,None])
                for j in range(d):
                    first[a,j]=points[a][(exps.sum(axis=0)==1)&(exps[j]==1)]
            F=faces(i)
            for J in (J for s in range(1,len(dead)+1) for J in combinations(dead,s)):
                unit=np.eye(len(J),dtype=int)[:,1:]
                supports=[np.column_stack([f,np.zeros((len(f),len(J)-1),dtype=int)]) for f in F]
                for j in range(d):
                    S=np.vstack([np.column_stack([first[a,j],np.tile(unit[b],(len(first[a,j]),1))]) for b,a in enumerate(J)])
                    if len(S):
                        supports.append(S)
                if _bernstein_solvable([S[1:]-S[0] for S in supports]):
                    out.append((tuple(int(x)+1 for x in c),"first-order terms vanish simultaneously"))
                    break
    return out

class _HypersurfaceView:
    """
    **Description:**
    A `CICY_orientifold` of a hypersurface (r = 1) with the interface of `FT_CY.CY_orientifold` that the hypersurface
    functions copied from `Uplift_functions` (suffix `_hypersurface`) use: `line_bundle()` is the single row of
    `divisors()`, and every other attribute is that of the orientifold.
    """
    def __init__(self,O: CICY_orientifold)->None:
        self._O=O

    def line_bundle(self)->np.ndarray|None:
        L=self._O.divisors()
        return None if L is None else np.asarray(L,dtype=int)[0]

    def __getattr__(self,name: str):
        return getattr(self._O,name)

def _hypersurface_method(O: CICY_orientifold,method: str,reflexive: bool=False)->bool:
    """
    **Description:**
    Decides whether a quantity of `O` is computed with the hypersurface function copied from `Uplift_functions` (suffix
    `_hypersurface`, as in `FT_CY.CY_orientifold`) or with the function of this module for complete intersections of
    any codimension. The hypersurface functions need a single defining equation (r = 1) in a four-dimensional ambient
    space.
    **Arguments:**
    - `O (CICY_orientifold)`: The Calabi-Yau complete intersection orientifold.
    - `method (str)`: "hypersurface", "general", or "auto": the hypersurface function if it applies, and the general one
      otherwise.
    - `reflexive (bool)`: Whether the hypersurface function also needs `O.polytope()` to be a reflexive polytope of
      dimension 4; "auto" then uses the general function if it is not. Defaults to `False`.
    **Raises:**
    - `ValueError`: Raised if the method is unknown, or is "hypersurface" for r > 1 or an ambient dimension other than 4.
    **Returns:**
    - `bool`: Whether the hypersurface function is used. `False` if the orientifold has no divisors (it is not regular),
      so that the general function raises the corresponding error.
    """
    if method not in ("auto","hypersurface","general"):
        raise ValueError(f"method {method!r} is unknown; choose from 'auto', 'hypersurface', 'general'.")
    L=O.divisors()
    if method=="general" or L is None:
        return False
    hypersurface=len(L)==1 and np.asarray(O.vectors_orbifold()).shape[1]==4
    if method=="hypersurface":
        if not hypersurface:
            raise ValueError("The hypersurface method needs a single defining equation (r = 1) in a four-dimensional ambient space.")
        return True
    return hypersurface and (not reflexive or (bool(O.polytope().is_reflexive()) and O.polytope().dim()==4))

def orientifold_fixed_locus(O: CICY_orientifold,check: bool=True,return_chi: bool=False,method: str="auto")->dict:
    """
    **Description:**
    Fixed locus of the orientifold involution on the Calabi-Yau complete intersection X, computed on B=X/sigma in the
    orbifold toric variety (`vectors_orbifold()`, `divisors()`, `orbifold_toric_fan()`); the generalization of
    `orientifold_fixed_locus_hypersurface` to codimension r, restricted to Calabi-Yau threefolds (dim B=3). X is the double cover of
    B for the toric data of `O`, i.e. if the orbifold fan was refined (nef decomposition, A1 blow-ups) the corresponding
    modification of the original complete intersection.
    The fixed locus is the disjoint union of the pieces B cap O(tau) over the pointwise fixed cones tau (`_fixed_cones`),
    since a torus element acts freely on all other orbits; their dimensions and Euler characteristics are those of generic
    complete intersections in tori (`_orbit_pieces`). The O7-planes are the closures of the 2-dimensional pieces: their
    minimal cones are the seeds (mu-odd rays meeting B, and fixed cones tau of dimension >= 2 on whose orbits |tau|-1 of
    the r sections vanish identically, the non-divisorial O7-planes; for r=1 the fixed 2-cones whose orbits lie in B), and
    the pieces on cones containing a seed add up to its Euler characteristic. All remaining pieces must be points, the
    O3-planes.
    **Arguments:**
    - `O (CICY_orientifold)`: The Calabi-Yau complete intersection orientifold.
    - `check (bool)`: Whether to verify that X is smooth (see `_singular_orbits`) and that the O7-planes are disjoint, as
      the O-plane interpretation requires. Defaults to `True`.
    - `return_chi (bool)`: Whether to also compute chi(B) and chi(X)=2chi(B)-chi_I by orbit sums over all cones, and to
      verify the Lefschetz congruence chi_I = chi(X) mod 4. Defaults to `False`.
    - `method (str)`: "hypersurface" computes the fixed locus with `orientifold_fixed_locus_hypersurface`, as
      `FT_CY.CY_orientifold` (only for r = 1; its key `O7_two_cones` is returned as `O7_cones`), "general" with the
      computation of this function for any codimension, and "auto" uses "hypersurface" for r = 1 and "general"
      otherwise (see `_hypersurface_method`). Defaults to `"auto"`.
    **Raises:**
    - `ValueError`: Raised if the orientifold is not regular, X is not a threefold, the fixed locus contains a fixed divisor
      inside B or a positive-dimensional piece that is not part of an O7-plane, or (with `check`) if X is singular or
      O7-planes meet, or (with `return_chi`) if the Lefschetz congruence fails, or if `method` is unknown or is
      "hypersurface" for r > 1.
    **Returns:**
    - `dict`: `O7_rays` (one-indexed labels of `vectors_orbifold()`), `O7_cones` (the seeds of the non-divisorial O7-planes,
      cones of dimension >= 2; `O7_two_cones` of `orientifold_fixed_locus_hypersurface`), `chi_O7_components` (O7-plane -> Euler
      characteristic), `O3_cones` (cone -> number of O3-planes on its orbit), `chi_O7`, `n_O3`, `chi_I` and, with
      `return_chi`, `chi_B` and `chi_X`.
    """
    if not O.is_regular():
        raise ValueError("The orientifold is not regular (no projected-in monomials, or the generic invariant complete intersection is degenerate, see `generic_sections_degenerate`), so B and its fixed locus are not defined.")
    if _hypersurface_method(O,method):
        out=orientifold_fixed_locus_hypersurface(_HypersurfaceView(O),check=check,return_chi=return_chi)
        return {("O7_cones" if key=="O7_two_cones" else key):value for key,value in out.items()}
    V=np.asarray(O.vectors_orbifold(),dtype=int)
    L=np.asarray(O.divisors(),dtype=int)
    if V.shape[1]-len(L)!=3:
        raise ValueError("The fixed locus is only implemented for Calabi-Yau threefolds (ambient dimension minus codimension 3).")
    NP,W,E=(list(x) for x in zip(*[_newton_data(V,l) for l in L]))
    mu=orientifold_mod2_character(O.xi())
    odd=(V@mu)%2==1
    r=1+odd.astype(int)
    cones_by_dim=_cones_by_dimension(O.orbifold_toric_fan())
    if check:
        Lam=lattice_refinement(np.asarray(O.xi(),dtype=int))
        V_up=np.rint(np.linalg.solve(Lam,(r[:,None]*V).T).T).astype(int)
        singular=_singular_orbits(V,V_up,r,L,NP,W,E,cones_by_dim)
        if singular:
            raise ValueError(f"The orientifold-invariant Calabi-Yau is singular along the orbits of {singular[:5]}{' ...' if len(singular)>5 else ''}.")
    cache={}
    fixed_pieces=[]
    for cones in cones_by_dim:
        fixed=cones[_fixed_cones(V,cones,mu,odd)]
        fixed_pieces.append((fixed,)+_orbit_pieces(V,fixed,W,E,cache))
    rays,_,_,n_vert=fixed_pieces[0]
    contained=(n_vert==0).all(axis=1)
    if contained.any():
        raise ValueError(f"The fixed divisors {rays[contained,0]+1} are contained in B.")
    seeds=[]
    for cones,_,dim,_ in fixed_pieces:
        for i in np.flatnonzero(dim==2):
            c=tuple(int(x) for x in cones[i])
            if not any(set(s)<=set(c) for s in seeds):
                seeds.append(c)
    chi_O7={s:0 for s in seeds}
    o3={}
    for cones,chi,dim,n_vert in fixed_pieces:
        owners=np.stack([np.isin(cones,s).sum(axis=1)==len(s) for s in seeds],axis=1) if seeds else np.zeros((len(cones),0),dtype=bool)
        for i in np.flatnonzero(dim>=0):
            label=tuple(int(x)+1 for x in cones[i])
            own=np.flatnonzero(owners[i])
            if len(own):
                if check and len(own)>1:
                    raise ValueError(f"The O7-planes {[tuple(x+1 for x in seeds[j]) for j in own]} meet on the orbit of cone {label}.")
                chi_O7[seeds[own[0]]]+=int(chi[i])
            elif dim[i]>0:
                raise ValueError(f"The fixed locus contains a {dim[i]}-dimensional piece on the orbit of cone {label} that is not part of an O7-plane.")
            elif chi[i]:
                o3[label]=int(chi[i])
    out={"O7_rays":np.array(sorted(s[0]+1 for s in seeds if len(s)==1),dtype=int),
         "O7_cones":sorted(tuple(x+1 for x in s) for s in seeds if len(s)>1),
         "chi_O7_components":{tuple(x+1 for x in s):v for s,v in chi_O7.items()},
         "O3_cones":o3,"chi_O7":int(sum(chi_O7.values())),"n_O3":int(sum(o3.values()))}
    out["chi_I"]=out["chi_O7"]+out["n_O3"]
    if return_chi:
        chi_B=int(sum(_orbit_pieces(V,cones,W,E,cache)[0].sum() for cones in [np.zeros((1,0),dtype=int)]+cones_by_dim))
        chi_X=2*chi_B-out["chi_I"]
        if chi_X%2 or (out["chi_I"]-chi_X)%4:
            raise ValueError(f"chi(X)={chi_X} and chi_I={out['chi_I']} violate the Lefschetz congruence chi_I = chi(X) = 0 mod 2, chi_I = chi(X) mod 4.")
        out["chi_B"]=chi_B
        out["chi_X"]=chi_X
    return out

def _fixed_locus(O: CICY_orientifold,return_chi: bool=False)->dict:
    """
    **Description:**
    Cached `orientifold_fixed_locus` of a Calabi-Yau complete intersection orientifold (`_fixed_locus_hypersurface`).
    **Arguments:**
    - `O (CICY_orientifold)`: The Calabi-Yau complete intersection orientifold.
    - `return_chi (bool)`: Whether chi(B) and chi(X) are needed. Defaults to `False`.
    **Returns:**
    - `dict`: The output of `orientifold_fixed_locus`.
    """
    return O.fixed_locus(return_chi=return_chi) if hasattr(O,"fixed_locus") else orientifold_fixed_locus(O,return_chi=return_chi)

def _cayley_vertex_data(fan: Fan,divisors: np.ndarray)->tuple[list[np.ndarray],np.ndarray,np.ndarray]:
    """
    **Description:**
    The vertices of the Newton polytopes of the divisors and their incidences with the rays
    (`_newton_polytope_incidence`), i.e. the lattice-independent part of `_cayley_face_data`: the vertices of the
    Cayley polytope Cay = conv(Delta_1 x {e_1}, ..., Delta_r x {e_r}) are those of the Delta_i, in this order.
    **Arguments:**
    - `fan (Fan)`: The fan; its rays are `fan.vectors()`.
    - `divisors (np.ndarray)`: The divisor coefficient vectors, one per row.
    **Raises:**
    - `ValueError`: Raised if a divisor has no global sections.
    **Returns:**
    - `tuple[list[np.ndarray],np.ndarray,np.ndarray]`: `(vertex_sets, piece, on_cay)`: the vertices of every Delta_i (one
      per row), piece[v] = i for the vertices v of Cay coming from Delta_i, and on_cay[j,v] = (vertex v lies on the face of
      Delta_i selected by ray j).
    """
    vertex_sets,incidences=[],[]
    for a in divisors:
        _,W,on_face=_newton_polytope_incidence(fan,a)
        vertex_sets.append(W)
        incidences.append(on_face)
    piece=np.concatenate([np.full(len(W),i,dtype=int) for i,W in enumerate(vertex_sets)])
    return vertex_sets,piece,np.concatenate(incidences,axis=1)

def _cayley_face_e(vertex_sets: list[np.ndarray],backend: str,lattice_basis: np.ndarray|None=None)->dict[frozenset[int],tuple[int,np.ndarray]]:
    """
    **Description:**
    E-polynomials of the nondegenerate torus hypersurfaces of all faces of the Cayley polytope of the polytopes with the
    given vertices (`_face_e_polynomials`), keyed by the vertex sets of the faces, with the vertices optionally
    written in the coordinates of another lattice first: with `lattice_basis` the matrix A of `_even_sublattice_basis`,
    the points m of M' become m@A in the lattice M of the double cover. The keys do not depend on the lattice.
    **Arguments:**
    - `vertex_sets (list[np.ndarray])`: The vertices of the polytopes Delta_i, one per row, see `_cayley_vertex_data`.
    - `backend (str)`: "faces", "normaliz" or "cytools".
    - `lattice_basis (np.ndarray|None)`: The (n,n) coordinate change applied to the vertices from the right. Defaults to
      `None` (none).
    **Returns:**
    - `dict[frozenset[int],tuple[int,np.ndarray]]`: For every face F of the Cayley polytope, (dim F, E(Z_F)).
    """
    unit=np.eye(len(vertex_sets),dtype=int)
    W_cay=np.vstack([np.column_stack([W if lattice_basis is None else W@lattice_basis,np.tile(unit[i],(len(W),1))]) for i,W in enumerate(vertex_sets)])
    return _face_e_polynomials(W_cay,backend)

def _cayley_face_data(fan: Fan,divisors: np.ndarray,backend: str,lattice_basis: np.ndarray|None=None)->tuple[np.ndarray,np.ndarray,dict[frozenset[int],tuple[int,np.ndarray]]]:
    """
    **Description:**
    `_cayley_face_data` of `CICY` with the vertices of the Newton polytopes optionally written in the coordinates of
    another lattice before the face E-polynomials are computed, as `HN._strata_e_functions_double_cover` does for one
    divisor: with `lattice_basis` the matrix A of `_even_sublattice_basis`, the points m of M' become m@A in the lattice
    M of the double cover. The ray-vertex incidences and the keys of the face E-polynomials (vertex sets) do not depend on
    the lattice (`_cayley_vertex_data`, `_cayley_face_e`).
    **Arguments:**
    - `fan (Fan)`: The fan; its rays are `fan.vectors()`.
    - `divisors (np.ndarray)`: The divisor coefficient vectors, one per row.
    - `backend (str)`: "faces", "normaliz" or "cytools".
    - `lattice_basis (np.ndarray|None)`: The (n,n) coordinate change applied to the vertices from the right. Defaults to
      `None` (none).
    **Raises:**
    - `ValueError`: Raised if a divisor has no global sections.
    **Returns:**
    - `tuple[np.ndarray,np.ndarray,dict]`: `(piece, on_cay, face_e)` as in `_cayley_face_data` of `CICY`.
    """
    vertex_sets,piece,on_cay=_cayley_vertex_data(fan,divisors)
    return piece,on_cay,_cayley_face_e(vertex_sets,backend,lattice_basis)

def _strata_e_sum(cones_by_dim: list[np.ndarray],n: int,on_cay: np.ndarray,piece: np.ndarray,r: int,face_e: dict[frozenset[int],tuple[int,np.ndarray]],size: int)->np.ndarray:
    """
    **Description:**
    Sum of the E-polynomials of the strata Z cap O(s) of a generic complete intersection Z over the given cones s, with
    `_stratum_e_polynomial` of `CICY` (as in `CICY.e_polynomial`).
    **Arguments:**
    - `cones_by_dim (list[np.ndarray])`: The cones, grouped by dimension, as rows of zero-based ray indices (the zero cone
      as an array of shape (1,0)).
    - `n (int)`: The dimension of the fan.
    - `on_cay (np.ndarray)`: Ray-vertex incidences of the Cayley polytope, see `_cayley_face_data`.
    - `piece (np.ndarray)`: piece[v] = i for the vertices v coming from the i-th Newton polytope.
    - `r (int)`: The number of divisors.
    - `face_e (dict)`: The per-face E-polynomials of the Cayley polytope.
    - `size (int)`: The size of the coefficient arrays.
    **Returns:**
    - `np.ndarray`: The (size,size) coefficient array of the sum.
    """
    e=np.zeros((size,size),dtype=object)
    for cones in cones_by_dim:
        if len(cones)==0:
            continue
        k=n-cones.shape[1]
        for mask,count in zip(*np.unique(on_cay[cones].all(axis=1),axis=0,return_counts=True)):
            e+=int(count)*cicy._stratum_e_polynomial(mask,k,piece,r,face_e,size)
    return e

def hodge_numbers_weighted(O: CICY_orientifold,backend: str="auto",check: bool=True,method: str="auto")->dict[str,int]:
    """
    **Description:**
    Computes the orientifold-even and -odd Hodge numbers `h^{1,1}_+`, `h^{1,1}_-`, `h^{2,1}_+` and `h^{2,1}_-` of the
    Calabi-Yau threefold complete intersection X of `O`; the generalization of `hodge_numbers_weighted_hypersurface`, by the method
    of `HN.hodge_numbers_weighted_quotient`. B=X/sigma is the generic complete intersection of the line bundles on the
    orbifold toric variety, and H^*(B,Q)=H^*(X,Q)^sigma with B a compact orbifold (X smooth), so h^{p,q}_+=h^{p,q}(B).
    X is cut out by the pull-backs of the same sections; on every orbit it is an etale double cover of B cap O(s) or
    isomorphic to it, hence a nondegenerate complete intersection with the same Newton polytopes written in the lattice M
    of X. Both E-polynomials follow from the orbit stratification with the Cayley trick (`CICY.e_polynomial`) on
    `O.orbifold_toric_fan()`, the second one with the vertices in M (`_even_sublattice_basis`), and
    h^{p,q}_-=h^{p,q}(X)-h^{p,q}(B).
    **Arguments:**
    - `O (CICY_orientifold)`: The Calabi-Yau complete intersection orientifold.
    - `backend (str)`: How the h*-polynomials of the faces are obtained, "faces" (face by face, without enumerating
          dilations), "normaliz", "cytools" or "auto" ("faces" if PyNormaliz is installed, see `_resolve_backend`). Defaults to
      `"auto"`.
    - `check (bool)`: Whether to verify that the fan is complete, that both diamonds are those of pure Hodge structures,
      that X is smooth and chi(B), chi(X) agree with the orbit sums of `orientifold_fixed_locus`, that
      h^{p,0}(B)=(1,0,0,0) (sigma acts on the holomorphic 3-form by -1), that the odd Hodge numbers are non-negative, and
      that h^{p,q}(X) equals the stringy Hodge numbers of the input complete intersection (`CICY.hodge_numbers`, which
      needs a reflexive Gorenstein Cayley polytope); the last test detects toric data describing a different Calabi-Yau.
      Defaults to `True`.
    - `method (str)`: "hypersurface" uses the closed formulas of `hodge_numbers_weighted_hypersurface`, as `FT_CY.CY_orientifold`:
      h^{1,1}_- from the interior points of the 2-faces of `O.polytope()` and of their dual faces, h^{2,1}_- from the
      Lefschetz fixed-point formula with `O.fixed_locus()`, and h^{1,1}, h^{2,1} of the polytope. It needs r = 1 and a
      reflexive `O.polytope()` of dimension 4, does not use `backend`, and always (also without `check`) verifies that
      chi(X) of the orbit sums is that of the polytope. "general" uses the E-polynomials described above, and "auto"
      uses "hypersurface" if it applies and "general" otherwise (see `_hypersurface_method`). Defaults to `"auto"`.
    **Raises:**
    - `ValueError`: Raised if the orientifold is not regular, X is not a threefold, or (with `check`) one of the tests
      fails, or if `method` is unknown or is "hypersurface" for r > 1 or a polytope that is not reflexive.
    **Returns:**
    - `dict[str,int]`: Dictionary with keys `"h11+"`, `"h11-"`, `"h21+"` and `"h21-"`.
    """
    if not O.is_regular():
        raise ValueError("The orientifold is not regular (no projected-in monomials, or the generic invariant complete intersection is degenerate, see `generic_sections_degenerate`), so B is not defined.")
    if _hypersurface_method(O,method,reflexive=True):
        return {key:int(value) for key,value in hodge_numbers_weighted_hypersurface(O).items()}
    L=np.asarray(O.divisors(),dtype=int)
    fan=O.orbifold_toric_fan()
    n=np.asarray(fan.vectors()).shape[1]
    r=len(L)
    d=n-r
    if d!=3:
        raise ValueError("Weighted Hodge numbers are only implemented for Calabi-Yau threefolds (ambient dimension minus codimension 3).")
    if check:
        _check_complete(fan)
    backend=_resolve_backend(backend)
    mu=orientifold_mod2_character(O.xi())
    vertex_sets,piece,on_cay=_cayley_vertex_data(fan,L)
    face_B=_cayley_face_e(vertex_sets,backend)
    face_X=_cayley_face_e(vertex_sets,backend,_even_sublattice_basis(mu,n))
    if face_X.keys()!=face_B.keys():
        raise RuntimeError("The face lattices in the two lattices differ.")
    cones_by_dim=[np.zeros((1,0),dtype=int)]+_cones_by_dimension(fan)
    diamonds=[]
    for face_e in (face_B,face_X):
        e=_strata_e_sum(cones_by_dim,n,on_cay,piece,r,face_e,n+r+1)
        if e[d+1:].any() or e[:,d+1:].any():
            raise RuntimeError(f"The E-polynomial has terms of degree > {d}:\n{e}")
        diamonds.append(_pure_hodge_numbers(e[:d+1,:d+1],check))
    hB,hX=diamonds
    h_odd=hX-hB
    if check:
        fixed=_fixed_locus(O,return_chi=True)
        sign=(-1)**np.add.outer(np.arange(d+1),np.arange(d+1))
        chi_B,chi_X=int((sign*hB).sum()),int((sign*hX).sum())
        if chi_B!=fixed["chi_B"] or chi_X!=fixed["chi_X"]:
            raise ValueError(f"chi(B)={chi_B}, chi(X)={chi_X} from the E-polynomials, but {fixed['chi_B']}, {fixed['chi_X']} from the orbit sums.")
        if hB[0,0]!=1 or hB[1:,0].any():
            raise ValueError(f"B has h^(p,0) = {hB[:,0]}, expected (1,0,0,0).")
        if (h_odd<0).any():
            raise ValueError(f"Negative odd Hodge numbers:\n{h_odd}")
        h_cicy=O.cicy().hodge_numbers()
        if (hX!=h_cicy).any():
            raise ValueError(f"The toric data of the orientifold describe a Calabi-Yau with Hodge numbers\n{hX}\nnot those of the CICY\n{h_cicy}.")
    return {"h11+":int(hB[1,1]),"h11-":int(h_odd[1,1]),"h21+":int(hB[2,1]),"h21-":int(h_odd[2,1])}

def O7_holomorphic_data(O: CICY_orientifold,check: bool=True,backend: str="auto",method: str="auto")->dict:
    """
    **Description:**
    Holomorphic invariants of the O7-planes S of `orientifold_fixed_locus` (keyed as in `chi_O7_components`): K_S^2,
    chi(O_S), the number of connected components, the irregularity q and the geometric genus p_g=h^{2,0}; the
    generalization of `O7_holomorphic_data_hypersurface`. An O7-plane with seed cone tau is S=B cap V(tau), the complete intersection
    of the toric divisors D_i, i in tau, and of the sections of the L_a not vanishing identically on O(tau) (the set A;
    all of them for a divisorial O7-plane). Adjunction in the orbifold toric variety gives
    K_S=(K_V+sum_{i in tau} D_i+sum_{a in A} L_a)|_S, so K_S^2 is an intersection number of `orbifold_toric_fan()`.
    As S is smooth and projective, its Hodge numbers follow from its E-polynomial, the sum of E(B cap O(s)) over the cones
    s containing tau (`CICY.e_polynomial`): components=h^{0,0}, q=h^{1,0}, p_g=h^{2,0} and chi(O_S)=components-q+p_g.
    **Arguments:**
    - `O (CICY_orientifold)`: The Calabi-Yau complete intersection orientifold.
    - `check (bool)`: Whether to verify (i) the purity of the Hodge numbers of S and chi(S) of `chi_O7_components`,
      (ii) K_S=D_i|_S/2 for divisorial O7-planes, which follows from -2K_B ~ sum of the disjoint O7-planes, (iii) the
      integrality of K_S^2 and Noether's formula 12chi(O_S)=K_S^2+chi(S), and (iv) the holomorphic Lefschetz formula
      chi_I=16+8*sum_S chi(O_S). Defaults to `True`.
    - `backend (str)`: How the h*-polynomials of the faces are obtained for the E-polynomials, "faces", "normaliz", "cytools" or "auto"
      (see `_resolve_backend`). Defaults to
      `"auto"`.
    - `method (str)`: "hypersurface" uses `O7_holomorphic_data_hypersurface`, as `FT_CY.CY_orientifold` (only for r = 1): K_S^2 from
      adjunction, chi(O_S) from Noether's formula, and the number of components and q from the face of the Newton
      polytope on the O7-plane; it does not use `backend` and its checks are those of `O7_holomorphic_data_hypersurface`.
      "general" uses the E-polynomials described above, and "auto" uses "hypersurface" for r = 1 and "general"
      otherwise (see `_hypersurface_method`). Defaults to `"auto"`.
    **Raises:**
    - `ValueError`: Raised (with `check`) if one of these tests fails, or if `method` is unknown or is "hypersurface"
      for r > 1.
    **Returns:**
    - `dict`: `p_g`, `chi_O`, `K2`, `components`, `q` (each a dict O7-plane -> int) and the totals `sum_p_g`,
      `sum_chi_O`.
    """
    if _hypersurface_method(O,method):
        return O7_holomorphic_data_hypersurface(_HypersurfaceView(O),check=check)
    fl=_fixed_locus(O)
    V=np.asarray(O.vectors_orbifold(),dtype=int)
    L=np.asarray(O.divisors(),dtype=int)
    n,r=V.shape[1],len(L)
    E=[_newton_data(V,l)[2] for l in L]
    IN={k:v for k,v in O.intersection_numbers_orbifold().items() if min(k)>=1}
    K=np.array(list(IN),dtype=int).reshape(-1,n)-1
    P=[K[:,s] for s in permutations(range(n))]
    w=np.array(list(IN.values()),dtype=float)/sum((p==K).all(axis=1) for p in P)
    T=lambda *x: float(sum((w*np.prod([x[j][p[:,j]] for j in range(n)],axis=0)).sum() for p in P))
    fan=O.orbifold_toric_fan()
    piece,on_cay,face_e=_cayley_face_data(fan,L,_resolve_backend(backend))
    cones_by_dim=[np.zeros((1,0),dtype=int)]+_cones_by_dimension(fan)
    sign=(-1)**np.add.outer(np.arange(3),np.arange(3))
    out={key:{} for key in ("p_g","chi_O","K2","components","q")}
    for S,chi_S in fl["chi_O7_components"].items():
        tau=np.array(S,dtype=int)-1
        alive=[a for a in range(r) if E[a][tau].all(axis=0).any()]
        unit=[np.eye(len(V))[i] for i in tau]
        a=sum(unit)+L[alive].sum(axis=0)-1
        factors=unit+[a,a]+[L[b] for b in alive]
        if len(factors)!=n:
            raise RuntimeError(f"O7-plane {S}: {len(factors)} intersection factors in dimension {n}.")
        K2=T(*factors)
        e=_strata_e_sum([c[np.isin(c,tau).sum(axis=1)==len(tau)] for c in cones_by_dim],n,on_cay,piece,r,face_e,n+r+1)
        if e[3:].any() or e[:,3:].any():
            raise RuntimeError(f"O7-plane {S}: the E-polynomial has terms of degree > 2:\n{e}")
        h=_pure_hodge_numbers(e[:3,:3],check)
        c,q,p_g=int(h[0,0]),int(h[1,0]),int(h[2,0])
        chi_O=c-q+p_g
        if check:
            if int((sign*h).sum())!=chi_S:
                raise ValueError(f"O7-plane {S}: chi(S)={int((sign*h).sum())} from the E-polynomial, {chi_S} from the fixed locus.")
            if len(S)==1 and not np.isclose(K2,T(*unit*3,*L)/4,atol=1e-6):
                raise ValueError(f"O7-plane {S}: adjunction gives K^2={K2}, the orientifold formula D^3 L_1...L_r/4={T(*unit*3,*L)/4}.")
            if not np.isclose(K2,round(K2),atol=1e-6):
                raise ValueError(f"O7-plane {S}: non-integral K^2={K2}.")
            if 12*chi_O!=round(K2)+chi_S:
                raise ValueError(f"O7-plane {S}: Noether's formula fails, 12chi(O)={12*chi_O} but K^2+chi={round(K2)+chi_S}.")
        out["p_g"][S],out["chi_O"][S],out["K2"][S],out["components"][S],out["q"][S]=p_g,chi_O,int(round(K2)),c,q
    out["sum_p_g"]=sum(out["p_g"].values())
    out["sum_chi_O"]=sum(out["chi_O"].values())
    if check and 16+8*out["sum_chi_O"]!=fl["chi_I"]:
        raise ValueError(f"Holomorphic Lefschetz fails: chi_I={fl['chi_I']}, but 16+8*sum chi(O_S)={16+8*out['sum_chi_O']}.")
    return out
