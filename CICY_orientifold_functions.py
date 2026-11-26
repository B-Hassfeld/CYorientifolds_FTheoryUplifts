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
# Description:  Versions of the orientifold functions of `Uplift_functions` for
#               Calabi-Yau complete intersections of any codimension r, used by
#               `CICY_orientifold.CICY_orientifold`. For r = 1 they agree with
#               the hypersurface functions of `Uplift_functions`.
# -----------------------------------------------------------------------------
from __future__ import annotations

import warnings
from collections.abc import Sequence
from itertools import combinations, permutations, product
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import qr

from cytools.vector_config.fan import Fan

from . import Uplift_functions as UF
from . import Hodge_Number_computations as HN
from .CICY import _partition_representatives, _stratum_e_polynomial

if TYPE_CHECKING:
    from .CICY_orientifold import CICY_orientifold


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
    - `polytopes (Sequence|None)`: The Newton polytopes of L_1,...,L_r on `points` (`UF.Newton_Polytope`), if they are
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
            NP=UF.Newton_Polytope(V,L) if polytopes is None else polytopes[a]
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
    the orbifold by the sections of L_a=(pts@m_a+D^(a))/rescalings (`UF.toric_orbifold`). If pts@m_a+D^(a) is odd on a ray
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
    - `polytopes (Sequence|None)`: The Newton polytopes of D^(1),...,D^(r) on `pts_CY_ambient` (`UF.Newton_Polytope`), as
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
    monomials=[np.asarray((UF.Newton_Polytope(pts,d) if P is None else P).points(),dtype=int).reshape(-1,pts.shape[1]) for d,P in zip(D,polytopes,strict=True)]
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
    B=_partition_representatives(V,A,S)
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
    monomial (no directions) never vanishes. The criterion of `UF._singular_orbits` and `UF.intersecting_cones`.
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
    {y in Z^|I| : sum y=0}, and 0 if Cay is not full-dimensional (as in `UF._orbit_pieces`, the case t=1). Every F[a] is
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
            chi+=(-1)**(k+s)*UF._normalized_volume(P,Kc)
    return int(chi)

def _orbit_pieces(V: np.ndarray,cones: np.ndarray,W: list[np.ndarray],E: list[np.ndarray],cache: dict)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    **Description:**
    Euler characteristics and dimensions of a generic complete intersection on the torus orbits O(cone) of
    equal-dimensional cones; the generalization of `UF._orbit_pieces`. On O(cone) the sections of L_a restrict to Laurent
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
                cache[key]=(k-len(alive),_complete_intersection_chi(F,UF._saturated_kernel(V[cones[i]])))
            else:
                cache[key]=(-1,0)
        dim[i],chi[i]=cache[key]
    return chi,dim,n_vert

def _singular_orbits(V: np.ndarray,V_up: np.ndarray,r: np.ndarray,L: np.ndarray,NP: list,W: list[np.ndarray],E: list[np.ndarray],cones_by_dim: list[np.ndarray])->list[tuple[tuple[int,...],str]]:
    """
    **Description:**
    Finds the torus orbits along which the orientifold-invariant complete intersection X is singular; the generalization of
    `UF._singular_orbits`. X is the double cover of B, cut out upstairs by the pull-backs of the sections of the L_a, whose
    exponent of x_i is r_i times the exponent on B. On an orbit O(tau) that X meets (the sections with non-empty faces have
    a common zero by Bernstein's theorem), X is singular if the upstairs cone is not unimodular (X passes through a
    quotient singularity). The restrictions of the sections with non-empty faces are generic, so X cap O(tau) is smooth and
    their differentials along O(tau) are independent. The sections a in the set D of sections vanishing identically on
    O(tau) are P_a=sum_{j in tau} x_j f_{a,j}+O(x^2) there, so X is singular exactly where the matrix (f_{a,j}) has rank
    <|D|, i.e. where sum_{a in J} lambda_a f_{a,j}=0 for all j in tau for some lambda in (C*)^J and non-empty J in D. For
    each J this is a generic system on O(tau) x (C*)^(|J|-1) together with the remaining sections (the supports of the
    f_{a,j} are the lattice points whose upstairs exponents along tau are the unit vector e_j, and each monomial lies in at
    most one of them), and it has a solution iff `_bernstein_solvable`. For r=1 this is the test of `UF._singular_orbits`.
    **Arguments:**
    - `V (np.ndarray)`: The orbifold rays, one per row.
    - `V_up (np.ndarray)`: The corresponding primitive rays of the Calabi-Yau ambient lattice N, one per row.
    - `r (np.ndarray)`: Ray rescalings (2 for mu-odd rays, else 1).
    - `L (np.ndarray)`: The (r,N) orientifold line bundles.
    - `NP (list)`: Their Newton polytopes.
    - `W (list[np.ndarray])`: Their vertices, one array per line bundle.
    - `E (list[np.ndarray])`: Vertex-face incidences, see `UF._newton_data`.
    - `cones_by_dim (list[np.ndarray])`: All cones of the fan, see `UF._cones_by_dimension`.
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
    functions of `Uplift_functions` use: `line_bundle()` is the single row of `divisors()`, and every other attribute is
    that of the orientifold.
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
    Decides whether a quantity of `O` is computed with the hypersurface function of `Uplift_functions` (as in
    `FT_CY.CY_orientifold`) or with the function of this module for complete intersections of any codimension. The
    hypersurface functions need a single defining equation (r = 1) in a four-dimensional ambient space.
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
    `UF.orientifold_fixed_locus` to codimension r, restricted to Calabi-Yau threefolds (dim B=3). X is the double cover of
    B for the toric data of `O`, i.e. if the orbifold fan was refined (nef decomposition, A1 blow-ups) the corresponding
    modification of the original complete intersection.
    The fixed locus is the disjoint union of the pieces B cap O(tau) over the pointwise fixed cones tau (`UF._fixed_cones`),
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
    - `method (str)`: "hypersurface" computes the fixed locus with `UF.orientifold_fixed_locus`, as
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
      cones of dimension >= 2; `O7_two_cones` of `UF.orientifold_fixed_locus`), `chi_O7_components` (O7-plane -> Euler
      characteristic), `O3_cones` (cone -> number of O3-planes on its orbit), `chi_O7`, `n_O3`, `chi_I` and, with
      `return_chi`, `chi_B` and `chi_X`.
    """
    if not O.is_regular():
        raise ValueError("The orientifold is not regular (no projected-in monomials, or the generic invariant complete intersection is degenerate, see `generic_sections_degenerate`), so B and its fixed locus are not defined.")
    if _hypersurface_method(O,method):
        out=UF.orientifold_fixed_locus(_HypersurfaceView(O),check=check,return_chi=return_chi)
        return {("O7_cones" if key=="O7_two_cones" else key):value for key,value in out.items()}
    V=np.asarray(O.vectors_orbifold(),dtype=int)
    L=np.asarray(O.divisors(),dtype=int)
    if V.shape[1]-len(L)!=3:
        raise ValueError("The fixed locus is only implemented for Calabi-Yau threefolds (ambient dimension minus codimension 3).")
    NP,W,E=(list(x) for x in zip(*[UF._newton_data(V,l) for l in L]))
    mu=UF.orientifold_mod2_character(O.xi())
    odd=(V@mu)%2==1
    r=1+odd.astype(int)
    cones_by_dim=UF._cones_by_dimension(O.orbifold_toric_fan())
    if check:
        Lam=UF.lattice_refinement(np.asarray(O.xi(),dtype=int))
        V_up=np.rint(np.linalg.solve(Lam,(r[:,None]*V).T).T).astype(int)
        singular=_singular_orbits(V,V_up,r,L,NP,W,E,cones_by_dim)
        if singular:
            raise ValueError(f"The orientifold-invariant Calabi-Yau is singular along the orbits of {singular[:5]}{' ...' if len(singular)>5 else ''}.")
    cache={}
    fixed_pieces=[]
    for cones in cones_by_dim:
        fixed=cones[UF._fixed_cones(V,cones,mu,odd)]
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
    Cached `orientifold_fixed_locus` of a Calabi-Yau complete intersection orientifold (`UF._fixed_locus`).
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
    (`HN._newton_polytope_incidence`), i.e. the lattice-independent part of `_cayley_face_data`: the vertices of the
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
        _,W,on_face=HN._newton_polytope_incidence(fan,a)
        vertex_sets.append(W)
        incidences.append(on_face)
    piece=np.concatenate([np.full(len(W),i,dtype=int) for i,W in enumerate(vertex_sets)])
    return vertex_sets,piece,np.concatenate(incidences,axis=1)

def _cayley_face_e(vertex_sets: list[np.ndarray],backend: str,lattice_basis: np.ndarray|None=None)->dict[frozenset[int],tuple[int,np.ndarray]]:
    """
    **Description:**
    E-polynomials of the nondegenerate torus hypersurfaces of all faces of the Cayley polytope of the polytopes with the
    given vertices (`HN._face_e_polynomials`), keyed by the vertex sets of the faces, with the vertices optionally
    written in the coordinates of another lattice first: with `lattice_basis` the matrix A of `HN._even_sublattice_basis`,
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
    return HN._face_e_polynomials(W_cay,backend)

def _cayley_face_data(fan: Fan,divisors: np.ndarray,backend: str,lattice_basis: np.ndarray|None=None)->tuple[np.ndarray,np.ndarray,dict[frozenset[int],tuple[int,np.ndarray]]]:
    """
    **Description:**
    `_cayley_face_data` of `CICY` with the vertices of the Newton polytopes optionally written in the coordinates of
    another lattice before the face E-polynomials are computed, as `HN._strata_e_functions_double_cover` does for one
    divisor: with `lattice_basis` the matrix A of `HN._even_sublattice_basis`, the points m of M' become m@A in the lattice
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
            e+=int(count)*_stratum_e_polynomial(mask,k,piece,r,face_e,size)
    return e

def hodge_numbers_weighted(O: CICY_orientifold,backend: str="auto",check: bool=True,method: str="auto")->dict[str,int]:
    """
    **Description:**
    Computes the orientifold-even and -odd Hodge numbers `h^{1,1}_+`, `h^{1,1}_-`, `h^{2,1}_+` and `h^{2,1}_-` of the
    Calabi-Yau threefold complete intersection X of `O`; the generalization of `UF.hodge_numbers_weighted`, by the method
    of `HN.hodge_numbers_weighted_quotient`. B=X/sigma is the generic complete intersection of the line bundles on the
    orbifold toric variety, and H^*(B,Q)=H^*(X,Q)^sigma with B a compact orbifold (X smooth), so h^{p,q}_+=h^{p,q}(B).
    X is cut out by the pull-backs of the same sections; on every orbit it is an etale double cover of B cap O(s) or
    isomorphic to it, hence a nondegenerate complete intersection with the same Newton polytopes written in the lattice M
    of X. Both E-polynomials follow from the orbit stratification with the Cayley trick (`CICY.e_polynomial`) on
    `O.orbifold_toric_fan()`, the second one with the vertices in M (`HN._even_sublattice_basis`), and
    h^{p,q}_-=h^{p,q}(X)-h^{p,q}(B).
    **Arguments:**
    - `O (CICY_orientifold)`: The Calabi-Yau complete intersection orientifold.
    - `backend (str)`: How the h*-polynomials of the faces are obtained, "faces" (face by face, without enumerating
          dilations), "normaliz", "cytools" or "auto" ("faces" if PyNormaliz is installed, see `HN._resolve_backend`). Defaults to
      `"auto"`.
    - `check (bool)`: Whether to verify that the fan is complete, that both diamonds are those of pure Hodge structures,
      that X is smooth and chi(B), chi(X) agree with the orbit sums of `orientifold_fixed_locus`, that
      h^{p,0}(B)=(1,0,0,0) (sigma acts on the holomorphic 3-form by -1), that the odd Hodge numbers are non-negative, and
      that h^{p,q}(X) equals the stringy Hodge numbers of the input complete intersection (`CICY.hodge_numbers`, which
      needs a reflexive Gorenstein Cayley polytope); the last test detects toric data describing a different Calabi-Yau.
      Defaults to `True`.
    - `method (str)`: "hypersurface" uses the closed formulas of `UF.hodge_numbers_weighted`, as `FT_CY.CY_orientifold`:
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
        return {key:int(value) for key,value in UF.hodge_numbers_weighted(O).items()}
    L=np.asarray(O.divisors(),dtype=int)
    fan=O.orbifold_toric_fan()
    n=np.asarray(fan.vectors()).shape[1]
    r=len(L)
    d=n-r
    if d!=3:
        raise ValueError("Weighted Hodge numbers are only implemented for Calabi-Yau threefolds (ambient dimension minus codimension 3).")
    if check:
        HN._check_complete(fan)
    backend=HN._resolve_backend(backend)
    mu=UF.orientifold_mod2_character(O.xi())
    vertex_sets,piece,on_cay=_cayley_vertex_data(fan,L)
    face_B=_cayley_face_e(vertex_sets,backend)
    face_X=_cayley_face_e(vertex_sets,backend,HN._even_sublattice_basis(mu,n))
    if face_X.keys()!=face_B.keys():
        raise RuntimeError("The face lattices in the two lattices differ.")
    cones_by_dim=[np.zeros((1,0),dtype=int)]+UF._cones_by_dimension(fan)
    diamonds=[]
    for face_e in (face_B,face_X):
        e=_strata_e_sum(cones_by_dim,n,on_cay,piece,r,face_e,n+r+1)
        if e[d+1:].any() or e[:,d+1:].any():
            raise RuntimeError(f"The E-polynomial has terms of degree > {d}:\n{e}")
        diamonds.append(HN._pure_hodge_numbers(e[:d+1,:d+1],check))
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
    generalization of `UF.O7_holomorphic_data`. An O7-plane with seed cone tau is S=B cap V(tau), the complete intersection
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
      (see `HN._resolve_backend`). Defaults to
      `"auto"`.
    - `method (str)`: "hypersurface" uses `UF.O7_holomorphic_data`, as `FT_CY.CY_orientifold` (only for r = 1): K_S^2 from
      adjunction, chi(O_S) from Noether's formula, and the number of components and q from the face of the Newton
      polytope on the O7-plane; it does not use `backend` and its checks are those of `UF.O7_holomorphic_data`.
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
        return UF.O7_holomorphic_data(_HypersurfaceView(O),check=check)
    fl=_fixed_locus(O)
    V=np.asarray(O.vectors_orbifold(),dtype=int)
    L=np.asarray(O.divisors(),dtype=int)
    n,r=V.shape[1],len(L)
    E=[UF._newton_data(V,l)[2] for l in L]
    IN={k:v for k,v in O.intersection_numbers_orbifold().items() if min(k)>=1}
    K=np.array(list(IN),dtype=int).reshape(-1,n)-1
    P=[K[:,s] for s in permutations(range(n))]
    w=np.array(list(IN.values()),dtype=float)/sum((p==K).all(axis=1) for p in P)
    T=lambda *x: float(sum((w*np.prod([x[j][p[:,j]] for j in range(n)],axis=0)).sum() for p in P))
    fan=O.orbifold_toric_fan()
    piece,on_cay,face_e=_cayley_face_data(fan,L,HN._resolve_backend(backend))
    cones_by_dim=[np.zeros((1,0),dtype=int)]+UF._cones_by_dimension(fan)
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
        h=HN._pure_hodge_numbers(e[:3,:3],check)
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
