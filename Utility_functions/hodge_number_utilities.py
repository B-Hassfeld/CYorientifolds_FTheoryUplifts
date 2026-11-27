"""
Hodge numbers from E-polynomials: stringy Hodge diamonds of reflexive Gorenstein polytopes (`hodge_diamond_stringy`,
`hodge_diamond_cy4`), the E-polynomials of the nondegenerate torus hypersurfaces of all faces of a lattice polytope, and
E-polynomials and Hodge numbers of generic hypersurfaces and of their double covers from the orbit stratification of a
toric fan. Collected from `Uplift_functions` (`hodge_diamond_stringy`, `hodge_diamond_cy4`, the `_hd_*` helpers,
`_saturated_kernel`, `_cones_by_dimension`) and `Hodge_Number_computations` (all functions except
`hodge_numbers_weighted_quotient`) on 2026-11-27, so that this module does not import `Uplift_functions`; those two
modules keep their own versions.
"""
from __future__ import annotations
# 'standard' imports
from itertools import combinations
from collections.abc import Callable, Iterable, Sequence
from fractions import Fraction
from math import ceil, comb
import warnings
from typing import Any, Literal
# 3rd party imports
import numpy as np
import os
import shutil
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor
import re
from pathlib import Path
from flint import fmpz_mat
# CYTools imports
from cytools import Cone, Polytope
from cytools.utils import lll_reduce
from cytools.vector_config.fan import Fan
from numpy.typing import ArrayLike
# helper modules
from .toric_geometry_utilities import Newton_Polytope

try:
    from PyNormaliz import Cone as NormalizCone
except ImportError:
    NormalizCone=None
try:
    from PyNormaliz import NmzSetNumberOfNormalizThreads
except ImportError:
    NmzSetNumberOfNormalizThreads=None

# Faces of normalized volume above this bound are handed to LattE by the "faces" backend (if LattE is installed): the
# cost of Normaliz' Hilbert series grows with the volume, that of LattE's Ehrhart polynomial does not.
_HD_FAT_FACE_VOLUME=2_000_000

# -----------------------------------------------------------------------------
# Stringy Hodge diamonds of polytopes and their helpers (from `Uplift_functions`)
# -----------------------------------------------------------------------------

def _cones_by_dimension(fan)->list[np.ndarray]:
    """
    **Description:**
    All cones of a simplicial fan, grouped by dimension, as arrays of zero-based ray indices (row d-1 holds the d-cones).
    **Arguments:**
    - `fan (Fan)`: A simplicial fan.
    **Returns:**
    - `list[np.ndarray]`: For d=1,...,dim the (C_d,d) arrays of d-cones.
    """
    maximal=[tuple(sorted(c)) for c in fan.cones()]
    n=max(len(c) for c in maximal)
    if n>fan.dim:
        raise ValueError("The fan is not simplicial.")
    return [np.array(sorted({f for c in maximal for f in combinations(c,d)}),dtype=int).reshape(-1,d)-1 for d in range(1,n+1)]

def _saturated_kernel(M: ArrayLike)->np.ndarray:
    """
    **Description:**
    Saturated integral basis of the kernel {x in Z^n : M x = 0}, computed by LLL on the embedding x -> (c*M x, x), so that
    the result does not depend on the version of `integral_nullspace` (older versions return a finite-index sublattice).
    **Arguments:**
    - `M (ArrayLike)`: Integer matrix of shape (m,n); m=0 is allowed.
    **Returns:**
    - `np.ndarray`: The kernel basis as columns, shape (n,n-rank(M)).
    """
    M=np.asarray(M,dtype=np.int64)
    m,n=M.shape
    if m==0:
        return np.eye(n,dtype=int)
    rank=np.linalg.matrix_rank(M)
    c=2**12*(1+int(np.abs(M).max()))
    while True:
        R=np.array(fmpz_mat(np.hstack([c*M.T,np.eye(n,dtype=np.int64)]).tolist()).lll().tolist(),dtype=object)
        K=R[np.all(R[:,:m]==0,axis=1)][:,m:]
        if len(K)==n-rank:
            return K.astype(np.int64).T
        c*=2**12

def hodge_diamond_stringy(P: Polytope|ArrayLike,backend: Literal["auto","faces","normaliz","latte","cytools"]="auto")->np.ndarray:
    """
    **Description:**
    Computes the stringy Hodge numbers of the Calabi-Yau complete intersection
    defined by the reflexive Gorenstein support P (e.g. the Cayley polytope of a
    nef partition).  For the Cayley polytope of the Newton polytopes Delta_i this is
    the Hodge diamond of the complete intersection itself; replacing P by its dual
    support (the Cayley polytope of the nabla_i) gives the mirror.
    **Arguments:**
    - `P (Polytope | ArrayLike)`: The support polytope (or its vertices, one per row).
    - `backend (Literal["auto", "faces", "normaliz", "latte", "cytools"])`: How the h*-polynomials of the faces are obtained.
        "faces" computes the h*-polynomial of every face on its own, without
        enumerating any dilation kP: from the Hilbert series of Normaliz
        (PyNormaliz), and for faces of very large volume from the Ehrhart
        polynomial of LattE if it is installed (`_hd_h_star_per_face`).
        "normaliz" (PyNormaliz) or "cytools" enumerate the dilations kP, whose
        number of lattice points grows quickly with the dimension;
        "latte" enumerates P only and computes the Ehrhart polynomials of all
        faces of dimension >= 3 one by one with LattE.
        "auto" picks "faces" if PyNormaliz is installed, and otherwise the
        first available of latte, cytools. Defaults to `"auto"`.
    **Raises:**
    - `ValueError`: Raised if P is not a reflexive Gorenstein support or the backend is unknown or not installed.
    - `RuntimeError`: Raised if the resulting Hodge numbers violate the Hodge symmetries or are negative.
    **Returns:**
    - `np.ndarray`: The integer Hodge diamond, h[p, q] = h^{p,q}.
    """
    V, W, index, backend=_hd_setup(P,backend,per_face=True)
    faces, duals, dims, leq=_hd_face_lattice(V,W)
    S=_hd_s_tilde(V,W,faces,duals,dims,leq,index,backend)
    S_dual=_hd_s_tilde(W,V,duals,faces,V.shape[1]-2-dims,leq.T,index,backend)
    cy_dim=V.shape[1]-2*index
    h=np.zeros((cy_dim+1,cy_dim+1),dtype=object)
    for d in np.unique(dims).tolist():
        # all faces of one dimension at once: M[i, j] = sum_F S~(F)_i S~(F*)_j, with exact (Python) integers
        on=np.flatnonzero(dims==d).tolist()
        M=np.array([S[f] for f in on],dtype=object).T@np.array([S_dual[f] for f in on],dtype=object)
        for i, j in zip(*np.nonzero(M)):
            p, q=d+1-i+j-index, i+j-index
            if 0<=p<=cy_dim and 0<=q<=cy_dim:
                h[p,q]+=M[i,j]
    if (h!=h.T).any() or (h!=h[::-1,::-1]).any() or (h<0).any():
        raise RuntimeError(f"inconsistent stringy Hodge numbers:\n{h}")
    return h.astype(int)

def hodge_diamond_cy4(P: Polytope|ArrayLike,backend: Literal["auto","normaliz","latte","cytools"]="auto")->np.ndarray:
    """
    **Description:**
    Fast Hodge diamond of a Calabi-Yau fourfold defined by a two-part nef
    partition, from the closed Batyrev--Borisov formulas for h^{1,1}, h^{2,1},
    h^{3,1} (as in the old, now commented-out ``h11_2_part_l_fast`` and
    ``h21_2_part_fast``) and h^{2,2} = 44 + 4 h^{1,1} - 2 h^{2,1} + 4 h^{3,1}.
    Same conventions as ``hodge_diamond_stringy``, which handles any dimension
    and any number of parts.
    **Arguments:**
    - `P (Polytope | ArrayLike)`: The Cayley polytope (or its vertices, one per row), e.g. F.Cayley_M().
    - `backend (Literal["auto", "faces", "normaliz", "latte", "cytools"])`: "normaliz" or "cytools" for enumerating P, 2P, P*, 2P*;
        "latte" also enumerates with CYTools here, and "faces" with normaliz (the per-face computation of
        `hodge_diamond_stringy` is not used by the closed formulas); "auto" prefers normaliz. Defaults to `"auto"`.
    **Raises:**
    - `ValueError`: Raised if P is not a reflexive Gorenstein support, is not the Cayley polytope of a two-part nef partition in six dimensions, or the backend is unknown or not installed.
    **Returns:**
    - `np.ndarray`: The 5 x 5 integer Hodge diamond, h[p, q] = h^{p,q}.
    """
    V, W, index, backend=_hd_setup(P,backend)
    if backend=="faces":
        backend="normaliz"
    if V.shape[1]!=8 or index!=2:
        raise ValueError("hodge_diamond_cy4 needs the Cayley polytope of a two-part nef partition in six dimensions")
    faces, duals, dims, leq=_hd_face_lattice(V,W)
    lstar, lstar_dual=_hd_interior_point_counts(V,W,duals,2,backend), _hd_interior_point_counts(W,V,faces,2,backend)

    def bb_h11(l: np.ndarray,ld: np.ndarray,dims: np.ndarray,leq: np.ndarray)->np.integer:
        """
        **Description:**
        Batyrev--Borisov formula for h^{1,1} (as in the old, now commented-out ``h11_2_part_l_fast``); with the roles of the faces and dual faces swapped it gives h^{3,1}.
        **Arguments:**
        - `l (np.ndarray)`: l[f, k] = l*(kF) for the faces F and k <= 2.
        - `ld (np.ndarray)`: ld[f, k] = l*(kF*) for the dual faces F*.
        - `dims (np.ndarray)`: The dimensions of the faces F.
        - `leq (np.ndarray)`: Boolean face order, leq[g, f] = (G <= F).
        **Returns:**
        - `np.integer`: The Hodge number.
        """
        on=lambda d: dims==d
        boundary=l[on(2),1]@leq[on(2)]
        return (ld[:,1].sum()-8-ld[on(0),2].sum()+(ld[on(1),1]+l[on(1),1]*ld[on(1),2]).sum()-(ld[on(2),1]*l[on(2),2]).sum()+(ld[on(3),1]*(l[on(3),2]-boundary[on(3)])).sum())

    def bb_h21(l: np.ndarray,ld: np.ndarray,dims: np.ndarray,leq: np.ndarray)->np.integer:
        """
        **Description:**
        Batyrev--Borisov formula for h^{2,1} (as in the old, now commented-out ``h21_2_part_fast``).
        **Arguments:**
        - `l (np.ndarray)`: l[f, k] = l*(kF) for the faces F and k <= 2.
        - `ld (np.ndarray)`: ld[f, k] = l*(kF*) for the dual faces F*.
        - `dims (np.ndarray)`: The dimensions of the faces F.
        - `leq (np.ndarray)`: Boolean face order, leq[g, f] = (G <= F).
        **Returns:**
        - `np.integer`: The Hodge number.
        """
        on=lambda d: dims==d
        boundary=lambda d: l[on(d),1]@leq[on(d)]
        return ((l[on(2),1]*ld[on(2),2]).sum()-(ld[on(3),1]*boundary(2)[on(3)]).sum()+(ld[on(4),1]*(l[on(4),2]-boundary(3)[on(4)])).sum())
    h11, h31=bb_h11(lstar,lstar_dual,dims,leq), bb_h11(lstar_dual,lstar,6-dims,leq.T)
    h21=bb_h21(lstar_dual,lstar,6-dims,leq.T)
    return np.array([[1,0,0,0,1],[0,h11,h21,h31,0],[0,h21,44+4*h11-2*h21+4*h31,h21,0],[0,h31,h21,h11,0],[1,0,0,0,1]],dtype=int)

def _hd_setup(P: Polytope|ArrayLike,backend: Literal["auto","faces","normaliz","latte","cytools"],per_face: bool=False)->tuple[np.ndarray,np.ndarray,int,Literal["faces","normaliz","latte","cytools"]]:
    """
    **Description:**
    Vertices V of P, vertices W of the dual support P* (the primitive rays of the
    dual cone), the Gorenstein index <m, n>, and the backend to use.  m and n are
    the integral degree vectors with <v, m> = 1 = <w, n>; they exist iff P is a
    reflexive Gorenstein support.
    **Arguments:**
    - `P (Polytope | ArrayLike)`: The support polytope (or its vertices, one per row).
    - `backend (Literal["auto", "faces", "normaliz", "latte", "cytools"])`: The requested backend; "auto" picks the first available of normaliz, latte, cytools (with a warning if PyNormaliz is missing). "faces" needs PyNormaliz.
    - `per_face (bool)`: Whether "auto" resolves to "faces" instead of "normaliz" if PyNormaliz is installed (for `hodge_diamond_stringy`). Defaults to `False`.
    **Raises:**
    - `ValueError`: Raised if P is not a reflexive Gorenstein support, or if the backend is unknown or not installed.
    **Returns:**
    - `tuple[np.ndarray, np.ndarray, int, Literal["faces", "normaliz", "latte", "cytools"]]`: `(V, W, index, backend)`: the integer vertex matrices of P and P*, the Gorenstein index and the resolved backend.
    """
    V=np.asarray(P.vertices() if isinstance(P,Polytope) else P,dtype=int)
    W=np.asarray(Cone(V).dual().extremal_rays(),dtype=int)
    W//=np.gcd.reduce(W,axis=1)[:,None]
    m, n=(np.rint(np.linalg.lstsq(X,np.ones(len(X)),rcond=None)[0]).astype(int) for X in (V,W))
    if np.linalg.matrix_rank(V)!=V.shape[1] or (V@m!=1).any() or (W@n!=1).any():
        raise ValueError("P is not a reflexive Gorenstein support (e.g. the Cayley polytope of a nef partition)")
    available={"normaliz": NormalizCone is not None,"latte": shutil.which("count") is not None,"cytools": True,"faces": NormalizCone is not None}
    if backend=="auto":
        backend=next(name for name,ok in available.items() if ok)
        if backend=="normaliz" and per_face:
            backend="faces"
        if backend not in ("normaliz","faces"):
            warnings.warn(f"PyNormaliz is not installed, so lattice points are counted with {backend}, which may be much slower. " "Install it with `pip install pynormaliz`.")
    elif not available.get(backend,False):
        raise ValueError(f"backend {backend!r} is unknown or not installed; choose from {[name for name,ok in available.items() if ok]}"+(" (install PyNormaliz with `pip install pynormaliz`)" if backend in ("normaliz","faces") else ""))
    return V, W, int(m@n), backend

def _hd_face_lattice(V: np.ndarray,W: np.ndarray)->tuple[list[frozenset[int]],list[frozenset[int]],np.ndarray,np.ndarray]:
    """
    **Description:**
    All faces F of P = conv(V), including the empty face and P itself, sorted by
    dimension: their vertex sets (rows of V), the vertex sets of the dual faces F*
    (rows of W), their dimensions, and leq[g, f] = (G <= F).  Faces are the
    intersections of facets, and the facets are dual to the vertices of P*.
    **Arguments:**
    - `V (np.ndarray)`: The vertices of P, one per row.
    - `W (np.ndarray)`: The vertices of the dual support P*, one per row.
    **Returns:**
    - `tuple[list[frozenset[int]], list[frozenset[int]], np.ndarray, np.ndarray]`: `(faces, duals, dims, leq)`: the row-index sets of the faces and of their dual faces (integer indices, partly NumPy integers), the integer array of face dimensions (-1 for the empty face) and the boolean face-order matrix.
    """
    zero=V@W.T==0
    facets={frozenset(np.flatnonzero(column)) for column in zero.T}
    faces, new=set(), {frozenset(range(len(V)))}
    while new:
        faces|=new
        new={F&facet for F in new for facet in facets}-faces
    dim={F: np.linalg.matrix_rank(V[list(F)])-1 for F in faces}
    faces=sorted(faces,key=dim.get)
    dims=np.array([dim[F] for F in faces])
    contains=np.zeros((len(faces),len(V)),dtype=int)
    for f, F in enumerate(faces):
        contains[f,list(F)]=1
    leq=contains@contains.T==contains.sum(axis=1)[:,None]
    return faces, [frozenset(np.flatnonzero(zero[list(F)].all(axis=0))) for F in faces], dims, leq

def _hd_interior_point_counts(X: np.ndarray,Y: np.ndarray,duals: Sequence[frozenset[int]],K: int,backend: Literal["normaliz","latte","cytools"])->np.ndarray:
    """
    **Description:**
    lstar[f, k] = l*(kF) for every face F of conv(X) and k <= K, where duals[f] is
    the set of rows of Y vanishing on F.  A lattice point x of kP lies in the
    relative interior of the face whose dual is {y : <x, y> = 0}, so one
    enumeration of kP counts all faces at once.
    **Arguments:**
    - `X (np.ndarray)`: The vertices of P = conv(X), one per row.
    - `Y (np.ndarray)`: The vertices of the dual support, one per row.
    - `duals (Sequence[frozenset[int]])`: For every face, the set of row indices of Y vanishing on it (as returned by `_hd_face_lattice`).
    - `K (int)`: The largest dilation factor.
    - `backend (Literal["normaliz", "latte", "cytools"])`: "normaliz" enumerates with PyNormaliz, any other value with CYTools.
    **Returns:**
    - `np.ndarray`: Integer array of shape (len(duals), K + 1) with lstar[f, k] = l*(kF) (column 0 is zero).
    """
    face_of_dual={dual: f for f,dual in enumerate(duals)}
    lstar=np.zeros((len(duals),K+1),dtype=int)
    for k in range(1,K+1):
        if backend=="normaliz":
            points=np.asarray(NormalizCone(polytope=(k*X).tolist()).LatticePoints(),dtype=int)[:,:-1]
        else:
            points=Polytope(k*X).points()
        zero_sets, counts=np.unique(points@Y.T==0,axis=0,return_counts=True)
        for zero_set, count in zip(zero_sets,counts):
            lstar[face_of_dual[frozenset(np.flatnonzero(zero_set))],k]=count
    return lstar

def _hd_s_tilde(X: np.ndarray,Y: np.ndarray,faces: Sequence[frozenset[int]],duals: Sequence[frozenset[int]],dims: np.ndarray,leq: np.ndarray,index: int,backend: Literal["faces","normaliz","latte","cytools"])->list[list[int]]:
    """
    **Description:**
    S~(F, t) = sum_{G <= F} (-1)^(dim F - dim G) h*_G(t) g([G, F], t) for every face F
    of conv(X) (dual support conv(Y); faces, duals, dims, leq as in _hd_face_lattice).
    With the backend "faces" the h*-polynomials come from `_hd_h_star_per_face`, without
    enumerating dilations. Otherwise h*_F of a face of dimension d follows from
    L_F(k) = #(kF cap Z) and l*(kF) = #(relint(kF) cap Z) for k <= ceil(d/2) through
        sum_k L_F(k) t^k = h*_F(t) / (1-t)^(d+1),
        sum_k l*(kF) t^k = t^(d+1) h*_F(1/t) / (1-t)^(d+1),
    using l*(kP) = L_P(k - index) for P itself (Gorenstein).  The sum over G with
    the g-polynomials of the intervals [G, F] is done by `_hd_s_tilde_from_h_star`.
    **Arguments:**
    - `X (np.ndarray)`: The vertices of P = conv(X), one per row.
    - `Y (np.ndarray)`: The vertices of the dual support, one per row.
    - `faces (Sequence[frozenset[int]])`: The row-index sets of the faces of P, sorted by dimension.
    - `duals (Sequence[frozenset[int]])`: The row-index sets (in Y) of the dual faces.
    - `dims (np.ndarray)`: The face dimensions.
    - `leq (np.ndarray)`: Boolean face order, leq[g, f] = (G <= F).
    - `index (int)`: The Gorenstein index.
    - `backend (Literal["faces", "normaliz", "latte", "cytools"])`: "faces" computes the h*-polynomial of every face on its own; "normaliz" or "cytools" enumerate dilations; "latte" computes Ehrhart polynomials of large faces with LattE's `count`.
    **Returns:**
    - `list[list[int]]`: For every face F, the integer coefficients of S~(F, t) (Python integers), constant term first.
    """
    if backend=="faces":
        return _hd_s_tilde_from_h_star(_hd_h_star_per_face(X,faces,dims),dims,leq)
    D=dims.max()
    needed=[-(-(D-index if d==D else d)//2) for d in dims]
    K=1 if backend=="latte" else max(needed)
    lstar=_hd_interior_point_counts(X,Y,duals,K,backend)
    L=leq.T.astype(int)@lstar
    L[:,0]=1
    one_minus_t_power=lambda series, d: _hd_mul(series,[(-1)**i*comb(d+1,i) for i in range(d+2)],len(series))
    h_star=[]
    for f, (F,d) in enumerate(zip(faces,dims)):
        if d<0:
            h_star.append([1])
        elif needed[f]>K:
            local=lll_reduce(X[list(F)]-X[min(F)])[:,-d:]
            with tempfile.TemporaryDirectory() as tmp:
                Path(tmp,"face.vrep").write_text(f"{len(local)} {d+1}\n"+"".join("1 "+" ".join(map(str,v))+"\n" for v in local))
                out=subprocess.run(["count","--ehrhart-polynomial","--vrep","face.vrep"],cwd=tmp,capture_output=True,text=True,check=True).stdout
            terms=re.findall(r"([+-])\s*(\d+(?:/\d+)?)\s*\*\s*t\^(\d+)",out.strip().splitlines()[-1])
            h_star.append([int(c) for c in one_minus_t_power([sum(Fraction(s+c)*k**int(e) for s,c,e in terms) for k in range(d+1)],d)])
        else:
            interior=[0]*index+L[f].tolist() if d==D else lstar[f].tolist()
            h=(one_minus_t_power(L[f].tolist(),d)+[0]*d)[: d+1]
            for i, c in enumerate(one_minus_t_power(interior,d)[1 : d+2],1):
                h[d+1-i]=c
            h_star.append(h)
    return _hd_s_tilde_from_h_star(h_star,dims,leq)

def _hd_g_polynomials(dims: np.ndarray,leq: np.ndarray,upper: bool=True)->tuple[np.ndarray,Callable[[int,int],tuple[int,...]]]:
    """
    **Description:**
    The g-polynomials of the intervals [G, F] of a face lattice, in one of two forms: with `upper` the recursion over
    the faces x with G < x <= F,
        g(G, F) = (1 - t) sum_x (t - 1)^(dim x - dim G - 1) g(x, F), truncated to degree < (rank + 1)//2,
    which is the one in S~ and in the stringy E-function (as in the old ``FaceLattice.g_polynomial``); without `upper`
    the same recursion in the opposite lattice, over the faces x with G <= x < F and with g(G, x), which is the
    g-polynomial of the interval itself in the E-polynomial of a torus hypersurface. In both, g = 1 for
    rank = dim F - dim G <= 2. An interval of rank rho with exactly rho atoms is the face lattice of a simplex (a
    polytope of dimension rho - 1 with rho vertices), i.e. a Boolean lattice, and then g = 1 in both forms as well. So
    only the remaining ("hard") intervals need the recursion, and for all others the sums over G weighted by g are
    plain sums.
    **Arguments:**
    - `dims (np.ndarray)`: The face dimensions (-1 for the empty face).
    - `leq (np.ndarray)`: Boolean face order, leq[g, f] = (G <= F).
    - `upper (bool)`: Which of the two recursions is used. Defaults to `True`.
    **Returns:**
    - `tuple[np.ndarray, Callable]`: `(hard, g)`: hard[lo, hi] = (lo <= hi and the interval is not known to have
      g = 1), and g(lo, hi), the coefficients of the g-polynomial (constant term first, up to degree (rank - 1)//2,
      memoized); g(lo, hi) = (1,) wherever hard[lo, hi] is False.
    """
    dims=np.asarray(dims,dtype=np.int64)
    rank=dims[None,:]-dims[:,None]
    atoms=np.rint((leq&(rank==1)).astype(np.float32)@leq.astype(np.float32)).astype(np.int64)
    hard=leq&(rank>=3)&(atoms!=rank)
    t_minus_one=[[(-1)**(e-i)*comb(e,i) for i in range(e+1)] for e in range(int(dims.max()-dims.min())+2)]
    memo: dict[tuple[int,int],tuple[int,...]]={}

    def g(lo: int,hi: int)->tuple[int,...]:
        """
        **Description:**
        Truncated g-polynomial of the interval [lo, hi] of the face lattice (memoized).
        **Arguments:**
        - `lo (int)`: Index of the lower face.
        - `hi (int)`: Index of the upper face.
        **Returns:**
        - `tuple[int, ...]`: Coefficients of g, constant term first, up to degree (rank - 1)//2.
        """
        if not hard[lo,hi]:
            return (1,)
        if (lo,hi) not in memo:
            n=int(dims[hi]-dims[lo]+1)//2
            h=[0]*n
            between=np.flatnonzero(leq[lo]&leq[:,hi])
            between=between[between!=(lo if upper else hi)]
            offset=dims[between]-dims[lo]-1 if upper else dims[hi]-dims[between]-1
            easy=~(hard[between,hi] if upper else hard[lo,between])
            # all faces x whose sub-interval has g = 1, of one dimension at once
            for e, count in enumerate(np.bincount(offset[easy]).tolist()):
                if count:
                    for i, c in enumerate(t_minus_one[e][:n]):
                        h[i]+=count*c
            for x, e in zip(between[~easy].tolist(),offset[~easy].tolist()):
                h=[a+b for a,b in zip(h,_hd_mul(t_minus_one[e],g(x,hi) if upper else g(lo,x),n))]
            memo[(lo,hi)]=tuple(_hd_mul([1,-1],h,n))
        return memo[(lo,hi)]
    return hard, g

def _hd_s_tilde_from_h_star(h_star: Sequence[Sequence[int]],dims: np.ndarray,leq: np.ndarray)->list[list[int]]:
    """
    **Description:**
    S~(F, t) = sum_{G <= F} (-1)^(dim F - dim G) h*_G(t) g([G, F], t) for every face F, from
    the h*-polynomials of the faces, with the g-polynomials of `_hd_g_polynomials` (the
    recursion over the faces x with G < x <= F).  For all intervals with g = 1 the sum over G
    is a single matrix product; only the remaining intervals contribute the corrections
    h*_G(t) (g([G, F], t) - 1).
    **Arguments:**
    - `h_star (Sequence[Sequence[int]])`: For every face, the coefficients of its h*-polynomial, constant term first (at most dim + 1 of them; [1] for the empty face).
    - `dims (np.ndarray)`: The face dimensions (-1 for the empty face).
    - `leq (np.ndarray)`: Boolean face order, leq[g, f] = (G <= F).
    **Returns:**
    - `list[list[int]]`: For every face F, the dim F + 2 integer coefficients of S~(F, t) (Python integers), constant term first.
    """
    dims=np.asarray(dims,dtype=np.int64)
    n_faces=len(dims)
    width=int(dims.max())+2
    largest=max((abs(int(c)) for h in h_star for c in h),default=1)
    # exact arithmetic: 64-bit integers if the sums cannot overflow, Python integers otherwise
    dtype=np.int64 if largest*n_faces<2**40 else object
    H=np.zeros((n_faces,width),dtype=dtype)
    for f, h in enumerate(h_star):
        H[f,:len(h)]=[int(c) for c in h]
    sign=np.where((dims[None,:]-dims[:,None])%2==0,1,-1)*leq
    S=sign.T.astype(dtype)@H
    hard, g=_hd_g_polynomials(dims,leq)
    for lo, hi in zip(*(index.tolist() for index in np.nonzero(hard))):
        g_poly=g(lo,hi)
        if any(g_poly[1:]):
            correction=_hd_mul([int(c) for c in H[lo]],(0,)+g_poly[1:],int(dims[hi])+2)
            for i, c in enumerate(correction):
                S[hi,i]+=int(sign[lo,hi])*c
    return [[int(c) for c in S[f,:int(dims[f])+2]] for f in range(n_faces)]

def _hd_h_star_per_face(X: np.ndarray,faces: Sequence[frozenset[int]],dims: np.ndarray)->list[list[int]]:
    """
    **Description:**
    The h*-polynomial of every face F of P = conv(X), computed face by face and without
    enumerating the lattice points of any dilation kP (the backend "faces" of
    `hodge_diamond_stringy`).  For a face of dimension d, taken in its own saturated lattice:
    - d <= 0, a segment, or normalized volume 1 (a unimodular simplex): h* = 1, or 1 + (Vol - 1) t for a segment;
    - otherwise the numerator of the Hilbert series of the cone over F from Normaliz, whose cost grows with
      the normalized volume of F;
    - for faces of normalized volume above `_HD_FAT_FACE_VOLUME`, if LattE is installed: from the Ehrhart
      polynomial L_F of LattE's `count` (Barvinok's algorithm, whose cost does not depend on the volume),
      through sum_k L_F(k) t^k = h*_F(t) / (1-t)^(d+1).  These faces are computed in parallel.
    Every result is checked against h*_F(1) = Vol(F).
    **Arguments:**
    - `X (np.ndarray)`: The vertices of P, one per row; they lie on a hyperplane <x, m> = 1 with m integral (Gorenstein support).
    - `faces (Sequence[frozenset[int]])`: The row-index sets of the faces (the empty face included).
    - `dims (np.ndarray)`: The face dimensions (-1 for the empty face).
    **Raises:**
    - `ValueError`: Raised if PyNormaliz is not installed or the vertices do not lie at height 1.
    - `RuntimeError`: Raised if the h*-polynomial of a face cannot be determined consistently.
    **Returns:**
    - `list[list[int]]`: For every face, the coefficients of its h*-polynomial, constant term first.
    """
    if NormalizCone is None:
        raise ValueError("The backend \"faces\" needs PyNormaliz (`pip install pynormaliz`).")
    X=np.asarray(X,dtype=int)
    grading=np.rint(np.linalg.lstsq(X.astype(float),np.ones(len(X)),rcond=None)[0]).astype(int)
    if (X@grading!=1).any():
        raise ValueError("The vertices do not lie on a hyperplane at integral height 1.")
    grading=[grading.tolist()]
    have_latte=shutil.which("count") is not None

    def hilbert_numerator(cone: Any,d: int,volume: int)->list[int]|None:
        """h* of a face from the Hilbert series of its cone, or `None` if the series is not of the form h*(t)/(1-t)^(d+1)."""
        numerator, denominator, shift=cone.HilbertSeries()
        if shift!=0 or list(denominator)!=[1]*(d+1) or len(numerator)>d+1 or sum(numerator)!=volume:
            return None
        return [int(c) for c in numerator]+[0]*(d+1-len(numerator))

    def latte_h_star(rows: np.ndarray,d: int,volume: int)->list[int]|None:
        """h* of a face from LattE's Ehrhart polynomial, or `None` if LattE fails or the result is inconsistent."""
        local=np.asarray(lll_reduce(rows-rows[0]),dtype=int)[:,-d:]
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,"face.vrep").write_text(f"{len(local)} {d+1}\n"+"".join("1 "+" ".join(map(str,v))+"\n" for v in local.tolist()))
            # signed decomposition in the primal space is much faster on these faces; LattE's default as a fallback
            for options in (("--irrational-all-primal","--maxdet=100","--exponential"),()):
                out=subprocess.run(["count","--ehrhart-polynomial",*options,"--vrep","face.vrep"],cwd=tmp,capture_output=True,text=True)
                lines=out.stdout.strip().splitlines()
                if out.returncode!=0 or not lines:
                    continue
                terms=re.findall(r"([+-])\s*(\d+(?:/\d+)?)\s*\*\s*t\^(\d+)",lines[-1])
                points=[sum(Fraction(s+c)*k**int(e) for s,c,e in terms) for k in range(d+1)]
                h=[sum((-1)**i*comb(d+1,i)*points[k-i] for i in range(k+1)) for k in range(d+1)]
                if terms and all(c.denominator==1 and c>=0 for c in h) and sum(h)==volume:
                    return [int(c) for c in h]
        return None
    h_star: list[list[int]|None]=[None]*len(faces)
    fat, cones=[], {}
    threads=NmzSetNumberOfNormalizThreads(1) if NmzSetNumberOfNormalizThreads is not None else None
    try:
        # the faces are small problems, for which starting Normaliz' threads costs more than it gains
        for f, (F,d) in enumerate(zip(faces,np.asarray(dims).tolist())):
            if d<=0:
                h_star[f]=[1]
                continue
            cone=NormalizCone(cone=X[sorted(F)].tolist(),grading=grading)
            volume=cone.Multiplicity()
            if isinstance(volume,(list,tuple)):
                if len(volume)!=2 or volume[1]!=1:
                    raise RuntimeError(f"Normaliz returned the non-integral volume {volume} for a lattice polytope.")
                volume=volume[0]
            volume=int(volume)
            if volume==1:
                h_star[f]=[1]+[0]*d
            elif d==1:
                h_star[f]=[1,volume-1]
            elif volume<=_HD_FAT_FACE_VOLUME:
                h_star[f]=hilbert_numerator(cone,d,volume)
                if h_star[f] is None:
                    fat.append((f,d,volume))
            else:
                fat.append((f,d,volume))
                cones[f]=cone
    finally:
        if threads is not None:
            NmzSetNumberOfNormalizThreads(threads)
    if fat and have_latte:
        with ThreadPoolExecutor(max_workers=min(8,os.cpu_count() or 1)) as pool:
            for (f,d,volume), h in zip(fat,pool.map(lambda job: latte_h_star(X[sorted(faces[job[0]])],job[1],job[2]),fat)):
                h_star[f]=h
    for f, d, volume in fat:
        if h_star[f] is None:
            # LattE is missing or failed: Normaliz with all its threads, however large the volume
            cone=cones[f] if f in cones else NormalizCone(cone=X[sorted(faces[f])].tolist(),grading=grading)
            h_star[f]=hilbert_numerator(cone,d,volume)
        if h_star[f] is None:
            raise RuntimeError(f"The h*-polynomial of the face with vertices {sorted(faces[f])} could not be determined.")
    return h_star

def _hd_mul(a: Sequence[int|Fraction],b: Sequence[int|Fraction],n: int)->list[int|Fraction]:
    """
    **Description:**
    First n coefficients of the product of the polynomials a and b.
    **Arguments:**
    - `a (Sequence[int | Fraction])`: Coefficients of the first polynomial, constant term first.
    - `b (Sequence[int | Fraction])`: Coefficients of the second polynomial, constant term first.
    - `n (int)`: Number of coefficients to keep.
    **Returns:**
    - `list[int | Fraction]`: The first n coefficients of a*b (integers for integer input).
    """
    c=[0]*n
    for i, x in enumerate(a[:n]):
        for j, y in enumerate(b[: n-i]):
            c[i+j]+=x*y
    return c

# -----------------------------------------------------------------------------
# E-polynomials and Hodge numbers of generic hypersurfaces and double covers
# (from `Hodge_Number_computations`)
# -----------------------------------------------------------------------------

def _uv_minus_one_power(k: int,size: int)->np.ndarray:
    """
    **Description:**
    The polynomial (uv-1)^k as a coefficient array, A[p,q] = coefficient of u^p v^q.
    **Arguments:**
    - `k (int)`: The exponent.
    - `size (int)`: The array size, at least k+1.
    **Returns:**
    - `np.ndarray`: (size,size) object array.
    """
    A=np.zeros((size,size),dtype=object)
    for i in range(k+1):
        A[i,i]=(-1)**(k-i)*comb(k,i)
    return A

def _poly_mul(A: np.ndarray,B: np.ndarray)->np.ndarray:
    """
    **Description:**
    Product of two polynomials in u,v given as square coefficient arrays of the same shape, truncated to that shape.
    **Arguments:**
    - `A (np.ndarray)`: The first factor.
    - `B (np.ndarray)`: The second factor.
    **Returns:**
    - `np.ndarray`: The truncated product.
    """
    m=A.shape[0]
    C=np.zeros_like(A)
    for p,q in zip(*np.nonzero(A)):
        for s,t in zip(*np.nonzero(B)):
            if p+s<m and q+t<m:
                C[p+s,q+t]+=A[p,q]*B[s,t]
    return C

def _polytope_face_data(W: np.ndarray,backend: str)->tuple[list[frozenset[int]],np.ndarray,np.ndarray,list[list[int]],list[list[int]],tuple[np.ndarray,Callable[[int,int],tuple[int,...]]]]:
    """
    **Description:**
    Combinatorial and Ehrhart data of the lattice polytope P=conv(W), taken in its own saturated affine lattice: the faces
    (as sets of rows of W, sorted by dimension, including the empty face and P), their dimensions, the face order, the
    h*-polynomials of all faces, the local h*-polynomials
        S~(F,t) = sum_{G<=F} (-1)^(dim F-dim G) h*_G(t) g([G,F]^*,t)
    (dual-interval g-polynomials, `_hd_s_tilde_from_h_star`) and the g-polynomials g([G,F],t) of the intervals
    themselves (`_hd_g_polynomials` without `upper`). The face lattice comes from `_hd_face_lattice` applied to the
    cone over P x {1}. With the backend "faces" the h*-polynomial of every face is computed on its own, without
    enumerating any dilation of P (`_hd_h_star_per_face`: Normaliz, and LattE for faces of very large volume).
    Otherwise the counts l*(kF) come from `_hd_interior_point_counts`: one enumeration of kP for k <= ceil(dim P/2)
    gives l*(kF) for all faces F at once, which by Ehrhart reciprocity fixes every h*_F; the number of points of kP
    grows quickly with dim P.
    **Arguments:**
    - `W (np.ndarray)`: The vertices of P, one per row.
    - `backend (str)`: "faces" (per-face h*-polynomials), or "normaliz" or "cytools" (lattice-point enumeration).
    **Returns:**
    - `tuple`: `(faces, dims, leq, h_star, s_tilde, g_data)` with leq[i,j] = (face i <= face j), polynomials as
      coefficient lists (constant term first), and `g_data = (hard, g)` as returned by `_hd_g_polynomials`: `g(i,j)`
      is the truncated g-polynomial of the interval [face i, face j], and it is 1 wherever `hard[i,j]` is False.
    """
    r=int(np.linalg.matrix_rank(W[1:]-W[0])) if len(W)>1 else 0
    if r==0:
        leq=np.array([[True,True],[False,True]])
        return [frozenset(),frozenset({0})],np.array([-1,0]),leq,[[1],[1]],[[1],[0,0]],(np.zeros((2,2),dtype=bool),lambda lo,hi: (1,))
    span=_saturated_kernel(_saturated_kernel(W-W[0]).T)
    Q=np.rint(np.linalg.lstsq(span.astype(float),(W-W[0]).T.astype(float),rcond=None)[0].T).astype(int)
    X=np.column_stack([Q,np.ones(len(Q),dtype=int)])
    Y=np.asarray(Cone(X).dual().extremal_rays(),dtype=int)
    Y//=np.gcd.reduce(Y,axis=1)[:,None]
    faces,duals,dims,leq=_hd_face_lattice(X,Y)
    if backend=="faces":
        h_star=_hd_h_star_per_face(X,faces,dims)
    else:
        K=ceil(r/2)
        lstar=_hd_interior_point_counts(X,Y,duals,K,backend)
        L=leq.T.astype(int)@lstar
        L[:,0]=1
        one_minus_t_power=lambda series,d: _hd_mul(series,[(-1)**i*comb(d+1,i) for i in range(d+2)],len(series))
        h_star=[]
        for f,d in enumerate(dims.tolist()):
            if d<=0:
                h_star.append([1])
                continue
            h=(one_minus_t_power(L[f].tolist(),d)+[0]*d)[:d+1]
            for i,c in enumerate(one_minus_t_power(lstar[f].tolist(),d)[1:d+2],1):
                h[d+1-i]=c
            h_star.append(h)
    return faces,dims,leq,h_star,_hd_s_tilde_from_h_star(h_star,dims,leq),_hd_g_polynomials(dims,leq,upper=False)

def _torus_hypersurface_e_polynomials(dims: np.ndarray,leq: np.ndarray,s_tilde: list[list[int]],g_data: tuple[np.ndarray,Callable[[int,int],tuple[int,...]]])->list[np.ndarray]:
    """
    **Description:**
    E-polynomials of nondegenerate hypersurfaces Z_F in the torus (C*)^dim F with Newton polytope F, for every face F of
    a lattice polytope (Danilov-Khovanskii, in the closed form of Batyrev-Borisov):
        uv E(Z_F;u,v) = (uv-1)^dim F + (-1)^(dim F+1) sum_{G<=F} u^(dim G+1) S~(G,v/u) g([G,F],uv),
    with g([G,F],t) the g-polynomial of the interval [G,F] itself (not of its dual, unlike in S~). The terms with g = 1
    (all intervals that are not `hard`) are summed for all faces F at once, one matrix product per dimension of G; the
    remaining intervals add the terms of g - 1.
    **Arguments:**
    - `dims (np.ndarray)`, `leq (np.ndarray)`, `s_tilde (list[list[int]])`, `g_data (tuple)`: As returned by
      `_polytope_face_data`.
    **Raises:**
    - `RuntimeError`: Raised if the right-hand side is not divisible by uv (inconsistent input).
    **Returns:**
    - `list[np.ndarray]`: For every face, E[p,q] = e^{p,q}(Z_F) as a (D+1,D+1) object array with D = dim P.
    """
    hard,g=g_data
    dims=np.asarray(dims,dtype=np.int64)
    D=int(dims.max())
    n_faces=len(dims)
    largest=max((abs(int(c)) for s in s_tilde for c in s),default=1)
    # exact arithmetic: 64-bit integers if the sums cannot overflow, Python integers otherwise
    dtype=np.int64 if largest*n_faces<2**40 else object
    A=np.zeros((n_faces,D+2,D+2),dtype=dtype)
    for d in np.unique(dims).tolist():
        on=np.flatnonzero(dims==d)
        below=leq[on].T.astype(dtype)@np.array([[int(c) for c in s_tilde[x]] for x in on.tolist()],dtype=dtype).reshape(len(on),d+2)
        for i in range(d+2):
            A[:,d+1-i,i]+=below[:,i]
    for x,f in zip(*(index.tolist() for index in np.nonzero(hard))):
        d=int(dims[x])
        for j,c in enumerate(g(x,f)[1:],1):
            if c:
                for i,s in enumerate(s_tilde[x]):
                    A[f,d+1-i+j,i+j]+=s*c
    out=[]
    for f,r in enumerate(dims.tolist()):
        if r<0:
            out.append(np.zeros((D+1,D+1),dtype=object))
            continue
        E=(-1)**(r+1)*A[f].astype(object)+_uv_minus_one_power(r,D+2)
        if E[0].any() or E[:,0].any():
            raise RuntimeError(f"E-polynomial of face {f} is not divisible by uv:\n{E}")
        out.append(E[1:,1:])
    return out

def _resolve_backend(backend: str)->str:
    """
    **Description:**
    Replaces the backend "auto" by "faces" if PyNormaliz is installed and by "cytools" otherwise. "faces" computes the
    h*-polynomial of every face of a Newton or Cayley polytope on its own, without enumerating dilations
    (`_hd_h_star_per_face`); "normaliz" and "cytools" enumerate the lattice points of the dilations.
    **Arguments:**
    - `backend (str)`: "faces", "normaliz", "cytools" or "auto".
    **Raises:**
    - `ValueError`: Raised if the backend is unknown, or needs PyNormaliz and it is not installed.
    **Returns:**
    - `str`: "faces", "normaliz" or "cytools".
    """
    if backend=="auto":
        return "faces" if NormalizCone is not None else "cytools"
    if backend not in ("faces","normaliz","cytools"):
        raise ValueError(f"backend {backend!r} is unknown; choose from 'auto', 'faces', 'normaliz', 'cytools'.")
    if backend!="cytools" and NormalizCone is None:
        raise ValueError(f"backend {backend!r} needs PyNormaliz (`pip install pynormaliz`).")
    return backend

def _newton_polytope_incidence(fan: Fan,divisor: ArrayLike)->tuple[int,np.ndarray,np.ndarray]:
    """
    **Description:**
    Vertices of the Newton polytope (integer hull) of a toric divisor and their incidences with the rays, which fix the
    face F_s of monomials surviving on every torus orbit O(s).
    **Arguments:**
    - `fan (Fan)`: A fan; its rays are `fan.vectors()`.
    - `divisor (ArrayLike)`: Integer coefficients of the divisor in the basis of the prime toric divisors.
    **Raises:**
    - `ValueError`: Raised if the divisor has no global sections.
    **Returns:**
    - `tuple[int,np.ndarray,np.ndarray]`: `(n, W, on_face)`: the dimension n of the fan, the vertices W (one per row) and
      on_face[i,j] = (<W_j,v_i> = -divisor[i]).
    """
    V=np.asarray(fan.vectors(),dtype=int)
    a=np.asarray(divisor,dtype=int)
    n=V.shape[1]
    NP=Newton_Polytope(V,a)
    if len(NP.points())==0:
        raise ValueError("The divisor has no global sections.")
    W=np.asarray(NP.vertices(),dtype=int).reshape(-1,n)
    return n,W,(V@W.T+a[:,None])==0

def _face_e_polynomials(W: np.ndarray,backend: str)->dict[frozenset[int],tuple[int,np.ndarray]]:
    """
    **Description:**
    E-polynomials of the nondegenerate torus hypersurfaces Z_F for all faces F of conv(W) (see
    `_torus_hypersurface_e_polynomials`), keyed by the vertex sets of the faces (sets of rows of W). These keys do not
    depend on the lattice in which the vertices are written, which allows comparing two lattices.
    **Arguments:**
    - `W (np.ndarray)`: The vertices, one per row.
    - `backend (str)`: "faces", "normaliz" or "cytools".
    **Returns:**
    - `dict[frozenset[int],tuple[int,np.ndarray]]`: For every face F, (dim F, E(Z_F)).
    """
    faces,dims,leq,_,s_tilde,g_data=_polytope_face_data(W,backend)
    E=_torus_hypersurface_e_polynomials(dims,leq,s_tilde,g_data)
    return {F:(int(d),e) for F,d,e in zip(faces,dims.tolist(),E)}

def _strata_summer(n: int,on_face: np.ndarray,face_e: dict[frozenset[int],tuple[int,np.ndarray]],orbit_term: bool)->Callable[[np.ndarray],np.ndarray]:
    """
    **Description:**
    The function that sums the E-polynomials of the strata Z cap O(s) over sets of cones s of one dimension k' (orbits
    of dimension k = n-k'): F_s empty gives (uv-1)^k (only if `orbit_term`), a point gives 0 and dim F_s = r > 0 gives
    E(Z_F) (uv-1)^(k-r) with E(Z_F) from `face_e`.
    **Arguments:**
    - `n (int)`: The dimension of the fan.
    - `on_face (np.ndarray)`: Ray-vertex incidences, see `_newton_polytope_incidence`.
    - `face_e (dict)`: The per-face E-polynomials, see `_face_e_polynomials`.
    - `orbit_term (bool)`: Whether orbits contained in Z contribute (uv-1)^k (False for the odd part of a double cover,
      on which the deck transformation acts trivially since it is a translation of a torus).
    **Returns:**
    - `Callable[[np.ndarray],np.ndarray]`: strata_sum(cones), the (n+1,n+1) coefficient array of sum_s E(Z cap O(s))
      over the rows s of `cones` (zero-based, all of one dimension).
    """
    def strata_sum(cones: np.ndarray)->np.ndarray:
        """Sum of E(Z cap O(s)) over the rows s of `cones` (zero-based, all of one dimension)."""
        e=np.zeros((n+1,n+1),dtype=object)
        if len(cones)==0:
            return e
        k=n-cones.shape[1]
        for mask,count in zip(*np.unique(on_face[cones].all(axis=1),axis=0,return_counts=True)):
            S=frozenset(np.flatnonzero(mask).tolist())
            if not S:
                if orbit_term:
                    e+=int(count)*_uv_minus_one_power(k,n+1)
                continue
            r,E=face_e[S]
            if r>0:
                Ef=np.zeros((n+1,n+1),dtype=object)
                Ef[:E.shape[0],:E.shape[1]]=E
                e+=int(count)*_poly_mul(Ef,_uv_minus_one_power(k-r,n+1))
        return e
    return strata_sum

def _strata_e_function(fan: Fan,divisor: ArrayLike,backend: str)->tuple[int,list[np.ndarray],Callable[[np.ndarray],np.ndarray]]:
    """
    **Description:**
    Set-up of the orbit stratification of the zero locus Z of a generic section of a toric divisor (see
    `e_polynomial_generic_hypersurface`): the cones of the fan grouped by dimension and a function that sums
    E(Z cap O(s)) over any set of cones s of one dimension.
    **Arguments:**
    - `fan (Fan)`: A simplicial fan; its rays are `fan.vectors()`.
    - `divisor (ArrayLike)`: Integer coefficients of the divisor in the basis of the prime toric divisors.
    - `backend (str)`: "faces", "normaliz", "cytools" or "auto" ("faces" if PyNormaliz is installed, see `_resolve_backend`).
    **Raises:**
    - `ValueError`: Raised if the divisor has no global sections or the fan is not simplicial.
    **Returns:**
    - `tuple[int,list[np.ndarray],Callable[[np.ndarray],np.ndarray]]`: `(n, cones_by_dim, strata_sum)`: the dimension n
      of the fan, cones_by_dim[d] the (C_d,d) array of zero-based d-cones (d=0,...,n, the zero cone included), and
      strata_sum(cones) the (n+1,n+1) coefficient array of sum_s E(Z cap O(s)) over the rows s of `cones`.
    """
    backend=_resolve_backend(backend)
    n,W,on_face=_newton_polytope_incidence(fan,divisor)
    face_e=_face_e_polynomials(W,backend)
    return n,[np.zeros((1,0),dtype=int)]+_cones_by_dimension(fan),_strata_summer(n,on_face,face_e,True)

def _even_sublattice_basis(mu: ArrayLike,n: int)->np.ndarray:
    """
    **Description:**
    Basis of the index-2 sublattice N = {x in Z^n : <mu,x> even} of N' = Z^n, as the columns of a matrix A (for the first
    mu-odd coordinate i: 2e_i, e_j - e_i for the other mu-odd j, e_j for the mu-even j). For a point m of the dual
    lattice M' = Z^n, m @ A are its coordinates in the dual lattice M of N. M contains M' with index 2,
    M = M' + Z*mu/2, so in these coordinates the points of M' + mu/2 become the integer points not in M'.
    **Arguments:**
    - `mu (ArrayLike)`: A non-zero mod-2 character of N'.
    - `n (int)`: The rank of N'.
    **Raises:**
    - `ValueError`: Raised if mu has the wrong length or is zero mod 2.
    **Returns:**
    - `np.ndarray`: The (n,n) integer matrix A, det A = 2.
    """
    mu=np.asarray(mu,dtype=int).reshape(-1)%2
    if len(mu)!=n or not mu.any():
        raise ValueError(f"mu = {mu.tolist()} must be a non-zero mod-2 vector of length {n}.")
    odd=np.flatnonzero(mu)
    A=np.identity(n,dtype=int)
    A[odd[0],odd[1:]]=-1
    A[odd[0],odd[0]]=2
    return A

def _strata_e_functions_double_cover(fan: Fan,divisor: ArrayLike,mu: ArrayLike,backend: str)->tuple[int,list[np.ndarray],Callable[[np.ndarray],np.ndarray],Callable[[np.ndarray],np.ndarray]]:
    """
    **Description:**
    Orbit stratification of the double cover X -> Z of the zero locus Z of a generic section of a toric divisor (see
    `e_polynomials_generic_double_cover`). The Newton polytope P of X in M is the integer hull of the divisor polytope
    in M' (X is cut out by the pull-backs of the sections of Z, not by all sections upstairs). Every orbit of X is
    either mapped isomorphically to its image (pointwise fixed by the deck transformation) or an etale double cover of
    it, and in both cases X cap O(s) is a nondegenerate hypersurface with Newton polytope F_s in M(s), since
    nondegeneracy lifts along etale maps. Hence E_-(X cap O(s)) = E(X cap O(s)) - E(Z cap O(s)) is the per-face
    difference E_M(Z_F) - E_M'(Z_F) of the Batyrev-Borisov formula in the two lattices, i.e. the formula with every
    h*-polynomial replaced by the h*-polynomial of the odd lattice points M' + mu/2. The (uv-1)^k torus terms cancel.
    **Arguments:**
    - `fan (Fan)`: A simplicial fan in N'; its rays are `fan.vectors()`.
    - `divisor (ArrayLike)`: Integer coefficients of the divisor on the toric variety of N'.
    - `mu (ArrayLike)`: The mod-2 character of N' that defines N = {<mu,n> even}.
    - `backend (str)`: "faces", "normaliz", "cytools" or "auto" (see `_resolve_backend`).
    **Raises:**
    - `ValueError`: Raised if the divisor has no global sections, the fan is not simplicial or mu is invalid.
    **Returns:**
    - `tuple`: `(n, cones_by_dim, even_sum, odd_sum)` as in `_strata_e_function`, with even_sum summing E(Z cap O(s))
      and odd_sum summing E_-(X cap O(s)).
    """
    backend=_resolve_backend(backend)
    n,W,on_face=_newton_polytope_incidence(fan,divisor)
    even=_face_e_polynomials(W,backend)
    cover=_face_e_polynomials(W@_even_sublattice_basis(mu,n),backend)
    if cover.keys()!=even.keys():
        raise RuntimeError("The face lattices in the two lattices differ.")
    odd={F:(r,cover[F][1]-E) for F,(r,E) in even.items()}
    cones_by_dim=[np.zeros((1,0),dtype=int)]+_cones_by_dimension(fan)
    return n,cones_by_dim,_strata_summer(n,on_face,even,True),_strata_summer(n,on_face,odd,False)

def _excision_masks(cones_by_dim: list[np.ndarray],excise: Iterable[Iterable[int]]|None,closure: bool)->list[np.ndarray]:
    """
    **Description:**
    Marks the cones whose torus orbits are excised. For every excised cone t this is either every cone containing t
    (closure=True: the orbit closure V(t) is removed, which leaves the toric variety of the fan without the cones
    containing t) or only t itself (closure=False: only the orbit O(t) is removed).
    **Arguments:**
    - `cones_by_dim (list[np.ndarray])`: All cones grouped by dimension, zero cone first, zero-based (see
      `_strata_e_function`).
    - `excise (Iterable[Iterable[int]]|None)`: The excised cones as one-indexed ray labels (as in `fan.cones()`), or a
      single such cone. `None` excises nothing.
    - `closure (bool)`: Whether to excise orbit closures or only orbits.
    **Raises:**
    - `ValueError`: Raised if an excised cone is not a cone of the fan.
    **Returns:**
    - `list[np.ndarray]`: One boolean array per dimension, True for the excised cones.
    """
    masks=[np.zeros(len(c),dtype=bool) for c in cones_by_dim]
    if excise is None:
        return masks
    excise=list(excise)
    if excise and np.isscalar(excise[0]):
        excise=[excise]
    for t in excise:
        t=np.array(sorted(int(x)-1 for x in t),dtype=int)
        d=len(t)
        if d>=len(cones_by_dim) or not (cones_by_dim[d]==t).all(axis=1).any():
            raise ValueError(f"{tuple(int(x)+1 for x in t)} is not a cone of the fan.")
        for c,m in zip(cones_by_dim,masks):
            if closure:
                m|=np.isin(c,t).sum(axis=1)==d
            elif c.shape[1]==d:
                m|=(c==t).all(axis=1)
    return masks

def e_polynomial_generic_hypersurface(fan: Fan,divisor: ArrayLike,backend: str="auto",excise: Iterable[Iterable[int]]|None=None,excise_closure: bool=True)->np.ndarray:
    """
    **Description:**
    Hodge-Deligne polynomial E(Z;u,v) = sum_{p,q} e^{p,q} u^p v^q of the zero locus Z of a generic section of the toric
    divisor D = sum_i divisor[i] D_i on the toric variety of a simplicial fan, optionally with toric strata excised. Z is
    stratified by the torus orbits O(s); the monomials surviving on O(s) are the lattice points of the face F_s of the
    Newton polytope of D on which <m,v_i> = -divisor[i] for all rays v_i of s. With k = dim O(s):
        F_s empty      ->  O(s) lies in Z,                E = (uv-1)^k,
        F_s a point    ->  Z misses O(s),                 E = 0,
        dim F_s = r>0  ->  Z cap O(s) = Z_F x (C*)^(k-r),  E = E(Z_F) (uv-1)^(k-r),
    with Z_F a nondegenerate hypersurface in (C*)^r with Newton polytope F_s (`_torus_hypersurface_e_polynomials`).
    E is additive over the strata, so excising strata just drops them from the sum: E(Z minus V(t)) is the sum over the
    cones s not containing t. This is the E-polynomial version of the Euler characteristic of `Uplift_functions._orbit_pieces`
    (E(Z;1,1) = chi(Z)). No compactness or smoothness is needed for E itself.
    **Arguments:**
    - `fan (Fan)`: A simplicial fan; its rays are `fan.vectors()` (labels 1,...,len(fan.vectors())).
    - `divisor (ArrayLike)`: Integer coefficients of D in the basis of the prime toric divisors.
    - `backend (str)`: How the h*-polynomials of the faces are obtained, "faces" (face by face, without enumerating
          dilations), "normaliz", "cytools" or "auto" ("faces" if PyNormaliz is installed, see `_resolve_backend`). Defaults to
      `"auto"`.
    - `excise (Iterable[Iterable[int]]|None)`: Cones of the fan (one-indexed ray labels, as in `fan.cones()`), or a
      single cone, whose strata are removed from Z. Defaults to `None` (nothing excised).
    - `excise_closure (bool)`: Whether to remove the orbit closure V(t) of every excised cone t (i.e. pass to the open
      toric subvariety of the fan without the cones containing t) or only the orbit O(t). Defaults to `True`.
    **Raises:**
    - `ValueError`: Raised if D has no global sections, the fan is not simplicial, or an excised cone is not in the fan.
    **Returns:**
    - `np.ndarray`: e[p,q] for 0 <= p,q <= dim Z, as an object array of Python integers.
    """
    n,cones_by_dim,strata_sum=_strata_e_function(fan,divisor,backend)
    removed=_excision_masks(cones_by_dim,excise,excise_closure)
    e=sum((strata_sum(c[~m]) for c,m in zip(cones_by_dim,removed)),np.zeros((n+1,n+1),dtype=object))
    if e[n].any() or e[:,n].any():
        raise RuntimeError(f"The E-polynomial has terms of degree > dim Z:\n{e}")
    return e[:n,:n]

def _check_complete(fan: Fan)->None:
    """
    **Description:**
    Verifies that a simplicial fan is complete: all maximal cones are full-dimensional and every wall lies in exactly
    two maximal cones.
    **Arguments:**
    - `fan (Fan)`: A simplicial fan.
    **Raises:**
    - `ValueError`: Raised if the fan is not complete.
    **Returns:**
    - `None`
    """
    n=np.asarray(fan.vectors()).shape[1]
    maximal=[tuple(sorted(c)) for c in fan.cones()]
    walls=[c[:i]+c[i+1:] for c in maximal for i in range(len(c))]
    if any(len(c)!=n for c in maximal) or (np.unique(walls,axis=0,return_counts=True)[1]!=2).any():
        raise ValueError("The fan is not complete and simplicial, so Z is not compact.")

def _pure_hodge_numbers(e: np.ndarray,check: bool)->np.ndarray:
    """
    **Description:**
    Hodge diamond h^{p,q} = (-1)^(p+q) e^{p,q} of a compact variety with pure cohomology from its E-polynomial.
    **Arguments:**
    - `e (np.ndarray)`: The E-polynomial coefficients e[p,q].
    - `check (bool)`: Whether to verify non-negativity, h^{p,q} = h^{q,p} and h^{p,q} = h^{d-p,d-q} (necessary for
      purity, not sufficient).
    **Raises:**
    - `ValueError`: Raised (with `check`) if one of the tests fails.
    **Returns:**
    - `np.ndarray`: The Hodge diamond h[p,q].
    """
    d=e.shape[0]-1
    h=np.array([[(-1)**(p+q)*e[p,q] for q in range(d+1)] for p in range(d+1)],dtype=object)
    if check and ((h<0).any() or (h!=h.T).any() or (h!=h[::-1,::-1]).any()):
        raise ValueError(f"The E-polynomial is not that of a pure Hodge structure (Z singular or not compact):\n{e}")
    return h.astype(int)

def _point_excised_hodge_numbers(h: np.ndarray,n_points: int,compact_support: bool=False)->np.ndarray:
    """
    **Description:**
    Mixed Hodge numbers of Z' = Z minus n_points points, for Z compact and connected of dimension d >= 1 with pure
    cohomology (Hodge diamond h) and Z' smooth or an orbifold. The long exact sequence of the pair (Z, points) gives
    H^k_c(Z') = H^k(Z) for k >= 2, H^0_c(Z') = 0 and 0 -> Q^(n_points-1) -> H^1_c(Z') -> H^1(Z) -> 0 with new classes
    of type (0,0). Poincare duality H^k(Z') = H^(2d-k)_c(Z')^*(-d) turns this into H^(2d)(Z') = 0 and n_points-1 new
    classes of type (d,d) in H^(2d-1)(Z'); all other groups are those of Z.
    The same sequence applies to the odd part of a double cover X -> Z (h = h_-(X), n_points = number of excised points
    of X minus that of Z, i.e. twice the number of excised points of Z off the branch locus): H^0(X)^- is 0 for X
    connected (h^{0,0} = 0, all n_points odd classes are new) and Q for X = Z + Z (h^{0,0} = 1, it maps injectively
    to H^0(points)^- as soon as a pair of points is excised).
    **Arguments:**
    - `h (np.ndarray)`: The Hodge diamond of Z.
    - `n_points (int)`: The number of excised points (0 leaves Z unchanged).
    - `compact_support (bool)`: Whether to return the compactly supported cohomology instead of the ordinary one.
      Defaults to `False`.
    **Raises:**
    - `ValueError`: Raised if points are excised and h^{0,0} > 1 or Z is zero-dimensional.
    **Returns:**
    - `np.ndarray`: H[k,p,q] = h^{p,q}(H^k(Z')) (or of H^k_c(Z')), shape (2d+1,d+1,d+1).
    """
    d=h.shape[0]-1
    H=np.zeros((2*d+1,d+1,d+1),dtype=int)
    for p in range(d+1):
        for q in range(d+1):
            H[p+q,p,q]=h[p,q]
    if n_points>0:
        if d==0 or h[0,0]>1:
            raise ValueError("Excising points is only implemented for positive dimension and h^{0,0} <= 1 (Z connected).")
        H[1,0,0]+=n_points-H[0,0,0]
        H[0,0,0]=0
    return H if compact_support else H[::-1,::-1,::-1].copy()

def _number_of_points(e: np.ndarray)->int:
    """
    **Description:**
    The number of points of a variety with E-polynomial e, which must be a finite set: a variety of dimension m >= 1 has
    a non-zero coefficient of (uv)^m (its number of m-dimensional components), so E is constant iff it is finite.
    **Arguments:**
    - `e (np.ndarray)`: The E-polynomial coefficients.
    **Raises:**
    - `ValueError`: Raised if the variety is not finite.
    **Returns:**
    - `int`: The number of points.
    """
    if np.count_nonzero(e)>(e[0,0]!=0):
        raise ValueError("The excised part of Z is not a finite set of points, so its Hodge numbers are not determined by "
                         f"the E-polynomial (use `e_polynomial_generic_hypersurface` instead). Its E-polynomial is\n{e}")
    return int(e[0,0])

def hodge_numbers_generic_hypersurface(fan: Fan,divisor: ArrayLike,backend: str="auto",check: bool=True,excise: Iterable[Iterable[int]]|None=None,excise_closure: bool=True,compact_support: bool=False)->np.ndarray:
    """
    **Description:**
    Hodge diamond h[p,q] = h^{p,q}(Z) of the zero locus Z of a generic section of a toric divisor on the toric variety of
    a complete simplicial fan, from `e_polynomial_generic_hypersurface`. The E-polynomial determines the Hodge numbers
    only if the cohomology of Z is pure, i.e. if Z is compact with at most quotient singularities (e.g. a generic member
    of a base-point free system on a simplicial toric variety, or the quotient of a smooth variety by a finite group);
    then h^{p,q} = (-1)^(p+q) e^{p,q}.
    With `excise`, the toric strata of the given cones are removed and the mixed Hodge numbers h^{p,q}(H^k(Z')) of the
    non-compact Z' are returned. This is only possible if the excised part of Z is a finite set of points (see
    `_point_excised_hodge_numbers`); for larger excised loci the E-polynomial does not fix the individual groups.
    **Arguments:**
    - `fan (Fan)`: A complete simplicial fan.
    - `divisor (ArrayLike)`: Integer coefficients of the divisor in the basis of the prime toric divisors.
    - `backend (str)`: See `e_polynomial_generic_hypersurface`. Defaults to `"auto"`.
    - `check (bool)`: Whether to verify that the fan is complete and that the Hodge numbers of Z are non-negative and
      satisfy h^{p,q} = h^{q,p} and Serre duality h^{p,q} = h^{d-p,d-q}, as purity requires (necessary, not
      sufficient). Defaults to `True`.
    - `excise (Iterable[Iterable[int]]|None)`: Cones of the fan (one-indexed ray labels), or a single cone, whose strata
      are removed, see `e_polynomial_generic_hypersurface`. Defaults to `None`.
    - `excise_closure (bool)`: Whether to remove orbit closures V(t) or only orbits O(t). Defaults to `True`.
    - `compact_support (bool)`: With `excise`, whether to return compactly supported instead of ordinary cohomology.
      Defaults to `False`.
    **Raises:**
    - `ValueError`: Raised (with `check`) if the fan is not complete or the E-polynomial of Z is not that of a pure
      Hodge structure, or (with `excise`) if the excised part of Z is not finite or Z is not connected.
    **Returns:**
    - `np.ndarray`: Without `excise` the Hodge diamond h[p,q] = h^{p,q}(Z), shape (d+1,d+1). With `excise` the mixed
      Hodge numbers H[k,p,q] = h^{p,q}(H^k(Z')) (or of H^k_c(Z')), shape (2d+1,d+1,d+1).
    """
    if check:
        _check_complete(fan)
    n,cones_by_dim,strata_sum=_strata_e_function(fan,divisor,backend)
    removed=_excision_masks(cones_by_dim,excise,excise_closure)
    zero=np.zeros((n+1,n+1),dtype=object)
    e_kept=sum((strata_sum(c[~m]) for c,m in zip(cones_by_dim,removed)),zero)
    e_cut=sum((strata_sum(c[m]) for c,m in zip(cones_by_dim,removed)),zero)
    e=e_kept+e_cut
    if e[n].any() or e[:,n].any():
        raise RuntimeError(f"The E-polynomial has terms of degree > dim Z:\n{e}")
    h=_pure_hodge_numbers(e[:n,:n],check)
    if excise is None:
        return h
    return _point_excised_hodge_numbers(h,_number_of_points(e_cut),compact_support)

def e_polynomials_generic_double_cover(fan: Fan,divisor: ArrayLike,mu: ArrayLike,backend: str="auto",excise: Iterable[Iterable[int]]|None=None,excise_closure: bool=True)->tuple[np.ndarray,np.ndarray]:
    """
    **Description:**
    Even and odd E-polynomials of a toric double cover. Z is the zero locus of a generic section of the divisor on the
    toric variety of the fan in N' = Z^n (as in `e_polynomial_generic_hypersurface`), and X its preimage under the
    quotient map from the toric variety of the same fan in the index-2 sublattice N = {n in N' : <mu,n> even}. The deck
    transformation sigma of X -> Z is the translation by the element of order 2 of the torus T_N with kernel N; it acts
    on the monomial t^m (m in M = M' + Z*mu/2) by -1 iff m is not in M' (the odd lattice points).
    Returns E_+ = E(Z) = E(X)^sigma and E_- = E(X) - E(Z), the E-polynomial of H^*_c(X)^-, which is the compactly
    supported cohomology of Z minus the branch locus with coefficients in the sign local system Q_- of the cover (its
    monodromy is mu, and the stalks of the anti-invariant part vanish on the branch locus). Per stratum, E_- is the
    Batyrev-Borisov formula with all h*-polynomials replaced by those of the odd lattice points (see
    `_strata_e_functions_double_cover`); pointwise fixed orbits, orbits contained in Z and tori contribute nothing.
    **Arguments:**
    - `fan (Fan)`: A simplicial fan in N'; its rays are `fan.vectors()` (labels 1,...,len(fan.vectors())).
    - `divisor (ArrayLike)`: Integer coefficients of the divisor in the basis of the prime toric divisors of N'.
    - `mu (ArrayLike)`: Non-zero mod-2 character of N' (e.g. `Uplift_functions.orientifold_mod2_character(xi)` for an orientifold).
    - `backend (str)`: How the h*-polynomials of the faces are obtained, "faces", "normaliz", "cytools" or "auto" (see `_resolve_backend`). Defaults to `"auto"`.
    - `excise (Iterable[Iterable[int]]|None)`: Cones whose strata are removed from Z and X, see
      `e_polynomial_generic_hypersurface`. Defaults to `None`.
    - `excise_closure (bool)`: Whether to remove orbit closures V(t) or only orbits O(t). Defaults to `True`.
    **Raises:**
    - `ValueError`: Raised if the divisor has no global sections, the fan is not simplicial, mu is invalid or an excised
      cone is not in the fan.
    **Returns:**
    - `tuple[np.ndarray,np.ndarray]`: (E_+, E_-), each with e[p,q] for 0 <= p,q <= dim Z.
    """
    n,cones_by_dim,even_sum,odd_sum=_strata_e_functions_double_cover(fan,divisor,mu,backend)
    removed=_excision_masks(cones_by_dim,excise,excise_closure)
    out=[]
    for strata_sum in (even_sum,odd_sum):
        e=sum((strata_sum(c[~m]) for c,m in zip(cones_by_dim,removed)),np.zeros((n+1,n+1),dtype=object))
        if e[n].any() or e[:,n].any():
            raise RuntimeError(f"The E-polynomial has terms of degree > dim Z:\n{e}")
        out.append(e[:n,:n])
    return out[0],out[1]

def _double_cover_hodge_numbers(fan: Fan,divisor: ArrayLike,mu: ArrayLike,backend: str,check: bool,excise: Iterable[Iterable[int]]|None,excise_closure: bool,compact_support: bool)->dict[str,int|np.ndarray]:
    """
    **Description:**
    Common part of `hodge_numbers_generic_double_cover` and `hodge_numbers_weighted_quotient`: the even and odd Hodge
    diamonds of X (the even one is that of Z) and, with `excise`, the even and odd mixed Hodge numbers after point
    excision together with the numbers of excised points.
    **Arguments:**
    - See `hodge_numbers_generic_double_cover`.
    **Raises:**
    - See `hodge_numbers_generic_double_cover`.
    **Returns:**
    - `dict[str,int|np.ndarray]`: `"h+"`, `"h-"` (diamonds) and with `excise` `"H+"`, `"H-"` (mixed Hodge numbers),
      `"n_cut+"` (excised points of Z) and `"n_cut-"` (excised points of X minus excised points of Z).
    """
    if check:
        _check_complete(fan)
    n,cones_by_dim,even_sum,odd_sum=_strata_e_functions_double_cover(fan,divisor,mu,backend)
    removed=_excision_masks(cones_by_dim,excise,excise_closure)
    zero=np.zeros((n+1,n+1),dtype=object)
    out={}
    for sign,strata_sum in (("+",even_sum),("-",odd_sum)):
        e_kept=sum((strata_sum(c[~m]) for c,m in zip(cones_by_dim,removed)),zero)
        e_cut=sum((strata_sum(c[m]) for c,m in zip(cones_by_dim,removed)),zero)
        e=e_kept+e_cut
        if e[n].any() or e[:,n].any():
            raise RuntimeError(f"The E-polynomial has terms of degree > dim Z:\n{e}")
        out["h"+sign]=_pure_hodge_numbers(e[:n,:n],check)
        if excise is not None:
            out["n_cut"+sign]=_number_of_points(e_cut)
            out["H"+sign]=_point_excised_hodge_numbers(out["h"+sign],out["n_cut"+sign],compact_support)
    return out

def hodge_numbers_generic_double_cover(fan: Fan,divisor: ArrayLike,mu: ArrayLike,backend: str="auto",check: bool=True,excise: Iterable[Iterable[int]]|None=None,excise_closure: bool=True,compact_support: bool=False)->tuple[np.ndarray,np.ndarray]:
    """
    **Description:**
    Even and odd Hodge numbers h^{p,q}_pm(X) of the toric double cover X -> Z of `e_polynomials_generic_double_cover`,
    with h_+ = h(Z). Both follow from E_pm by h^{p,q}_pm = (-1)^(p+q) e^{p,q}_pm if X is compact with pure cohomology
    (e.g. X smooth or an orbifold; then Z = X/sigma is an orbifold too). With `excise`, the strata of the given cones are
    removed from Z and X and the mixed Hodge numbers are returned (the excised parts must be finite, see
    `_point_excised_hodge_numbers`).
    **Arguments:**
    - `fan (Fan)`: A complete simplicial fan in N'.
    - `divisor (ArrayLike)`: Integer coefficients of the divisor in the basis of the prime toric divisors of N'.
    - `mu (ArrayLike)`: Non-zero mod-2 character of N' defining N = {<mu,n> even}.
    - `backend (str)`: See `e_polynomial_generic_hypersurface`. Defaults to `"auto"`.
    - `check (bool)`: Whether to verify that the fan is complete and that both diamonds are non-negative and symmetric
      (necessary for purity, not sufficient). Defaults to `True`.
    - `excise (Iterable[Iterable[int]]|None)`: Cones (one-indexed ray labels), or a single cone, whose strata are
      removed. Defaults to `None`.
    - `excise_closure (bool)`: Whether to remove orbit closures V(t) or only orbits O(t). Defaults to `True`.
    - `compact_support (bool)`: With `excise`, whether to return compactly supported instead of ordinary cohomology.
      Defaults to `False`.
    **Raises:**
    - `ValueError`: Raised (with `check`) if the fan is not complete or a diamond is not that of a pure Hodge structure,
      or (with `excise`) if the excised part is not finite.
    **Returns:**
    - `tuple[np.ndarray,np.ndarray]`: (h_+, h_-) as (d+1,d+1) diamonds, or with `excise` the mixed Hodge numbers
      (H_+, H_-) with H[k,p,q] = h^{p,q}_pm(H^k(X')) (or of H^k_c(X')), shape (2d+1,d+1,d+1).
    """
    out=_double_cover_hodge_numbers(fan,divisor,mu,backend,check,excise,excise_closure,compact_support)
    return (out["h+"],out["h-"]) if excise is None else (out["H+"],out["H-"])
