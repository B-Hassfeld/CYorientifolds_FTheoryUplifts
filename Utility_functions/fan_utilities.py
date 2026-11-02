"""
Toric fan utilities: refinements (stellar subdivisions, pulling triangulations, common refinements), divisor tests
(Cartier, nef, ample), intersection numbers and Chern classes of complete intersections, and divisor bases.
Functions that live in `Uplift_functions` (which imports this module) are imported inside the functions using them,
to avoid a circular import.
"""
from .matrix_utilities import *
import warnings
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import chain, combinations, combinations_with_replacement, islice, permutations
from math import factorial, prod
from typing import Any
import numpy as np
import scipy.sparse as sps
from numpy.typing import ArrayLike
from flint import fmpz_mat
from cytools import Cone
from cytools.cone import dualize
from cytools.vector_config import VectorConfiguration
from cytools.vector_config.fan import Fan


def _exact_scaled_coeffs(Mi,det,x_int,coef_float):
    """
    **Description:**
    Computes `det` times the solution s of `Mi s = x_int` exactly; by Cramer's rule this is an integer vector. The float
    solution scaled by `det` is rounded and certified by the exact int64 identity `Mi s = det x_int`; if the certificate
    fails, the system is solved exactly with FLINT. The entries of `Mi @ s` are assumed to fit into int64.
    **Arguments:**
    - `Mi (np.ndarray)`: Square int64 matrix whose columns are the rays of a simplicial cone.
    - `det (int)`: The determinant of `Mi`.
    - `x_int (np.ndarray)`: The int64 right-hand side.
    - `coef_float (np.ndarray)`: A floating point approximation of the solution of `Mi s = x_int`.
    **Returns:**
    - `np.ndarray`: The int64 vector `det * Mi^{-1} x_int`.
    """
    s=np.rint(coef_float*det).astype(np.int64)
    if np.array_equal(Mi@s,det*x_int):
        return s
    num,den=fmpz_mat(Mi.tolist()).solve(fmpz_mat([[int(v)] for v in x_int])).numer_denom()   # exact solution num/den, den > 0
    t=[int(det)*int(v) for v in num.entries()]; den=int(den)                               # det * solution = t/den
    return np.array([abs(v)//den*(1 if v>=0 else -1) for v in t],dtype=np.int64)          # truncated toward zero, like int() of a rational

def refine_fan(fan,blowups_or_all_vectors=None,tol=1e-9,chunk=256,stats=None):
    """
    **Description:**
    Refines a fan by successive stellar subdivisions at the vectors of its configuration that are not yet rays, after
    appending the new vectors of `blowups_or_all_vectors`. The fan may be simplicial or not and need not span the ambient
    space; in the latter case all vectors are first written exactly in integer coordinates with respect to a lattice basis
    of their span (Hermite normal form), which does not change the combinatorics.
    A new ray x is inserted by determining the face of the fan that contains x in its relative interior and replacing every
    cone containing that face by the joins of x with those facets of the cone that do not contain the face (for a simplicial
    cone: drop one ray of the face at a time). The rays are located quickly: all of them are first located in the input cones
    (the simplicial ones with one matrix product per `chunk` rays, the others with their facet normals), and every subdivided
    cone keeps its children, which cover it exactly, so a ray is found by walking down this history from its starting cone
    instead of testing every cone. The face is decided exactly: in a simplicial cone from the integer coefficients of
    `_exact_scaled_coeffs`, in a non-simplicial cone from its integer facet normals (`dualize`); floats with tolerance `tol`
    only guide the walk through simplicial cones. Cones are assumed to be listed by their extremal rays and to be
    full-dimensional in the span of the fan, and the vectors to carry the default labels 1, 2, ....
    **Arguments:**
    - `fan (Fan)`: The fan to refine.
    - `blowups_or_all_vectors (ArrayLike|None)`: Vectors to add as rays, one per row; vectors already in the configuration
      are ignored, so either the new rays alone or all vectors may be passed. Defaults to `None`.
    - `tol (float)`: Tolerance of the floating point containment tests in simplicial cones. Defaults to `1e-9`.
    - `chunk (int)`: Number of rays located per matrix product against the simplicial input cones; the temporary array has
      (number of simplicial input cones) x dim x `chunk` entries. Defaults to `256`.
    - `stats (dict|None)`: If given, updated with the number of inserted rays (`rays`) and the total number of steps down the
      history (`depth`). Defaults to `None`.
    **Raises:**
    - `ValueError`: Raised if the new vectors do not lie in the span of the fan, cannot be written in a lattice basis of it,
      or a new ray is not contained in the support of the fan.
    - `RuntimeError`: Raised if the point location fails, i.e. the located cone does not contain the ray exactly.
    **Returns:**
    - `Fan`: The refined fan over the enlarged vector configuration, every vector of which is a ray; the fan over the
      enlarged configuration unchanged if there is nothing to insert.
    """
    if blowups_or_all_vectors is not None:
        blowups=row_difference(blowups_or_all_vectors,fan.vc.vectors())
        vc=VectorConfiguration(np.concatenate((fan.vc.vectors(),blowups),axis=0))
        new_fan=Fan(vc,cones=fan.cones())
    else:
        new_fan=fan; vc=fan.vc
    to_be_refined=set(vc.vectors_to_labels(row_difference(vc.vectors(),new_fan.vectors())))
    if len(to_be_refined)==0:
        return new_fan
    vectors=vc.vectors(); dim=fan.dim
    if vc.dim!=dim: raise ValueError("The new rays do not lie in the span of the fan.")
    Vi=np.asarray(vectors,dtype=np.int64)
    if dim<Vi.shape[1]:                                   # exact integer coordinates in the span of the fan
        B=hnf(Vi)[:dim]
        Y=np.rint(Vi@np.linalg.pinv(B.astype(float))).astype(np.int64)
        if not np.array_equal(Y@B,Vi): raise ValueError("Could not express the vectors in a lattice basis of their span.")
        Vi=Y
    coords=Vi.astype(float)
    cones=set(new_fan.cones())
    inv={}; hrep={}                                       # simplicial: float inverse; otherwise: (facets, integer normals)
    def add_hrep(c):
        H=np.asarray(dualize(Vi[np.array(c)-1]),dtype=np.int64)
        on=H@Vi[np.array(c)-1].T==0
        hrep[c]=([tuple(l for l,o in zip(c,row) if o) for row in on],H)
    def add_cones(new):
        simp=[c for c in new if len(c)==dim]
        if simp:
            I=np.linalg.inv(coords[np.array(simp)-1].transpose(0,2,1))
            for k,c in enumerate(simp): inv[c]=I[k]
        for c in new:
            if len(c)>dim: add_hrep(c)
    def contains(c,x,xi):
        return bool(np.all(inv[c]@x>=-tol)) if len(c)==dim else bool(np.all(hrep[c][1]@xi>=0))
    add_cones(list(cones))
    labels=list(to_be_refined)
    # initial location: simplicial cones with one matrix product per chunk, the others exactly
    start={}
    simp0=[c for c in cones if len(c)==dim]; big0=[c for c in cones if len(c)>dim]
    if simp0:
        flat=np.stack([inv[c] for c in simp0]).reshape(-1,dim); rays=coords[np.array(labels)-1]
        for s in range(0,len(labels),chunk):
            hits=np.all((flat@rays[s:s+chunk].T).reshape(len(simp0),dim,-1)>=-tol,axis=1)
            for c,n in zip(*np.nonzero(hits)): start.setdefault(labels[s+n],simp0[c])
    for c in big0:
        inside=np.all(hrep[c][1]@Vi[np.array(labels)-1].T>=0,axis=0)
        for n in np.flatnonzero(inside): start.setdefault(labels[n],c)
    children={}; l2c={}
    for c in cones:
        for l in c: l2c.setdefault(l,set()).add(c)
    depth=0
    for label in labels:
        x=coords[label-1]; xi=Vi[label-1]
        c=start.get(label)
        if c is None: raise ValueError(f"The ray {vectors[label-1]} is not contained in the support of the fan.")
        while c in children:                               # walk down the history: the children of a cone cover it
            ch=children[c]; depth+=1
            nxt=next((k for k in ch if contains(k,x,xi)),None)
            if nxt is None:                                # float tolerance: take the simplicial child closest to containing x
                simp=[k for k in ch if len(k)==dim]
                if not simp: raise RuntimeError("Point location failed.")
                nxt=simp[int(np.argmax([np.min(inv[k]@x) for k in simp]))]
            c=nxt
        if len(c)==dim:                                    # face of c containing x in its interior, decided exactly
            Mi=Vi[np.array(c)-1].T
            det=int(round(np.linalg.det(Mi.astype(float))))
            sgn=np.sign(_exact_scaled_coeffs(Mi,det,xi,inv[c]@x))*np.sign(det)
            if (sgn<0).any(): raise RuntimeError("Exact check: the ray is not in the located cone.")
            face=tuple(l for l,v in zip(c,sgn) if v>0)
        else:
            facets,H=hrep[c]; val=H@xi
            if (val<0).any(): raise RuntimeError("Exact check: the ray is not in the located cone.")
            face=set(c)
            for F,v in zip(facets,val):
                if v==0: face&=set(F)
            face=tuple(sorted(face))
        star=set.intersection(*(l2c[l] for l in face))     # all cones containing that face
        created=[]
        for c_old in star:
            cones.discard(c_old)
            for l in c_old: l2c[l].discard(c_old)
            if len(c_old)==dim:                            # simplicial: drop one ray of the face
                rest=tuple(l for l in c_old if l not in face)
                kids=[tuple(sorted(comb+rest+(label,))) for comb in combinations(face,len(face)-1)]
            else:                                          # pyramids over the facets not containing the face
                facets,H=hrep[c_old]
                kids=[tuple(sorted(F+(label,))) for F,v in zip(facets,H@xi) if v>0]
            for c_new in kids:
                cones.add(c_new)
                for l in c_new: l2c.setdefault(l,set()).add(c_new)
            children[c_old]=kids; created+=kids
        add_cones(created)
    if stats is not None: stats.update(depth=depth,rays=len(labels))
    return Fan(vc=vc,cones=cones)


def minimal_cone_relations(fan,points,tol=1e-9,chunk=128):
    """
    **Description:**
    For every point p in the support of the simplicial, full-dimensional fan `fan`: a maximal cone C containing p, the
    smallest positive integer m such that m p is an integer combination of the rays of C, and these coefficients
    m p = sum_j s_j v_j (s_j >= 0, zero outside the face of C containing p in its interior, so the relation does not
    depend on the choice of C). The cones are located with floats (one matrix product per `chunk` points against all
    cones); m and s are exact: FLINT solves C x = p over the rationals, m is the (minimal) common denominator of x and s
    its numerators. If the located cone does not contain p exactly, the other cones are tried, most likely first.
    **Arguments:**
    - `fan (Fan)`: A simplicial fan spanning the ambient space.
    - `points (ArrayLike)`: Integer points in the support of the fan, one per row.
    - `tol (float)`: Tolerance of the floating point containment test. Defaults to `1e-9`.
    - `chunk (int)`: Points located per matrix product; the temporary array has (number of cones) x dim x `chunk`
      entries. Defaults to `128`.
    **Raises:**
    - `ValueError`: Raised if the fan is not simplicial and full-dimensional, or a point lies in no cone.
    **Returns:**
    - `tuple[np.ndarray,np.ndarray,np.ndarray]`: The labels of the cone of each point (shape (N, dim)), the integer
      coefficients s (shape (N, dim), in the order of the cone labels) and the multipliers m (shape (N,)).
    """
    row={l:i for i,l in enumerate(fan.vc.labels)}
    X=np.asarray(fan.vc.vectors(),dtype=np.int64); d=X.shape[1]
    cones=np.array(fan.cones())
    if cones.ndim!=2 or cones.shape[1]!=d: raise ValueError("The fan must be simplicial and full-dimensional.")
    A=X[np.vectorize(row.get)(cones)].transpose(0,2,1)                # (C,d,d), columns are the rays
    inv=np.linalg.inv(A.astype(float)); flat=inv.reshape(-1,d)
    P=np.asarray(points,dtype=np.int64)
    loc=np.empty(len(P),dtype=np.intp)
    for s0 in range(0,len(P),chunk):                                  # first cone containing each point (floats)
        inside=np.all((flat@P[s0:s0+chunk].T.astype(float)).reshape(len(cones),d,-1)>=-tol,axis=1)
        loc[s0:s0+chunk]=np.where(inside.any(axis=0),inside.argmax(axis=0),-1)
    def relation(c,p):                                                # exact: C x = p, x = s/m
        num,den=fmpz_mat(A[c].tolist()).solve(fmpz_mat([[int(v)] for v in p])).numer_denom()
        return np.array([int(v) for v in num.entries()],dtype=np.int64),int(den)
    S=np.empty((len(P),d),dtype=np.int64); M=np.empty(len(P),dtype=np.int64)
    for i,(c,p) in enumerate(zip(loc,P)):
        s,m=relation(c,p) if c>=0 else (None,0)
        if s is None or (s<0).any():                                  # rare: float location failed, try all cones
            for c in np.argsort(-(inv@p.astype(float)).min(axis=1)):
                s,m=relation(c,p)
                if (s>=0).all(): break
            else: raise ValueError(f"The point {tuple(p)} lies in no cone of the fan.")
        loc[i]=c; S[i]=s; M[i]=m
    return cones[loc],S,M

def get_lower_dimensional_cones(cones_or_fan: Fan|Iterable[Iterable[int]],d: int)->list[tuple[int,...]]:
    """
    **Description:**
    Computes all `d`-element subsets of the ray labels of the given cones; for simplicial cones these are the
    `d`-dimensional faces.
    **Arguments:**
    - `cones_or_fan (Fan|Iterable[Iterable[int]])`: A fan, whose maximal cones are used, or a collection of cones, each
      represented by a tuple of one-indexed ray labels.
    - `d (int)`: Number of rays in the lower-dimensional cones to extract.
    **Returns:**
    - `list[tuple[int, ...]]`: The list of distinct `d`-ray cones (in no particular order).
    """
    match cones_or_fan:
        case Fan():
            cones_or_fan=cones_or_fan.cones()
    return list({combo for row in cones_or_fan for combo in combinations(row,d)})

#lower_case

def is_Cartier(toric_fan: Fan,weights: ArrayLike,return_Q_Cartier_data: bool=False,decimals: int=10)->tuple[bool,list[np.ndarray|None]|None]:
    """
    **Description:**
    Determines whether a toric divisor is Cartier and optionally returns
    local Q-Cartier data.
    **Arguments:**
    - `toric_fan (Fan)`: The toric fan.
    - `weights (ArrayLike)`: Divisor coefficients in the toric prime divisor basis.
    - `return_Q_Cartier_data (bool)`: Whether to continue past cones on which the divisor is not Cartier and return the local data for all cones. Defaults to `False`.
    - `decimals (int)`: Number of decimals to which non-integral local data are rounded. Defaults to `10`.
    **Returns:**
    - `tuple[bool,list[np.ndarray|None]|None]`: A pair `(is_cartier, cartier_data)`. `cartier_data` contains, for each maximal cone, the local Cartier data `m_sigma` (an integer array if integral, a rounded float array if only Q-Cartier, `None` if not Q-Cartier on that cone). If `return_Q_Cartier_data=False`, the function returns `(False, None)` as soon as a cone is found on which the divisor is not Cartier.
    """
    weights=np.array(weights)
    cartier_data=[]
    is_cartier=True
    for c in toric_fan.cones():
        arr=toric_fan.vectors(c)
        cone_gen_weights=weights[np.array(c)-1]
        y=-np.linalg.lstsq(arr,cone_gen_weights,rcond=None)[0]
        if not np.allclose(arr@y,-cone_gen_weights,rtol=0,atol=1e-10):
            if return_Q_Cartier_data:
                cartier_data.append(None)
                is_cartier=False
                continue
            return False, None
        if np.allclose(y,np.round(y),rtol=0,atol=1e-10):
            cartier_data.append(np.round(y).astype(int))
        else:
            is_cartier=False
            if return_Q_Cartier_data:
                cartier_data.append(np.round(y,decimals=decimals))
            else:
                return False, None
    return is_cartier, cartier_data

def is_nef(toric_fan: Fan,weights: ArrayLike)->np.bool_:
    """
    **Description:**
    Tests whether the divisor coefficient vector lies in the nef cone by pairing with the secondary-cone hyperplanes.
    **Arguments:**
    - `toric_fan (Fan)`: The toric fan.
    - `weights (ArrayLike)`: Divisor coefficients in the toric prime divisor basis.
    **Returns:**
    - `np.bool_`: `True` if the divisor is nef, otherwise `False`.
    """
    return np.all(np.array(toric_fan.secondary_cone_hyperplanes())@weights>=0)

def is_ample(toric_fan: Fan,weights: ArrayLike)->np.bool_:
    """
    **Description:**
    Tests whether the divisor coefficient vector lies in the interior of the nef cone by strict pairing with the secondary-cone hyperplanes.
    **Arguments:**
    - `toric_fan (Fan)`: The toric fan.
    - `weights (ArrayLike)`: Divisor coefficients in the toric prime divisor basis.
    **Returns:**
    - `np.bool_`: `True` if the divisor is ample, otherwise `False`.
    """
    return np.all(np.array(toric_fan.secondary_cone_hyperplanes())@weights>0)

def general_divisor_intersection(fan: Fan,divisors: Sequence[ArrayLike])->float:
    """
    **Description:**
    Computes the intersection number D_1*...*D_n of n=`fan.dim` divisors in the toric variety of `fan`.
    **Arguments:**
    - `fan (Fan)`: The toric fan; must be a triangulation.
    - `divisors (Sequence[ArrayLike])`: `fan.dim` divisors, each a coefficient vector over the prime toric divisors, ordered as `fan.labels`.
    **Raises:**
    - `ValueError`: Raised if the number of divisors is not `fan.dim` or a divisor does not have one coefficient per label.
    **Returns:**
    - `float`: The intersection number of the divisors.
    """
    if len(divisors)!=fan.dim:
        raise ValueError(f"Need {fan.dim} divisors, got {len(divisors)}.")
    divisors=np.asarray(divisors)
    if divisors.shape[1]!=len(fan.labels):
        raise ValueError(f"Divisors need {len(fan.labels)} coefficients, got {divisors.shape[1]}.")
    label_idx={l:idx for idx,l in enumerate(fan.labels)}
    supports=[set(np.nonzero(div)[0]) for div in divisors]
    total=0
    #keys are sorted label tuples, so sum over all distinct orderings of each key
    for key,value in fan.intersection_numbers(copy=False).items():
        if 0 in key:
            continue
        idx=tuple(label_idx[l] for l in key)
        if not all(support.intersection(idx) for support in supports):
            continue
        for perm in set(permutations(idx)):
            total+=value*prod(divisors[a][i] for a,i in enumerate(perm))
    return total

def localization_weights(fan: Fan)->tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    **Description:**
    Torus-fixed point data of the simplicial toric variety of `fan` for a generic one-parameter subgroup xi=(1,k,k^2,...).
    At the fixed point of a maximal cone sigma the weights of the prime toric divisors D_i (i in sigma) are the
    coordinates of xi in the ray basis of sigma; for every polynomial P in the D_i,
    int P = sum_sigma P(weights)/(|det sigma| prod weights).
    **Arguments:**
    - `fan (Fan)`: A complete simplicial fan.
    **Raises:**
    - `ValueError`: Raised if no generic xi is found.
    **Returns:**
    - `tuple[np.ndarray,np.ndarray,np.ndarray]`: (cones, weights, denominators): the maximal cones as zero-based ray indices (C,d), the weights (C,d) and |det sigma| prod weights (C,).
    """
    vecs=np.asarray(fan.vectors(),float)
    cones=np.asarray(fan.cones(as_inds=True))
    d=vecs.shape[1]
    M=vecs[cones].transpose(0,2,1)
    Minv=np.linalg.inv(M)
    for k in range(2,50):
        weights=Minv@(float(k)**np.arange(d))
        if np.all(np.abs(weights)>1e-10):
            return cones,weights,np.abs(np.linalg.det(M))*np.prod(weights,axis=1)
    raise ValueError("No generic one-parameter subgroup found.")

def toric_chern_class(fan: Fan,divisors: ArrayLike,n: int)->int|dict[tuple[int,...],int]|dict[int,int]:
    """
    **Description:**
    Chern tensor c_n(X) of the complete intersection X=L_1∩...∩L_r in the simplicial toric variety of `fan`, of any
    dimension, computed by torus localization on the maximal cones of the fan. With the weights w of the prime toric
    divisors and l_k of the L_k at a fixed point, c(X)=prod(1+w)/prod(1+l_k), so c_n follows from the power sums
    p_j=sum w^j-sum l_k^j by Newton's identities. The tensor of rank dim X-n is
    T_{a_1..a_m}=int_X c_n(X) D_{a_1}...D_{a_m}. Prime-toric divisor indices are 1-based.
    **Arguments:**
    - `fan (Fan)`: The complete simplicial toric fan of the ambient variety.
    - `divisors (ArrayLike)`: The coefficient vectors of L_1,...,L_r in the prime toric divisor basis, one per row.
    - `n (int)`: Degree of the Chern class (0 to dim X).
    **Raises:**
    - `ValueError`: Raised if `n` is not between 0 and dim X.
    **Returns:**
    - `int|dict[tuple[int,...],int]|dict[int,int]`: The rounded Chern tensor: a number if n=dim X, {i: value} if n=dim X-1, and {sorted label tuple: value} otherwise.
    """
    # from ..Uplift_functions import localization_weights   # local import: Uplift_functions imports this module
    divs=np.atleast_2d(np.asarray(divisors,float))
    cones,weights,den=localization_weights(fan)
    d=weights.shape[1]
    rank=d-len(divs)-n
    if n<0 or rank<0:
        raise ValueError(f"The degree must be between 0 and {d-len(divs)}.")
    ell=np.einsum("kci,ci->ck",divs[:,cones],weights)
    p=[None]+[np.sum(weights**j,axis=1)-np.sum(ell**j,axis=1) for j in range(1,n+1)]
    e=[np.ones(len(cones))]
    for m in range(1,n+1):
        e.append(sum((-1)**(i-1)*e[m-i]*p[i] for i in range(1,m+1))/m)
    base=e[n]*np.prod(ell,axis=1)/den
    if rank==0:
        return round(np.sum(base))
    result={}
    for cone,w,coeff in zip(cones,weights,base):
        for inds in combinations_with_replacement(range(d),rank):
            key=tuple(sorted(cone[list(inds)]+1))
            result[key]=result.get(key,0)+coeff*np.prod(w[list(inds)])
    result={key: round(value) for key,value in result.items() if abs(value)>1e-8}
    if rank==1:
        result={key[0]: value for key,value in result.items()}
    return result

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

def subvariety_intersection_numbers(fan: Fan,divisors: Sequence[ArrayLike]|None=None,basis: ArrayLike|None=None,Q: ArrayLike|None=None,in_basis: bool=False,tol: float|None=1e-5,decimals: int=10)->dict[tuple[int,...],int|float]:
    """
    **Description:**
    Intersection numbers of the restricted toric divisors on the complete intersection X = D^(1) ∩ ... ∩ D^(r) of `divisors` in the toric variety of `fan`, in terms of all prime toric divisors, of a prime toric basis, or of a GLSM basis (see `_subvariety_intersection_numbers_all_prime_torics` and `tensor_symm_in_basis`).
    No Calabi-Yau condition is assumed: the result is the ambient intersection product with the class D^(1)...D^(r), which gives the intersection numbers of X whenever X is a transverse complete intersection (e.g. for generic sections of base-point-free divisors).
    **Arguments:**
    - `fan (Fan)`: The toric fan of the ambient variety.
    - `divisors (Sequence[ArrayLike]|None)`: Coefficient vectors of the divisors cutting out X, ordered by ray label. If `None`, the anticanonical hypersurface is used. Defaults to `None`.
    - `basis (ArrayLike|None)`: One-indexed labels of the prime toric basis divisors. If `None` and `in_basis` is `True` or `Q` is given, `basis_homology(fan)` is used. Defaults to `None`.
    - `Q (ArrayLike|None)`: GLSM charge matrix; if given, the result is expressed in the GLSM basis. Defaults to `None`.
    - `in_basis (bool)`: Whether to express the result in a prime toric basis. Defaults to `False`.
    - `tol (float|None)`: Rounding tolerance, see `_subvariety_intersection_numbers_all_prime_torics`. Defaults to `1e-5`.
    - `decimals (int)`: Number of decimals to round to. Defaults to `10`.
    **Returns:**
    - `dict[tuple[int,...],int|float]`: The intersection numbers keyed by sorted one-indexed label tuples (prime torics, basis positions or GLSM indices); absent keys are zero.
    """
    out=_subvariety_intersection_numbers_all_prime_torics(fan,divisors,tol=tol,decimals=decimals)
    if in_basis or Q is not None:
        if basis is None:
            basis=basis_homology(fan)
    if basis is not None:
        return tensor_symm_in_basis(out,basis,Q,in_prime_basis=False)
    return out

def _subvariety_intersection_numbers_all_prime_torics(fan: Fan,divisors: Sequence[ArrayLike]|None=None,tol: float|None=1e-5,decimals: int=10)->dict[tuple[int,...],float]:
    """
    **Description:**
    Intersection numbers of the restricted prime toric divisors on the complete intersection X = D^(1) ∩ ... ∩ D^(r)
    (of any dimension d = dim(fan) - r, no Calabi-Yau condition assumed), via the ambient intersection numbers K:
        kappa_{a_1..a_d} = sum_{i_1..i_r} c^(1)_{i_1} ... c^(r)_{i_r} K_{i_1..i_r a_1..a_d}
    **Arguments:**
    - `fan (Fan)`: CYTools Fan. Ray labels 1,...,N; fan.intersection_numbers() returns {sorted tuple: value},
        where label 0 is the canonical divisor (ignored here).
    - `divisors (Sequence[ArrayLike]|None)`: Coefficients of D^(k) = sum_i c^(k)_i D_i, each of shape (N,), ordered by label 1,...,N.
        If `None`, the anticanonical hypersurface (all c_i = 1) is used. Defaults to `None`.
    - `tol (float|None)`: If given, drop entries with |value| <= tol and round to `decimals` (values within `tol` of an integer are rounded to it). Defaults to `1e-5`.
    - `decimals (int)`: Number of decimals to round to. Defaults to `10`.
    **Returns:**
    - `dict[tuple[int,...],float]`: {sorted tuple of labels in 1,...,N: value}; absent keys are zero.
    """
    if divisors is None:
        divisors=[np.ones(len(fan.vectors()),dtype=int)]
    K=fan.intersection_numbers()
    keys=np.array(list(K),dtype=np.int64)
    vals=np.fromiter(K.values(),dtype=float,count=len(K))
    keep=keys[:,0]!=0
    keys, vals=keys[keep], vals[keep]
    n, r=keys.shape[1], len(divisors)
    C=np.zeros((r,len(fan.vc.labels)+1))
    C[:,1:]=divisors
    same_prev=np.zeros(keys.shape,dtype=bool)
    same_prev[:,1:]=keys[:,1:]==keys[:,:-1]
    out_keys, out_vals=[], []
    for pos in permutations(range(n),r):
        rows=np.ones(len(vals),dtype=bool)
        for k, p in enumerate(pos):
            if p>0 and p-1 not in pos[:k]:
                rows&=~same_prev[:,p]
        w=vals[rows]*np.prod([C[k,keys[rows,p]] for k,p in enumerate(pos)],axis=0)
        nz=w!=0
        out_keys.append(keys[rows][nz][:,[q for q in range(n) if q not in pos]])
        out_vals.append(w[nz])
    ukeys, inv=np.unique(np.concatenate(out_keys),axis=0,return_inverse=True)
    sums=np.bincount(inv.ravel(),weights=np.concatenate(out_vals),minlength=len(ukeys))
    if tol is not None:
        sums=np.round(sums,decimals)
        nearest=np.rint(sums)
        sums=np.where(np.abs(sums-nearest)<=tol,nearest,sums)
    return {tuple(key): float(v) for key,v in zip(ukeys.tolist(),sums) if tol is None or abs(v)>tol}

def reorder_fan(fan: Fan,vectors: ArrayLike)->Fan:
    """
    **Description:**
    Returns the same fan over a vector configuration with the given order of its vectors, i.e. the cones are relabelled
    such that label i refers to `vectors[i-1]`.
    **Arguments:**
    - `fan (Fan)`: The toric fan.
    - `vectors (ArrayLike)`: The vectors of `fan.vc`, one per row, in the new order.
    **Raises:**
    - `ValueError`: Raised if `vectors` is not a reordering of the vectors of `fan.vc`.
    **Returns:**
    - `Fan`: The fan over `VectorConfiguration(vectors)`.
    """
    vectors=np.asarray(vectors)
    old=fan.vc.vectors()
    if old.shape!=vectors.shape or not np.array_equal(np.unique(old,axis=0),np.unique(vectors,axis=0)) or len(np.unique(vectors,axis=0))!=len(vectors):
        raise ValueError("vectors must be a reordering of the vectors of the fan.")
    vc=VectorConfiguration(vectors)
    new_label=dict(zip(fan.vc.labels,np.atleast_1d(vc.vectors_to_labels(old)).tolist()))
    return Fan(vc=vc,cones=[tuple(new_label[l] for l in c) for c in fan.cones()])

def _facets(fan: Fan, c: tuple) -> list[tuple]:
    """
    **Description:**
    Facets of the pointed cone of `fan` with extremal rays `c`, each as the tuple of its labels, from one PPL call (`dualize`).
    The lineality equations that `dualize` emits for fans that do not span the ambient space vanish on all of `c` and are dropped.
    **Arguments:**
    - `fan (Fan)`: The fan providing the ray vectors.
    - `c (tuple[int,...])`: Labels of the extremal rays of the cone.
    **Returns:**
    - `list[tuple[int,...]]`: The facets as label tuples, in the order of `c`.
    """
    W=np.asarray(fan.vectors(c),dtype=np.int64)
    return [F for F in (tuple(l for l,o in zip(c,h@W.T==0) if o) for h in dualize(W)) if len(F)<len(c)]

def _pull(c: tuple, facets: list[tuple], k: int, pos) -> set[tuple]:
    """
    **Description:**
    Pulling triangulation of the k-dimensional pointed cone `c` with facets `facets` (label tuples): the ray v of smallest
    `pos` is joined with the pulling triangulations of the facets not containing v, recursively. The facets of a facet F are
    read off combinatorially as the maximal intersections of F with the other facets (every ridge lies in exactly two facets),
    so no further PPL calls are needed. A cone with exactly k rays is a simplex and is returned as it is.
    **Arguments:**
    - `c (tuple[int,...])`: Sorted labels of the extremal rays of the cone.
    - `facets (list[tuple[int,...]])`: Its facets as label tuples, e.g. from `_facets`.
    - `k (int)`: Dimension of the cone.
    - `pos (Callable[[int],int])`: Position of a label in the pulling order; smaller is pulled first.
    **Returns:**
    - `set[tuple[int,...]]`: The simplices of the triangulation as sorted label tuples.
    """
    if len(c)==k: return {c}
    v=min(c,key=pos); out=set()
    for F in facets:
        if v in F: continue
        if len(F)==k-1: out.add(tuple(sorted(F+(v,)))); continue
        R={tuple(l for l in F if l in G) for G in facets if G!=F}
        R=[r for r in R if not any(set(r)<set(s) for s in R)]
        out.update(tuple(sorted(T+(v,))) for T in _pull(F,R,k-1,pos))
    return out

def pull_ray(fan: Fan, label: int) -> Fan:
    """
    **Description:**
    Stellar subdivision (pulling refinement) of `fan` at its existing ray `label`: every non-simplicial cone containing the ray
    is replaced by the joins of the ray with its facets not containing the ray. Simplicial cones and cones without the ray are
    unchanged. The result is regular whenever `fan` is regular (lower the height of the pulled ray slightly). Cones are assumed
    to be listed by their extremal rays and to be full-dimensional in the span of the fan.
    **Arguments:**
    - `fan (Fan)`: The fan to subdivide.
    - `label (int)`: Label of the ray to pull.
    **Returns:**
    - `Fan`: The subdivided fan over the same vector configuration; `fan` itself if there is nothing to subdivide.
    """
    d=fan.dim; big=[c for c in fan.cones() if label in c and len(c)>d]
    if not big: return fan
    cones=set(fan.cones()).difference(big)
    for c in big: cones.update(tuple(sorted(F+(label,))) for F in _facets(fan,c) if label not in F)
    return Fan(vc=fan.vc,cones=cones)

def make_simplicial(fan: Fan, order=None) -> Fan:
    """
    **Description:**
    Simplicial refinement of a fan by the pulling triangulation of every non-simplicial cone, with the rays pulled in the order
    `order`; equals the iterated `pull_ray` over `order`. Each non-simplicial cone is triangulated recursively by `_pull` after
    a single PPL call for its facets. The result is a fan: the triangulation induced on a face is the pulling triangulation of
    that face in the same order, whichever cone it is computed from, so shared walls are subdivided consistently. It is regular
    whenever `fan` is regular. Cones are assumed to be listed by their extremal rays and to be full-dimensional in the span of
    the fan; simplicial cones are left unchanged.
    **Arguments:**
    - `fan (Fan)`: The fan to refine.
    - `order (Iterable[int]|None)`: Labels in the order in which they are pulled. Defaults to `None` (label order of the configuration).
    **Returns:**
    - `Fan`: The simplicial fan over the same vector configuration; `fan` itself if it is already simplicial.
    """
    d=fan.dim; big=[c for c in fan.cones() if len(c)>d]
    if not big: return fan
    pos={l:i for i,l in enumerate(fan.vc.labels if order is None else order)}.__getitem__
    cones=set(fan.cones()).difference(big)
    for c in big: cones.update(_pull(c,_facets(fan,c),d,pos))
    return Fan(vc=fan.vc,cones=cones)

def common_fan_refinement(fan1,fan2,vectors_fine_to_add=None,simplicial=False):
    """
    **Description:**
    Common refinement of two fans of the same ambient dimension. Neither fan needs to be simplicial or
    complete; the result is supported on the intersection of the two supports. A cone C1 of fan1 lying in
    a single cone C2 of fan2 is kept as it is. Every other C1 is intersected exactly (PPL) with each
    C2 that is not separated from it by a facet of C1 or of C2; the full-dimensional intersections are
    the remaining cones. The vector configuration of the result consists of the vectors of fan1, then
    the vectors of fan2 not already present, then the rays of the intersections that are vectors of
    neither fan, in this order and with the default labels 1,2,...; the labels of the input fans are not
    used. Rays are identified by exact comparison of integer vectors, so both fans should store
    primitive ray generators. If `vectors_fine_to_add` is None, a warning lists all appended rays;
    otherwise a warning is issued only for appended rays that are not in `vectors_fine_to_add`.
    Used, e.g., to refine the blow-down fan {rho_+, rho_-} * facefan(nabla_fib) by facefan(nabla^circ).
    **Arguments:**
    - `fan1 (Fan)`: A fan all of whose maximal cones are full-dimensional and pointed.
    - `fan2 (Fan)`: A fan of the same ambient dimension, with the same requirements on its cones.
    - `vectors_fine_to_add (ArrayLike, optional)`: Rays of neither fan that are expected to be needed
      by the refinement. Appended rays contained in this list do not trigger a warning.
    - `simplicial (bool)`: Whether to make the result simplicial with `make_simplicial`. Defaults to `False`.
    **Raises:**
    - `ValueError`: Raised if the fans have different dimensions or a cone is not full-dimensional.
    **Returns:**
    - `Fan`: The common refinement (in general not simplicial unless `simplicial=True`).
    """
    d=fan1.dim
    if fan2.dim!=d: raise ValueError("The fans must have the same dimension.")
    vecs=[tuple(map(int,v)) for v in fan1.vc.vectors()]
    vecs+=[t for t in (tuple(map(int,v)) for v in fan2.vc.vectors()) if t not in vecs]
    row={t:k for k,t in enumerate(vecs)}
    def label(w):
        t=tuple(map(int,w))
        if t not in row: row[t]=len(row)
        return row[t]+1
    def facets(fan):
        W=[np.array(fan.vectors(c),dtype=np.int64) for c in fan.cones()]
        if any(np.linalg.matrix_rank(w)!=d for w in W): raise ValueError("All cones must be full-dimensional.")
        return W,[np.asarray(dualize(w),dtype=np.int64) for w in W]
    W1,H1=facets(fan1)
    W2,H2=facets(fan2)
    cones=[]
    for c,w1,h1 in zip(fan1.cones(),W1,H1):
        if any((h2@w1.T>=0).all() for h2 in H2):
            cones.append(tuple(sorted(label(v) for v in w1))); continue
        for w2,h2 in zip(W2,H2):
            # separated by a facet of C2 or of C1: no full-dimensional intersection
            if (h2@w1.T<=0).all(axis=1).any() or (h1@w2.T<=0).all(axis=1).any(): continue
            rays=dualize(np.vstack((h1,h2)))
            if len(rays) and np.linalg.matrix_rank(rays)==d:
                cones.append(tuple(sorted(label(r) for r in rays)))
    if len(row)>len(vecs):
        added=list(row)[len(vecs):]
        if vectors_fine_to_add is None:
            warnings.warn(f"The refinement needs {len(added)} rays of neither fan: {added}")
        else:
            expected={tuple(map(int,v)) for v in vectors_fine_to_add}
            unexpected=[t for t in added if t not in expected]
            if unexpected: warnings.warn(f"The refinement needs {len(unexpected)} rays of neither fan that are not in vectors_fine_to_add: {unexpected}")
    if simplicial:
        return make_simplicial(Fan(vc=VectorConfiguration(np.array(list(row))),cones=cones))
    return Fan(vc=VectorConfiguration(np.array(list(row))),cones=cones)

def basis_homology(toric_fan_or_points,integral=True,chunk=1024,max_subsets=10**6):
    """
    **Description:**
    Labels of prime toric divisors forming a basis of H^2 (the divisor classes): the complement of k rays, k the
    dimension of their span, that form a basis of the lattice span ∩ Z^m (`integral=True`, a Z-basis) or are only
    linearly independent (`integral=False`, a Q-basis). For a fan the maximal cones are tried first in
    `fan.cones()` ; then all k-subsets of rays in lexicographic
    order, which for points and `integral=False` reproduces `basis(points)`. Floats only pre-screen candidates;
    the decision is exact (Smith normal form).
    **Arguments:**
    - `toric_fan_or_points (Fan|ArrayLike)`: The toric fan, or integral ray generators one per row.
    - `integral (bool)`: Whether to require a Z-basis. Defaults to `True`.
    - `chunk (int)`: Number of candidates screened per vectorised determinant call. Defaults to `1024`.
    - `max_subsets (int)`: Maximal number of k-subsets of rays tried after the cones. Defaults to `10**6`.
    **Raises:**
    - `ValueError`: Raised if no suitable k rays were found.
    **Returns:**
    - `np.ndarray`: One-indexed labels of the basis divisors (the fan's labels, or row index + 1 for points).
    """
    fan=toric_fan_or_points if isinstance(toric_fan_or_points,Fan) else None
    labels=np.array(fan.used_labels) if fan else np.arange(1,len(toric_fan_or_points)+1)
    V=np.asarray(fan.vc.vectors(which=labels) if fan else toric_fan_or_points,dtype=np.int64)
    k=np.linalg.matrix_rank(V)
    row={l:i for i,l in enumerate(labels)}
    cones=(tuple(row[l] for l in c) for c in fan.cones() if len(c)==k) if fan else ()
    candidates=chain(cones,islice(combinations(range(len(V)),k),max_subsets))
    while len(group:=np.array(list(islice(candidates,chunk)),dtype=np.intp)):
        if k==V.shape[1]:
            det=np.abs(np.linalg.det(V[group].astype(float)))
            group=group[det<2.5 if integral else det>0.5]
        for c in group:
            snf=fmpz_mat(V[c].tolist()).snf()
            index=abs(prod(int(snf[i,i]) for i in range(k)))
            if index==1 or (index and not integral):
                return np.delete(labels,c)
    raise ValueError("No basis found: no k rays form a lattice basis (or none among the candidates tried); use integral=False for a Q-basis.")

def moving_cone(toric_variety: Fan)->Cone:
    """
    **Description:**
    Computes the moving cone from the GLSM charges by intersecting the cones obtained after deleting each toric ray.
    **Arguments:**
    - `toric_variety (Fan)`: The toric fan or toric variety object.
    **Returns:**
    - `Cone`: The moving cone.
    """
    rays=toric_variety.vectors()
    glsm=np.array(integral_nullspace(np.asarray(rays.T,dtype=int))).T
    h_planes=np.array([h for i in range(len(rays)) for h in Cone(np.delete(glsm.T,i,0)).hyperplanes()])@glsm
    mov=Cone(hyperplanes=h_planes)
    return mov

def cone_indices(fan: Fan,cones: Iterable[Iterable[int]]|None=None)->list[int]:
    """
    **Description:**
    Computes the index [N ∩ span(σ) : Z⟨rays of σ⟩] of every cone σ, i.e. the product of the nonzero Smith normal form invariants of its ray matrix. For a simplicial cone this is its multiplicity (|det| of the rays for a full-dimensional one), and the cone is smooth iff the index is 1. For a non-simplicial cone it is only the index of the lattice generated by its rays; such a cone is never smooth.
    The product of the nonzero invariants of a rank-r matrix is its r-th determinantal divisor, the gcd of all r×r minors, so all cones with the same number of rays are handled at once with batched floating-point determinants (exact as long as the minors stay well below 2^53). The rank is found by lowering r from min(#rays, dim) until the gcd is nonzero, which only costs extra for non-simplicial cones.
    **Arguments:**
    - `fan (Fan)`: The toric fan.
    - `cones (Iterable[Iterable[int]])`: The cones, as tuples of one-indexed ray labels of `fan`.
    **Returns:**
    - `list[int]`: The index of each cone, in the order of `cones`.
    """
    if cones is None:
        cones=fan.cones()
    V=np.asarray(fan.vectors())
    row={l:i for i,l in enumerate(fan.used_labels)}
    cones=[[row[l] for l in c] for c in cones]
    out=np.ones(len(cones),dtype=np.int64)
    groups=defaultdict(list)
    for i,c in enumerate(cones):
        groups[len(c)].append(i)
    for k,idx in groups.items():
        if k==0:
            continue
        idx=np.array(idx)
        R=V[np.array([cones[i] for i in idx])]
        for r in range(min(k,V.shape[1]),0,-1):
            minors=np.stack([np.linalg.det(R[:,list(rows)][:,:,list(cols)]) for rows in combinations(range(k),r) for cols in combinations(range(V.shape[1]),r)],axis=1)
            g=np.gcd.reduce(np.abs(np.rint(minors)).astype(np.int64),axis=1)
            out[idx[g>0]]=g[g>0]
            idx,R=idx[g==0],R[g==0]
            if len(idx)==0:
                break
    return out.tolist()

def find_cone(new_ray: ArrayLike,current_cones: Iterable[tuple[int,...]],all_vectors: np.ndarray,tol: float=1e-10)->frozenset[int]|None:
    """
    **Description:**
    Searches the current (full-dimensional, simplicial) cones for the first one containing `new_ray` by solving for its coefficients in the generators of all cones at once, and returns the labels of the generators with strictly positive coefficient, i.e. the carrier face of `new_ray`.
    **Arguments:**
    - `new_ray (ArrayLike)`: The ray to locate.
    - `current_cones (Iterable[tuple[int,...]])`: Current cones, represented by tuples of one-indexed ray labels; each must have as many rays as the ambient dimension.
    - `all_vectors (np.ndarray)`: Full ray matrix.
    - `tol (float)`: Numerical tolerance for the sign of the coefficients. Defaults to `1e-10`.
    **Raises:**
    - `numpy.linalg.LinAlgError`: Raised if a cone is not simplicial and full-dimensional (singular or non-square generator matrix).
    **Returns:**
    - `frozenset[int]|None`: The carrier face labels, or `None` if no cone contains `new_ray`.
    """
    cones=list(current_cones)
    if len(cones)==0:
        return None
    try:
        A=all_vectors[np.array(cones)-1].transpose(0,2,1).astype(float)
        coeffs=np.linalg.solve(A,np.asarray(new_ray,dtype=float)[:,None])[...,0]
    except (ValueError,np.linalg.LinAlgError):
        for cone in cones:
            numpy_indices=np.array(cone)-1
            A=all_vectors[numpy_indices].T
            coeffs=np.linalg.solve(A,new_ray)
            if np.all(coeffs>=-tol):
                return frozenset(idx for idx,coeff in zip(cone,coeffs) if coeff>tol)
        return None
    hit=np.flatnonzero(np.all(coeffs>=-tol,axis=1))
    if len(hit)==0:
        return None
    return frozenset(idx for idx,coeff in zip(cones[hit[0]],coeffs[hit[0]]) if coeff>tol)