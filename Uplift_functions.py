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
# Description:  This module contains tools designed for Calabi-Yau hypersurface
#               computations.
# -----------------------------------------------------------------------------
from __future__ import annotations
# 'standard' imports
from itertools import combinations, combinations_with_replacement, permutations, product
from collections.abc import Iterable, Mapping, Sequence
from collections import defaultdict, Counter
from fractions import Fraction
from functools import cache
from math import comb,factorial, prod
import warnings
from typing import TYPE_CHECKING, Any, Literal, overload
import math
# 3rd party imports
import numpy as np
import shutil
import tempfile
import subprocess
import re
from pathlib import Path
from flint import fmpz_mat
from sympy import Matrix
from sympy.matrices.normalforms import smith_normal_decomp
from scipy.optimize import nnls
from scipy.linalg import schur, qr
import scipy.sparse as sps
from scipy.optimize import milp, LinearConstraint, linprog
from scipy.spatial import ConvexHull
# CYTools imports
from cytools import Polytope, h_polytope, Cone
from cytools.utils import integral_nullspace, lll_reduce
from cytools.vector_config.fan import Fan
from cytools.vector_config import VectorConfiguration
import ppl
from cygv import compute_gv
from numpy.typing import ArrayLike
if TYPE_CHECKING:
    from .FT_CY import CY_orientifold, F_Theory_Uplift

def compute_partition(divisors: Sequence[np.ndarray],rays: np.ndarray)->tuple[bool,np.ndarray|None]:
    """
    **Description:**
    Uses linear equivalence to shift a collection of toric divisors into a representation whose coefficients form a partition of the anticanonical divisor, if such a representation exists. The shift is found by solving a mixed-integer linear program; the current implementation supports exactly two divisors.
    **Arguments:**
    - `divisors (Sequence[np.ndarray])`: The divisor coefficient vectors (one integer array per divisor, in the toric prime divisor basis).
    - `rays (np.ndarray)`: The primitive ray generators of the toric fan, one per row.
    **Returns:**
    - `tuple[bool, np.ndarray | None]`: A pair `(exists, shifted_divisors)`, where `exists` is a boolean and `shifted_divisors` is the integer array of shifted divisors (one per row) if it exists, otherwise `None`.
    """
    null=np.zeros([rays.shape[0],rays.shape[1]],dtype=int)
    linear1=np.tensordot(np.identity(2,dtype=int),rays,axes=0)
    linear1=linear1.transpose(0,2,1,3).reshape(len(divisors)*rays.shape[0],len(divisors)*rays.shape[1])
    linear2=np.hstack([rays]*len(divisors))
    affine1=np.concatenate(divisors)
    affine2=sum(divisors)-1
    lower1=-affine1
    upper1=np.full_like(affine1,np.inf,dtype=float)
    lower2=-affine2
    upper2=-affine2
    linear=np.vstack([linear1,linear2])
    lower=np.concatenate([lower1,lower2])
    upper=np.concatenate([upper1,upper2])
    c=np.zeros(linear.shape[1])
    integrality=np.ones_like(c)
    constraints=LinearConstraint(linear,lower,upper)
    res=milp(c=np.zeros(linear.shape[1]),constraints=constraints,integrality=np.ones(linear.shape[1]),bounds=(-np.inf,np.inf))
    if not res.success or res.x is None:
        return (False,None)
    sol=np.rint(res.x.reshape(len(divisors),len(rays[0]))@(rays.T)+np.array(divisors)).astype(int)
    return (True,sol)

def contains_row(arr: np.ndarray,target: ArrayLike)->np.bool_:
    """
    **Description:**
    Determines whether the one-dimensional array `target` occurs as a row of the two-dimensional array `arr`.
    **Arguments:**
    - `arr (np.ndarray)`: The two-dimensional array to search.
    - `target (ArrayLike)`: The one-dimensional row to find.
    **Returns:**
    - `np.bool_`: `True` if `target` is a row of `arr`, otherwise `False`.
    """
    return np.any(np.all(arr==target,axis=1))

def contains_rows(arr: np.ndarray,targets: np.ndarray)->np.bool_:
    """
    **Description:**
    Determines whether every row of `targets` occurs as a row of `arr`.
    **Arguments:**
    - `arr (np.ndarray)`: The two-dimensional array to search.
    - `targets (np.ndarray)`: The two-dimensional array of target rows.
    **Returns:**
    - `np.bool_`: `True` if every target row occurs in `arr`, otherwise `False`.
    """
    return np.any((targets[:,None,:]==arr[None,:,:]).all(axis=2),axis=1).all()

def get_same_rows(A: np.ndarray,B: np.ndarray)->np.ndarray:
    """
    **Description:**
    Computes the rows of `A` that also occur as rows of `B`.
    **Arguments:**
    - `A (np.ndarray)`: First two-dimensional array.
    - `B (np.ndarray)`: Second two-dimensional array.
    **Returns:**
    - `np.ndarray`: The rows of `A` that also occur in `B`.
    """
    return A[np.where((A[:,None,:]==B[None,:,:]).all(axis=2))[0]]

def same_rows(A: ArrayLike,B: ArrayLike)->bool:
    """
    **Description:**
    Determines whether `A` and `B` contain the same rows with the same multiplicities, independent of row order.
    **Arguments:**
    - `A (ArrayLike)`: First two-dimensional array.
    - `B (ArrayLike)`: Second two-dimensional array.
    **Returns:**
    - `bool`: `True` if `A` and `B` have the same rows with the same multiplicities, otherwise `False`.
    """
    rowsA, countsA=np.unique(A,axis=0,return_counts=True)
    rowsB, countsB=np.unique(B,axis=0,return_counts=True)
    return np.array_equal(rowsA,rowsB) and np.array_equal(countsA,countsB)
# ---- Obsolete: superseded by hodge_diamond_stringy / hodge_diamond_cy4 / F_Theory_Uplift.hodge_numbers_CY (kept as comments for reference) ----
# def dual_face_Cayley_polytope(Cdvert: np.ndarray,f: PolytopeFace)->Polytope:
#     """
#     **Description:**
#     Computes the face of the dual Cayley polytope whose vertices pair trivially with all vertices of the input face `f`.
#     **Arguments:**
#     - `Cdvert (np.ndarray)`: Vertices of the dual Cayley polytope, one per row.
#     - `f (PolytopeFace)`: A face of the Cayley polytope.
#     **Returns:**
#     - `Polytope`: The dual face.
#     """
#     return Polytope(Cdvert[np.all(Cdvert@f.vertices().T==0,axis=1)])
# def h11_2_part(Cay: Polytope,Cayd: Polytope,det: bool=False)->int:
#     """
#     **Description:**
#     Computes the Hodge number `h^{1,1}` of a complete intersection Calabi-Yau described by a two-part nef partition using the associated Cayley polytope and its dual.
#     **Arguments:**
#     - `Cay (Polytope)`: The Cayley polytope.
#     - `Cayd (Polytope)`: The dual Cayley polytope.
#     - `det (bool)`: Whether to print intermediate contributions. Defaults to `False`.
#     **Returns:**
#     - `int`: The Hodge number `h^{1,1}`.
#     """
#     Cdvert=Cayd.vertices()
#     n=Cay.dim()-1
#     h11_ret=len(Cayd.points())-n-2
#     if det:
#         print("Trivial term: ",h11_ret)
#     for f in Cayd.faces(n):
#         h11_ret=h11_ret-len(Polytope(2*(f.vertices())).interior_points())
#     if det:
#         print("After 2*dual facets: ",h11_ret)
#     for f in Cayd.faces(n-1):
#         h11_ret=h11_ret+len(f.interior_points())
#     if det:
#         print("After dual codim-2: ",h11_ret)
#     for f in Cay.faces(1):
#         k=len(f.interior_points())
#         if k>0:
#             h11_ret=h11_ret+(k*(len(Polytope(2*(dual_face_Cayley_polytope(Cdvert,f).vertices())).interior_points())))
#     if det: 
#         print("After 1-face/2*codim-2 dual face: ",h11_ret)
#     for f in Cay.faces(2):
#         k=len(Polytope(2*(f.vertices())).interior_points())
#         if k>0:
#             h11_ret=h11_ret-(k*len(dual_face_Cayley_polytope(Cdvert,f).interior_points()))
#     if det:
#         print("After 2*(2-face)/codim-3 dual face: ",h11_ret)
#     for f in Cay.faces(3):
#         k=len(Polytope(2*(f.vertices())).interior_points())
#         if k>0:
#             h11_ret=h11_ret+(k*len(dual_face_Cayley_polytope(Cdvert,f).interior_points()))
#         k=len(dual_face_Cayley_polytope(Cdvert,f).interior_points())
#         if k>0:
#             for g in f.faces(2):
#                 h11_ret=h11_ret-(len(g.interior_points())*k)
#     return h11_ret
# def h21_2_part(Cay: Polytope,Cayd: Polytope,det: bool=False)->int:
#     """
#     **Description:**
#     Computes the Hodge number `h^{2,1}` of a complete intersection Calabi-Yau described by a two-part nef partition in a six-dimensional ambient variety.
#     **Arguments:**
#     - `Cay (Polytope)`: The Cayley polytope.
#     - `Cayd (Polytope)`: The dual Cayley polytope.
#     - `det (bool)`: Whether to print intermediate contributions. Defaults to `False`.
#     **Returns:**
#     - `int`: The Hodge number `h^{2,1}`.
#     """
#     Cdvert=Cayd.vertices()
#     n=Cay.dim()-1
#     h21_ret=0
#     for f in Cay.faces(2):
#         h21_ret=h21_ret+len(Polytope(2*(dual_face_Cayley_polytope(Cdvert,f).vertices())).interior_points())*len(f.interior_points())
#     if det:
#         print(h21_ret)
#     for f in Cay.faces(4):
#         h21_ret=h21_ret+len(Polytope(2*(f.vertices())).interior_points())*len(dual_face_Cayley_polytope(Cdvert,f).interior_points())
#     if det:
#         print(h21_ret)
#     for f in Cay.faces(3):
#         k=len(dual_face_Cayley_polytope(Cdvert,f).interior_points())
#         if k>0:
#             for g in f.faces(2):
#                 h21_ret=h21_ret-len(g.interior_points())*k
#     if det: 
#         print(h21_ret)
#     for f in Cay.faces(4):
#         k=len(dual_face_Cayley_polytope(Cdvert,f).interior_points())
#         if k>0:
#             for g in f.faces(3):
#                 h21_ret=h21_ret-len(g.interior_points())*k
#     return h21_ret
# ---- End of obsolete block ----

def get_indices(arr: np.ndarray,targets: ArrayLike)->np.ndarray:
    """
    **Description:**
    Finds the indices of rows of `arr` that also occur among the rows of `targets`.
    **Arguments:**
    - `arr (np.ndarray)`: The two-dimensional array to search.
    - `targets (ArrayLike)`: The target row (one-dimensional) or rows (two-dimensional).
    **Returns:**
    - `np.ndarray`: The indices of matching rows in `arr`.
    """
    return  np.where(np.any((arr[:,None,:]==targets).all(axis=2),axis=1))[0]

def get_index(arr: np.ndarray,target: ArrayLike)->np.ndarray:
    """
    **Description:**
    Finds the indices of rows of `arr` equal to the one-dimensional array `target`.
    **Arguments:**
    - `arr (np.ndarray)`: The two-dimensional array to search.
    - `target (ArrayLike)`: The target row.
    **Returns:**
    - `np.ndarray`: The indices of rows of `arr` equal to `target`.
    """
    return  np.where(np.all(arr==target,axis=1))[0]

@overload
def snf(Q: ArrayLike,return_decomposition: Literal[False]=...)->np.ndarray: ...

@overload
def snf(Q: ArrayLike,return_decomposition: Literal[True])->tuple[np.ndarray,np.ndarray,np.ndarray]: ...

@overload
def snf(Q: ArrayLike,return_decomposition: bool=...)->np.ndarray|tuple[np.ndarray,np.ndarray,np.ndarray]: ...

def snf(Q: ArrayLike,return_decomposition: bool=False)->np.ndarray|tuple[np.ndarray,np.ndarray,np.ndarray]:
    """
    **Description:**
    Computes the Smith normal form of an integer matrix.
    **Arguments:**
    - `Q (ArrayLike)`: The integer matrix.
    - `return_decomposition (bool)`: Whether to return the decomposition (computed with sympy, since FLINT provides no SNF transforms). Defaults to `False`.
    **Returns:**
    - `np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]`: The Smith normal form (computed with FLINT), or, if `return_decomposition=True`, the triple `(A, S, T)` returned by sympy's `smith_normal_decomp`, with `A = S @ Q @ T`.
    """
    if return_decomposition:
        A,S,T=smith_normal_decomp(Matrix(Q))
        return np.array(A,dtype=int),np.array(S,dtype=int),np.array(T,dtype=int)
    return _fmpz_to_np(_np_to_fmpz(Q).snf())

@overload
def hnf(Q: ArrayLike,return_transform: Literal[False]=...)->np.ndarray: ...

@overload
def hnf(Q: ArrayLike,return_transform: Literal[True])->tuple[np.ndarray,np.ndarray]: ...

@overload
def hnf(Q: ArrayLike,return_transform: bool=...)->np.ndarray|tuple[np.ndarray,np.ndarray]: ...

def hnf(Q: ArrayLike,return_transform: bool=False)->np.ndarray|tuple[np.ndarray,np.ndarray]:
    """
    **Description:**
    Computes the (row-style) Hermite normal form H of an integer matrix with FLINT.
    **Arguments:**
    - `Q (ArrayLike)`: The integer matrix.
    - `return_transform (bool)`: Whether to also return the unimodular matrix U with U @ Q = H. Defaults to `False`.
    **Returns:**
    - `np.ndarray | tuple[np.ndarray, np.ndarray]`: H, or the pair (H, U).
    """
    if return_transform:
        H,U=_np_to_fmpz(Q).hnf(transform=True)
        return _fmpz_to_np(H),_fmpz_to_np(U)
    return _fmpz_to_np(_np_to_fmpz(Q).hnf())

def _np_to_fmpz(Q: ArrayLike)->fmpz_mat:
    """
    **Description:**
    Converts an integer matrix into a FLINT `fmpz_mat`.
    **Arguments:**
    - `Q (ArrayLike)`: The two-dimensional integer matrix.
    **Returns:**
    - `fmpz_mat`: The same matrix as a FLINT integer matrix.
    """
    Q=np.asarray(Q,dtype=int)
    return fmpz_mat(*Q.shape,Q.ravel().tolist())

def _fmpz_to_np(M: fmpz_mat)->np.ndarray:
    """
    **Description:**
    Converts a FLINT `fmpz_mat` into a two-dimensional integer NumPy array.
    **Arguments:**
    - `M (fmpz_mat)`: The FLINT integer matrix.
    **Returns:**
    - `np.ndarray`: The same matrix as an integer array of shape `(M.nrows(), M.ncols())`.
    """
    return np.array(M.tolist(),dtype=int).reshape(M.nrows(),M.ncols())

def integral_kernel(M: ArrayLike)->np.ndarray:
    """
    **Description:**
    Computes a saturated, LLL-reduced integral basis of the kernel {x in Z^n : M x = 0}, using the Hermite normal form (via `integral_nullspace`) and LLL from FLINT.
    **Arguments:**
    - `M (ArrayLike)`: The integer matrix whose kernel is wanted.
    **Returns:**
    - `np.ndarray`: The kernel basis, one vector per row.
    """
    K=integral_nullspace(np.asarray(M,dtype=int)).T
    if K.shape[0]==0:
        return K.astype(int)
    return np.array(fmpz_mat([[int(x) for x in row] for row in K.tolist()]).lll().tolist(),dtype=int)

def glsm_from_points(pts: ArrayLike)->np.ndarray:
    """
    **Description:**
    Computes an integral basis of linear relations among the input toric points, i.e. the saturated integral kernel of `pts.T`.
    **Arguments:**
    - `pts (ArrayLike)`: The toric point configuration, one point per row.
    **Returns:**
    - `np.ndarray`: A GLSM charge matrix for the point configuration, one relation per row.
    """
    return integral_kernel(np.asarray(pts).T)

def points_from_glsm(glsm: ArrayLike)->np.ndarray:
    """
    **Description:**
    Computes an integral point configuration whose relations are described by the input GLSM charge matrix, i.e. the saturated integral kernel of `glsm`.
    **Arguments:**
    - `glsm (ArrayLike)`: The GLSM charge matrix, one relation per row.
    **Returns:**
    - `np.ndarray`: A toric point configuration, one point per row.
    """
    return integral_kernel(glsm).T
####Make one function accepting both a polytope and a list of vertices
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


# def find_trilayer_vertex_polytope(p: Polytope,as_index: bool=False)->np.ndarray|np.intp:
#     """
#     **Description:**
#     Uses the GLSM charge matrix of the vertices to identify the vertex corresponding to half the anticanonical class.
#     **Arguments:**
#     - `p (Polytope)`: The trilayer polytope.
#     - `as_index (bool)`: Whether to return the vertex index rather than the vertex coordinates. Defaults to `False`.
#     **Returns:**
#     - `np.ndarray | np.intp`: The distinguished vertex, or its index in `p.vertices()` if `as_index=True`.
#     """
#     glsm_vert=glsm_from_points(p.vertices())
#     half_anticanon=np.sum(glsm_vert,axis=1)//2
#     index=get_indices(glsm_vert.T,np.array([half_anticanon]))[0]
#     if as_index:
#         return index
#     else:
#         return p.vertices()[index]

# def find_trilayer_vertex_vertices(V: np.ndarray,as_vertex_index: bool=False)->np.ndarray|np.intp:
#     """
#     **Description:**
#     Uses the GLSM charge matrix of the vertex set to identify the vertex corresponding to half the anticanonical class.
#     **Arguments:**
#     - `V (np.ndarray)`: The vertices of a trilayer polytope, one per row.
#     - `as_vertex_index (bool)`: Whether to return the vertex index rather than the vertex coordinates. Defaults to `False`.
#     **Returns:**
#     - `np.ndarray | np.intp`: The distinguished vertex, or its row index in `V` if `as_vertex_index=True`.
#     """
#     glsm_vert=glsm_from_points(V)
#     half_anticanon=np.sum(glsm_vert,axis=1)//2
#     index=get_indices(glsm_vert.T,np.array([half_anticanon]))[0]
#     if as_vertex_index:
#         return index
#     else:
#         return V[index]

def trilayer_normal_form(p: Polytope)->Polytope:
    """
    **Description:**
    Applies an integral change of basis that moves the distinguished trilayer vertex into a standard position.
    **Arguments:**
    - `p (Polytope)`: The trilayer polytope.
    **Returns:**
    - `Polytope`: The polytope in trilayer normal form.
    """
    verts=p.vertices()
    index=find_trilayer_vertex(verts,as_index=True)
    verts[[0,index]]=verts[[index,0]]
    a,s,t=snf(verts,return_decomposition=True)
    b=np.ones(len(verts),dtype=int)
    b[0]=-1
    c=s@b
    y=np.zeros(a.shape[1],dtype=int)
    for ii in range(len(y)):
        if a[ii][ii]!=0:
            y[ii]=c[ii]/a[ii][ii]
    r=t@y
    a2,s2,t2=snf(r[:,None],return_decomposition=True)
    U0=np.round(np.linalg.inv(s2).T).astype(int)
    M=U0@verts.T
    for i in range(1,p.ambient_dim()):
        U0[i]=U0[i]+M[i,0]*U0[0]
    return Polytope((U0@verts.T).T)
###Make lowercase Newton_Polytope->newton_polytope

def Newton_Polytope(pts: ArrayLike,weights: ArrayLike)->h_polytope.HPolytope:
    """
    **Description:**
    Constructs the Newton polytope associated with a toric divisor with coefficient vector `weights` on the fan with rays `pts`.
    **Arguments:**
    - `pts (ArrayLike)`: Matrix whose rows are the rays of the toric fan.
    - `weights (ArrayLike)`: Divisor coefficients in the toric prime divisor basis.
    **Returns:**
    - `h_polytope.HPolytope`: The Newton polytope of the divisor (a `Polytope` subclass defined by inequalities).
    """
    return h_polytope.HPolytope(np.column_stack([pts,weights]).astype(int))

def row_difference(A: np.ndarray,B: np.ndarray)->np.ndarray:
    """
    **Description:**
    Computes all rows of `A` that do not occur as rows of `B`.
    **Arguments:**
    - `A (np.ndarray)`: First two-dimensional array.
    - `B (np.ndarray)`: Second two-dimensional array.
    **Returns:**
    - `np.ndarray`: Rows of `A` that are not rows of `B`.
    """
    return A[~np.any((A[:,None,:]==B[None,:,:]).all(axis=2),axis=1)]
##Obsolete I think

def points_not_interior_to_facets_and_codim2_faces(p: Polytope)->np.ndarray:
    """
    **Description:**
    Computes the lattice points of a polytope after removing points interior to facets and codimension-two faces.
    **Arguments:**
    - `p (Polytope)`: The polytope.
    **Returns:**
    - `np.ndarray`: The selected lattice points, one per row.
    """
    pts=p.points_not_interior_to_facets()
    for f in p.faces(p.dim()-2):
        if len(f.interior_points())>0:
            pts=np.delete(pts,get_indices(pts,f.interior_points()),axis=0)
    return pts

def get_lower_dimensional_cones(cones: Iterable[Iterable[int]],d: int)->list[tuple[int,...]]:
    """
    **Description:**
    Computes all `d`-element faces of the given maximal cones.
    **Arguments:**
    - `cones (Iterable[Iterable[int]])`: Collection of cones, each represented by a tuple of one-indexed ray labels.
    - `d (int)`: Number of rays in the lower-dimensional cones to extract.
    **Returns:**
    - `list[tuple[int, ...]]`: The list of distinct `d`-ray cones (in no particular order).
    """
    return list({combo for row in cones for combo in combinations(row,d)})

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

def O3O7_line_bundle(pts_CY_ambient: np.ndarray,q: ArrayLike,rescalings: ArrayLike)->np.ndarray|None:
    """
    **Description:**
    Determines the divisor coefficients of the orientifold line bundle by selecting a projected-in monomial of the Calabi-Yau hypersurface Newton polytope and rescaling divisor classes under the orbifold map.
    **Arguments:**
    - `pts_CY_ambient (np.ndarray)`: Rays of the original Calabi-Yau ambient toric fan, one per row.
    - `q (ArrayLike)`: Integer vector defining the `Z_2` action.
    - `rescalings (ArrayLike)`: Rescalings of toric divisor classes under the orbifold map.
    **Returns:**
    - `np.ndarray | None`: The O3/O7 line-bundle coefficients, or `None` if no projected-in monomial is found.
    """
    CY3_equation_newton_polytope=Newton_Polytope(pts_CY_ambient,[1]*len(pts_CY_ambient))
    projected_in_monomial_indices=np.where(np.mod(CY3_equation_newton_polytope.points()@q,2)==1)[0]
    if len(projected_in_monomial_indices)==0:
        return None
    arbitrary_monomial_point=CY3_equation_newton_polytope.points()[projected_in_monomial_indices[0]]
    line_bundle_weights_CYhypersurface=pts_CY_ambient@arbitrary_monomial_point+1
    line_bundle_weights_FtheoryBase=np.rint(line_bundle_weights_CYhypersurface/rescalings).astype(int)
    return line_bundle_weights_FtheoryBase

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

def inequivalent_Z2_actions(lattice_symmetries: Sequence[np.ndarray]|np.ndarray)->np.ndarray:
    """
    **Description:**
    Enumerates half-integer lattice points defining `Z_2` torus actions modulo the action of the supplied lattice symmetry group.
    **Arguments:**
    - `lattice_symmetries (Sequence[np.ndarray] | np.ndarray)`: List or three-dimensional array of square integer matrices acting from the left (e.g. `Polytope.automorphisms(action="left")`).
    **Returns:**
    - `np.ndarray`: Inequivalent integer representatives `q` (one per row, entries in {0, 1}) such that `q/2` defines a `Z_2` action.
    """
    dim=lattice_symmetries[0].shape[0]
    t_possibilities={t for t0 in combinations_with_replacement([0,1],dim) for t in permutations(t0)}
    t_possibilities=[t for t in t_possibilities]
    t_possibilities=np.delete(t_possibilities,t_possibilities.index(tuple([0]*dim)),0)
    inequivalent_t_possibilities={frozenset([tuple(y) for y in x]) for x in  np.transpose(np.array([np.mod(s@(t_possibilities.T),2) for s in lattice_symmetries]),[2,0,1])}
    inequivalent_t_possibilities=np.array([[y for y in x][0] for x in inequivalent_t_possibilities])
    return inequivalent_t_possibilities
#lower case

def linebundle_weights_from_Newton_Polytope(vectors: np.ndarray,Newton_polytope: Polytope)->np.ndarray:
    """
    **Description:**
    Recovers the toric divisor coefficients whose Newton polytope is `Newton_polytope` by maximizing the corresponding inequalities over its lattice points.
    **Arguments:**
    - `vectors (np.ndarray)`: Rays of the toric fan, one per row.
    - `Newton_polytope (Polytope)`: The Newton polytope.
    **Returns:**
    - `np.ndarray`: The divisor coefficient vector.
    """
    return np.max(-(vectors@Newton_polytope.points().T),axis=1)
#whats the diff?

def is_Gorenstein_full_dim(cone: Cone)->tuple[bool,np.ndarray|None]:
    """
    **Description:**
    Tests whether the cone is Gorenstein and returns its Gorenstein vector. The vector is obtained from a least-squares solution of `M @ n = 1` over the extremal rays `M`, so this is only reliable for full-dimensional cones.
    **Arguments:**
    - `cone (Cone)`: The cone to test.
    **Returns:**
    - `tuple[bool, np.ndarray | None]`: A pair `(is_gorenstein, n)`, where `n` is the integral Gorenstein vector with `<ray, n> = 1` for all extremal rays if the cone is Gorenstein, otherwise `None`.
    """
    M=np.asarray(cone.extremal_rays(),dtype=np.int64)
    n=np.rint(np.linalg.lstsq(M,np.ones(len(M)),rcond=None)[0]).astype(np.int64)
    return (True,n) if np.all(M@n==1) else (False,None)

def is_Gorenstein(cone: Cone)->tuple[bool,np.ndarray|None]:
    """
    **Description:**
    Tests whether a cone is Gorenstein and returns a Gorenstein vector, i.e. an integral vector `n` with `<ray, n> = 1` for all extremal rays. The integral linear system is solved exactly by successive unimodular column operations.
    **Arguments:**
    - `cone (Cone)`: The cone to test.
    **Returns:**
    - `tuple[bool, np.ndarray | None]`: A pair `(is_gorenstein, n)`, where `n` is a Gorenstein vector if the cone is Gorenstein, otherwise `None`.
    """
    M=np.asarray(cone.extremal_rays(),dtype=int)
    n=np.zeros(M.shape[1],dtype=int)
    B=np.eye(M.shape[1],dtype=int)
    for a in M:
        c, b=a@B, int(1-a@n)
        if not np.any(c):
            if b:
                return False, None
            continue
        for j in range(1,len(c)):
            if c[j]:
                x, y=int(c[0]), int(c[j])
                g=math.gcd(x,y)
                u=pow(x//g,-1,abs(y//g))
                v=(g-u*x)//y
                T=np.array([[u,-y//g],[v,x//g]],dtype=int)
                B[:,[0,j]]=B[:,[0,j]]@T
                c[0], c[j]=g, 0
        g=int(c[0])
        if b%g:
            return False, None
        n+=B[:,0]*(b//g)
        B=B[:,1:]
    return True, n

def is_reflexive_Gorenstein(cone: Cone)->bool:
    """
    **Description:**
    Determines whether both the cone and its dual are Gorenstein.
    **Arguments:**
    - `cone (Cone)`: The cone to test.
    **Returns:**
    - `bool`: `True` if the cone is reflexive Gorenstein, otherwise `False`.
    """
    if is_Gorenstein(cone)[0]:
        dual_cone=cone.dual()
        if is_Gorenstein(dual_cone)[0]:
            return True
    return False
#lower_case

def Gorenstein_index(cone: Cone)->np.integer:
    """
    **Description:**
    Computes the pairing of the Gorenstein generators of a reflexive Gorenstein cone and its dual.
    **Arguments:**
    - `cone (Cone)`: The cone to test.
    **Raises:**
    - `ValueError`: Raised if the cone is not reflexive Gorenstein.
    **Returns:**
    - `np.integer`: The Gorenstein index.
    """
    if is_reflexive_Gorenstein(cone):
        dual_cone=cone.dual()
        return is_Gorenstein(cone)[1]@is_Gorenstein(dual_cone)[1]
    raise ValueError("Cone is not reflexive Gorenstein")
#lower_case

def Cartier_index(toric_fan: Fan,weights: ArrayLike)->int|None:
    """
    **Description:**
    Computes the Cartier index of a Q-Cartier toric divisor, i.e. the smallest positive integer `k` such that `k` times the divisor is Cartier.
    Returns None if the divisor is not Q-Cartier.
    **Arguments:**
    - `toric_fan (Fan)`: The toric fan.
    - `weights (ArrayLike)`: Divisor coefficients in the toric prime divisor basis.
    **Returns:**
    - `int|None`: The Cartier index, or `None` if the divisor is not Q-Cartier.
    """
    weights=np.asarray(weights)
    indices=[]
    for c in toric_fan.cones():
        A=toric_fan.vectors(c)
        b=weights[np.asarray(c)-1]
        y=-np.linalg.lstsq(A,b,rcond=None)[0]
        if not np.allclose(A@y,-b,rtol=0,atol=1e-10):
            return None
        k=1
        while not np.allclose(k*y,np.rint(k*y),rtol=0,atol=1e-10):
            k+=1
        indices.append(k)
    return math.lcm(*indices)
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

def is_effective(points: ArrayLike,weights: ArrayLike)->bool:
    """
    **Description:**
    Determines whether the Newton polytope of a divisor has at least one lattice point.
    **Arguments:**
    - `points (ArrayLike)`: Rays of the toric fan.
    - `weights (ArrayLike)`: Divisor coefficients in the toric prime divisor basis.
    **Returns:**
    - `bool`: `True` if the divisor has a nonzero section, otherwise `False`.
    """
    try:
        NP=Newton_Polytope(points,weights)
        if len(NP.points())>0:
            return True
        return False
    except ValueError:
        return False

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

def generic_section_factorizes(points: np.ndarray,linebundle_weights: np.ndarray)->bool|np.bool_:
    """
    **Description:**
    Determines whether the generic section of a toric divisor factorizes by testing whether every toric coordinate appears nontrivially in some section.
    **Arguments:**
    - `points (np.ndarray)`: Rays of the toric fan.
    - `linebundle_weights (np.ndarray)`: Divisor coefficients in the toric prime divisor basis.
    **Returns:**
    - `bool|np.bool_`: `True` if the generic section factorizes (also if the divisor has no sections), otherwise `False`.
    """
    try:
        NP=Newton_Polytope(points,linebundle_weights)
    except ValueError:
        return True
    return~np.all(np.any(points@NP.points().T+linebundle_weights[:,None]!=0,axis=1))
#obsolete

def attempt_to_make_nef(toric_variety: Fan,line_bundle: ArrayLike,epsilon: float=1e-5)->Fan:
    """
    **Description:**
    Perturbs the triangulation heights in the direction of the given line bundle in order to find a triangulation for which the line bundle is nef.
    **Arguments:**
    - `toric_variety (Fan)`: A regular triangulation (toric fan) of a vector configuration; its heights `toric_variety.heights()` and vector configuration `toric_variety.vc` are used.
    - `line_bundle (ArrayLike)`: Divisor coefficients in the toric prime divisor basis.
    - `epsilon (float)`: Magnitude of the perturbation used to obtain a triangulation rather than a subdivision. Defaults to `1e-5`.
    **Returns:**
    - `Fan`: A triangulation of the vector configuration.
    """
    line_bundle=np.array(line_bundle)
    hts0=toric_variety.heights()+epsilon*np.array([(np.sin(i+1)+1) for i in range(len(line_bundle))])
    hts1=line_bundle
    return toric_variety.vc.triangulate(heights=hts1/epsilon+hts0)
#Make one function

def basis(points: np.ndarray)->list[int]:
    """
    **Description:**
    Returns indices of rows of `points` forming a basis for the row span.
    **Arguments:**
    - `points (np.ndarray)`: Matrix whose rows are candidate basis vectors.
    **Raises:**
    - `ValueError`: Raised if no basis can be found.
    **Returns:**
    - `list[int]`: Indices of basis rows.
    """
    n=points.shape[0]
    d=np.linalg.matrix_rank(points)
    basis_indices=[]
    for i in range(n):
        test_indices=basis_indices+[i]
        if np.linalg.matrix_rank(points[test_indices])==len(test_indices):
            basis_indices.append(i)
        if len(basis_indices)==d:
            return basis_indices
    raise ValueError("No basis could be found")

def basis_H2_toric_fan(toric_fan: Fan)->np.ndarray:
    """
    **Description:**
    Finds a smooth maximal cone and returns the complementary ray labels, giving a convenient GLSM or curve-homology basis.
    **Arguments:**
    - `toric_fan (Fan)`: The toric fan.
    **Raises:**
    - `ValueError`: Raised if no smooth maximal cone is found.
    **Returns:**
    - `np.ndarray`: One-indexed ray labels forming the basis.
    """
    for c in toric_fan.cones():
        if Cone(toric_fan.vectors(c)).is_smooth():
            mask=np.ones(len(toric_fan.vectors()),dtype=bool)
            mask[np.array(c)-1]=False
            basis=np.arange(1,len(mask)+1)[mask]
            return basis
    raise ValueError("No basis could be found")

def cone_indices(fan: Fan,cones: Iterable[Iterable[int]])->list[int]:
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

def attempt_to_make_Cartier(tri: Fan,D: np.ndarray)->tuple[Fan,np.ndarray]:
    """
    **Description:**
    If the divisor is not Cartier on the given fan, this function adds rays from the Newton-polytope inequalities and attempts to triangulate the refined configuration so that the divisor becomes Cartier and nef.
    **Arguments:**
    - `tri (Fan)`: The initial toric fan.
    - `D (np.ndarray)`: Divisor coefficients in the toric prime divisor basis.
    **Returns:**
    - `tuple[Fan,np.ndarray]`: A pair `(new_fan, new_D)` consisting of the refined fan and the updated divisor coefficients (the inputs unchanged if `D` is already Cartier).
    """
    if is_Cartier(tri,D)[0]:
        return (tri,D)
    else:
        NP=Newton_Polytope(tri.vectors(),D)
        inequalities=NP.inequalities()
        new_points=row_difference(inequalities[:,:-1],tri.vectors())
        indices_new_points=get_indices(inequalities[:,:-1],new_points)
        new_vectors=np.concatenate((tri.vectors(),new_points),axis=0)
        new_vc=VectorConfiguration(new_vectors)
        new_D=np.concatenate((D,inequalities[:,-1][indices_new_points]))
        tri_Cartier=attempt_to_make_nef(new_vc.triangulate(),new_D)
        return (tri_Cartier,new_D)
#make one function

def BL(fan: Fan,lb: np.ndarray)->list[tuple[int,...]]:
    """
    **Description:**
    Computes sections of the divisor with weights `lb` and returns the corresponding base locus in the given fan.
    **Arguments:**
    - `fan (Fan)`: The toric fan.
    - `lb (np.ndarray)`: Divisor coefficients in the toric prime divisor basis.
    **Returns:**
    - `list[tuple[int,...]]`: Cones (tuples of one-indexed ray labels) defining the base locus.
    """
    return base_locus(sections(fan.vectors(),lb),cones=fan.cones())

def base_locus(sections: np.ndarray,cones: Sequence[tuple[int,...]]|None=None,dim: int=4)->list[tuple[int,...]]:
    """
    **Description:**
    Computes minimal coordinate strata on which all sections vanish. If cones are provided, the search is restricted to strata of the corresponding toric fan.
    **Arguments:**
    - `sections (np.ndarray)`: Section exponent matrix (rows: toric coordinates, columns: sections).
    - `cones (Sequence[tuple[int,...]]|None)`: Cones of the toric fan, as tuples of one-indexed ray labels. If `None`, all coordinate strata up to dimension `dim` are considered. Defaults to `None`.
    - `dim (int)`: Dimension used when `cones=None`. Defaults to `4`.
    **Returns:**
    - `list[tuple[int,...]]`: Cones (tuples of one-indexed ray labels) defining the base locus.
    """
    num_coords,num_sections=sections.shape
    B=sections>0 
    minimal_hitting_sets=[]
    if type(cones)==type(None):
        for codim in range(1,dim+1):
            for combo in combinations(range(num_coords),codim):
                combo_set=set(combo)
                if any(set(mhs).issubset(combo_set) for mhs in minimal_hitting_sets):
                    continue
                if B[list(combo),:].any(axis=0).all():
                    minimal_hitting_sets.append(combo)            
        return [tuple(x+1 for x in mhs) for mhs in minimal_hitting_sets]
    else:
        for codim in range(1,len(cones[0])+1):
            for combo in get_lower_dimensional_cones(cones,codim):
                combo_set=set(combo)
                if any(set(mhs).issubset(combo_set) for mhs in minimal_hitting_sets):
                    continue
                if B[np.array(combo)-1,:].any(axis=0).all():
                    minimal_hitting_sets.append(combo)            
        return [mhs for mhs in minimal_hitting_sets]

@overload
def normal_fan(polytopes: Polytope|list[Polytope],inequalities: ArrayLike|None=...,maximal_refinement: Literal[False]=...,triangulate_refinement: bool=...,return_unrefined_fan: bool=...)->tuple[Fan,np.ndarray,list[list[int]]]: ...

@overload
def normal_fan(polytopes: Polytope|list[Polytope],inequalities: ArrayLike|None=...,maximal_refinement: Literal[True]=...,triangulate_refinement: Literal[False]=...,return_unrefined_fan: Literal[False]=...)->tuple[np.ndarray,np.ndarray]|tuple[None,None]: ...

@overload
def normal_fan(polytopes: Polytope|list[Polytope],inequalities: ArrayLike|None=...,maximal_refinement: Literal[True]=...,triangulate_refinement: Literal[False]=...,return_unrefined_fan: Literal[True]=...)->tuple[np.ndarray,np.ndarray,Fan]|tuple[None,None,None]: ...

@overload
def normal_fan(polytopes: Polytope|list[Polytope],inequalities: ArrayLike|None=...,maximal_refinement: Literal[True]=...,triangulate_refinement: Literal[True]=...,return_unrefined_fan: Literal[False]=...)->tuple[Fan,np.ndarray]|tuple[None,None]: ...

@overload
def normal_fan(polytopes: Polytope|list[Polytope],inequalities: ArrayLike|None=...,maximal_refinement: Literal[True]=...,triangulate_refinement: Literal[True]=...,return_unrefined_fan: Literal[True]=...)->tuple[Fan,np.ndarray,Fan]|tuple[None,None,None]: ...

@overload
def normal_fan(polytopes: Polytope|list[Polytope],inequalities: ArrayLike|None=...,maximal_refinement: bool=...,triangulate_refinement: bool=...,return_unrefined_fan: bool=...)->tuple[Fan,np.ndarray,list[list[int]]]|tuple[np.ndarray,np.ndarray]|tuple[np.ndarray,np.ndarray,Fan]|tuple[Fan,np.ndarray]|tuple[Fan,np.ndarray,Fan]|tuple[None,None]|tuple[None,None,None]: ...

def normal_fan(polytopes: Polytope|list[Polytope],inequalities: ArrayLike|None=None,maximal_refinement: bool=False,triangulate_refinement: bool=False,return_unrefined_fan: bool=False)->tuple[Fan,np.ndarray,list[list[int]]]|tuple[np.ndarray,np.ndarray]|tuple[np.ndarray,np.ndarray,Fan]|tuple[Fan,np.ndarray]|tuple[Fan,np.ndarray,Fan]|tuple[None,None]|tuple[None,None,None]:
    """
    **Description:**
    Constructs the normal fan of a lattice polytope, or of the Minkowski sum of a list of lattice polytopes. Optionally constructs a maximal refinement subject to the specified inequalities.
    **Arguments:**
    - `polytopes (Polytope|list[Polytope])`: A lattice polytope, or a list of lattice polytopes whose Minkowski sum is used. The maximal refinement requires a list.
    - `inequalities (ArrayLike|None)`: Inequality data used for maximal refinement. Required if `maximal_refinement=True`. Defaults to `None`.
    - `maximal_refinement (bool)`: Whether to construct the maximal refinement. Defaults to `False`.
    - `triangulate_refinement (bool)`: Whether to triangulate the refined vector configuration. Defaults to `False`.
    - `return_unrefined_fan (bool)`: Whether to also return the unrefined normal fan. Defaults to `False`.
    **Raises:**
    - `Exception`: Raised if `maximal_refinement=True` but no inequalities are provided.
    **Returns:**
    - `tuple[Fan,np.ndarray,list[list[int]]]|tuple[np.ndarray,np.ndarray]|tuple[np.ndarray,np.ndarray,Fan]|tuple[Fan,np.ndarray]|tuple[Fan,np.ndarray,Fan]|tuple[None,None]|tuple[None,None,None]`: If `maximal_refinement=False`: `(fan, weights, cones)` with the (triangulated) normal fan, the line-bundle weights and its maximal cones as lists of one-indexed ray labels. Otherwise `(vectors, weights)` (refined ray vectors) if `triangulate_refinement=False`, or `(refined_fan, weights)` if `triangulate_refinement=True`, each followed by the unrefined normal fan if `return_unrefined_fan=True`; all entries are `None` if the inequalities cannot be satisfied.
    """
    if type(polytopes)==type([]):
        msum_vertices=nested_sum([p.vertices() for p in polytopes])
        p=Polytope(np.unique(flatten(msum_vertices,len(polytopes)-1),axis=0))
        vertex_split=np.array([np.array(np.where(np.all(np.array(msum_vertices)-v==0,axis=-1))).T[0] for v in p.vertices()])
    else:
        p=polytopes
        weights=p.inequalities().T[-1]
    hyperplane_saturations=[p.inequalities()[np.where(x==0)[0]] for x in (np.vstack([p.vertices().T,[1]*len(p.vertices())]).T@(p.inequalities().T))]
    normal_fan_edges=np.delete(p.inequalities().T,-1,0).T
    normal_fan_vc=VectorConfiguration(normal_fan_edges)
    cones=[[int(np.where(np.all(normal_fan_edges-x==0,axis=1))[0][0])+1 for x in np.delete(s.T,-1,0).T] for s in hyperplane_saturations]
    n_fan=Fan(vc=normal_fan_vc,cones=cones)
    if type(polytopes)==type([]):
        vertices_to_vertices_map=[vertex_split[np.where([i in c for c in cones])[0][0]] for i in range(1,len(normal_fan_edges)+1)]
        weights=np.array([-np.array([polytopes[j].vertices()[x] for j,x in enumerate(pointers)])@(normal_fan_edges[i]) for i,pointers in enumerate(vertices_to_vertices_map)])
    if not maximal_refinement:
        return (normal_fan_vc.triangulate(cells=cones),weights,cones)
    if type(inequalities)==type(None):
        raise Exception('Inequalities must be given to construct maximal refinement')
    inequalities=np.array(inequalities)
    n_vectors=n_fan.vectors()
    if np.max((1-weights[:,0])*inequalities[0]-weights[:,1])>=inequalities[-1]:
        if return_unrefined_fan:
            return (None,None,None)
        else:
            return (None,None)
    if np.min((1-weights[:,0])*inequalities[0]-weights[:,1])<0:
        if return_unrefined_fan:
            return (None,None,None)
        else:
            return (None,None)
    maximal_blow_ups=[h_polytope.HPolytope(np.vstack([[np.concatenate([np.delete(inequalities,-1,0)@np.array([polytopes[j].vertices()[x] for j,x in enumerate(vertex_split[i])]),[inequalities[-1]]])],np.vstack([(p.vertices()-m).T,[0]*len(p.vertices())]).T ])).points() for i,m in enumerate(p.vertices())]
    maximal_blow_ups=[np.unique([np.rint(x/np.gcd.reduce(x)).astype(int) for x in np.delete(b,np.where(np.all(b==0,axis=1))[0][0],0)],axis=0) for b in maximal_blow_ups]
    all_vectors=np.unique(np.array([y for x in maximal_blow_ups for y in x]),axis=0)
    all_weights=np.array([-np.array([(pol.vertices()[vertex_split[np.where([np.any(np.all(y-x==0,axis=-1)) for y in maximal_blow_ups])[0][0]][j]])@x for x in all_vectors]) for j,pol in enumerate(polytopes)])
    old_indices=np.where([type(n_fan.vc.vectors_to_labels(v))!=type(None) for v in all_vectors])[0]
    blow_up_weights=np.delete(all_weights.T,old_indices,0)
    blow_up_vectors=np.delete(all_vectors,old_indices,0)
    all_vectors=np.vstack([n_fan.vectors(),blow_up_vectors])
    all_weights=np.vstack([weights,blow_up_weights])
    if not triangulate_refinement:
        if return_unrefined_fan:
            return (all_vectors,all_weights,n_fan)
        else:
            return (all_vectors,all_weights)
    if return_unrefined_fan:
        return (refine_fan(make_simplicial(n_fan),all_vectors),all_weights,n_fan)
    else:
        return (refine_fan(make_simplicial(n_fan),all_vectors),all_weights)

def nested_sum(lists: Sequence[Iterable[Any]],depth: int=0,acc: Any=0)->Any:
    """
    **Description:**
    Recursively forms all sums obtained by choosing one element from each list in `lists`.
    **Arguments:**
    - `lists (Sequence[Iterable[Any]])`: A list of lists or arrays whose elements are to be summed.
    - `depth (int)`: Recursion depth. Defaults to `0`.
    - `acc (Any)`: Accumulated partial sum. Defaults to `0`.
    **Returns:**
    - `Any`: The nested list of sums (nesting depth `len(lists)-depth`), or the accumulated sum `acc` if `depth==len(lists)`.
    """
    if depth==len(lists):
        return acc
    return [nested_sum(lists,depth+1,acc+x) for x in lists[depth]]

def flatten(lst: list[Any],depth: int)->list[Any]:
    """
    **Description:**
    Recursively flattens a nested list by the specified number of levels.
    **Arguments:**
    - `lst (list[Any])`: The nested list to flatten.
    - `depth (int)`: Number of nesting levels to flatten.
    **Returns:**
    - `list[Any]`: The flattened list.
    """
    if depth==0:
        return lst
    result=[]
    for x in lst:
        if isinstance(x,list):
            result.extend(flatten(x,depth-1))
        else:
            result.append(x)
    return result
#obsolete

def O7_cones(vc_orbifold: Fan,O7_labels: Iterable[int],d: int)->list[tuple[int,...]]:
    """
    **Description:**
    Computes the `d`-ray cones of a toric fan whose rays are all contained in the set of O7-plane labels.
    **Arguments:**
    - `vc_orbifold (Fan)`: The toric fan of the orbifold.
    - `O7_labels (Iterable[int])`: One-indexed labels of the O7 divisors.
    - `d (int)`: Number of rays in the cones to consider.
    **Returns:**
    - `list[tuple[int,...]]`: Cones whose rays are contained in `O7_labels`.
    """
    d_cones=get_lower_dimensional_cones(vc_orbifold.cones(),d)
    relevant_d_cones=[t for t in d_cones if set(t).issubset(O7_labels)]
    return relevant_d_cones

def trilayer_5d_Ftheory_uplift(p: Polytope,verbosity: int=1)->Polytope:
    """
    **Description:**
    Constructs the five-dimensional polytope associated with the F-theory uplift of a trilayer orientifold in the limit where all mid-layer divisors are blown down.
    **Arguments:**
    - `p (Polytope)`: The reflexive trilayer polytope.
    - `verbosity (int)`: Verbosity level controlling printed blowdown information. Defaults to `1`.
    **Raises:**
    - `Exception`: Raised if `p` is not trilayer.
    **Returns:**
    - `Polytope`: The five-dimensional F-theory uplift polytope.
    """
    if not p.is_trilayer():
        raise Exception("Polytope is not trilayer.")
    p=trilayer_normal_form(p)
    mid_layer_points=np.where(p.points().T[0]==0)[0]
    mid_layer_points=np.array([i for i in set(mid_layer_points).intersection(set(p.points_to_indices(p.points_not_interior_to_facets())))])
    if len(mid_layer_points)>1 and verbosity>0:
        print(f"{len(mid_layer_points)-1} exceptional divisors have been blown down.")
    p_dim=p.dim()
    p3=Polytope(np.delete(p.points()[np.where(p.points().T[0]==1)[0]].T,0,0).T)
    p2KB=Newton_Polytope(p3.points(),len(p3.points())*[2])
    n_fan,wts,cns=normal_fan(p2KB)
    cone_hyperplanes=[Cone(n_fan.vectors(c)).dual().rays() for c in cns]
    blown_up=[h_polytope.HPolytope(np.vstack([np.vstack([h.T,[0]*len(h)]).T,[np.concatenate([p2KB.vertices()[i],[2]]),np.concatenate([-p2KB.vertices()[i],[-1]])]])).points() for i,h in enumerate(cone_hyperplanes)]
    blown_up=[[np.rint(x/np.gcd.reduce(x)).astype(int) for x in b] for b in blown_up]
    blown_up=[np.unique(b,axis=0) for b in blown_up]
    all_vecs=np.unique([i for b in blown_up for i in b],axis=0)
    full_vc=VectorConfiguration(all_vecs)
    monomials=Newton_Polytope(full_vc.vectors(),[2]*len(full_vc.vectors())).points()@(full_vc.vectors().T)+np.array([2]*len(full_vc.vectors()))
    O7pos=np.where(np.array([min(m) for m in monomials.T])==1)[0]
    vx=np.concatenate([[0]*(p_dim-1),[3,1]])
    vy=np.concatenate([[0]*(p_dim-1),[-2,-1]])
    vz=np.concatenate([[0]*(p_dim-1),[0,1]])
    uplift_vecs_singular=np.vstack([np.vstack([full_vc.vectors().T,[[0]*len(full_vc.vectors()),[1]*len(full_vc.vectors())]]).T,[vx,vy,vz]])
    D4res1=uplift_vecs_singular[O7pos]+vx+2*vy
    D4res2=2*uplift_vecs_singular[O7pos]+2*vx+3*vy
    uplift_vecs=np.vstack([uplift_vecs_singular,D4res1,D4res2])
    p5=Polytope(uplift_vecs)
    return p5

def sections(points: np.ndarray,divisor: np.ndarray|None=None,polytope: Polytope|None=None)->np.ndarray:
    """
    **Description:**
    Computes the exponent vectors of all monomial sections of the divisor with coefficient vector `divisor` on the fan with rays `points`.
    **Arguments:**
    - `points (np.ndarray)`: Rays of the toric fan.
    - `divisor (np.ndarray|None)`: Divisor coefficients in the toric prime divisor basis. If `None`, they are recovered from `polytope`. Defaults to `None`.
    - `polytope (Polytope|None)`: Newton polytope of the divisor. If `None`, it is computed from `points` and `divisor`. Defaults to `None`.
    **Raises:**
    - `ValueError`: Raised if neither `divisor` nor `polytope` is given.
    **Returns:**
    - `np.ndarray`: Matrix of section exponents (rows: toric coordinates, columns: sections), or an empty array if there are no sections.
    """
    if divisor is None:
        if polytope is None:
            raise ValueError("At least divisor or polytope must be defined")
        divisor=linebundle_weights_from_Newton_Polytope(points,polytope)
    else:
        if polytope is None:
            polytope=Newton_Polytope(points,divisor)
    if len(polytope.points())==0:
        return np.array([])
    else:
        return points@polytope.points().T+divisor[:,None]

def solve_over_integers(M: ArrayLike,b: ArrayLike)->tuple[bool,np.ndarray|None]:
    """
    **Description:**
    Uses Smith normal form to determine whether the equation `Mx=b` has an integral solution, and returns one if it exists.
    **Arguments:**
    - `M (ArrayLike)`: Integer matrix.
    - `b (ArrayLike)`: Integer inhomogeneous term.
    **Returns:**
    - `tuple[bool,np.ndarray|None]`: A pair `(has_solution, x)`, where `x` is an integral solution if one exists, otherwise `None`.
    """
    a,s,t=snf(M,return_decomposition=True)
    c=s@np.asarray(b,dtype=int)
    y=np.zeros(a.shape[1],dtype=int)
    for ii in range(np.min(a.shape)):
        if a[ii,ii]!=0:
            if c[ii]%a[ii,ii]!=0:
                return (False,None)
            y[ii]=c[ii]//a[ii,ii]
    if np.all(a@y==c):
        return (True,t@y)
    else:
        return (False,None)

def make_simplicial(fan: Fan)->Fan:
    """
    **Description:**
    Replaces non-simplicial cones by cones obtained from a fine triangulation of the corresponding vector configuration, leaving simplicial cones unchanged.
    **Arguments:**
    - `fan (Fan)`: The toric fan to refine.
    **Returns:**
    - `Fan`: A simplicial refinement of the fan.
    """
    new_cones=set(fan.cones())
    dim=fan.dim
    for c in fan.cones():
        if len(c)>dim:
            new_cones.discard(c)
            carr=np.array(c)
            vc_tri=VectorConfiguration(fan.vectors(c)).triangulate(make_fine=True)
            for co in vc_tri.cones():
                new_cones.add(tuple(carr[np.array(co)-1]))
    return Fan(vc=fan.vc,cones=new_cones)

def refine_fan(fan,blowups_or_all_vectors=None,tol=1e-10):
    if blowups_or_all_vectors is not None:
        blowups=row_difference(blowups_or_all_vectors,fan.vc.vectors())
        all_vectors=np.concatenate((fan.vc.vectors(),blowups),axis=0)
        vc=VectorConfiguration(all_vectors)
        new_fan=Fan(vc,cones=fan.cones())
    else:
        new_fan=fan
        vc=fan.vc
    vec_diff=row_difference(vc.vectors(),new_fan.vectors())
    to_be_refined=set(vc.vectors_to_labels(vec_diff))
    if len(to_be_refined)==0:
        return new_fan
    vectors=vc.vectors()
    dim=fan.dim
    full_dim=dim==vectors.shape[1]
    if vc.dim!=dim:
        raise ValueError("The new rays do not lie in the span of the fan.")
    cones=set(new_fan.cones())
    l2c={}
    for c in cones:
        for l in c:
            l2c.setdefault(l,set()).add(c)
    fast=all(len(c)==dim for c in cones)
    if fast:
        coords=vectors.astype(float) if full_dim else vectors@np.linalg.svd(vectors.astype(float))[2][:dim].T
        cone_list=list(cones)
        cone_idx={c:i for i,c in enumerate(cone_list)}
        try:
            inv=np.linalg.inv(coords[np.array(cone_list)-1].transpose(0,2,1))
        except np.linalg.LinAlgError:
            fast=False
        alive=np.ones(len(cone_list),dtype=bool)
    for label in to_be_refined:
        ray=vc.vectors(label)
        link_base=None
        if fast:
            alive_idx=np.flatnonzero(alive)
            hit=np.flatnonzero(np.all(inv[alive_idx]@coords[label-1]>=-tol,axis=1))
            if len(hit)>0:
                c=cone_list[alive_idx[hit[0]]]
                coeffs=np.linalg.solve(coords[np.array(c)-1].T,coords[label-1])
                link_base=tuple(l for l,x in zip(c,coeffs) if x>tol)
        elif full_dim:
            link_base=find_cone(ray,cones,vectors)
        else:
            link_base=find_cone_general(ray,cones,vectors)
        if link_base is None:
            raise ValueError(f"The ray {ray} is not contained in the support of the fan.")
        link_base=tuple(link_base)
        star=set.intersection(*(l2c[l] for l in link_base))
        new_cones=[]
        for c_old in star:
            cones.discard(c_old)
            for l in c_old:
                l2c[l].discard(c_old)
            rest=tuple(l for l in c_old if l not in link_base)
            for comb in combinations(link_base,len(link_base)-1):
                new_cones.append(tuple(sorted(comb+rest+(label,))))
        for c_new in new_cones:
            cones.add(c_new)
            for l in c_new:
                l2c.setdefault(l,set()).add(c_new)
        if fast:
            alive[[cone_idx.pop(c_old) for c_old in star]]=False
            for c_new in new_cones:
                cone_idx[c_new]=len(cone_list)
                cone_list.append(c_new)
            try:
                inv=np.concatenate((inv,np.linalg.inv(coords[np.array(new_cones)-1].transpose(0,2,1))))
                alive=np.concatenate((alive,np.ones(len(new_cones),dtype=bool)))
            except np.linalg.LinAlgError:
                fast=False
    return Fan(vc=vc,cones=cones)
# def refine_fan_old(fan: Fan,blowups_or_all_vectors: np.ndarray|None=None)->Fan:
#     """
#     **Description:**
#     Adds new rays to a fan and star-subdivides the cones containing them. If no new vectors are given, the vector configuration of the fan is used to detect rays not already present in the fan.
#     **Arguments:**
#     - `fan (Fan)`: The toric fan to refine.
#     - `blowups_or_all_vectors (np.ndarray|None)`: Blowup rays or a full vector configuration containing the old rays and new rays. Defaults to `None`.
#     **Returns:**
#     - `Fan`: The refined toric fan.
#     """
#     if blowups_or_all_vectors is not None:
#         blowups=row_difference(blowups_or_all_vectors,fan.vc.vectors())
#         all_vectors=np.concatenate((fan.vc.vectors(),blowups),axis=0)
#         vc_all=VectorConfiguration(all_vectors)
#         new_fan=Fan(vc_all,cones=fan.cones())
#     else: 
#         new_fan=fan
#     vec_diff=row_difference(new_fan.vc.vectors(),new_fan.vectors())
#     to_be_refined=set(new_fan.vc.vectors_to_labels(vec_diff))
#     if fan.dim==len(new_fan.vc.vectors()[0]):
#         for label in to_be_refined:
#             all_cones=set(new_fan.cones())
#             link_base=tuple(find_cone(new_fan.vc.vectors(label),all_cones,new_fan.vc.vectors()))
#             link_base_len=len(link_base)
#             for c in new_fan.link(link_base):
#                 all_cones.discard(tuple(sorted(link_base+c)))
#                 for comb in combinations(link_base,link_base_len-1):
#                     all_cones.add(tuple(sorted(comb+c+(label,))))
#             new_fan=Fan(vc=new_fan.vc,cones=all_cones)
#         return new_fan
#     else:
#         for label in to_be_refined:
#             all_cones=set(new_fan.cones())
#             link_base=tuple(find_cone_general(new_fan.vc.vectors(label),all_cones,all_vectors))
#             link_base_len=len(link_base)
#             for c in new_fan.link(link_base):
#                 all_cones.discard(tuple(sorted(link_base+c)))
#                 for comb in combinations(link_base,link_base_len-1):
#                     all_cones.add(tuple(sorted(comb+c+(label,))))
#             new_fan=Fan(vc=new_fan.vc,cones=all_cones)
#         return new_fan
#Make just one find_cone function that can accept both with option full_dim=True/False

def find_cone_general(new_ray: ArrayLike,current_cones: Iterable[tuple[int,...]],all_vectors: np.ndarray)->frozenset[int]|None:
    """
    **Description:**
    Searches the current cones (of any dimension, in any ambient space) for the first one containing `new_ray`, and returns the labels of its generators with strictly positive coefficient. Cones with linearly independent generators are tested all at once by a batched least-squares solve, cones with linearly dependent generators by non-negative least squares.
    For a simplicial cone the coefficients are unique, so the result is the carrier face of `new_ray`. For a non-simplicial cone the coefficients are not unique and the returned set can be a proper subset of the carrier face (e.g. a diagonal of a square face), which is then not a face of the fan.
    **Arguments:**
    - `new_ray (ArrayLike)`: The ray to locate.
    - `current_cones (Iterable[tuple[int,...]])`: Current cones, represented by tuples of one-indexed ray labels.
    - `all_vectors (np.ndarray)`: Full ray matrix.
    **Returns:**
    - `frozenset[int]|None`: The carrier face labels, or `None` if no carrier cone is found.
    """

    def nnls_face(cone):
        cone_list=list(cone)
        A=all_vectors[[idx-1 for idx in cone_list]].T
        x, residual=nnls(A,new_ray)
        if residual<1e-10:
            return frozenset(cone_list[i] for i,coeff in enumerate(x) if coeff>1e-10)
        return None
    cones=list(current_cones)
    for cone in cones[:20]:
        face=nnls_face(cone)
        if face is not None:
            return face
    cones=cones[20:]
    if len(cones)==0:
        return None
    b=np.asarray(new_ray,dtype=float)
    lengths=np.fromiter(map(len,cones),dtype=int,count=len(cones))
    independent=np.zeros(len(cones),dtype=bool)
    hit=np.zeros(len(cones),dtype=bool)
    coeffs={}
    for k in np.unique(lengths):
        if k>len(b):
            continue
        idx=np.flatnonzero(lengths==k)
        C=np.array(cones) if len(idx)==len(cones) else np.array([cones[i] for i in idx])
        A=all_vectors[C-1].transpose(0,2,1).astype(float)
        G=A.transpose(0,2,1)@A
        ind=np.linalg.det(G)>1e-9*np.prod(np.diagonal(G,axis1=1,axis2=2),axis=1)
        if not np.any(ind):
            continue
        idx,A,G=idx[ind],A[ind],G[ind]
        independent[idx]=True
        x=np.linalg.solve(G,(b@A)[...,None])[...,0]
        r=b-np.einsum('nij,nj->ni',A,x)
        cand=np.flatnonzero((np.linalg.norm(r,axis=1)<1e-6)&np.all(x>=-1e-6,axis=1))
        if len(cand)==0:
            continue
        A,G,x=A[cand],G[cand],x[cand]+np.linalg.solve(G[cand],np.einsum('nij,ni->nj',A[cand],r[cand])[...,None])[...,0]
        residual=np.linalg.norm(np.einsum('nij,nj->ni',A,x)-b,axis=1)
        ok=(residual<1e-10)&np.all(x>=-1e-10,axis=1)
        hit[idx[cand[ok]]]=True
        coeffs.update(zip(idx[cand[ok]],x[ok]))
    for i in np.flatnonzero(hit|~independent):
        if independent[i]:
            return frozenset(l for l,coeff in zip(cones[i],coeffs[i]) if coeff>1e-10)
        face=nnls_face(cones[i])
        if face is not None:
            return face
    return None

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

def array_to_latex(arr: np.ndarray,as_fractions: bool=False,max_denominator: int=10**6)->str:
    """
    **Description:**
    Converts a two-dimensional NumPy array into a LaTeX `pmatrix` string.
    **Arguments:**
    - `arr (np.ndarray)`: The two-dimensional array to convert.
    - `as_fractions (bool)`: Whether to write the entries as fractions `\\frac{p}{q}` instead of decimals. Defaults to `False`.
    - `max_denominator (int)`: Largest denominator used when approximating floating point entries by fractions. Defaults to `10**6`.
    **Raises:**
    - `ValueError`: Raised if `arr` has more than two dimensions.
    **Returns:**
    - `str`: A LaTeX `pmatrix` representation of the array.
    """
    if len(arr.shape)>2:
        raise ValueError("Only 2D matrices are supported.")

    def to_frac(x):
        f=Fraction(x) if isinstance(x,(int,np.integer,Fraction)) else Fraction(float(x)).limit_denominator(max_denominator)
        if f.denominator==1:
            return str(f.numerator)
        return ("-" if f<0 else "")+rf"\frac{{{abs(f.numerator)}}}{{{f.denominator}}}"
    fmt=to_frac if as_fractions else str
    lines=["            "+" & ".join(map(fmt,row))+r" \\" for row in arr]
    return "\\begin{pmatrix}\n"+"\n".join(lines)+"\n        \\end{pmatrix}"

def integral_gale_transform(points: ArrayLike)->np.ndarray:
    """
    **Description:**
    Lifts the input points by appending a column of ones, and computes a saturated, LLL-reduced integral basis of the linear relations among the lifted points (the integral Gale transform).
    **Arguments:**
    - `points (ArrayLike)`: Point configuration with shape `(n, d)`.
    **Raises:**
    - `ValueError`: Raised if the number of points is not greater than `d+1`.
    **Returns:**
    - `np.ndarray`: The integral Gale transform.
    """
    points=np.array(points)
    n, d=points.shape
    if n<=d+1:
        raise ValueError(f"Need strictly more points (n={n}) than dimensions + 1 (d+1={d+1}).")
    lifted_points=np.hstack((points,np.ones((n,1),dtype=int)))
    return integral_kernel(lifted_points.T)

def divisor_intersections(fan: Fan,intersection_dict: Mapping[tuple[int,...],float],divisors: Sequence[np.ndarray],basis_set: set[int],as_LLL: bool=True)->np.ndarray:
    """
    **Description:**
    Computes the curve classes obtained by intersecting a list of divisors with toric strata, expressed in a chosen basis of curve homology.
    **Arguments:**
    - `fan (Fan)`: The toric fan.
    - `intersection_dict (Mapping[tuple[int,...],float])`: Dictionary of toric intersection numbers, keyed by sorted tuples of one-indexed ray labels.
    - `divisors (Sequence[np.ndarray])`: List of divisor coefficient vectors.
    - `basis_set (set[int])`: Set of one-indexed ray labels used as the homology basis.
    - `as_LLL (bool)`: Whether to LLL-reduce the resulting lattice basis. Defaults to `True`.
    **Returns:**
    - `np.ndarray`: The divisor-intersection curve classes (integer matrix, one row per curve), optionally LLL-reduced with zero rows removed.
    """
    codim_cicy=len(divisors)
    simplices=get_lower_dimensional_cones(fan.cones(),fan.dim-codim_cicy-1)
    divisor_nonvanishing_sets=[]
    for div in divisors:
        divisor_nonvanishing_sets.append(set(np.where(div!=0)[0]+1))
    curves_homology_in_basis=np.zeros((len(simplices),len(basis_set)),dtype=float)
    basis_idx_map={b: idx for idx,b in enumerate(sorted(basis_set))}
    for s_idx, s in enumerate(simplices):
        star_s=fan.star(s)
        link_rays={item for sub_tuple in star_s for item in sub_tuple}
        valid_intersections=[]
        for div_set in divisor_nonvanishing_sets:
            valid_intersections.append(div_set.intersection(link_rays))
        valid_i=basis_set.intersection(link_rays)
        if not (all(valid_intersections) and valid_i):
            continue
        for i in valid_i:
            i_idx=basis_idx_map.get(i)
            total_intersection=0
            for ts in product(*valid_intersections):
                key=tuple(sorted(s+ts+(i,)))
                coefficient=np.prod([divisors[a][ray-1]for a,ray in enumerate(ts)])
                total_intersection+=coefficient*intersection_dict.get(key,0)
            curves_homology_in_basis[s_idx,i_idx]=total_intersection
    rounded=np.rint(curves_homology_in_basis)
    if not np.allclose(curves_homology_in_basis,rounded,atol=1e-6):
        warnings.warn("divisor_intersections: non-integral curve classes encountered, rounding to the nearest integer.")
    curves_homology_in_basis=rounded.astype(int)
    if as_LLL:
        reduced=np.array(lll_reduce(curves_homology_in_basis.T)).T
        return reduced[np.any(reduced!=0,axis=1)]
    return curves_homology_in_basis

def make_rows_integer(arr: ArrayLike,den: int=2)->np.ndarray:
    """
    **Description:**
    Multiplies every non-integral row of `arr` that becomes integral after multiplication by `den` with `den`, and rounds the result to integers.
    **Arguments:**
    - `arr (ArrayLike)`: Two-dimensional array of (rational) row vectors.
    - `den (int)`: Denominator by which non-integral rows are rescaled. Defaults to `2`.
    **Returns:**
    - `np.ndarray`: Integer array of the same shape as `arr`.
    """
    arr=np.asarray(arr).copy()
    integer=np.all(np.isclose(arr,np.rint(arr)),axis=1)
    den_integer=np.all(np.isclose(den*arr,np.rint(den*arr)),axis=1)
    mask=~integer&den_integer
    arr[mask]*=den
    return np.rint(arr).astype(int)

def hodge_numbers_weighted(O: CY_orientifold)->dict[str,int|np.integer]:
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
    fixed=_fixed_locus(O,return_chi=True)
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

def _orbit_pieces(V: np.ndarray,cones: np.ndarray,W: np.ndarray,E: np.ndarray,cache: dict)->tuple[np.ndarray,np.ndarray,np.ndarray]:
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

def chi_hypersurface(V: ArrayLike,cones_by_dim: list[np.ndarray],divisor: ArrayLike)->int:
    """
    **Description:**
    Euler characteristic of the generic hypersurface in the class `divisor` of a complete simplicial toric variety, as the
    sum of the Kouchnirenko Euler characteristics of its intersections with all torus orbits (Danilov-Khovanskii).
    **Arguments:**
    - `V (ArrayLike)`: The rays of the fan, one per row.
    - `cones_by_dim (list[np.ndarray])`: All cones of the fan of dimension 1,...,dim, as returned by `_cones_by_dimension`.
    - `divisor (ArrayLike)`: Divisor coefficients in the basis of the rays.
    **Returns:**
    - `int`: The Euler characteristic.
    """
    V=np.asarray(V,dtype=int)
    _,W,E=_newton_data(V,np.asarray(divisor,dtype=int))
    cache={}
    return int(sum(_orbit_pieces(V,cones,W,E,cache)[0].sum() for cones in [np.zeros((1,0),dtype=int)]+cones_by_dim))

def find_O7_planes(O: CY_orientifold|F_Theory_Uplift)->np.ndarray:
    """
    **Description:**
    Finds the divisorial O7-planes from the orbifold rays alone (no fan): the rays of `vectors_orbifold()` that are odd
    under `orientifold_mod2_character` (their divisors are pointwise fixed) and whose divisor meets the orientifold B,
    i.e. at least two monomials of the orientifold line bundle survive on it. Non-divisorial O7-planes (fixed 2-cones
    whose stratum lies in B) need the fan, see `orientifold_fixed_locus`.
    **Arguments:**
    - `O (CY_orientifold|F_Theory_Uplift)`: The Calabi-Yau orientifold or an F-theory uplift.
    **Raises:**
    - `ValueError`: Raised if a pointwise fixed divisor is contained in B (B reducible).
    **Returns:**
    - `np.ndarray`: Sorted one-indexed labels of `vectors_orbifold()`.
    """
    O=O.orientifold() if hasattr(O,"orientifold") else O
    if O.line_bundle() is None:
        raise ValueError("The orientifold is not regular (no projected-in monomial), so B and its O7-planes are not defined.")
    V=np.asarray(O.vectors_orbifold(),dtype=int)
    _,_,E=_newton_data(V,np.asarray(O.line_bundle(),dtype=int))
    n_face=E.sum(axis=1)
    odd=(V@orientifold_mod2_character(O.xi()))%2==1
    if np.any(odd&(n_face==0)):
        raise ValueError(f"The fixed divisors {np.flatnonzero(odd&(n_face==0))+1} are contained in B.")
    return np.flatnonzero(odd&(n_face>=2))+1

def _singular_orbits(V: np.ndarray,V_up: np.ndarray,r: np.ndarray,L: np.ndarray,NP: h_polytope.HPolytope,E: np.ndarray,cones_by_dim: list[np.ndarray])->list[tuple[tuple[int,...],str]]:
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

def orientifold_fixed_locus(O: CY_orientifold|F_Theory_Uplift,check: bool=True,return_chi: bool=False)->dict:
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
    - `check (bool)`: Whether to verify that X is smooth (see `_singular_orbits`) and that the O7-planes are disjoint, as
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
        singular=_singular_orbits(V,V_up,r,L,NP,E,cones_by_dim)
        if singular:
            raise ValueError(f"The orientifold-invariant Calabi-Yau is singular along the orbits of {singular[:5]}{' ...' if len(singular)>5 else ''}.")
    cache={}
    fixed_pieces=[]
    for cones in cones_by_dim:
        fixed=cones[_fixed_cones(V,cones,mu,odd)]
        fixed_pieces.append((fixed,)+_orbit_pieces(V,fixed,W,E,cache))
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
        chi_B=int(sum(_orbit_pieces(V,cones,W,E,cache)[0].sum() for cones in [np.zeros((1,0),dtype=int)]+cones_by_dim))
        chi_X=2*chi_B-out["chi_I"]
        if chi_X%2 or (out["chi_I"]-chi_X)%4:
            raise ValueError(f"chi(X)={chi_X} and chi_I={out['chi_I']} violate the Lefschetz congruence chi_I = chi(X) = 0 mod 2, chi_I = chi(X) mod 4.")
        out["chi_B"]=chi_B
        out["chi_X"]=chi_X
    return out

def _fixed_locus(O: CY_orientifold|F_Theory_Uplift,return_chi: bool=False)->dict:
    """
    **Description:**
    Cached `orientifold_fixed_locus` of a Calabi-Yau orientifold or of the orientifold of an F-theory uplift.
    **Arguments:**
    - `O (CY_orientifold|F_Theory_Uplift)`: The Calabi-Yau orientifold or F-theory uplift.
    - `return_chi (bool)`: Whether chi(B) and chi(X) are needed. Defaults to `False`.
    **Returns:**
    - `dict`: The output of `orientifold_fixed_locus`.
    """
    O=O.orientifold() if hasattr(O,"orientifold") else O
    return O.fixed_locus(return_chi=return_chi) if hasattr(O,"fixed_locus") else orientifold_fixed_locus(O,return_chi=return_chi)

def chi_I(O: CY_orientifold|F_Theory_Uplift)->int:
    """
    **Description:**
    Euler characteristic of the fixed locus of the orientifold involution, chi_O7 plus the number of O3-planes
    (see `orientifold_fixed_locus`).
    **Arguments:**
    - `O (CY_orientifold|F_Theory_Uplift)`: The Calabi-Yau orientifold or F-theory uplift.
    **Returns:**
    - `int`: The Euler characteristic of the fixed locus.
    """
    return _fixed_locus(O)["chi_I"]

def chi_O7(O: CY_orientifold|F_Theory_Uplift)->int:
    """
    **Description:**
    Total Euler characteristic of the O7-planes, divisorial and non-divisorial (see `orientifold_fixed_locus`).
    **Arguments:**
    - `O (CY_orientifold|F_Theory_Uplift)`: The Calabi-Yau orientifold or F-theory uplift.
    **Returns:**
    - `int`: The Euler characteristic of the O7-planes.
    """
    return _fixed_locus(O)["chi_O7"]

def chi_O3(O: CY_orientifold|F_Theory_Uplift)->int:
    """
    **Description:**
    Number of O3-planes, i.e. the Euler characteristic of the pointlike part of the fixed locus, including O3-planes at
    torus-fixed points of the ambient space (see `orientifold_fixed_locus`).
    **Arguments:**
    - `O (CY_orientifold|F_Theory_Uplift)`: The Calabi-Yau orientifold or F-theory uplift.
    **Returns:**
    - `int`: The number of O3-planes.
    """
    return _fixed_locus(O)["n_O3"]

def general_divisor_intersection(fan: Fan,divisors: Sequence[ArrayLike])->float:
    """
    **Description:**
    Computes the intersection number D_1*...*D_n of n=`fan.dim` divisors in the toric variety of `fan`.
    **Arguments:**
    - `fan (Fan)`: The toric fan; must be a triangulation.
    - `divisors (Sequence[ArrayLike])`: `fan.dim` divisors, each a coefficient vector over the prime toric divisors, ordered as `fan.labels`.
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

def intersection_test(fan: Fan,intersection_dict: Mapping[tuple[int,...],float],divisors: Sequence[np.ndarray],target_divisors: Sequence[int]|np.ndarray)->np.ndarray:
    """
    **Description:**
    Tests, for each target toric divisor `D_t`, whether the complete intersection cut out by `divisors` meets `D_t`, i.e. whether `D_t` times the product of `divisors` times some 3-ray stratum in the star of `D_t` has a nonzero intersection number.
    **Arguments:**
    - `fan (Fan)`: The toric fan.
    - `intersection_dict (Mapping[tuple[int,...],float])`: Dictionary of toric intersection numbers, keyed by sorted tuples of one-indexed ray labels.
    - `divisors (Sequence[np.ndarray])`: Divisor coefficient vectors of the complete intersection.
    - `target_divisors (Sequence[int]|np.ndarray)`: One-indexed ray labels of the toric divisors to be tested.
    **Returns:**
    - `np.ndarray`: Boolean array with one entry per target divisor, `True` if the intersection is nonzero.
    """
    codim_cicy=len(divisors)
    divisor_nonvanishing_sets=[]
    for div in divisors:
        divisor_nonvanishing_sets.append(set(np.where(div!=0)[0]+1))
    curves_homology_in_basis=np.zeros( len(target_divisors),dtype=bool)
    for i, d in enumerate(target_divisors):
        stari=fan.star((d,))
        stari_set={item for sub_tuple in stari for item in sub_tuple}
        simps=get_lower_dimensional_cones(stari,fan.dim-codim_cicy-1)
        valid_intersections=[]
        for div_set in divisor_nonvanishing_sets:
            valid_intersections.append(div_set.intersection(stari_set))
        for s in simps:
            total_intersection=0
            for ts in product(*valid_intersections):
                key=tuple(sorted(s+ts+(d,)))
                coefficient=np.prod([divisors[a][ray-1]for a,ray in enumerate(ts)])
                total_intersection+=coefficient*intersection_dict.get(key,0)
            if abs(total_intersection)>1e-8:
                curves_homology_in_basis[i]=True
                break
    return curves_homology_in_basis

def intersection_test_uplift(F: F_Theory_Uplift,target_divisors: Sequence[int]|np.ndarray)->np.ndarray:
    """
    **Description:**
    Tests, for each target divisor of the smooth uplift ambient variety, whether it has a nonvanishing intersection with the elliptic Calabi-Yau fourfold. The fourfold is cut out by the base line bundle `F.line_bundle_base_N()` and by the divisor `3*D_x` of the Weierstrass coordinate `x`; the test is delegated to `intersection_test`.
    **Arguments:**
    - `F (F_Theory_Uplift)`: The F-theory uplift.
    - `target_divisors (Sequence[int]|np.ndarray)`: One-indexed ray labels of the toric divisors to be tested.
    **Returns:**
    - `np.ndarray`: Boolean array with one entry per target divisor, `True` if the corresponding divisor has a nonzero intersection with the Calabi-Yau.
    """
    LBW=np.zeros(len(F.line_bundle_base_N()),dtype=int)
    xlab=len(F.vectors_singular_uplift_ambient())-3
    LBW[xlab]=3
    return intersection_test(F.smooth_uplift_ambient_toric_fan(),F.intersection_numbers_smooth_uplift_ambient(),[F.line_bundle_base_N(),LBW],target_divisors)
#Should be obsolete as we have the Z2 fixed locus

# ---- Obsolete: superseded by hodge_diamond_stringy / hodge_diamond_cy4 / F_Theory_Uplift.hodge_numbers_CY (kept as comments for reference) ----
# def _vertex_key(vertices: ArrayLike)->tuple[tuple[int,...],...]:
#     """
#     **Description:**
#     Hashable key for a face, independent of the ordering of its vertices.
#     **Arguments:**
#     - `vertices (ArrayLike)`: The vertices of the face, one per row.
#     **Returns:**
#     - `tuple[tuple[int,...],...]`: The sorted tuple of vertex tuples.
#     """
#     vertices=np.asarray(vertices,dtype=np.int64)
#     return tuple(sorted(map(tuple,vertices.tolist())))
# def h11_2_part_l_fast(Cay: Polytope,Cayd: Polytope,det: bool=False)->int:
#     """
#     **Description:**
#     Faster implementation of h11_2_part_l.
#     The implementation avoids constructing a new Polytope for every face.
#     It assumes the Cayley dual-face relation used in the original code:
#         <v_dual, v_face> = 0
#     for every vertex v_face of the original face.
#     **Arguments:**
#     - `Cay (Polytope)`: The Cayley polytope.
#     - `Cayd (Polytope)`: The dual Cayley polytope (same dimension as `Cay`).
#     - `det (bool)`: If `True`, print the individual contributions. Defaults to `False`.
#     **Returns:**
#     - `int`: The Hodge number `h^{1,1}`.
#     """
#     dim=Cay.dim()
#     if Cayd.dim()!=dim:
#         raise ValueError("Cay and Cayd must have the same dimension, got "f"{dim} and {Cayd.dim()}.")
#     n=dim-1
#     if n<3:
#         raise ValueError("This implementation expects Cayley polytopes for which 1-, 2-, and 3-faces occur.")
#     cay_faces_1=Cay.faces(1)
#     cay_faces_2=Cay.faces(2)
#     cay_faces_3=Cay.faces(3)
#     dual_dimensions={n-1,n-2,n-3}
#     cayd_faces={d: Cayd.faces(d) for d in dual_dimensions}
#     cayd_face_lookup={d: {_vertex_key(face.vertices()): face for face in faces} for d,faces in cayd_faces.items()}
#     cay_vertices=np.asarray(Cay.vertices(),dtype=int)
#     cayd_vertices=np.asarray(Cayd.vertices(),dtype=int)
#     zero_pairing=(cayd_vertices@cay_vertices.T==0)
#     point_to_vertex_index=np.full(len(Cay.points()),-1,dtype=int)
#     cay_vertex_point_indices=np.asarray(Cay.vertices(as_indices=True),dtype=int)
#     point_to_vertex_index[cay_vertex_point_indices]=np.arange(len(cay_vertices),dtype=int)
#     def dual_face_data(face: PolytopeFace)->tuple[PolytopeFace,np.ndarray]:
#         """
#         **Description:**
#         Return the existing Cayd PolytopeFace dual to `face`, together
#         with its dual vertices.
#         **Arguments:**
#         - `face (PolytopeFace)`: A face of `Cay`.
#         **Returns:**
#         - `tuple[PolytopeFace,np.ndarray]`: A pair `(dual_face, dual_vertices)` of the dual face of `Cayd` and its vertices.
#         """
#         face_point_indices=np.asarray(face.vertices(as_indices=True),dtype=np.intp)
#         face_vertex_indices=point_to_vertex_index[face_point_indices]
#         if np.any(face_vertex_indices<0):
#             raise RuntimeError("A face vertex could not be matched to Cay.vertices().")
#         mask=np.all(zero_pairing[:,face_vertex_indices],axis=1)
#         dual_vertices=cayd_vertices[mask]
#         if len(dual_vertices)==0:
#             raise RuntimeError("The zero-pairing condition produced no dual vertices.")
#         dual_dimension=n-face.dim()
#         key=_vertex_key(dual_vertices)
#         try:
#             dual_face=cayd_face_lookup[dual_dimension][key]
#         except KeyError as exc:
#             raise RuntimeError("The zero-pairing vertex set was not found among the "+f"{dual_dimension}-faces of Cayd.") from exc
#         return dual_face, dual_vertices
#     doubled_cayd=Polytope(2*cayd_vertices)
#     doubled_dual_codim2_lookup={_vertex_key(face.vertices()): face for face in doubled_cayd.faces(n-1)}
#     trivial_term=len(Cayd.points())-n-2
#     doubled_dual_facet_term=-sum(len(face.labels_int) for face in doubled_cayd.faces(n))
#     dual_codim2_term=sum(len(face.labels_int) for face in cayd_faces[n-1])
#     edge_term=0
#     nonzero_edges=0
#     for face in cay_faces_1:
#         lstar_face=len(face.labels_int)
#         if lstar_face==0:
#             continue
#         _, dual_vertices=dual_face_data(face)
#         doubled_dual_key=_vertex_key(2*dual_vertices)
#         try:
#             doubled_dual_face=doubled_dual_codim2_lookup[doubled_dual_key]
#         except KeyError as exc:
#             raise RuntimeError("Could not match a dual face with its face in 2*Cayd.") from exc
#         edge_term+=lstar_face*len(doubled_dual_face.labels_int)
#         nonzero_edges+=1
#     face_2_term=0
#     nonzero_2faces=0
#     for face in cay_faces_2:
#         dual_face, _=dual_face_data(face)
#         lstar_dual=len(dual_face.labels_int)
#         if lstar_dual==0:
#             continue
#         lstar_face=len(face.labels_int)
#         lattice_points_face=len(face.labels)
#         lstar_doubled_face=3*lstar_face+lattice_points_face-3
#         face_2_term-=lstar_dual*lstar_doubled_face
#         nonzero_2faces+=1
#     face_3_term=0
#     nonzero_3faces=0
#     for face in cay_faces_3:
#         dual_face, _=dual_face_data(face)
#         lstar_dual=len(dual_face.labels_int)
#         if lstar_dual==0:
#             continue
#         lstar_face=len(face.labels_int)
#         lattice_points_face=len(face.labels)
#         normalized_volume_raw=face.as_polytope().volume()
#         normalized_volume=int(round(float(normalized_volume_raw)))
#         if not np.isclose(float(normalized_volume_raw),normalized_volume):
#             raise RuntimeError("The normalized volume of a lattice 3-face was not integral: "+f"{normalized_volume_raw}.")
#         lstar_doubled_face=normalized_volume-lattice_points_face+3*lstar_face+3
#         subface_correction=sum(len(subface.labels_int) for subface in face.faces(2))
#         face_3_term+=lstar_dual*(lstar_doubled_face-subface_correction)
#         nonzero_3faces+=1
#     result=trivial_term+doubled_dual_facet_term+dual_codim2_term+edge_term+face_2_term+face_3_term
#     if det:
#         print("Trivial term:                         ",trivial_term)
#         print("Doubled dual facets:                 ",doubled_dual_facet_term)
#         print("Dual codimension-2 faces:            ",dual_codim2_term)
#         print("1-face / doubled dual-face term:     ",edge_term)
#         print("2-face / dual-face term:             ",face_2_term)
#         print("3-face and subface correction term:  ",face_3_term)
#         print("Nonzero contributing edges:          ",nonzero_edges)
#         print("Nonzero contributing 2-faces:        ",nonzero_2faces)
#         print("Nonzero contributing 3-faces:        ",nonzero_3faces)
#         print("h11:                                  ",result)
#     return int(result)
# def h21_2_part_fast(Cay: Polytope,Cayd: Polytope,det: bool=False)->int:
#     """
#     **Description:**
#     Fast implementation of h21_2_part.
#     This is algebraically equivalent to
#         sum_{dim f=2} l*(f) l*(2 f*)
#       + sum_{dim f=4} l*(2f) l*(f*)
#       - sum_{dim f=3} l*(f*) sum_{g< f, dim g=2} l*(g)
#       - sum_{dim f=4} l*(f*) sum_{g< f, dim g=3} l*(g),
#     where f* is the dual Cayley face.
#     Optimizations:
#     1. Dual faces are found using the zero-pairing matrix rather than by
#        repeatedly calling dual_face_Cayley_polytope.
#     2. 2*Cay and 2*Cayd are constructed only once.
#     3. Interior lattice-point counts use ``labels_int`` directly.
#     4. The two contributions involving 4-faces are evaluated in the same loop.
#     **Arguments:**
#     - `Cay (Polytope)`: Cayley polytope.
#     - `Cayd (Polytope)`: Dual Cayley polytope.
#     - `det (bool)`: Print individual contributions. Defaults to `False`.
#     **Returns:**
#     - `int`: h^{2,1}.
#     """
#     dim=Cay.dim()
#     if Cayd.dim()!=dim:
#         raise ValueError("Cay and Cayd must have the same dimension, got "f"{dim} and {Cayd.dim()}.")
#     n=dim-1
#     if n<4:
#         raise ValueError("This implementation expects Cayley polytopes with 4-faces.")
#     cay_faces_2=Cay.faces(2)
#     cay_faces_3=Cay.faces(3)
#     cay_faces_4=Cay.faces(4)
#     dual_dimensions={n-2,n-3,n-4}
#     cay_vertices=np.asarray(Cay.vertices(),dtype=np.int64)
#     cayd_vertices=np.asarray(Cayd.vertices(),dtype=np.int64)
#     zero_pairing=(cayd_vertices@cay_vertices.T==0)
#     cay_vertex_lookup={tuple(map(int,v)): i for i,v in enumerate(cay_vertices)}
#     def vertex_key(vertices: ArrayLike)->tuple[tuple[int,...],...]:
#         """
#         **Description:**
#         Hashable, ordering-independent key of a vertex set.
#         **Arguments:**
#         - `vertices (ArrayLike)`: The vertices, one per row.
#         **Returns:**
#         - `tuple[tuple[int,...],...]`: The sorted tuple of vertex tuples.
#         """
#         return tuple(sorted(tuple(map(int,row)) for row in np.asarray(vertices)))
#     cayd_lstar_lookup={d: {vertex_key(face.vertices()): len(face.labels_int) for face in Cayd.faces(d)} for d in dual_dimensions}
#     def dual_face_key_and_lstar(face: PolytopeFace)->tuple[tuple[tuple[int,...],...],int]:
#         """
#         **Description:**
#         Finds the face of `Cayd` dual to `face` via the zero-pairing matrix and returns its vertex key together with its number of interior lattice points.
#         **Arguments:**
#         - `face (PolytopeFace)`: A face of `Cay`.
#         **Returns:**
#         - `tuple[tuple[tuple[int,...],...],int]`: A pair `(key, lstar)` of the vertex key of the dual face and its number of interior lattice points.
#         """
#         fv=np.asarray(face.vertices(),dtype=np.int64)
#         try:
#             face_vertex_indices=np.fromiter((cay_vertex_lookup[tuple(map(int,v))] for v in fv),dtype=np.intp,count=len(fv))
#         except KeyError as exc:
#             raise RuntimeError("A face vertex could not be matched to Cay.vertices().") from exc
#         mask=np.all(zero_pairing[:,face_vertex_indices],axis=1)
#         dual_vertices=cayd_vertices[mask]
#         if len(dual_vertices)==0:
#             raise RuntimeError("The zero-pairing condition produced no dual vertices.")
#         dual_dim=n-face.dim()
#         key=vertex_key(dual_vertices)
#         try:
#             lstar=cayd_lstar_lookup[dual_dim][key]
#         except KeyError as exc:
#             raise RuntimeError("The zero-pairing vertex set was not found among the " f"{dual_dim}-faces of Cayd.") from exc
#         return key, lstar
#     doubled_cay=Polytope(2*cay_vertices)
#     doubled_cayd=Polytope(2*cayd_vertices)
#     doubled_cay_4_lstar={}
#     for face in doubled_cay.faces(4):
#         doubled_vertices=np.asarray(face.vertices(),dtype=np.int64)
#         if np.any(doubled_vertices%2):
#             raise RuntimeError("A vertex of 2*Cay was unexpectedly not divisible by 2.")
#         original_key=vertex_key(doubled_vertices//2)
#         doubled_cay_4_lstar[original_key]=len(face.labels_int)
#     doubled_dual_dim=n-2
#     doubled_cayd_lstar={}
#     for face in doubled_cayd.faces(doubled_dual_dim):
#         doubled_vertices=np.asarray(face.vertices(),dtype=np.int64)
#         if np.any(doubled_vertices%2):
#             raise RuntimeError("A vertex of 2*Cayd was unexpectedly not divisible by 2.")
#         original_key=vertex_key(doubled_vertices//2)
#         doubled_cayd_lstar[original_key]=len(face.labels_int)
#     face_2_term=0
#     nonzero_2faces=0
#     for face in cay_faces_2:
#         lstar_face=len(face.labels_int)
#         if lstar_face==0:
#             continue
#         dual_key, _=dual_face_key_and_lstar(face)
#         try:
#             lstar_doubled_dual=doubled_cayd_lstar[dual_key]
#         except KeyError as exc:
#             raise RuntimeError("Could not match a dual face with its corresponding face in 2*Cayd.") from exc
#         if lstar_doubled_dual==0:
#             continue
#         face_2_term+=lstar_face*lstar_doubled_dual
#         nonzero_2faces+=1
#     face_3_term=0
#     nonzero_3faces=0
#     for face in cay_faces_3:
#         _, lstar_dual=dual_face_key_and_lstar(face)
#         if lstar_dual==0:
#             continue
#         subface_correction=sum(len(subface.labels_int) for subface in face.faces(2))
#         if subface_correction==0:
#             continue
#         face_3_term-=lstar_dual*subface_correction
#         nonzero_3faces+=1
#     face_4_term=0
#     nonzero_4faces=0
#     for face in cay_faces_4:
#         _, lstar_dual=dual_face_key_and_lstar(face)
#         if lstar_dual==0:
#             continue
#         face_key=vertex_key(face.vertices())
#         try:
#             lstar_doubled_face=doubled_cay_4_lstar[face_key]
#         except KeyError as exc:
#             raise RuntimeError("Could not match a Cay 4-face with its corresponding face in 2*Cay.") from exc
#         subface_correction=sum(len(subface.labels_int) for subface in face.faces(3))
#         face_4_term+=lstar_dual*(lstar_doubled_face-subface_correction)
#         nonzero_4faces+=1
#     result=face_2_term+face_3_term+face_4_term
#     if det:
#         print("2-face / doubled dual-face term:      ",face_2_term)
#         print("3-face subface correction term:       ",face_3_term)
#         print("4-face combined term:                  ",face_4_term)
#         print("Nonzero contributing 2-faces:          ",nonzero_2faces)
#         print("Nonzero contributing 3-faces:          ",nonzero_3faces)
#         print("Nonzero contributing 4-faces:          ",nonzero_4faces)
#         print("h21:                                     ",result)
#     return int(result)
# ---- End of obsolete block ----

def latte_lattice_points(vertices: ArrayLike|None=None,A: ArrayLike|None=None,b: ArrayLike|None=None,latte_options: Sequence[str]=("--redundancy-check=none",))->tuple[int,int]:
    """
    **Description:**
    Counts the lattice points and the interior lattice points of a polytope with LattE (the `count` executable must be on the PATH). The polytope is given either by its vertices or by the inequalities `A x <= b`.
    - Vertices: a single LattE call computes the Ehrhart polynomial L(t) of the lattice polytope; the number of lattice points is L(1) and, by Ehrhart-Macdonald reciprocity, the number of interior lattice points is (-1)^d L(-1). Lower-dimensional polytopes are first moved to intrinsic (unimodularly LLL-reduced) lattice coordinates, so "interior" means relative interior.
    - Inequalities: two LattE counts, of `A x <= b` and of `A x <= b - 1`; for integral `A` and `b` the latter are exactly the points with all inequalities strict, i.e. the interior in R^n (zero if the polytope is not full-dimensional).
    **Arguments:**
    - `vertices (ArrayLike|None)`: Integral vertices (or any integral points whose convex hull is the polytope), one per row. Defaults to `None`.
    - `A (ArrayLike|None)`: Integral inequality matrix of shape (number of inequalities, dimension). Defaults to `None`.
    - `b (ArrayLike|None)`: Integral right-hand side of `A x <= b`. Defaults to `None`.
    - `latte_options (Sequence[str])`: Extra options passed to LattE's `count`. Defaults to `("--redundancy-check=none",)`.
    **Raises:**
    - `ValueError`: Raised if not exactly one of `vertices` and `(A, b)` is given, or if the input is not integral.
    - `RuntimeError`: Raised if LattE fails.
    **Returns:**
    - `tuple[int,int]`: The number of lattice points and the number of (relative) interior lattice points.
    """

    def latte(rows: np.ndarray,*options: str)->str|None:
        """Last line of LattE's output for the polytope written as `rows`, or `None` if LattE reports it empty."""
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp,"polytope").write_text(f"{len(rows)} {rows.shape[1]}\n"+"".join(" ".join(map(str,row))+"\n" for row in rows.tolist()))
            out=subprocess.run(["count",*options,*latte_options,"polytope"],cwd=tmp,capture_output=True,text=True)
        if out.returncode!=0:
            if "Empty polytope" in out.stdout+out.stderr:
                return None
            raise RuntimeError(f"LattE failed:\n{out.stderr}")
        return out.stdout.strip().splitlines()[-1]
    if (vertices is None)==(A is None or b is None):
        raise ValueError("give either the vertices or the inequalities A x <= b")
    data=[np.asarray(x) for x in ((vertices,) if vertices is not None else (A,b))]
    if any(not np.array_equal(x,np.rint(x)) for x in data):
        raise ValueError("LattE needs integral input")
    if vertices is None:
        A,b=(np.rint(x).astype(int) for x in data)
        total,interior=(latte(np.column_stack([b-shift,-A])) for shift in (0,1))
        if total is None:
            raise ValueError("the inequalities A x <= b define an empty or unbounded polyhedron")
        return int(total.split()[-1]),int(interior.split()[-1]) if interior else 0
    V=np.rint(data[0]).astype(int)
    d=np.linalg.matrix_rank(V-V[0])
    if d==0:
        return 1,1
    local=V-V[0] if d==V.shape[1] else lll_reduce(V-V[0])[:,-d:]
    terms=re.findall(r"([+-])\s*(\d+(?:/\d+)?)\s*\*\s*t\^(\d+)",latte(np.column_stack([np.ones(len(local),dtype=int),local]),"--ehrhart-polynomial","--vrep"))
    L=lambda t: sum(Fraction(sign+c)*t**int(e) for sign,c,e in terms)
    return int(L(1)),int((-1)**d*L(-1))
# ---- Obsolete: superseded by latte_lattice_points (kept as comments for reference) ----
# def count_interior_lattice_points_hrep(A: ArrayLike,b: ArrayLike,count_bin: str="count",latte_options: Sequence[str]=("--redundancy-check=none",))->int:
#     """
#     **Description:**
#     Count interior lattice points of
#         P = {x : A x <= b}
#     assuming A and b are primitive integer facet inequalities.
#     Interior lattice points satisfy
#         A x <= b - 1.
#     **Arguments:**
#     - `A (ArrayLike)`: Integer inequality matrix of shape (n_ineqs, dim).
#     - `b (ArrayLike)`: Integer right-hand side of shape (n_ineqs,).
#     - `count_bin (str)`: Name of LattE's `count` executable if it is on PATH, or its full path. Defaults to `"count"`.
#     - `latte_options (Sequence[str])`: Extra command-line options passed to LattE. Defaults to `("--redundancy-check=none",)`.
#     **Returns:**
#     - `int`: The number of interior lattice points of P.
#     """
#     A=np.asarray(A,dtype=object)
#     b=np.asarray(b,dtype=object)
#     return count_lattice_points_hrep(A,b-1,count_bin=count_bin,latte_options=latte_options)
# def parse_latte_output(stdout: str,stderr: str="")->int:
#     """
#     **Description:**
#     Robustly parse LattE's output.
#     In LattE integrale 1.7.6, stdout is often simply
#     77
#     while stderr contains all diagnostic information.
#     Raises `RuntimeError` if no lattice-point count can be found.
#     **Arguments:**
#     - `stdout (str)`: Standard output of the LattE `count` call.
#     - `stderr (str)`: Standard error of the LattE `count` call. Defaults to `""`.
#     **Returns:**
#     - `int`: The number of lattice points reported by LattE.
#     """
#     stdout_clean=stdout.strip()
#     if re.fullmatch(r"[+-]?\d+",stdout_clean):
#         return int(stdout_clean)
#     output=stdout+"\n"+stderr
#     m=re.search(r"Total number of lattice points(?:\s+is)?\s*:?\s*([+-]?\d+)",output)
#     if m:
#         return int(m.group(1))
#     m=re.search(r"The number of lattice points is.*?\n\s*([+-]?\d+)\s*(?:\n|$)",output,flags=re.DOTALL)
#     if m:
#         return int(m.group(1))
#     candidates=[]
#     for line in stdout.splitlines():
#         line=line.strip()
#         if re.fullmatch(r"[+-]?\d+",line):
#             candidates.append(int(line))
#     if candidates:
#         return candidates[-1]
#     raise RuntimeError(f"Could not parse LattE output.\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")
# def count_lattice_points_hrep(A: ArrayLike,b: ArrayLike,count_bin: str="count",latte_options: Sequence[str]=("--redundancy-check=none",),linearity_rows: Sequence[int]|None=None,keep_file: bool=False,filename: str|Path="polytope.hrep.latte")->int:
#     """
#     **Description:**
#     Count lattice points in
#         P = {x in Z^d : A x <= b}
#     using LattE.
#     **Arguments:**
#     - `A (ArrayLike)`: Integer inequality matrix of shape (n_ineqs, dim).
#     - `b (ArrayLike)`: Integer right-hand side of shape (n_ineqs,).
#     - `count_bin (str)`: Either "count" if it is on PATH, or full path to LattE's count binary. Defaults to `"count"`.
#     - `latte_options (Sequence[str])`: Tuple/list of extra LattE options. Defaults to `("--redundancy-check=none",)`.
#     - `linearity_rows (Sequence[int]|None)`: Optional list of 1-indexed rows treated as equalities by LattE. Defaults to `None`.
#     - `keep_file (bool)`: If True, write the LattE input file to `filename` and keep it. Defaults to `False`.
#     - `filename (str|Path)`: Output filename used if keep_file=True. Defaults to `"polytope.hrep.latte"`.
#     **Returns:**
#     - `int`: Number of lattice points.
#     """
#     if shutil.which(count_bin) is None and not Path(count_bin).exists():
#         raise FileNotFoundError(f"Could not find LattE executable '{count_bin}'. Use command-v count in the shell, or pass the full path as count_bin.")
#     if keep_file:
#         hrep_file=Path(filename)
#         write_latte_hrep(A,b,hrep_file,linearity_rows=linearity_rows)
#         cmd=[count_bin,*latte_options,str(hrep_file)]
#         proc=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,)
#         if proc.returncode!=0:
#             raise RuntimeError(f"LattE failed.\nCommand: {' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}")
#         return parse_latte_output(proc.stdout,proc.stderr)
#     with tempfile.TemporaryDirectory() as tmpdir:
#         hrep_file=Path(tmpdir)/"polytope.hrep.latte"
#         write_latte_hrep(A,b,hrep_file,linearity_rows=linearity_rows)
#         cmd=[count_bin,*latte_options,str(hrep_file)]
#         proc=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,)
#         if proc.returncode!=0:
#             raise RuntimeError("LattE failed.\n"f"Command: {' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}")
#         return parse_latte_output(proc.stdout,proc.stderr)
# def write_latte_hrep(A: ArrayLike,b: ArrayLike,filename: str|Path,linearity_rows: Sequence[int]|None=None)->None:
#     """
#     **Description:**
#     Write the H-representation
#         A x <= b
#     to a LattE .hrep.latte file.
#     LattE row convention:
#         b_i  -A_i1  -A_i2  ...  -A_id
#     **Arguments:**
#     - `A (ArrayLike)`: Integer inequality matrix of shape (n_ineqs, dim).
#     - `b (ArrayLike)`: Integer right-hand side of shape (n_ineqs,).
#     - `filename (str|Path)`: Output filename.
#     - `linearity_rows (Sequence[int]|None)`: Optional list of 1-indexed row numbers that should be treated as equalities by LattE. Usually leave as None. Defaults to `None`.
#     **Returns:**
#     - `None`
#     """
#     A=np.asarray(A,dtype=object)
#     b=np.asarray(b,dtype=object)
#     if A.ndim!=2:
#         raise ValueError("A must be a 2D array.")
#     if b.ndim!=1:
#         raise ValueError("b must be a 1D array.")
#     if A.shape[0]!=b.shape[0]:
#         raise ValueError("A and b have incompatible shapes.")
#     n_ineqs, dim=A.shape
#     with open(filename,"w") as f:
#         f.write(f"{n_ineqs} {dim+1}\n")
#         for Ai, bi in zip(A,b):
#             row=[int(bi)]+[-int(x) for x in Ai]
#             f.write(" ".join(map(str,row))+"\n")
#         if linearity_rows is not None and len(linearity_rows)>0:
#             linearity_rows=list(map(int,linearity_rows))
#             f.write("linearity "+str(len(linearity_rows))+" "+" ".join(map(str,linearity_rows))+"\n")
# def count_interior_lattice_points_vrep(vertices: ArrayLike,backend: str='ppl',count_bin: str="count",latte_options: Sequence[str]=("--redundancy-check=none",))->int:
#     """
#     **Description:**
#     Counts the interior lattice points of the convex hull of `vertices` by converting it to an H-representation with CYTools' `poly_v_to_h` and calling `count_interior_lattice_points_hrep`.
#     **Arguments:**
#     - `vertices (ArrayLike)`: The vertices (or points) of the polytope, one per row.
#     - `backend (str)`: Backend of `poly_v_to_h` ("ppl", "qhull" or "palp"). Defaults to `'ppl'`.
#     - `count_bin (str)`: Name of LattE's `count` executable if it is on PATH, or its full path. Defaults to `"count"`.
#     - `latte_options (Sequence[str])`: Extra command-line options passed to LattE. Defaults to `("--redundancy-check=none",)`.
#     **Returns:**
#     - `int`: The number of interior lattice points.
#     """
#     out=poly_v_to_h(vertices,backend=backend)[0]
#     A=out[:,:-1]
#     b=out[:,-1]
#     return count_interior_lattice_points_hrep(A,b,count_bin=count_bin,latte_options=latte_options)
# def count_lattice_points_vrep(vertices: ArrayLike,count_bin: str="count",latte_options: Sequence[str]=("--redundancy-check=none",),keep_file: bool=False,filename: str|Path="polytope.vrep.latte",)->int:
#     """
#     **Description:**
#     Count lattice points in the convex hull of `vertices`
#     using LattE directly in V-representation.
#     **Arguments:**
#     - `vertices (ArrayLike)`: Integer vertices of the full-dimensional polytope P, one per row.
#     - `count_bin (str)`: Name of LattE's `count` executable if it is on PATH, or its full path. Defaults to `"count"`.
#     - `latte_options (Sequence[str])`: Extra command-line options passed to LattE. Defaults to `("--redundancy-check=none",)`.
#     - `keep_file (bool)`: If True, write the LattE input file to `filename` and keep it. Defaults to `False`.
#     - `filename (str|Path)`: Output filename used if keep_file=True. Defaults to `"polytope.vrep.latte"`.
#     **Returns:**
#     - `int`: Number of lattice points in P, including the boundary.
#     """
#     vertices=np.asarray(vertices,dtype=object)
#     if shutil.which(count_bin) is None and not Path(count_bin).exists():
#         raise FileNotFoundError(f"Could not find LattE executable '{count_bin}'. Use `command -v count` in the shell, or pass the full path as count_bin.")
#     if keep_file:
#         vrep_file=Path(filename)
#         write_latte_vrep(vertices,vrep_file)
#         cmd=[count_bin,"--vrep",*latte_options,str(vrep_file),]
#         proc=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,)
#         if proc.returncode!=0:
#             raise RuntimeError("LattE failed.\n"f"Command: {' '.join(cmd)}\n\n"f"STDOUT:\n{proc.stdout}\n\n"f"STDERR:\n{proc.stderr}")
#         return parse_latte_output(proc.stdout,proc.stderr)
#     with tempfile.TemporaryDirectory() as tmpdir:
#         vrep_file=Path(tmpdir)/"polytope.vrep.latte"
#         write_latte_vrep(vertices,vrep_file)
#         cmd=[count_bin,"--vrep",*latte_options,str(vrep_file),]
#         proc=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,)
#         if proc.returncode!=0:
#             raise RuntimeError("LattE failed.\n"f"Command: {' '.join(cmd)}\n\n"f"STDOUT:\n{proc.stdout}\n\n"f"STDERR:\n{proc.stderr}")
#         return parse_latte_output(proc.stdout,proc.stderr)
# def count_interior_lattice_points_vrep_ehrhart(vertices: ArrayLike,count_bin: str="count",latte_options: Sequence[str]=("--redundancy-check=none",),keep_file: bool=False,filename: str|Path="polytope.vrep.latte",)->int:
#     """
#     **Description:**
#     Count interior lattice points of a full-dimensional lattice polytope
#     given by its vertices, using Ehrhart-Macdonald reciprocity.
#     If L_P(t) = #(tP ∩ Z^d),
#     then #(int(P) ∩ Z^d) = (-1)^d L_P(-1).
#     **Arguments:**
#     - `vertices (ArrayLike)`: Integer vertices of the polytope, one per row (2D).
#     - `count_bin (str)`: Name of LattE's `count` executable if it is on PATH, or its full path. Defaults to `"count"`.
#     - `latte_options (Sequence[str])`: Extra command-line options passed to LattE. Defaults to `("--redundancy-check=none",)`.
#     - `keep_file (bool)`: If True, write the LattE input file to `filename` and keep it. Defaults to `False`.
#     - `filename (str|Path)`: Output filename used if keep_file=True. Defaults to `"polytope.vrep.latte"`.
#     **Returns:**
#     - `int`: The number of interior lattice points.
#     """
#     vertices=np.asarray(vertices,dtype=object)
#     if vertices.ndim!=2:
#         raise ValueError("vertices must be a 2D array.")
#     dim=vertices.shape[1]
#     if shutil.which(count_bin) is None and not Path(count_bin).exists():
#         raise FileNotFoundError(f"Could not find LattE executable '{count_bin}'.")
#     def run(vrep_file: Path)->int:
#         """
#         **Description:**
#         Writes the LattE input file, computes the Ehrhart polynomial and evaluates the reciprocity formula.
#         **Arguments:**
#         - `vrep_file (Path)`: Path of the LattE V-representation file to write.
#         **Returns:**
#         - `int`: The number of interior lattice points.
#         """
#         write_latte_vrep(vertices,vrep_file)
#         cmd=[count_bin,"--ehrhart-polynomial","--vrep",*latte_options,str(vrep_file),]
#         proc=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,)
#         if proc.returncode!=0:
#             raise RuntimeError("LattE failed.\n"f"Command: {' '.join(cmd)}\n\n"f"STDOUT:\n{proc.stdout}\n\n"f"STDERR:\n{proc.stderr}")
#         poly=_extract_latte_ehrhart_polynomial(proc.stdout)
#         L_minus_one=_evaluate_latte_polynomial(poly,-1)
#         result=((-1)**dim)*L_minus_one
#         if result.denominator!=1:
#             raise RuntimeError("Ehrhart reciprocity produced a non-integral result: "f"{result}\n"f"Ehrhart polynomial: {poly}")
#         return int(result)
#     if keep_file:
#         return run(Path(filename))
#     with tempfile.TemporaryDirectory() as tmpdir:
#         return run(Path(tmpdir)/"polytope.vrep.latte")
# def _extract_latte_ehrhart_polynomial(stdout: str)->str:
#     """
#     **Description:**
#     Extract the Ehrhart polynomial from LattE stdout.
#     **Arguments:**
#     - `stdout (str)`: Standard output of a LattE `--ehrhart-polynomial` call.
#     **Returns:**
#     - `str`: The line of the output containing the Ehrhart polynomial.
#     """
#     lines=[line.strip() for line in stdout.splitlines() if line.strip()]
#     for line in reversed(lines):
#         if re.search(r"\bt(?:\^?\d+)?\b",line):
#             return line
#     for line in reversed(lines):
#         if re.fullmatch(r"[+-]?\d+(?:/\d+)?",line):
#             return line
#     raise RuntimeError("Could not extract Ehrhart polynomial from LattE output.\n\n"f"STDOUT:\n{stdout}")
# def _evaluate_latte_polynomial(poly: str,x: int|Fraction)->Fraction:
#     """
#     **Description:**
#     Evaluate a univariate LattE Ehrhart polynomial exactly.
#     **Arguments:**
#     - `poly (str)`: The polynomial in the variable `t`, as printed by LattE.
#     - `x (int|Fraction)`: The value at which the polynomial is evaluated.
#     **Returns:**
#     - `Fraction`: The exact value of the polynomial at `x`.
#     """
#     poly=poly.replace(" ","")
#     poly=poly.replace("*","")
#     poly=poly.replace("T","t")
#     if not poly.startswith(("+","-")):
#         poly="+"+poly
#     terms=re.findall(r"([+-])([^+-]+)",poly)
#     value=Fraction(0)
#     for sign, term in terms:
#         sgn=1 if sign=="+" else-1
#         if "t" in term:
#             coeff_part, power_part=term.split("t",1)
#             if coeff_part=="":
#                 coeff=Fraction(1)
#             else:
#                 coeff=Fraction(coeff_part.strip("()"))
#             if power_part=="":
#                 power=1
#             elif power_part.startswith("^"):
#                 power=int(power_part[1:])
#             else:
#                 raise RuntimeError(f"Could not parse polynomial term: {term}")
#         else:
#             coeff=Fraction(term.strip("()"))
#             power=0
#         value+=sgn*coeff*(x**power)
#     return value
# ---- End of obsolete block ----
# ---- Obsolete: superseded by hodge_diamond_stringy / hodge_diamond_cy4 / F_Theory_Uplift.hodge_numbers_CY (kept as comments for reference) ----
# def Cayley_N(F: F_Theory_Uplift)->Polytope:
#     """
#     **Description:**
#     Constructs the Cayley polytope conv((pol_W_N, 1, 0) union (pol_B_N, 0, 1)) of the N-lattice polytopes of the Weierstrass and base parts of the dual nef partition of the uplift.
#     **Arguments:**
#     - `F (F_Theory_Uplift)`: The F-theory uplift.
#     **Returns:**
#     - `Polytope`: The Cayley polytope in the N lattice.
#     """
#     return Polytope(np.concatenate((np.column_stack([F.pol_W_N().points(),np.ones(len(F.pol_W_N().points()),dtype=int),np.zeros(len(F.pol_W_N().points()),dtype=int)]),np.column_stack([F.pol_B_N().points(),np.zeros(len(F.pol_B_N().points()),dtype=int),np.ones(len(F.pol_B_N().points()),dtype=int)])),axis=0))
# def Cayley_M(F: F_Theory_Uplift)->Polytope:
#     """
#     **Description:**
#     Constructs the Cayley polytope conv((pol_W_M, 1, 0) union (pol_B_M, 0, 1)) of the M-lattice Newton polytopes of the Weierstrass and base line bundles of the uplift.
#     **Arguments:**
#     - `F (F_Theory_Uplift)`: The F-theory uplift.
#     **Returns:**
#     - `Polytope`: The Cayley polytope in the M lattice.
#     """
#     return Polytope(np.concatenate((np.column_stack([F.pol_W_M().points(),np.ones(len(F.pol_W_M().points()),dtype=int),np.zeros(len(F.pol_W_M().points()),dtype=int)]),np.column_stack([F.pol_B_M().points(),np.zeros(len(F.pol_B_M().points()),dtype=int),np.ones(len(F.pol_B_M().points()),dtype=int)])),axis=0))
# """Batyrev--Nill stringy E-functions for Gorenstein support polytopes.
# Specialized to reflexive Gorenstein Cayley cones.  The primary entry point
# accepts an already-computed Cayley support P, its cone C, the dual cone C_dual,
# and the two Gorenstein degree vectors.
# Input may be a CYTools Polytope or an integer array whose rows are points.
# NumPy arrays are converted to CYTools Polytope objects internally.  All cone
# duality, face-lattice, and lattice-point geometry is performed by CYTools.
# The *dual support* is required: its vertices must be in the dual lattice and
# all primal/dual pairings must be nonnegative.  This is cone-support duality,
# not ordinary polar duality after translating the degree-two slice.
# Example
# -------
#     from stringy_e_cayley import stringy_e_from_cones
#     ans = stringy_e_from_cones(P, C, C_dual, degree_C, degree_C_dual)
#     print(ans.e_polynomial())
#     print(ans.hodge_numbers)
#     print(ans.primal_completely_split, ans.dual_completely_split)
#     print(ans.interpretation())
# If only the primal cone is completely split, ``ans.hodge_numbers`` applies to
# that complete-intersection side.  The dual polynomial is still the canonical
# Gorenstein-cone stringy E-invariant, but it must not be advertised as the
# Hodge diamond of an ordinary Batyrev--Borisov mirror complete intersection.
# The expensive part is the facewise Ehrhart/h* computation.  When LattE's
# ``count`` executable is available, the default backend computes one exact
# Ehrhart polynomial per nontrivial face instead of enumerating every dilation
# separately.  Cache the returned result (or the FaceLattice objects) for large
# examples.
# """
# Poly1=Tuple[int,...] 
# Laurent2=Dict[Tuple[int,int],int]
# _HSTAR_CACHE: Dict[Tuple[int,Tuple[Tuple[int,...],...]],Poly1]={}
# def clear_hstar_cache()->None:
#     """
#     **Description:**
#     Clear the process-wide exact h* cache used across repeated notebook runs.
#     **Returns:**
#     - `None`
#     """
#     _HSTAR_CACHE.clear()
# def hstar_cache_size()->int:
#     """
#     **Description:**
#     Return the number of face h* polynomials held in the global cache.
#     **Returns:**
#     - `int`: The number of cached h* polynomials.
#     """
#     return len(_HSTAR_CACHE)
# def _trim(a: Iterable[int])->Poly1:
#     """
#     **Description:**
#     Converts a coefficient sequence into a polynomial tuple and removes trailing zero coefficients. Polynomials are stored as `Poly1 = tuple[int, ...]` of coefficients in increasing degree.
#     **Arguments:**
#     - `a (Iterable[int])`: Coefficients in increasing degree.
#     **Returns:**
#     - `Poly1`: The trimmed coefficient tuple (`(0,)` for the zero polynomial).
#     """
#     z=list(map(int,a))
#     while len(z)>1 and z[-1]==0:
#         z.pop()
#     return tuple(z or [0])
# def _add(a: Poly1,b: Poly1)->Poly1:
#     """
#     **Description:**
#     Adds two integer polynomials. Polynomials are stored as `Poly1 = tuple[int, ...]` of coefficients in increasing degree.
#     **Arguments:**
#     - `a (Poly1)`: First summand.
#     - `b (Poly1)`: Second summand.
#     **Returns:**
#     - `Poly1`: The trimmed sum `a+b`.
#     """
#     return _trim((a[i] if i<len(a) else 0)+(b[i] if i<len(b) else 0) for i in range(max(len(a),len(b))))
# def _scale(a: Poly1,c: int)->Poly1:
#     """
#     **Description:**
#     Multiplies an integer polynomial by a scalar. Polynomials are stored as `Poly1 = tuple[int, ...]` of coefficients in increasing degree.
#     **Arguments:**
#     - `a (Poly1)`: The polynomial.
#     - `c (int)`: The scalar factor.
#     **Returns:**
#     - `Poly1`: The trimmed product `c*a`.
#     """
#     return _trim(c*x for x in a)
# def _mul(a: Poly1,b: Poly1)->Poly1:
#     """
#     **Description:**
#     Multiplies two integer polynomials. Polynomials are stored as `Poly1 = tuple[int, ...]` of coefficients in increasing degree.
#     **Arguments:**
#     - `a (Poly1)`: First factor.
#     - `b (Poly1)`: Second factor.
#     **Returns:**
#     - `Poly1`: The trimmed product `a*b`.
#     """
#     z=[0]*(len(a)+len(b)-1)
#     for i, x in enumerate(a):
#         for j, y in enumerate(b):
#             z[i+j]+=x*y
#     return _trim(z)
# def _t_minus_one_pow(k: int)->Poly1:
#     """
#     **Description:**
#     Coefficients of the polynomial `(t-1)^k`. Polynomials are stored as `Poly1 = tuple[int, ...]` of coefficients in increasing degree.
#     **Arguments:**
#     - `k (int)`: The nonnegative exponent.
#     **Returns:**
#     - `Poly1`: The coefficients of `(t-1)^k`.
#     """
#     return tuple(comb(k,i)*(-1)**(k-i) for i in range(k+1))
# def _h_star_from_counts(counts: Sequence[int],d: int)->Poly1:
#     """
#     **Description:**
#     Recover h* from L_P(0),...,L_P(d) exactly.
#     **Arguments:**
#     - `counts (Sequence[int])`: The Ehrhart values L_P(0),...,L_P(d) (at least d+1 entries).
#     - `d (int)`: The dimension of the polytope P.
#     **Returns:**
#     - `Poly1`: The trimmed h*-polynomial coefficients in increasing degree.
#     """
#     if len(counts)<d+1:
#         raise ValueError(f"need at least {d+1} Ehrhart values")
#     return _trim(sum((-1)**(j-i)*comb(d+1,j-i)*int(counts[i]) for i in range(j+1)) for j in range(d+1))
# ---- Obsolete: superseded by latte_lattice_points (kept as comments for reference) ----
# def write_latte_vrep(vertices: ArrayLike|Polytope|PolytopeFace,filename: str|Path)->None:
#     """
#     **Description:**
#     Write integral vertices in LattE's native homogenized V-representation.
#     Each vertex ``v`` is written as ``1 v_1 ... v_d``.  Native LattE V-input
#     must be full-dimensional in the supplied coordinates; lower-dimensional
#     faces are first put into intrinsic lattice coordinates by
#     :func:`_intrinsic_lattice_vertices`.
#     **Arguments:**
#     - `vertices (ArrayLike|Polytope|PolytopeFace)`: Integer vertices, one per row, or a CYTools object whose `vertices()` are used.
#     - `filename (str|Path)`: Output filename.
#     **Returns:**
#     - `None`
#     """
#     vertices=_integer_points(vertices)
#     n_vertices, dim=vertices.shape
#     with open(filename,"w",encoding="utf-8") as stream:
#         stream.write(f"{n_vertices} {dim+1}\n")
#         for row in vertices:
#             stream.write("1 "+" ".join(map(str,map(int,row)))+"\n")
# ---- End of obsolete block ----
# def _intrinsic_lattice_vertices(vertices: ArrayLike|Polytope|PolytopeFace,d: int)->np.ndarray:
#     """
#     **Description:**
#     Put a d-dimensional lattice face into full-dimensional Z^d coordinates.
#     LattE's native V-representation does not handle a polytope that is
#     lower-dimensional in its supplied ambient coordinates.  CYTools already
#     uses an exact unimodular LLL transformation for this situation.  Reusing
#     that transformation preserves the affine lattice and hence the Ehrhart
#     polynomial, while avoiding construction/enumeration of a new Polytope.
#     **Arguments:**
#     - `vertices (ArrayLike|Polytope|PolytopeFace)`: Integer vertices of the face, one per row, or a CYTools object whose `vertices()` are used.
#     - `d (int)`: The intrinsic dimension of the face.
#     **Returns:**
#     - `np.ndarray`: Integer array of shape (n, d) with the vertices in intrinsic lattice coordinates.
#     """
#     verts=_integer_points(vertices)
#     if d<0:
#         raise ValueError("intrinsic dimension must be nonnegative")
#     if d==0:
#         return np.zeros((len(verts),0),dtype=np.int64)
#     if d>verts.shape[1]:
#         raise ValueError("intrinsic dimension exceeds ambient dimension")
#     translated=np.asarray(verts-verts[0],dtype=np.int64)
#     if d==verts.shape[1] and _affine_dim(translated)==d:
#         return translated
#     try:
#         from cytools.utils import lll_reduce
#     except ImportError as e:
#         raise ImportError("lower-dimensional LattE V-input requires cytools.utils.lll_reduce") from e
#     reduced=np.asarray(lll_reduce(translated),dtype=np.int64)
#     if reduced.ndim!=2 or reduced.shape!=translated.shape:
#         raise RuntimeError("CYTools lll_reduce returned an unexpected shape")
#     intrinsic=reduced[:,reduced.shape[1]-d :]
#     if intrinsic.shape[1]!=d or _affine_dim(intrinsic)!=d:
#         raise RuntimeError("failed to obtain full-dimensional intrinsic lattice coordinates")
#     return np.unique(intrinsic,axis=0)
# def _cytools_work_is_small(vertices: np.ndarray,d: int,cutoff: int)->bool:
#     """
#     **Description:**
#     Cheap upper-bound proxy for the total lattice-point enumeration problem
#     CYTools faces when computing L_P(1), ..., L_P(d).
#     The face must already be in intrinsic Z^d coordinates.
#     Returns True as soon as the total size of the axis-aligned bounding boxes
#     is known to be <= cutoff, and exits early once the cutoff is exceeded.
#     **Arguments:**
#     - `vertices (np.ndarray)`: Vertices of the face in intrinsic coordinates, one per row.
#     - `d (int)`: The dimension of the face.
#     - `cutoff (int)`: Maximal total number of bounding-box points considered small.
#     **Returns:**
#     - `bool`: `True` if the enumeration work is at most `cutoff`, otherwise `False`.
#     """
#     if d<=0:
#         return True
#     widths=[int(w) for w in np.ptp(vertices,axis=0)]
#     total=0
#     for k in range(1,d+1):
#         box_points=1
#         for w in widths:
#             box_points*=k*w+1
#             if box_points>cutoff:
#                 return False
#         total+=box_points
#         if total>cutoff:
#             return False
#     return True
# def _parse_latte_ehrhart_polynomial(stdout: str,d: int)->tuple[Fraction,...]:
#     """
#     **Description:**
#     Parse LattE's exact univariate Ehrhart polynomial into coefficients.
#     LattE normally prints, for example,
#         + 1 * t^0 + 10/3 * t^1 + 8 * t^2 + 20/3 * t^3
#     The returned tuple is ``(c_0,...,c_d)`` for ``sum c_i t^i``.
#     **Arguments:**
#     - `stdout (str)`: Standard output of a LattE `--ehrhart-polynomial` call.
#     - `d (int)`: The dimension of the polytope (expected degree).
#     **Returns:**
#     - `tuple[Fraction,...]`: The coefficients ``(c_0,...,c_d)``.
#     """
#     lines=[line.strip() for line in stdout.splitlines() if line.strip()]
#     candidates=[line for line in lines if re.search(r"\bt(?:\s*\^\s*\d+)?\b",line)]
#     if not candidates:
#         raise RuntimeError(f"could not find an Ehrhart polynomial in LattE output\n\nSTDOUT:\n{stdout}")
#     expr=candidates[-1].replace("T","t")
#     if not expr.startswith(("+","-")):
#         expr="+"+expr
#     terms=re.findall(r"([+-])\s*([^+-]+)",expr)
#     coeffs=[Fraction(0) for _ in range(d+1)]
#     seen=False
#     for sign, body in terms:
#         body=re.sub(r"\s+","",body)
#         if not body:
#             continue
#         sgn=1 if sign=="+" else-1
#         if "t" in body:
#             m=re.fullmatch(r"(?:(\d+(?:/\d+)?)\*?)?t(?:\^(\d+))?",body,)
#             if m is None:
#                 continue
#             coefficient=Fraction(m.group(1) or "1")*sgn
#             power=int(m.group(2) or "1")
#         else:
#             try:
#                 coefficient=Fraction(body)*sgn
#             except ValueError:
#                 continue
#             power=0
#         if power>d:
#             raise RuntimeError(f"LattE returned Ehrhart degree {power}, expected at most {d}: {expr}")
#         coeffs[power]+=coefficient
#         seen=True
#     if not seen:
#         raise RuntimeError(f"could not parse LattE Ehrhart polynomial: {expr!r}")
#     if coeffs[0]!=1:
#         raise RuntimeError(f"unexpected Ehrhart constant coefficient {coeffs[0]} (expected 1): {expr}")
#     if d>0 and coeffs[d]==0:
#         raise RuntimeError(f"LattE returned a polynomial of degree below face dimension {d}: {expr}")
#     return tuple(coeffs)
# def _ehrhart_values_from_coefficients(coefficients: Sequence[Fraction],d: int)->tuple[int,...]:
#     """
#     **Description:**
#     Evaluate an exact Ehrhart polynomial at 0,...,d.
#     **Arguments:**
#     - `coefficients (Sequence[Fraction])`: The coefficients ``(c_0,...,c_d)`` of the Ehrhart polynomial.
#     - `d (int)`: The dimension of the polytope.
#     **Returns:**
#     - `tuple[int,...]`: The values L_P(0),...,L_P(d).
#     """
#     values: List[int]=[]
#     for k in range(d+1):
#         value=sum(c*(k**j) for j,c in enumerate(coefficients))
#         if value.denominator!=1:
#             raise RuntimeError(f"Ehrhart polynomial evaluated non-integrally at k={k}: {value}")
#         values.append(int(value))
#     return tuple(values)
# def ehrhart_polynomial_latte_vrep(vertices: ArrayLike|Polytope|PolytopeFace,*,intrinsic_dim: int|None=None,count_bin: str="count",latte_options: Sequence[str]=("--redundancy-check=none",),)->tuple[Fraction,...]:
#     """
#     **Description:**
#     Compute an exact Ehrhart polynomial with LattE from vertex data.
#     Returns coefficients ``(c_0,...,c_d)`` of ``L_P(t)``.  If the vertices are
#     embedded in a larger ambient lattice (as every proper face of a support
#     polytope is), pass its intrinsic dimension; the face is then moved to
#     full-dimensional intrinsic lattice coordinates before LattE is called.
#     **Arguments:**
#     - `vertices (ArrayLike|Polytope|PolytopeFace)`: Integer vertices, one per row, or a CYTools object whose `vertices()` are used.
#     - `intrinsic_dim (int|None)`: The intrinsic dimension of the polytope; if `None` it is computed from the vertices. Defaults to `None`.
#     - `count_bin (str)`: Name of LattE's `count` executable if it is on PATH, or its full path. Defaults to `"count"`.
#     - `latte_options (Sequence[str])`: Extra command-line options passed to LattE. Defaults to `("--redundancy-check=none",)`.
#     **Returns:**
#     - `tuple[Fraction,...]`: The coefficients ``(c_0,...,c_d)`` of ``L_P(t)``.
#     """
#     verts=_integer_points(vertices)
#     d=_affine_dim(verts) if intrinsic_dim is None else int(intrinsic_dim)
#     if d<0:
#         raise ValueError("intrinsic_dim must be nonnegative")
#     if d==0:
#         return (Fraction(1),)
#     executable=shutil.which(count_bin)
#     if executable is None and Path(count_bin).exists():
#         executable=str(Path(count_bin).resolve())
#     if executable is None:
#         raise FileNotFoundError(f"Could not find LattE executable {count_bin!r}. Pass its full path " "as count_bin or put 'count' on PATH.")
#     intrinsic=_intrinsic_lattice_vertices(verts,d)
#     with tempfile.TemporaryDirectory(prefix="stringy_e_latte_") as tmp:
#         path=Path(tmp)/"face.vrep.latte"
#         write_latte_vrep(intrinsic,path)
#         cmd=[executable,"--vrep","--ehrhart-polynomial",*tuple(latte_options),str(path),]
#         process=subprocess.run(cmd,cwd=tmp,capture_output=True,text=True,check=False,)
#         if process.returncode!=0:
#             raise RuntimeError("LattE failed to compute an Ehrhart polynomial.\n" f"Command: {' '.join(cmd)}\n\n" f"STDOUT:\n{process.stdout}\n\nSTDERR:\n{process.stderr}")
#         return _parse_latte_ehrhart_polynomial(process.stdout,d)
# ---- Obsolete: superseded by latte_lattice_points (kept as comments for reference) ----
# def _integer_points(a: ArrayLike|Polytope|PolytopeFace)->np.ndarray:
#     """
#     **Description:**
#     Extract integer vertices/points from an array or CYTools-like object.
#     **Arguments:**
#     - `a (ArrayLike|Polytope|PolytopeFace)`: A nonempty two-dimensional integral array (or list/tuple), or an object with a `vertices()` method.
#     **Returns:**
#     - `np.ndarray`: The points as an `int64` array of shape (n, dim).
#     """
#     if isinstance(a,np.ndarray) or isinstance(a,(list,tuple)):
#         x=np.asarray(a)
#     elif hasattr(a,"vertices"):
#         x=np.asarray(a.vertices())
#     else:
#         raise TypeError("expected an integer array or a CYTools Polytope-like object")
#     if x.ndim!=2 or len(x)==0:
#         raise ValueError("points must be a nonempty two-dimensional array")
#     if not np.all(np.equal(x,np.rint(x))):
#         raise ValueError("all coordinates must be integers")
#     return np.asarray(x,dtype=object).astype(np.int64)
# ---- End of obsolete block ----
# def _integer_vector(a: ArrayLike,*,name: str,length: int|None=None)->np.ndarray:
#     """
#     **Description:**
#     Validates that `a` is a one-dimensional integral vector (of the given length) and converts it to an `int64` array.
#     **Arguments:**
#     - `a (ArrayLike)`: The vector.
#     - `name (str)`: Name of the vector used in error messages.
#     - `length (int|None)`: Required length; not checked if `None`. Defaults to `None`.
#     **Returns:**
#     - `np.ndarray`: The vector as an `int64` array.
#     """
#     x=np.asarray(a)
#     if x.ndim!=1:
#         raise ValueError(f"{name} must be a one-dimensional vector")
#     if length is not None and len(x)!=length:
#         raise ValueError(f"{name} must have length {length}")
#     if not np.all(np.equal(x,np.rint(x))):
#         raise ValueError(f"{name} must be integral")
#     return np.asarray(x,dtype=object).astype(np.int64)
# def cayley_support(delta1: ArrayLike|Polytope,delta2: ArrayLike|Polytope,*,as_cytools: bool=False,backend: str|None=None)->np.ndarray|Polytope:
#     """
#     **Description:**
#     Return conv((Delta1,1,0) union (Delta2,0,1)).
#     ``delta1`` and ``delta2`` must use the same lattice coordinates (normally
#     six columns).  By default an integer NumPy array is returned; request a
#     CYTools Polytope with ``as_cytools=True``.
#     **Arguments:**
#     - `delta1 (ArrayLike|Polytope)`: First Cayley summand (its vertices or points, one per row).
#     - `delta2 (ArrayLike|Polytope)`: Second Cayley summand (its vertices or points, one per row).
#     - `as_cytools (bool)`: If `True`, return a CYTools Polytope. Defaults to `False`.
#     - `backend (str|None)`: Polytope backend used if `as_cytools=True`. Defaults to `None`.
#     **Returns:**
#     - `np.ndarray|Polytope`: The unique Cayley points as an `int64` array, or the Cayley Polytope if `as_cytools=True`.
#     """
#     a, b=_integer_points(delta1), _integer_points(delta2)
#     if a.shape[1]!=b.shape[1]:
#         raise ValueError("Delta_1 and Delta_2 must have the same ambient dimension")
#     ca=np.hstack((a,np.tile([[1,0]],(len(a),1))))
#     cb=np.hstack((b,np.tile([[0,1]],(len(b),1))))
#     out=np.unique(np.vstack((ca,cb)),axis=0)
#     if not as_cytools:
#         return out
#     try:
#         from cytools import Polytope
#     except ImportError as e:
#         raise ImportError("CYTools is required for as_cytools=True") from e
#     return Polytope(out,**({"backend": backend} if backend else {}))
# def cayley_support_2part(delta1: ArrayLike|Polytope,delta2: ArrayLike|Polytope,**kwargs: Any)->np.ndarray|Polytope:
#     """
#     **Description:**
#     Descriptive alias for :func:`cayley_support`.
#     **Arguments:**
#     - `delta1 (ArrayLike|Polytope)`: First Cayley summand.
#     - `delta2 (ArrayLike|Polytope)`: Second Cayley summand.
#     - `kwargs (Any)`: Keyword arguments passed to `cayley_support` (`as_cytools`, `backend`).
#     **Returns:**
#     - `np.ndarray|Polytope`: The output of `cayley_support`.
#     """
#     return cayley_support(delta1,delta2,**kwargs)
# def _primitive_rows(rays: ArrayLike)->np.ndarray:
#     """
#     **Description:**
#     Primitive, unique integral ray generators, with their signs preserved.
#     **Arguments:**
#     - `rays (ArrayLike)`: Nonzero integral ray generators, one per row.
#     **Returns:**
#     - `np.ndarray`: The sorted unique primitive rays as an `int64` array.
#     """
#     r=_integer_points(rays)
#     if np.any(np.all(r==0,axis=1)):
#         raise ValueError("a cone ray cannot be zero")
#     out=[]
#     for row in r:
#         g=int(np.gcd.reduce(np.abs(row)))
#         out.append(tuple(map(int,row//g)))
#     return np.asarray(sorted(set(out)),dtype=np.int64)
# @dataclass
# class CayleyGorensteinData:
#     """CYTools objects for a two-part Cayley cone and its Gorenstein dual."""
#     primal_cone: Any
#     dual_cone: Any
#     primal_support: Any
#     dual_support: Any
#     primal_vertices: np.ndarray
#     dual_vertices: np.ndarray
#     degree: np.ndarray
# @dataclass
# class GorensteinConeData:
#     """Validated precomputed reflexive Gorenstein cone data.
#     ``degree_C`` lies in the lattice dual to ``C`` and has value one on
#     primitive extremal rays of ``C``.  ``degree_C_dual`` satisfies the
#     analogous condition on ``C_dual``.
#     """
#     primal_cone: Any
#     dual_cone: Any
#     primal_support: Any
#     dual_support: Any
#     primal_vertices: np.ndarray
#     dual_vertices: np.ndarray
#     degree_C: np.ndarray
#     degree_C_dual: np.ndarray
#     index: int
# def cayley_gorenstein_data(delta1: ArrayLike|Polytope,delta2: ArrayLike|Polytope,*,interior_point: ArrayLike|None=None,poly_backend: str|None=None,check: bool=True,)->CayleyGorensteinData:
#     """
#     **Description:**
#     Build primal/dual cones and degree-one supports using CYTools only.
#     CYTools performs cone dualization lazily; the expensive conversion from the
#     dual H-representation to rays occurs at ``dual_cone.rays()``.  For a
#     reflexive Gorenstein cone, its primitive extremal rays all have degree one
#     and therefore are precisely the vertices of the dual support.
#     ``interior_point`` is the unique interior lattice point of Delta1+Delta2.
#     By default it is inferred with CYTools.  The primal degree element is
#     ``(interior_point,1,1)``.
#     **Arguments:**
#     - `delta1 (ArrayLike|Polytope)`: First Cayley summand (Polytope or its points, one per row).
#     - `delta2 (ArrayLike|Polytope)`: Second Cayley summand (Polytope or its points, one per row).
#     - `interior_point (ArrayLike|None)`: The integral interior point of Delta1+Delta2; inferred if `None`. Defaults to `None`.
#     - `poly_backend (str|None)`: Backend for the CYTools Polytopes that are constructed. Defaults to `None`.
#     - `check (bool)`: Whether to verify reflexivity, dimensions and the Gorenstein property. Defaults to `True`.
#     **Returns:**
#     - `CayleyGorensteinData`: The primal/dual cones, supports, their vertices and the primal degree vector.
#     """
#     try:
#         from cytools import Cone, Polytope
#     except ImportError as e:
#         raise ImportError("cayley_gorenstein_data requires CYTools") from e
#     def cy_poly(obj: Polytope|ArrayLike)->Polytope:
#         """
#         **Description:**
#         Returns `obj` unchanged if it is already a CYTools polytope, and otherwise builds the CYTools `Polytope` spanned by its rows (using `poly_backend` if given).
#         **Arguments:**
#         - `obj (Polytope|ArrayLike)`: A CYTools polytope or an integer array whose rows are points.
#         **Returns:**
#         - `Polytope`: The corresponding CYTools polytope.
#         """
#         if hasattr(obj,"vertices") and hasattr(obj,"faces"):
#             return obj
#         kw={"backend": poly_backend} if poly_backend else {}
#         return Polytope(_integer_points(obj),**kw)
#     poly1, poly2=cy_poly(delta1), cy_poly(delta2)
#     d1, d2=_integer_points(poly1), _integer_points(poly2)
#     primal_vertices=cayley_support(d1,d2)
#     base_dim=d1.shape[1]
#     if d2.shape[1]!=base_dim:
#         raise ValueError("Delta_1 and Delta_2 must have the same ambient dimension")
#     minkowski_sum=poly1.minkowski_sum(poly2)
#     if int(minkowski_sum.dimension())!=base_dim:
#         raise ValueError("Delta_1 + Delta_2 is not full-dimensional in the base lattice")
#     if interior_point is None:
#         if minkowski_sum.is_reflexive():
#             p0=np.zeros(base_dim,dtype=np.int64)
#         else:
#             interior=np.asarray(minkowski_sum.interior_points(),dtype=np.int64)
#             if interior.ndim!=2:
#                 interior=interior.reshape((-1,base_dim))
#             if len(interior)!=1:
#                 same=np.array_equal(np.asarray(sorted(map(tuple,d1))),np.asarray(sorted(map(tuple,d2))),)
#                 hint=(" The two inputs are identical; stringy_e_cayley expects the " "two Cayley summands, not a primal/dual pair or two copies of " "the anticanonical polytope." if same else "")
#                 raise ValueError("Delta_1 + Delta_2 must have exactly one interior lattice " f"point, but CYTools found {len(interior)}.{hint}")
#             p0=interior[0]
#     else:
#         p0=np.asarray(interior_point)
#     if p0.shape!=(base_dim,) or not np.all(np.equal(p0,np.rint(p0))):
#         raise ValueError(f"interior_point must be an integral vector of length {base_dim}")
#     if check:
#         centered_sum=Polytope(_integer_points(minkowski_sum)-np.asarray(p0,dtype=np.int64),**({"backend": poly_backend} if poly_backend else {}),)
#         if not centered_sum.is_reflexive():
#             raise ValueError("Delta_1 + Delta_2, translated by its interior point, is not " "reflexive; the inputs do not define the required Gorenstein " "Cayley construction")
#     degree=np.concatenate((np.asarray(p0,dtype=np.int64),[1,1]))
#     primal_cone=Cone(rays=primal_vertices,check=check)
#     dual_cone=primal_cone.dual_cone()
#     dual_vertices=_primitive_rows(dual_cone.extremal_rays())
#     pair=primal_vertices@dual_vertices.T
#     for j in range(pair.shape[1]):
#         if np.all(pair[:,j]<=0) and np.any(pair[:,j]<0):
#             dual_vertices[j]*=-1
#             pair[:,j]*=-1
#     if np.any(pair<0):
#         i, j=map(int,np.argwhere(pair<0)[0])
#         raise RuntimeError("CYTools cone dualization returned an invalid ray orientation: " f"<primal_ray[{i}], dual_ray[{j}]> = {int(pair[i,j])}")
#     degrees=dual_vertices@degree
#     if np.any(degrees!=1):
#         values=sorted(set(map(int,degrees)))
#         raise ValueError("the dual cone is not Gorenstein for degree " f"{degree.tolist()}; dual-ray degrees are {values}")
#     kw={"backend": poly_backend} if poly_backend else {}
#     primal_support=Polytope(primal_vertices,**kw)
#     dual_support=Polytope(dual_vertices,**kw)
#     if check:
#         expected_dim=base_dim+1
#         if int(primal_cone.dimension())!=base_dim+2:
#             raise ValueError("the Cayley cone is not full-dimensional")
#         if int(primal_support.dimension())!=expected_dim:
#             raise ValueError(f"primal support must have affine dimension {expected_dim}")
#         if int(dual_support.dimension())!=expected_dim:
#             raise ValueError(f"dual support must have affine dimension {expected_dim}")
#         if np.any(~np.any(pair==0,axis=0)) or np.any(~np.any(pair==0,axis=1)):
#             raise ValueError("some support vertex lies on no dual facet")
#     return CayleyGorensteinData(primal_cone=primal_cone,dual_cone=dual_cone,primal_support=primal_support,dual_support=dual_support,primal_vertices=primal_vertices,dual_vertices=dual_vertices,degree=degree,)
# def dual_support_vertices_cytools(delta1: Polytope|ArrayLike,delta2: Polytope|ArrayLike,*,interior_point: ArrayLike|None=None,check: bool=True,)->np.ndarray:
#     """
#     **Description:**
#     Fast CYTools computation of dual degree-one support vertices.
#     The vertices are the primitive extremal rays of the dual of the two-part Cayley cone of `delta1` and `delta2`, see :func:`cayley_gorenstein_data`.
#     **Arguments:**
#     - `delta1 (Polytope|ArrayLike)`: The first Cayley summand, as a CYTools polytope or an integer array of points.
#     - `delta2 (Polytope|ArrayLike)`: The second Cayley summand, as a CYTools polytope or an integer array of points.
#     - `interior_point (ArrayLike|None)`: The unique interior lattice point of `delta1 + delta2`. If `None`, it is inferred with CYTools. Defaults to `None`.
#     - `check (bool)`: Whether to run the reflexivity and dimension checks. Defaults to `True`.
#     **Raises:**
#     - `ImportError`: Raised if CYTools is not available.
#     - `ValueError`: Raised if the summands do not define a reflexive Gorenstein Cayley cone.
#     **Returns:**
#     - `np.ndarray`: Integer array whose rows are the vertices of the dual degree-one support.
#     """
#     return cayley_gorenstein_data(delta1,delta2,interior_point=interior_point,check=check).dual_vertices
# def validate_gorenstein_cone_data(P: Polytope|ArrayLike,C: Cone,C_dual: Cone,degree_C: ArrayLike,degree_C_dual: ArrayLike,*,pairing: ArrayLike|None=None,expected_index: int|None=None,poly_backend: str|None=None,check: bool=True,)->GorensteinConeData:
#     """
#     **Description:**
#     Validate precomputed cone data and construct the dual support polytope.
#     Convention: ``degree_C`` grades ``C`` and therefore lies in the lattice of
#     ``C_dual``; ``degree_C_dual`` grades ``C_dual`` and lies in the lattice of
#     ``C``.  If the two supplied vectors satisfy the equations only after being
#     swapped, they are swapped automatically.
#     **Arguments:**
#     - `P (Polytope|ArrayLike)`: The degree-one support of `C`, as a CYTools polytope or an integer array of points. Its vertices must be the primitive extremal rays of `C`.
#     - `C (Cone)`: The reflexive Gorenstein cone (CYTools `Cone`).
#     - `C_dual (Cone)`: The dual cone of `C` (CYTools `Cone`).
#     - `degree_C (ArrayLike)`: Integer Gorenstein degree vector grading `C`.
#     - `degree_C_dual (ArrayLike)`: Integer Gorenstein degree vector grading `C_dual`.
#     - `pairing (ArrayLike|None)`: Integer square matrix `B` defining the pairing `<x,y> = x B y`. If `None`, the identity is used. Defaults to `None`.
#     - `expected_index (int|None)`: If given, the Gorenstein index must equal this value. Defaults to `None`.
#     - `poly_backend (str|None)`: Backend passed to the CYTools `Polytope` constructor. Defaults to `None`.
#     - `check (bool)`: Whether to also check full-dimensionality and the support dimensions. Defaults to `True`.
#     **Raises:**
#     - `ImportError`: Raised if CYTools is not available.
#     - `TypeError`: Raised if `C` or `C_dual` is not a CYTools cone.
#     - `ValueError`: Raised if the data are inconsistent (wrong dimensions, non-integral pairing, non-Gorenstein degree vectors, non-dual cones, wrong index, or wrong support).
#     **Returns:**
#     - `GorensteinConeData`: The validated cones, supports, degree vectors and Gorenstein index.
#     """
#     try:
#         from cytools import Polytope
#     except ImportError as e:
#         raise ImportError("validate_gorenstein_cone_data requires CYTools") from e
#     for name, cone in (("C",C),("C_dual",C_dual)):
#         if not hasattr(cone,"extremal_rays") or not hasattr(cone,"dimension"):
#             raise TypeError(f"{name} must be a CYTools Cone")
#     primal_rays=_primitive_rows(C.extremal_rays())
#     dual_rays=_primitive_rows(C_dual.extremal_rays())
#     ambient=primal_rays.shape[1]
#     if dual_rays.shape[1]!=ambient:
#         raise ValueError("C and C_dual have different ambient dimensions")
#     B=np.eye(ambient,dtype=np.int64) if pairing is None else np.asarray(pairing)
#     if B.shape!=(ambient,ambient):
#         raise ValueError(f"pairing must have shape {(ambient,ambient)}")
#     if not np.all(np.equal(B,np.rint(B))):
#         raise ValueError("pairing matrix must be integral")
#     B=np.asarray(B,dtype=np.int64)
#     dC=_integer_vector(degree_C,name="degree_C",length=ambient)
#     dD=_integer_vector(degree_C_dual,name="degree_C_dual",length=ambient)
#     def heights_on_C(v: np.ndarray)->np.ndarray:
#         """
#         **Description:**
#         Pairings of the primitive extremal rays of `C` with `v`.
#         **Arguments:**
#         - `v (np.ndarray)`: An integer vector in the lattice of `C_dual`.
#         **Returns:**
#         - `np.ndarray`: The values `<ray, v>` for all primitive extremal rays of `C`.
#         """
#         return primal_rays@B@v
#     def heights_on_dual(v: np.ndarray)->np.ndarray:
#         """
#         **Description:**
#         Pairings of the primitive extremal rays of `C_dual` with `v`.
#         **Arguments:**
#         - `v (np.ndarray)`: An integer vector in the lattice of `C`.
#         **Returns:**
#         - `np.ndarray`: The values `<v, ray>` for all primitive extremal rays of `C_dual`.
#         """
#         return dual_rays@B.T@v
#     direct=np.all(heights_on_C(dC)==1) and np.all(heights_on_dual(dD)==1)
#     swapped=np.all(heights_on_C(dD)==1) and np.all(heights_on_dual(dC)==1)
#     if not direct and swapped:
#         dC, dD=dD, dC
#         direct=True
#     if not direct:
#         raise ValueError("the supplied vectors are not Gorenstein degree vectors for these " "cones. Required: every primitive extremal ray of C has degree 1 " "under degree_C, and every primitive extremal ray of C_dual has " "degree 1 under degree_C_dual. Observed degree sets: " f"C={sorted(set(map(int,heights_on_C(dC))))}, " f"C_dual={sorted(set(map(int,heights_on_dual(dD))))}")
#     cross=primal_rays@B@dual_rays.T
#     if np.any(cross<0):
#         i, j=map(int,np.argwhere(cross<0)[0])
#         raise ValueError("C_dual is not dual to C for the supplied pairing: " f"pairing of extremal rays ({i},{j}) is {int(cross[i,j])}")
#     index=int(dD@B@dC)
#     if index<1:
#         raise ValueError(f"the Gorenstein index must be positive, got {index}")
#     if expected_index is not None and index!=int(expected_index):
#         raise ValueError(f"expected Gorenstein index {expected_index}, got {index}")
#     kw={"backend": poly_backend} if poly_backend else {}
#     primal_support=P if hasattr(P,"faces") else Polytope(_integer_points(P),**kw)
#     support_vertices=_integer_points(primal_support)
#     if set(map(tuple,support_vertices))!=set(map(tuple,primal_rays)):
#         raise ValueError("vertices of P are not exactly the primitive extremal rays of C")
#     dual_support=Polytope(dual_rays,**kw)
#     if check:
#         cone_dim=int(C.dimension())
#         if cone_dim!=ambient or int(C_dual.dimension())!=ambient:
#             raise ValueError("C and C_dual must both be full-dimensional")
#         expected_support_dim=ambient-1
#         if int(primal_support.dimension())!=expected_support_dim:
#             raise ValueError(f"P must have affine dimension {expected_support_dim}")
#         if int(dual_support.dimension())!=expected_support_dim:
#             raise ValueError(f"the dual degree-one support must have dimension {expected_support_dim}")
#         if np.any(~np.any(cross==0,axis=0)) or np.any(~np.any(cross==0,axis=1)):
#             raise ValueError("some extremal ray lies on no facet of the dual cone")
#     return GorensteinConeData(primal_cone=C,dual_cone=C_dual,primal_support=primal_support,dual_support=dual_support,primal_vertices=support_vertices,dual_vertices=dual_rays,degree_C=dC,degree_C_dual=dD,index=index,)
# def complete_split_witness_index_two(opposite_support: Polytope|ArrayLike,degree: ArrayLike,)->tuple[np.ndarray,np.ndarray]|None:
#     """
#     **Description:**
#     Return a two-point complete-splitting witness, if one exists.
#     For a reflexive Gorenstein cone ``C`` of index two, the Batyrev--Nill
#     criterion says that ``C`` is completely split exactly when its Gorenstein
#     degree element is a sum of two lattice points of the degree-one support of
#     ``C.dual()``.  Consequently, to test ``C`` pass the *opposite* support and
#     the degree vector that grades ``C``.
#     The result is ``(e1, e2)`` with ``e1 + e2 == degree``, or ``None``.  This
#     is an exact lattice test; it does not try to recognize a Cayley embedding
#     from floating-point affine geometry.
#     **Arguments:**
#     - `opposite_support (Polytope|ArrayLike)`: The degree-one support of the dual cone, as a CYTools polytope or an integer array of points.
#     - `degree (ArrayLike)`: The integer Gorenstein degree vector grading `C`.
#     **Raises:**
#     - `ImportError`: Raised if array input is given and CYTools is not available.
#     - `ValueError`: Raised if the support has no or non-integral lattice points, or `degree` has the wrong length or is not integral.
#     **Returns:**
#     - `tuple[np.ndarray,np.ndarray]|None`: The witness `(e1, e2)` of distinct lattice points with `e1 + e2 == degree`, or `None` if `C` is not completely split.
#     """
#     if hasattr(opposite_support,"points"):
#         points=np.asarray(opposite_support.points())
#     else:
#         try:
#             from cytools import Polytope
#         except ImportError as e:
#             raise ImportError("array input to complete_split_witness_index_two requires CYTools") from e
#         points=np.asarray(Polytope(_integer_points(opposite_support)).points())
#     if points.ndim!=2 or len(points)==0:
#         raise ValueError("opposite support has no lattice points")
#     if not np.all(np.equal(points,np.rint(points))):
#         raise ValueError("opposite support returned nonintegral lattice points")
#     points=np.asarray(points,dtype=np.int64)
#     target=_integer_vector(degree,name="degree",length=points.shape[1])
#     point_set={tuple(map(int,p)) for p in points}
#     for first in sorted(point_set):
#         second=tuple(int(target[i])-first[i] for i in range(len(target)))
#         if second in point_set and second!=first:
#             return (np.asarray(first,dtype=np.int64),np.asarray(second,dtype=np.int64),)
#     return None
# def _affine_dim(v: np.ndarray)->int:
#     """
#     **Description:**
#     Affine dimension of a point set, i.e. the rank of the differences to its first point.
#     **Arguments:**
#     - `v (np.ndarray)`: Array whose rows are the points.
#     **Returns:**
#     - `int`: The dimension of the affine span (0 for at most one point).
#     """
#     if len(v)<=1:
#         return 0
#     return int(np.linalg.matrix_rank(np.asarray(v[1:]-v[0],dtype=float)))
# @dataclass(frozen=True)
# class Face:
#     id: int
#     vertices: FrozenSet[int]
#     dim: int
# class FaceLattice:
#     """Face poset plus exact Ehrhart and Stanley-polynomial operations."""
#     def __init__(self,poly: Polytope|ArrayLike,*,ehrhart_backend: str="auto",latte_count_bin: str="count",latte_options: Sequence[str]=("--redundancy-check=none",),cytools_work_cutoff: int=100_0)->None:
#         """
#         **Description:**
#         Builds the face lattice of a lattice polytope with CYTools and sets up the Ehrhart backend used for the h*-polynomials of its faces.
#         Faces are sorted by dimension, from the empty face to the full polytope.
#         **Arguments:**
#         - `poly (Polytope|ArrayLike)`: The polytope, as a CYTools polytope or an integer array of points.
#         - `ehrhart_backend (str)`: How Ehrhart h*-polynomials of faces are computed: "auto", "latte", "normaliz" or "cytools". Defaults to "auto".
#         - `latte_count_bin (str)`: Name or path of LattE's `count` executable. Defaults to "count".
#         - `latte_options (Sequence[str])`: Extra command-line options passed to LattE. Defaults to `("--redundancy-check=none",)`.
#         - `cytools_work_cutoff (int)`: Work threshold below which the "auto" backend counts lattice points with CYTools instead of LattE. Defaults to `1000`.
#         **Raises:**
#         - `ImportError`: Raised if CYTools is not available, or if the "normaliz" backend is requested without PyNormaliz or the normaliz executable.
#         - `ValueError`: Raised if `ehrhart_backend` is not one of the allowed values.
#         - `FileNotFoundError`: Raised if the "latte" backend is requested but LattE's count executable is not found.
#         **Returns:**
#         - `None`.
#         """
#         if not hasattr(poly,"faces"):
#             try:
#                 from cytools import Polytope
#             except ImportError as e:
#                 raise ImportError("FaceLattice requires CYTools") from e
#             poly=Polytope(_integer_points(poly))
#         self.source=poly
#         self.vertices=_integer_points(poly)
#         self.dim=int(poly.dimension())
#         if ehrhart_backend not in ("auto","latte","normaliz","cytools"):
#             raise ValueError("ehrhart_backend must be 'auto', 'latte', 'normaliz', or 'cytools'")
#         self.ehrhart_backend=ehrhart_backend
#         self._latte_count_bin=latte_count_bin
#         self._latte_options=tuple(latte_options)
#         self._latte_executable=shutil.which(latte_count_bin)
#         if self._latte_executable is None and Path(latte_count_bin).exists():
#             self._latte_executable=str(Path(latte_count_bin).resolve())
#         self._normaliz_cone=None
#         self._normaliz_executable=None
#         if ehrhart_backend=="latte":
#             if self._latte_executable is None:
#                 raise FileNotFoundError(f"ehrhart_backend='latte' requires LattE's " f"{latte_count_bin!r} executable")
#         elif ehrhart_backend=="normaliz":
#             try:
#                 from PyNormaliz import Cone as NormalizCone
#                 self._normaliz_cone=NormalizCone
#             except ImportError:
#                 self._normaliz_executable=shutil.which("normaliz")
#                 if self._normaliz_executable is None:
#                     raise ImportError("ehrhart_backend='normaliz' requires " "PyNormaliz or the normaliz executable")
#         self._cytools_work_cutoff=cytools_work_cutoff
#         self._ehrhart_usage={"analytic": 0,"cytools": 0,"latte": 0,"normaliz": 0,"normaliz-cli": 0,}
#         face_dims=self._face_sets_cytools(poly)
#         face_dims[frozenset()]=-1
#         face_dims[frozenset(range(len(self.vertices)))]=self.dim
#         raw=sorted(face_dims,key=lambda s: (face_dims[s],tuple(sorted(s))))
#         self.faces=tuple(Face(i,s,face_dims[s]) for i,s in enumerate(raw))
#         self.by_vertices={f.vertices: f.id for f in self.faces}
#         self.by_dim: Dict[int,Tuple[int,...]]={d: tuple(f.id for f in self.faces if f.dim==d) for d in range(-1,self.dim+1)}
#         self.empty=self.by_vertices[frozenset()]
#         self.full=self.by_vertices[frozenset(range(len(self.vertices)))]
#     def _face_sets_cytools(self,poly: Polytope)->dict[frozenset[int],int]:
#         """
#         **Description:**
#         Collects all nonempty faces of `poly` as sets of vertex indices (into `self.vertices`) together with their dimensions.
#         **Arguments:**
#         - `poly (Polytope)`: The CYTools polytope whose faces are enumerated.
#         **Returns:**
#         - `dict[frozenset[int],int]`: Map from the vertex-index set of each face to its dimension.
#         """
#         lookup={tuple(map(int,x)): i for i,x in enumerate(self.vertices)}
#         ans: Dict[FrozenSet[int],int]={}
#         for dim, level in enumerate(poly.faces()):
#             for face in level:
#                 key=frozenset(lookup[tuple(map(int,x))] for x in np.asarray(face.vertices()))
#                 ans[key]=dim
#         return ans
#     @lru_cache(maxsize=None)
#     def intrinsic_vertices(self,face_id: int)->np.ndarray:
#         """
#         **Description:**
#         Vertices of a face in its intrinsic lattice Z^d.
#         **Arguments:**
#         - `face_id (int)`: Index of the face in `self.faces`.
#         **Returns:**
#         - `np.ndarray`: Integer array of shape `(n_vertices, d)` with `d = max(dim, 0)`.
#         """
#         face=self.faces[face_id]
#         if face.dim<=0:
#             return np.zeros((len(face.vertices),max(face.dim,0)),dtype=np.int64,)
#         verts=self.vertices[list(face.vertices)]
#         return _intrinsic_lattice_vertices(verts,face.dim,)
#     def _ehrhart_backend_for_face(self,face_id: int,intrinsic: np.ndarray,)->str:
#         """
#         **Description:**
#         Select the Ehrhart backend for one individual face.
#         **Arguments:**
#         - `face_id (int)`: Index of the face in `self.faces`.
#         - `intrinsic (np.ndarray)`: The face's vertices in intrinsic lattice coordinates.
#         **Returns:**
#         - `str`: The backend name: `self.ehrhart_backend` unless it is "auto", otherwise "cytools" or "latte".
#         """
#         if self.ehrhart_backend!="auto":
#             return self.ehrhart_backend
#         d=self.faces[face_id].dim
#         if self._latte_executable is None:
#             return "cytools"
#         if _cytools_work_is_small(intrinsic,d,self._cytools_work_cutoff,):
#             return "cytools"
#         return "latte"
#     @lru_cache(maxsize=None)
#     def interval(self,lo: int,hi: int)->tuple[int,...]:
#         """
#         **Description:**
#         Face interval [lo,hi] of the face poset.
#         **Arguments:**
#         - `lo (int)`: Index of the lower face.
#         - `hi (int)`: Index of the upper face.
#         **Raises:**
#         - `ValueError`: Raised if face `lo` is not contained in face `hi`.
#         **Returns:**
#         - `tuple[int,...]`: Indices of all faces F with lo <= F <= hi.
#         """
#         a, b=self.faces[lo].vertices, self.faces[hi].vertices
#         if not a.issubset(b):
#             raise ValueError("not a face interval")
#         return tuple(f.id for f in self.faces if a.issubset(f.vertices) and f.vertices.issubset(b))
#     @lru_cache(maxsize=None)
#     def g_polynomial(self,lo: int,hi: int)->Poly1:
#         """
#         **Description:**
#         Stanley's g-polynomial of the Eulerian interval [lo,hi].
#         **Arguments:**
#         - `lo (int)`: Index of the lower face.
#         - `hi (int)`: Index of the upper face.
#         **Returns:**
#         - `Poly1`: Coefficient tuple of g, constant term first.
#         """
#         rank=self.faces[hi].dim-self.faces[lo].dim
#         if rank==0:
#             return (1,)
#         h: Poly1=(0,)
#         for x in self.interval(lo,hi):
#             if x==lo:
#                 continue
#             exponent=self.faces[x].dim-self.faces[lo].dim-1
#             h=_add(h,_mul(_t_minus_one_pow(exponent),self.g_polynomial(x,hi)))
#         one_minus_t_h=_mul((1,-1),h)
#         cutoff=(rank-1)//2
#         return _trim(one_minus_t_h[: cutoff+1])
#     def _count_dilate_cytools(self,verts: np.ndarray,k: int)->int:
#         """
#         **Description:**
#         Counts the lattice points of the k-th dilate of a polytope with CYTools.
#         **Arguments:**
#         - `verts (np.ndarray)`: Integer array whose rows are the vertices.
#         - `k (int)`: The dilation factor.
#         **Raises:**
#         - `ImportError`: Raised if CYTools is not available.
#         - `RuntimeError`: Raised if CYTools fails to enumerate the points.
#         **Returns:**
#         - `int`: The number of lattice points of `k*conv(verts)`.
#         """
#         try:
#             from cytools import Polytope
#         except ImportError as e:
#             raise ImportError("lattice-point enumeration requires CYTools") from e
#         try:
#             return int(len(Polytope(k*verts).points()))
#         except Exception as e:
#             raise RuntimeError("CYTools failed while enumerating lattice points of a dilated face") from e
#     @staticmethod
#     def _h_star_from_series(series: list[Any]|tuple[Any,...],d: int)->Poly1:
#         """
#         **Description:**
#         Decode a Normaliz series and recover h* exactly through degree d.
#         **Arguments:**
#         - `series (list[Any]|tuple[Any,...])`: Normaliz Ehrhart series `[numerator, denominator_exponents, shift]`, with integer coefficient/exponent lists and an integer shift.
#         - `d (int)`: Dimension of the polytope.
#         **Raises:**
#         - `RuntimeError`: Raised if the series has an unexpected format or a nonpositive denominator exponent.
#         **Returns:**
#         - `Poly1`: Coefficient tuple of the h*-polynomial, constant term first.
#         """
#         if not isinstance(series,(list,tuple)) or len(series)!=3:
#             raise RuntimeError(f"unexpected Normaliz Ehrhart-series output: {series!r}")
#         numerator, denominator_exponents, shift=series
#         numerator=[int(x) for x in numerator]
#         denominator_exponents=[int(x) for x in denominator_exponents]
#         shift=int(shift)
#         inverse_denominator=[0]*(d+1)
#         inverse_denominator[0]=1
#         for exponent in denominator_exponents:
#             if exponent<=0:
#                 raise RuntimeError("Normaliz returned a nonpositive denominator exponent")
#             for k in range(exponent,d+1):
#                 inverse_denominator[k]+=inverse_denominator[k-exponent]
#         counts=[]
#         for k in range(d+1):
#             counts.append(sum(coefficient*inverse_denominator[k-shift-i] for i,coefficient in enumerate(numerator) if 0<=k-shift-i<=d))
#         return _h_star_from_counts(counts,d)
#     def _h_star_latte(self,verts: ArrayLike,d: int)->Poly1:
#         """
#         **Description:**
#         Compute h* from one exact LattE Ehrhart-polynomial call.
#         **Arguments:**
#         - `verts (ArrayLike)`: Integer vertices of a full-dimensional polytope in its intrinsic lattice.
#         - `d (int)`: Dimension of the polytope.
#         **Raises:**
#         - `FileNotFoundError`: Raised if LattE's count executable cannot be found.
#         - `RuntimeError`: Raised if the LattE call fails.
#         **Returns:**
#         - `Poly1`: Coefficient tuple of the h*-polynomial, constant term first.
#         """
#         coefficients=ehrhart_polynomial_latte_vrep(verts,intrinsic_dim=d,count_bin=self._latte_executable or self._latte_count_bin,latte_options=self._latte_options,)
#         return _h_star_from_counts(_ehrhart_values_from_coefficients(coefficients,d),d)
#     def _h_star_normaliz(self,verts: ArrayLike,d: int)->Poly1:
#         """
#         **Description:**
#         Compute h* with the in-process PyNormaliz binding.
#         **Arguments:**
#         - `verts (ArrayLike)`: Integer vertices of a full-dimensional polytope in its intrinsic lattice.
#         - `d (int)`: Dimension of the polytope.
#         **Raises:**
#         - `RuntimeError`: Raised if the PyNormaliz backend was not initialized or returns unexpected output.
#         **Returns:**
#         - `Poly1`: Coefficient tuple of the h*-polynomial, constant term first.
#         """
#         if self._normaliz_cone is None:
#             raise RuntimeError("PyNormaliz backend was not initialized")
#         cone=self._normaliz_cone(polytope=np.asarray(verts,dtype=int).tolist())
#         series=(cone.EhrhartSeries() if hasattr(cone,"EhrhartSeries") else cone.HilbertSeries())
#         return self._h_star_from_series(series,d)
#     def _h_star_normaliz_cli(self,verts: ArrayLike,d: int)->Poly1:
#         """
#         **Description:**
#         Compute h* with the Normaliz executable when PyNormaliz is absent.
#         **Arguments:**
#         - `verts (ArrayLike)`: Integer vertices of a full-dimensional polytope in its intrinsic lattice.
#         - `d (int)`: Dimension of the polytope.
#         **Raises:**
#         - `RuntimeError`: Raised if the executable was not initialized, fails, or its output cannot be parsed.
#         **Returns:**
#         - `Poly1`: Coefficient tuple of the h*-polynomial, constant term first.
#         """
#         if self._normaliz_executable is None:
#             raise RuntimeError("Normaliz executable backend was not initialized")
#         vertices=np.asarray(verts,dtype=np.int64)
#         with tempfile.TemporaryDirectory(prefix="stringy_e_nmz_") as tmp:
#             input_path=f"{tmp}/face.in"
#             output_path=f"{tmp}/face.out"
#             with open(input_path,"w",encoding="utf-8") as stream:
#                 stream.write(f"amb_space {vertices.shape[1]+1}\n")
#                 stream.write(f"polytope {len(vertices)}\n")
#                 for row in vertices:
#                     stream.write(" ".join(map(str,map(int,row)))+"\n")
#             process=subprocess.run([self._normaliz_executable,"-q",input_path],cwd=tmp,capture_output=True,text=True,check=False,)
#             if process.returncode!=0 or not Path(output_path).exists():
#                 raise RuntimeError("Normaliz failed to compute an Ehrhart series: "+(process.stderr.strip() or process.stdout.strip()))
#             with open(output_path,encoding="utf-8") as stream:
#                 lines=stream.readlines()
#         start=next((i for i,line in enumerate(lines) if "Hilbert series:" in line or "Ehrhart series:" in line),None,)
#         if start is None:
#             raise RuntimeError("Normaliz output contains no Ehrhart/Hilbert series")
#         numerator=[int(x) for x in lines[start+1].split()]
#         denominator_exponents: List[int]=[]
#         shift=next((int(line.split("=",1)[1].strip()) for line in lines[start+2 :] if "shift" in line and "=" in line),0,)
#         for line in lines[start+2 :]:
#             for exponent, multiplicity in re.findall(r"(-?\d+)\s*:\s*(\d+)",line):
#                 denominator_exponents.extend([int(exponent)]*int(multiplicity))
#             if denominator_exponents and not line.strip():
#                 break
#         if not denominator_exponents:
#             raise RuntimeError("could not parse the Normaliz series denominator")
#         return self._h_star_from_series([numerator,denominator_exponents,shift],d)
#     @lru_cache(maxsize=None)
#     def lattice_count(self,face_id: int,k: int)->int:
#         """
#         **Description:**
#         Number of lattice points of the k-th dilate of a face.
#         **Arguments:**
#         - `face_id (int)`: Index of the face in `self.faces`.
#         - `k (int)`: Nonnegative dilation factor.
#         **Raises:**
#         - `ValueError`: Raised if `k` is negative.
#         **Returns:**
#         - `int`: The lattice-point count (0 for the empty face, 1 for `k = 0`).
#         """
#         if k<0:
#             raise ValueError("dilation must be nonnegative")
#         face=self.faces[face_id]
#         if face.dim==-1:
#             return 0
#         if k==0:
#             return 1
#         verts=self.vertices[list(face.vertices)]
#         return self._count_dilate_cytools(verts,k)
#     @lru_cache(maxsize=None)
#     def h_star(self,face_id: int)->Poly1:
#         """
#         **Description:**
#         Ehrhart h*-polynomial; h*_empty=1.
#         **Arguments:**
#         - `face_id (int)`: Index of the face in `self.faces`.
#         **Raises:**
#         - `RuntimeError`: Raised if the selected Ehrhart backend is unknown or fails.
#         **Returns:**
#         - `Poly1`: Coefficient tuple of the h*-polynomial, constant term first.
#         """
#         face=self.faces[face_id]
#         d=face.dim
#         if d==-1:
#             return (1,)
#         if d==0:
#             return (1,)
#         verts=self.vertices[list(face.vertices)]
#         cache_key=(d,tuple(sorted(tuple(map(int,row)) for row in verts)),)
#         if cache_key in _HSTAR_CACHE:
#             return _HSTAR_CACHE[cache_key]
#         if d==1:
#             delta=np.asarray(verts[1]-verts[0],dtype=np.int64,)
#             lattice_length=int(np.gcd.reduce(np.abs(delta)))
#             answer=_trim((1,lattice_length-1))
#             self._ehrhart_usage["analytic"]+=1
#         else:
#             intrinsic=self.intrinsic_vertices(face_id)
#             backend=self._ehrhart_backend_for_face(face_id,intrinsic,)
#             if backend=="latte":
#                 answer=self._h_star_latte(intrinsic,d,)
#             elif backend=="normaliz":
#                 if self._normaliz_cone is not None:
#                     answer=self._h_star_normaliz(intrinsic,d,)
#                 else:
#                     answer=self._h_star_normaliz_cli(intrinsic,d,)
#                     backend="normaliz-cli"
#             elif backend=="cytools":
#                 counts=[1]
#                 for k in range(1,d+1):
#                     counts.append(self._count_dilate_cytools(intrinsic,k,))
#                 answer=_h_star_from_counts(counts,d,)
#             else:
#                 raise RuntimeError(f"unknown Ehrhart backend {backend!r}")
#             self._ehrhart_usage[backend]+=1
#         _HSTAR_CACHE[cache_key]=answer
#         return answer
#     def precompute_s_tilde(self,*,verbose: bool=False,label: str="")->None:
#         """
#         **Description:**
#         Populate all h*, g, and S-tilde caches, optionally showing progress.
#         **Arguments:**
#         - `verbose (bool)`: Whether to print progress. Defaults to `False`.
#         - `label (str)`: Prefix for the progress messages. Defaults to `""`.
#         **Returns:**
#         - `None`.
#         """
#         total=len(self.faces)
#         report_every=max(1,total//20)
#         for i, face in enumerate(self.faces,1):
#             self.s_tilde(face.id)
#             if verbose and (i==1 or i==total or i%report_every==0):
#                 prefix=f"{label}: " if label else ""
#                 print(f"{prefix}{i}/{total} faces",flush=True)
#     @lru_cache(maxsize=None)
#     def s_tilde(self,face_id: int)->Poly1:
#         """
#         **Description:**
#         Borisov--Mavlyutov/Batyrev--Nill S-tilde polynomial.
#         **Arguments:**
#         - `face_id (int)`: Index of the face in `self.faces`.
#         **Returns:**
#         - `Poly1`: Coefficient tuple of S-tilde, constant term first.
#         """
#         p=self.faces[face_id]
#         total: Poly1=(0,)
#         for g in self.interval(self.empty,face_id):
#             fg=self.faces[g]
#             sign=(-1)**(p.dim-fg.dim)
#             total=_add(total,_scale(_mul(self.h_star(g),self.g_polynomial(g,face_id)),sign))
#         return total
# def _translated_hstar_key(verts: ArrayLike,d: int)->tuple[int,tuple[tuple[int,...],...]]:
#     """
#     **Description:**
#     Cheap translation-invariant cache key.
#     **Arguments:**
#     - `verts (ArrayLike)`: Integer vertices of the face.
#     - `d (int)`: Dimension of the face.
#     **Returns:**
#     - `tuple[int,tuple[tuple[int,...],...]]`: The pair `(d, sorted vertices translated so that the first vertex is the origin)`.
#     """
#     v=np.asarray(verts,dtype=np.int64)
#     v=v-v[0]
#     return (d,tuple(sorted(tuple(map(int,row)) for row in v)),)
# def pair_dual_faces(primal: FaceLattice,dual: FaceLattice,pairing: ArrayLike|None=None,*,check: bool=True,)->dict[int,int]:
#     """
#     **Description:**
#     Pair faces by annihilation: F* has vertices y with <x,y>=0 for all x in F.
#     **Arguments:**
#     - `primal (FaceLattice)`: Face lattice of the primal support.
#     - `dual (FaceLattice)`: Face lattice of the dual support.
#     - `pairing (ArrayLike|None)`: Integer matrix `B` defining `<x,y> = x B y`. If `None`, the standard dot product is used. Defaults to `None`.
#     - `check (bool)`: Whether to check nonnegativity, dimension reversal and bijectivity. Defaults to `True`.
#     **Raises:**
#     - `ValueError`: Raised if the pairing is not integral, the dimensions do not match, or the supports are not dual.
#     **Returns:**
#     - `dict[int,int]`: Map from primal face index to dual face index.
#     """
#     if pairing is None:
#         if primal.vertices.shape[1]!=dual.vertices.shape[1]:
#             raise ValueError("coordinate dimensions differ; supply an explicit pairing matrix")
#         pair=primal.vertices@dual.vertices.T
#     else:
#         b=np.asarray(pairing)
#         pair=primal.vertices@b@dual.vertices.T
#     if not np.all(np.equal(pair,np.rint(pair))):
#         raise ValueError("pairings must be integral")
#     pair=np.asarray(pair,dtype=np.int64)
#     if check and np.any(pair<0):
#         ij=tuple(map(int,np.argwhere(pair<0)[0]))
#         raise ValueError(f"negative primal/dual support pairing at vertex pair {ij}")
#     out: Dict[int,int]={}
#     all_dual=frozenset(range(len(dual.vertices)))
#     for f in primal.faces:
#         if f.dim==-1:
#             star=all_dual
#         else:
#             rows=list(f.vertices)
#             star=frozenset(np.flatnonzero(np.all(pair[rows,:]==0,axis=0)).tolist())
#         if star not in dual.by_vertices:
#             raise ValueError(f"zero set for primal face {f.id} is not a dual face")
#         out[f.id]=dual.by_vertices[star]
#         if check and f.dim+dual.faces[out[f.id]].dim!=primal.dim-1:
#             raise ValueError(f"dimension-reversing duality fails at primal face {f.id}")
#     if check and len(set(out.values()))!=len(dual.faces):
#         raise ValueError("face pairing is not a bijection; supports/pairing are inconsistent")
#     return out
# @dataclass
# class StringyEResult:
#     coefficients: Laurent2
#     hodge_numbers: Dict[Tuple[int,int],int]
#     index: int
#     polytope_dim: int
#     cy_dim: int
#     primal: FaceLattice
#     dual: FaceLattice
#     dual_faces: Dict[int,int]
#     primal_completely_split: Optional[bool]=None
#     dual_completely_split: Optional[bool]=None
#     primal_split_witness: Optional[Tuple[np.ndarray,np.ndarray]]=None
#     dual_split_witness: Optional[Tuple[np.ndarray,np.ndarray]]=None
#     cayley_data: Optional[CayleyGorensteinData]=None
#     cone_data: Optional[GorensteinConeData]=None
#     def e_polynomial(self)->str:
#         """
#         **Description:**
#         Human-readable form of the stringy E-function.
#         **Returns:**
#         - `str`: The E-polynomial as formatted by :func:`format_laurent`.
#         """
#         return format_laurent(self.coefficients)
#     def hodge_diamond(self,fill: int=0)->list[list[int]]:
#         """
#         **Description:**
#         Hodge diamond as a nested list.
#         **Arguments:**
#         - `fill (int)`: Value used for entries not stored in `hodge_numbers`. Defaults to `0`.
#         **Returns:**
#         - `list[list[int]]`: Square `(cy_dim+1) x (cy_dim+1)` list with entry `[p][q] = h^{p,q}`.
#         """
#         return [[self.hodge_numbers.get((p,q),fill) for q in range(self.cy_dim+1)] for p in range(self.cy_dim+1)]
#     @property
#     def formal_hodge_numbers(self)->dict[tuple[int,int],int]:
#         """
#         **Description:**
#         Alias emphasizing that a geometric realization needs a split cone.
#         **Returns:**
#         - `dict[tuple[int,int],int]`: The stringy Hodge numbers `{(p, q): h^{p,q}}`.
#         """
#         return self.hodge_numbers
#     def interpretation(self)->str:
#         """
#         **Description:**
#         Describe what the two cone supports geometrically justify.
#         **Returns:**
#         - `str`: A sentence describing the complete-splitting status of the primal and dual cones.
#         """
#         p, q=self.primal_completely_split, self.dual_completely_split
#         if p is True and q is True:
#             return ("both cones are completely split: this is the usual " "Batyrev--Borisov complete-intersection mirror setting")
#         if p is True and q is False:
#             return ("the primal cone gives a complete intersection, but the dual " "cone has no complete-intersection realization from a splitting; " "on the dual side the result is the Gorenstein-cone/generalized-CY " "stringy E-invariant")
#         if p is False and q is True:
#             return ("the dual cone gives a complete intersection, but the primal " "cone has no complete-intersection realization from a splitting")
#         if p is False and q is False:
#             return ("neither cone is completely split: the result is the formal " "Batyrev--Nill stringy E-invariant of the Gorenstein polytope")
#         return "complete-splitting was not tested for this result"
# @dataclass
# class StringyHodgeResult:
#     """Hodge-only result that does not materialize the stringy E-polynomial."""
#     hodge_numbers: Dict[Tuple[int,int],int]
#     index: int
#     polytope_dim: int
#     cy_dim: int
#     primal: FaceLattice
#     dual: FaceLattice
#     dual_faces: Dict[int,int]
#     def hodge_diamond(self,fill: int=0)->list[list[int]]:
#         """
#         **Description:**
#         Hodge diamond as a nested list.
#         **Arguments:**
#         - `fill (int)`: Value used for entries not stored in `hodge_numbers`. Defaults to `0`.
#         **Returns:**
#         - `list[list[int]]`: Square `(cy_dim+1) x (cy_dim+1)` list with entry `[p][q] = h^{p,q}`.
#         """
#         return [[self.hodge_numbers.get((p,q),fill) for q in range(self.cy_dim+1)] for p in range(self.cy_dim+1)]
# def _accumulate_term(out: Laurent2,a: Poly1,b: Poly1,power_u: int,sign: int,shift: int)->None:
#     """
#     **Description:**
#     Adds `sign * u^power_u * a(v/u) * b(uv) * (uv)^(-shift)` to the Laurent polynomial `out` in place, dropping zero coefficients.
#     **Arguments:**
#     - `out (Laurent2)`: Coefficients `{(i, j): c}` of `u^i v^j`; modified in place.
#     - `a (Poly1)`: Coefficient tuple of the first factor.
#     - `b (Poly1)`: Coefficient tuple of the second factor.
#     - `power_u (int)`: Power of `u` of the term.
#     - `sign (int)`: Overall sign of the term.
#     - `shift (int)`: Power of `uv` that is divided out.
#     **Returns:**
#     - `None`.
#     """
#     for i, x in enumerate(a):
#         for j, y in enumerate(b):
#             key=(power_u-i+j-shift,i+j-shift)
#             out[key]=out.get(key,0)+sign*x*y
#             if out[key]==0:
#                 del out[key]
# def _accumulate_hodge_term(out: Laurent2,a: Poly1,b: Poly1,power_u: int,shift: int)->None:
#     """
#     **Description:**
#     Accumulate Hodge numbers directly, without first storing E coefficients.
#     For a face of rank ``power_u``, the E-term sign and the extraction sign
#     cancel because the parity of the two resulting exponents is
#     ``power_u``. Thus each product ``a_i*b_j`` contributes directly.
#     **Arguments:**
#     - `out (Laurent2)`: Hodge numbers `{(p, q): h^{p,q}}`; modified in place.
#     - `a (Poly1)`: Coefficient tuple of the primal S-tilde polynomial.
#     - `b (Poly1)`: Coefficient tuple of the dual S-tilde polynomial.
#     - `power_u (int)`: Rank of the face (dimension plus one).
#     - `shift (int)`: The Gorenstein index, i.e. the power of `uv` that is divided out.
#     **Returns:**
#     - `None`.
#     """
#     for i, x in enumerate(a):
#         if not x:
#             continue
#         for j, y in enumerate(b):
#             if not y:
#                 continue
#             key=(power_u-i+j-shift,i+j-shift)
#             out[key]=out.get(key,0)+x*y
#             if out[key]==0:
#                 del out[key]
# def stringy_hodge(primal_poly: Polytope|ArrayLike,dual_poly: Polytope|ArrayLike,*,index: int=2,pairing: ArrayLike|None=None,check: bool=True,require_nonnegative_hodge: bool=True,ehrhart_backend: str="auto",latte_count_bin: str="count",latte_options: Sequence[str]=("--redundancy-check=none",),verbose: bool=False,cytools_work_cutoff: int=100_0)->StringyHodgeResult:
#     """
#     **Description:**
#     Compute only the stringy Hodge numbers/Hodge diamond.
#     This uses the same exact facewise S-tilde formula as :func:`stringy_e`,
#     but avoids allocating the E-polynomial and avoids assembling the dual
#     E-polynomial solely for the optional mirror-reciprocity check.
#     **Arguments:**
#     - `primal_poly (Polytope|ArrayLike)`: The primal Gorenstein support polytope, as a CYTools polytope or an integer array of points.
#     - `dual_poly (Polytope|ArrayLike)`: The dual Gorenstein support polytope, as a CYTools polytope or an integer array of points.
#     - `index (int)`: The Gorenstein index. Defaults to `2`.
#     - `pairing (ArrayLike|None)`: Integer matrix `B` defining `<x,y> = x B y` between the supports. If `None`, the standard dot product is used. Defaults to `None`.
#     - `check (bool)`: Whether to check the face duality and the symmetries of the result. Defaults to `True`.
#     - `require_nonnegative_hodge (bool)`: Whether negative Hodge numbers raise an error (only with `check`). Defaults to `True`.
#     - `ehrhart_backend (str)`: How Ehrhart h*-polynomials of faces are computed: "auto", "latte", "normaliz" or "cytools". Defaults to "auto".
#     - `latte_count_bin (str)`: Name or path of LattE's `count` executable. Defaults to "count".
#     - `latte_options (Sequence[str])`: Extra command-line options passed to LattE. Defaults to `("--redundancy-check=none",)`.
#     - `verbose (bool)`: Whether to print progress information. Defaults to `False`.
#     - `cytools_work_cutoff (int)`: Work threshold below which the "auto" backend counts lattice points with CYTools instead of LattE. Defaults to `1000`.
#     **Raises:**
#     - `ValueError`: Raised if `index` is not positive, the supports are not dual, or (with `check`) a consistency check fails.
#     **Returns:**
#     - `StringyHodgeResult`: The stringy Hodge numbers together with the face lattices and face pairing.
#     """
#     if index<1:
#         raise ValueError("index must be positive")
#     started=perf_counter()
#     p=FaceLattice(primal_poly,ehrhart_backend=ehrhart_backend,latte_count_bin=latte_count_bin,latte_options=latte_options,cytools_work_cutoff=cytools_work_cutoff)
#     q=FaceLattice(dual_poly,ehrhart_backend=ehrhart_backend,latte_count_bin=latte_count_bin,latte_options=latte_options,cytools_work_cutoff=cytools_work_cutoff)
#     if p.dim!=q.dim:
#         raise ValueError("dual Gorenstein support polytopes must have equal dimension")
#     star=pair_dual_faces(p,q,pairing,check=check)
#     if verbose:
#         print(f"face lattices: {len(p.faces)} primal, {len(q.faces)} dual; Ehrhart backend: {p.ehrhart_backend}",flush=True)
#     cy_dim=p.dim+1-2*index
#     hodge: Laurent2={(p_,q_): 0 for p_ in range(cy_dim+1) for q_ in range(cy_dim+1)}
#     for face in p.faces:
#         rank=face.dim+1
#         _accumulate_hodge_term(hodge,p.s_tilde(face.id),q.s_tilde(star[face.id]),rank,index,)
#     result=StringyHodgeResult(hodge,index,p.dim,cy_dim,p,q,star)
#     if check:
#         _check_hodge_result(result,require_nonnegative_hodge=require_nonnegative_hodge)
#     if verbose:
#         print(f"completed Hodge-only computation in {perf_counter()-started:.2f} seconds",flush=True)
#     return result
# def stringy_e(primal_poly: Polytope|ArrayLike,dual_poly: Polytope|ArrayLike,*,index: int=2,pairing: ArrayLike|None=None,check: bool=True,require_nonnegative_hodge: bool=True,ehrhart_backend: str="auto",latte_count_bin: str="count",latte_options: Sequence[str]=("--redundancy-check=none",),verbose: bool=False,cytools_work_cutoff: int=100_0)->StringyEResult:
#     """
#     **Description:**
#     Compute the Batyrev--Nill stringy E-function and stringy Hodge numbers.
#     **Arguments:**
#     - `primal_poly (Polytope|ArrayLike)`: The primal Gorenstein support polytope, as a CYTools polytope or an integer array of points.
#     - `dual_poly (Polytope|ArrayLike)`: The dual Gorenstein support polytope, as a CYTools polytope or an integer array of points.
#     - `index (int)`: The Gorenstein index. Defaults to `2`.
#     - `pairing (ArrayLike|None)`: Integer matrix `B` defining `<x,y> = x B y` between the supports. If `None`, the standard dot product is used. Defaults to `None`.
#     - `check (bool)`: Whether to check the face duality and the symmetries and mirror reciprocity of the result. Defaults to `True`.
#     - `require_nonnegative_hodge (bool)`: Whether negative Hodge numbers raise an error (only with `check`). Defaults to `True`.
#     - `ehrhart_backend (str)`: How Ehrhart h*-polynomials of faces are computed: "auto", "latte", "normaliz" or "cytools". Defaults to "auto".
#     - `latte_count_bin (str)`: Name or path of LattE's `count` executable. Defaults to "count".
#     - `latte_options (Sequence[str])`: Extra command-line options passed to LattE. Defaults to `("--redundancy-check=none",)`.
#     - `verbose (bool)`: Whether to print progress information. Defaults to `False`.
#     - `cytools_work_cutoff (int)`: Work threshold below which the "auto" backend counts lattice points with CYTools instead of LattE. Defaults to `1000`.
#     **Raises:**
#     - `ValueError`: Raised if `index` is not positive, the supports are not dual, or (with `check`) a consistency check fails.
#     **Returns:**
#     - `StringyEResult`: The E-function coefficients, stringy Hodge numbers, face lattices and face pairing.
#     """
#     if index<1:
#         raise ValueError("index must be positive")
#     started=perf_counter()
#     p=FaceLattice(primal_poly,ehrhart_backend=ehrhart_backend,latte_count_bin=latte_count_bin,latte_options=latte_options,cytools_work_cutoff=cytools_work_cutoff)
#     q=FaceLattice(dual_poly,ehrhart_backend=ehrhart_backend,latte_count_bin=latte_count_bin,latte_options=latte_options,cytools_work_cutoff=cytools_work_cutoff)
#     if p.dim!=q.dim:
#         raise ValueError("dual Gorenstein support polytopes must have equal dimension")
#     star=pair_dual_faces(p,q,pairing,check=check)
#     if verbose:
#         print(f"face lattices: {len(p.faces)} primal, {len(q.faces)} dual; " f"Ehrhart backend: {p.ehrhart_backend}",flush=True,)
#     p.precompute_s_tilde(verbose=verbose,label="primal")
#     q.precompute_s_tilde(verbose=verbose,label="dual")
#     e: Laurent2={}
#     for f in p.faces:
#         k=f.dim+1
#         _accumulate_term(e,p.s_tilde(f.id),q.s_tilde(star[f.id]),k,(-1)**k,index)
#     cy_dim=p.dim+1-2*index
#     hodge={(a,b): (-1)**(a+b)*c for (a,b),c in e.items()}
#     result=StringyEResult(e,hodge,index,p.dim,cy_dim,p,q,star)
#     if check:
#         _check_result(result,require_nonnegative_hodge=require_nonnegative_hodge)
#     if verbose:
#         print(f"completed in {perf_counter()-started:.2f} seconds",flush=True)
#     return result
# def stringy_e_from_summands(delta1: Polytope|ArrayLike,delta2: Polytope|ArrayLike,*,interior_point: ArrayLike|None=None,index: int=2,poly_backend: str|None=None,check: bool=True,require_nonnegative_hodge: bool=True,ehrhart_backend: str="auto",latte_count_bin: str="count",latte_options: Sequence[str]=("--redundancy-check=none",),verbose: bool=False,)->StringyEResult:
#     """
#     **Description:**
#     Compatibility entry point that constructs cones from two summands.
#     This constructs and dualizes the Cayley cone natively, verifies that the
#     dual rays form a degree-one Gorenstein support, and computes its stringy
#     E-function.  ``index=2`` is retained as an argument only for explicitness
#     and testing; a two-part Cayley Gorenstein construction should use 2.
#     **Arguments:**
#     - `delta1 (Polytope|ArrayLike)`: The first Cayley summand, as a CYTools polytope or an integer array of points.
#     - `delta2 (Polytope|ArrayLike)`: The second Cayley summand, as a CYTools polytope or an integer array of points.
#     - `interior_point (ArrayLike|None)`: The unique interior lattice point of `delta1 + delta2`. If `None`, it is inferred with CYTools. Defaults to `None`.
#     - `index (int)`: The Gorenstein index; must be 2 when `check` is set. Defaults to `2`.
#     - `poly_backend (str|None)`: Backend passed to the CYTools `Polytope` constructor. Defaults to `None`.
#     - `check (bool)`: Whether to check the Cayley construction (`index == 2`, reflexivity of `delta1 + delta2`, dimensions), the face duality and the symmetries of the result. Defaults to `True`.
#     - `require_nonnegative_hodge (bool)`: Whether negative Hodge numbers raise an error (only with `check`). Defaults to `True`.
#     - `ehrhart_backend (str)`: How Ehrhart h*-polynomials of faces are computed: "auto", "latte", "normaliz" or "cytools". Defaults to "auto".
#     - `latte_count_bin (str)`: Name or path of LattE's `count` executable. Defaults to "count".
#     - `latte_options (Sequence[str])`: Extra command-line options passed to LattE. Defaults to `("--redundancy-check=none",)`.
#     - `verbose (bool)`: Whether to print progress information. Defaults to `False`.
#     **Raises:**
#     - `ImportError`: Raised if CYTools is not available.
#     - `ValueError`: Raised if `index != 2` (with `check`) or the summands do not define a reflexive Gorenstein Cayley cone.
#     **Returns:**
#     - `StringyEResult`: The stringy E-function result, with `cayley_data` set.
#     """
#     if check and index!=2:
#         raise ValueError("a two-part Cayley support has Gorenstein index 2")
#     data=cayley_gorenstein_data(delta1,delta2,interior_point=interior_point,poly_backend=poly_backend,check=check,)
#     result=stringy_e(data.primal_support,data.dual_support,index=index,check=check,require_nonnegative_hodge=require_nonnegative_hodge,ehrhart_backend=ehrhart_backend,latte_count_bin=latte_count_bin,latte_options=latte_options,verbose=verbose,)
#     result.cayley_data=data
#     return result
# def stringy_hodge_from_summands(delta1: Polytope|ArrayLike,delta2: Polytope|ArrayLike,*,interior_point: ArrayLike|None=None,index: int=2,poly_backend: str|None=None,check: bool=True,require_nonnegative_hodge: bool=True,ehrhart_backend: str="auto",latte_count_bin: str="count",latte_options: Sequence[str]=("--redundancy-check=none",),verbose: bool=False,)->StringyHodgeResult:
#     """
#     **Description:**
#     Hodge-only counterpart of :func:`stringy_e_from_summands`.
#     **Arguments:**
#     - `delta1 (Polytope|ArrayLike)`: The first Cayley summand, as a CYTools polytope or an integer array of points.
#     - `delta2 (Polytope|ArrayLike)`: The second Cayley summand, as a CYTools polytope or an integer array of points.
#     - `interior_point (ArrayLike|None)`: The unique interior lattice point of `delta1 + delta2`. If `None`, it is inferred with CYTools. Defaults to `None`.
#     - `index (int)`: The Gorenstein index; must be 2 when `check` is set. Defaults to `2`.
#     - `poly_backend (str|None)`: Backend passed to the CYTools `Polytope` constructor. Defaults to `None`.
#     - `check (bool)`: Whether to check the Cayley construction (`index == 2`, reflexivity of `delta1 + delta2`, dimensions), the face duality and the symmetries of the result. Defaults to `True`.
#     - `require_nonnegative_hodge (bool)`: Whether negative Hodge numbers raise an error (only with `check`). Defaults to `True`.
#     - `ehrhart_backend (str)`: How Ehrhart h*-polynomials of faces are computed: "auto", "latte", "normaliz" or "cytools". Defaults to "auto".
#     - `latte_count_bin (str)`: Name or path of LattE's `count` executable. Defaults to "count".
#     - `latte_options (Sequence[str])`: Extra command-line options passed to LattE. Defaults to `("--redundancy-check=none",)`.
#     - `verbose (bool)`: Whether to print progress information. Defaults to `False`.
#     **Raises:**
#     - `ImportError`: Raised if CYTools is not available.
#     - `ValueError`: Raised if `index != 2` (with `check`) or the summands do not define a reflexive Gorenstein Cayley cone.
#     **Returns:**
#     - `StringyHodgeResult`: The stringy Hodge numbers of the Cayley construction.
#     """
#     if check and index!=2:
#         raise ValueError("a two-part Cayley support has Gorenstein index 2")
#     data=cayley_gorenstein_data(delta1,delta2,interior_point=interior_point,poly_backend=poly_backend,check=check,)
#     return stringy_hodge(data.primal_support,data.dual_support,index=index,check=check,require_nonnegative_hodge=require_nonnegative_hodge,ehrhart_backend=ehrhart_backend,latte_count_bin=latte_count_bin,latte_options=latte_options,verbose=verbose,)
# def stringy_e_from_cones(P: Polytope|ArrayLike,C: Cone,C_dual: Cone,degree_C: ArrayLike,degree_C_dual: ArrayLike,*,pairing: ArrayLike|None=None,expected_index: int|None=None,poly_backend: str|None=None,check: bool=True,require_nonnegative_hodge: bool=True,ehrhart_backend: str="auto",latte_count_bin: str="count",latte_options: Sequence[str]=("--redundancy-check=none",),verbose: bool=False,)->StringyEResult:
#     """
#     **Description:**
#     Compute the stringy E-function from fully precomputed Cayley cone data.
#     No Cayley cone, dual cone, Gorenstein degree vector, or index is
#     recomputed.  Only primitive extremal rays are requested from the supplied
#     cones, since these are needed to construct and validate the two support
#     polytopes.
#     **Arguments:**
#     - `P (Polytope|ArrayLike)`: The degree-one support of `C` (e.g. the Cayley polytope), as a CYTools polytope or an integer array of points.
#     - `C (Cone)`: The reflexive Gorenstein cone over `P` (CYTools `Cone`).
#     - `C_dual (Cone)`: The dual cone of `C` (CYTools `Cone`).
#     - `degree_C (ArrayLike)`: Integer Gorenstein degree vector grading `C`.
#     - `degree_C_dual (ArrayLike)`: Integer Gorenstein degree vector grading `C_dual`.
#     - `pairing (ArrayLike|None)`: Integer matrix `B` defining `<x,y> = x B y`. If `None`, the identity is used. Defaults to `None`.
#     - `expected_index (int|None)`: If given, the Gorenstein index must equal this value. Defaults to `None`.
#     - `poly_backend (str|None)`: Backend passed to the CYTools `Polytope` constructor. Defaults to `None`.
#     - `check (bool)`: Whether to check the full-dimensionality of the cones, the dimensions of the supports, the face duality and the symmetries of the result. Defaults to `True`.
#     - `require_nonnegative_hodge (bool)`: Whether negative Hodge numbers raise an error (only with `check`). Defaults to `True`.
#     - `ehrhart_backend (str)`: How Ehrhart h*-polynomials of faces are computed: "auto", "latte", "normaliz" or "cytools". Defaults to "auto".
#     - `latte_count_bin (str)`: Name or path of LattE's `count` executable. Defaults to "count".
#     - `latte_options (Sequence[str])`: Extra command-line options passed to LattE. Defaults to `("--redundancy-check=none",)`.
#     - `verbose (bool)`: Whether to print progress information. Defaults to `False`.
#     **Raises:**
#     - `ImportError`: Raised if CYTools is not available.
#     - `TypeError`: Raised if `C` or `C_dual` is not a CYTools cone.
#     - `ValueError`: Raised if the cone data are inconsistent or (with `check`) a consistency check of the result fails.
#     **Returns:**
#     - `StringyEResult`: The stringy E-function result, with `cone_data` set and, for index 2, the complete-splitting witnesses and flags.
#     """
#     data=validate_gorenstein_cone_data(P,C,C_dual,degree_C,degree_C_dual,pairing=pairing,expected_index=expected_index,poly_backend=poly_backend,check=check,)
#     result=stringy_e(data.primal_support,data.dual_support,index=data.index,pairing=pairing,check=check,require_nonnegative_hodge=require_nonnegative_hodge,ehrhart_backend=ehrhart_backend,latte_count_bin=latte_count_bin,latte_options=latte_options,verbose=verbose,)
#     result.cone_data=data
#     if data.index==2:
#         result.primal_split_witness=complete_split_witness_index_two(data.dual_support,data.degree_C)
#         result.dual_split_witness=complete_split_witness_index_two(data.primal_support,data.degree_C_dual)
#         result.primal_completely_split=result.primal_split_witness is not None
#         result.dual_completely_split=result.dual_split_witness is not None
#         if verbose:
#             print("complete splitting: " f"C={result.primal_completely_split}, " f"C_dual={result.dual_completely_split}",flush=True,)
#             print(result.interpretation(),flush=True)
#     return result
# def stringy_hodge_from_cones(P: Polytope|ArrayLike,C: Cone,C_dual: Cone,degree_C: ArrayLike,degree_C_dual: ArrayLike,*,pairing: ArrayLike|None=None,expected_index: int|None=None,poly_backend: str|None=None,check: bool=True,require_nonnegative_hodge: bool=True,ehrhart_backend: str="auto",latte_count_bin: str="count",latte_options: Sequence[str]=("--redundancy-check=none",),verbose: bool=False,)->StringyHodgeResult:
#     """
#     **Description:**
#     Compute only the Hodge diamond from precomputed Cayley cone data.
#     **Arguments:**
#     - `P (Polytope|ArrayLike)`: The degree-one support of `C` (e.g. the Cayley polytope), as a CYTools polytope or an integer array of points.
#     - `C (Cone)`: The reflexive Gorenstein cone over `P` (CYTools `Cone`).
#     - `C_dual (Cone)`: The dual cone of `C` (CYTools `Cone`).
#     - `degree_C (ArrayLike)`: Integer Gorenstein degree vector grading `C`.
#     - `degree_C_dual (ArrayLike)`: Integer Gorenstein degree vector grading `C_dual`.
#     - `pairing (ArrayLike|None)`: Integer matrix `B` defining `<x,y> = x B y`. If `None`, the identity is used. Defaults to `None`.
#     - `expected_index (int|None)`: If given, the Gorenstein index must equal this value. Defaults to `None`.
#     - `poly_backend (str|None)`: Backend passed to the CYTools `Polytope` constructor. Defaults to `None`.
#     - `check (bool)`: Whether to check the full-dimensionality of the cones, the dimensions of the supports, the face duality and the symmetries of the result. Defaults to `True`.
#     - `require_nonnegative_hodge (bool)`: Whether negative Hodge numbers raise an error (only with `check`). Defaults to `True`.
#     - `ehrhart_backend (str)`: How Ehrhart h*-polynomials of faces are computed: "auto", "latte", "normaliz" or "cytools". Defaults to "auto".
#     - `latte_count_bin (str)`: Name or path of LattE's `count` executable. Defaults to "count".
#     - `latte_options (Sequence[str])`: Extra command-line options passed to LattE. Defaults to `("--redundancy-check=none",)`.
#     - `verbose (bool)`: Whether to print progress information. Defaults to `False`.
#     **Raises:**
#     - `ImportError`: Raised if CYTools is not available.
#     - `TypeError`: Raised if `C` or `C_dual` is not a CYTools cone.
#     - `ValueError`: Raised if the cone data are inconsistent or (with `check`) a consistency check of the result fails.
#     **Returns:**
#     - `StringyHodgeResult`: The stringy Hodge numbers.
#     """
#     data=validate_gorenstein_cone_data(P,C,C_dual,degree_C,degree_C_dual,pairing=pairing,expected_index=expected_index,poly_backend=poly_backend,check=check,)
#     return stringy_hodge(data.primal_support,data.dual_support,index=data.index,pairing=pairing,check=check,require_nonnegative_hodge=require_nonnegative_hodge,ehrhart_backend=ehrhart_backend,latte_count_bin=latte_count_bin,latte_options=latte_options,verbose=verbose,)
# stringy_e_cayley=stringy_e_from_cones
# def _check_hodge_result(r: StringyHodgeResult,*,require_nonnegative_hodge: bool)->None:
#     """
#     **Description:**
#     Checks that the Hodge numbers lie in the CY range and satisfy Hodge symmetry and Poincare duality.
#     **Arguments:**
#     - `r (StringyHodgeResult)`: The result to check.
#     - `require_nonnegative_hodge (bool)`: Whether negative Hodge numbers are an error.
#     **Raises:**
#     - `ValueError`: Raised if a check fails.
#     **Returns:**
#     - `None`.
#     """
#     if r.cy_dim<0:
#         raise ValueError("negative Calabi--Yau dimension")
#     bad=[key for key in r.hodge_numbers if min(key)<0 or max(key)>r.cy_dim]
#     if bad:
#         raise ValueError(f"Hodge numbers fall outside the CY range: {bad[:5]}")
#     for (p,q), value in r.hodge_numbers.items():
#         if r.hodge_numbers.get((q,p),0)!=value:
#             raise ValueError("Hodge symmetry check failed")
#         if r.hodge_numbers.get((r.cy_dim-p,r.cy_dim-q),0)!=value:
#             raise ValueError("Poincare duality check failed")
#     if require_nonnegative_hodge and any(value<0 for value in r.hodge_numbers.values()):
#         raise ValueError("negative extracted stringy Hodge numbers")
# def _check_result(r: StringyEResult,*,require_nonnegative_hodge: bool)->None:
#     """
#     **Description:**
#     Checks that the E-function is a CY-range polynomial with u-v symmetry, Poincare duality and primal/dual mirror reciprocity.
#     **Arguments:**
#     - `r (StringyEResult)`: The result to check.
#     - `require_nonnegative_hodge (bool)`: Whether negative Hodge numbers are an error.
#     **Raises:**
#     - `ValueError`: Raised if a check fails.
#     **Returns:**
#     - `None`.
#     """
#     if r.cy_dim<0:
#         raise ValueError("negative Calabi--Yau dimension")
#     bad=[k for k in r.coefficients if min(k)<0 or max(k)>r.cy_dim]
#     if bad:
#         raise ValueError(f"E-function did not become a CY-range polynomial; bad exponents: {bad[:5]}")
#     for (a,b), c in r.coefficients.items():
#         if r.coefficients.get((b,a),0)!=c:
#             raise ValueError("u-v symmetry check failed")
#         if r.coefficients.get((r.cy_dim-a,r.cy_dim-b),0)!=c:
#             raise ValueError("Poincare duality check failed")
#     inverse_star={dual_id: primal_id for primal_id,dual_id in r.dual_faces.items()}
#     if len(inverse_star)!=len(r.dual.faces):
#         raise ValueError("cannot verify mirror reciprocity: face duality is not bijective")
#     dual_e: Laurent2={}
#     for face in r.dual.faces:
#         k=face.dim+1
#         _accumulate_term(dual_e,r.dual.s_tilde(face.id),r.primal.s_tilde(inverse_star[face.id]),k,(-1)**k,r.index,)
#     expected_from_dual={(r.cy_dim-a,b): (-1)**r.cy_dim*coefficient for (a,b),coefficient in dual_e.items() if coefficient}
#     if expected_from_dual!=r.coefficients:
#         raise ValueError("primal/dual mirror-reciprocity check failed")
#     if require_nonnegative_hodge:
#         bad_h={k: v for k,v in r.hodge_numbers.items() if v<0}
#         if bad_h:
#             raise ValueError(f"negative extracted stringy Hodge numbers: {bad_h}")
# def extract_hodge_numbers(e: Mapping[tuple[int,int],int],*,check_nonnegative: bool=True)->dict[tuple[int,int],int]:
#     """
#     **Description:**
#     Extract h^{p,q} from E=sum (-1)^(p+q) h^{p,q} u^p v^q.
#     **Arguments:**
#     - `e (Mapping[tuple[int,int],int])`: Coefficients `{(p, q): c}` of `u^p v^q` in the E-polynomial.
#     - `check_nonnegative (bool)`: Whether negative Hodge numbers raise an error. Defaults to `True`.
#     **Raises:**
#     - `ValueError`: Raised if `check_nonnegative` is set and some Hodge number is negative.
#     **Returns:**
#     - `dict[tuple[int,int],int]`: The nonzero Hodge numbers `{(p, q): h^{p,q}}`.
#     """
#     h={(p,q): (-1)**(p+q)*int(c) for (p,q),c in e.items() if c}
#     if check_nonnegative and any(x<0 for x in h.values()):
#         raise ValueError("coefficients do not define nonnegative stringy Hodge numbers")
#     return h
# def format_laurent(poly: Mapping[Tuple[int,int],int])->str:
#     """
#     **Description:**
#     Human-readable exact two-variable Laurent polynomial.
#     **Arguments:**
#     - `poly (Mapping[Tuple[int,int],int])`: The coefficients `{(a, b): c}` of the monomials `c*u^a*v^b`.
#     **Returns:**
#     - `str`: The polynomial as a string, e.g. `"u*v + 2*u - 1"`; `"0"` for the zero polynomial.
#     """
#     if not poly:
#         return "0"
#     terms: List[str]=[]
#     for (a,b), c in sorted(poly.items(),key=lambda z: (sum(z[0]),z[0]),reverse=True):
#         factors=(["u" if a==1 else f"u^{a}"] if a else [])+(["v" if b==1 else f"v^{b}"] if b else [])
#         mon="*".join(factors)
#         mag=abs(c)
#         body=mon if mag==1 and mon else str(mag)+(("*"+mon) if mon else "")
#         terms.append(("- " if c<0 else "+ ")+body)
#     s=" ".join(terms)
#     return s[2:] if s.startswith("+ ") else "-"+s[2:] if s.startswith("- ") else s
# def check_two_part_cayley(points: Polytope|ArrayLike,*,base_dim: int=6)->None:
#     """
#     **Description:**
#     Strong shape/height check for the convention (Delta_i,e_i).
#     **Arguments:**
#     - `points (Polytope|ArrayLike)`: The Cayley polytope, as an integer array of points (one per row) or a CYTools polytope (its vertices are used).
#     - `base_dim (int)`: Dimension of the polytopes Delta_i. Defaults to `6`.
#     **Raises:**
#     - `ValueError`: Raised if the points do not have `base_dim+2` coordinates, if the last two coordinates are not exactly (1,0) and (0,1) (both present), or if the affine dimension is not `base_dim+1`.
#     **Returns:**
#     - `None`.
#     """
#     v=_integer_points(points)
#     if v.shape[1]!=base_dim+2:
#         raise ValueError(f"expected {base_dim+2} coordinates")
#     heights={tuple(map(int,x[-2:])) for x in v}
#     if not heights.issubset({(1,0),(0,1)}) or heights!={(1,0),(0,1)}:
#         raise ValueError("Cayley heights must be exactly (1,0) and (0,1), both present")
#     if _affine_dim(v)!=base_dim+1:
#         raise ValueError(f"expected affine dimension {base_dim+1}")
# __all__=["CayleyGorensteinData","GorensteinConeData","Face","FaceLattice","StringyEResult","StringyHodgeResult","cayley_support","cayley_support_2part","cayley_gorenstein_data","check_two_part_cayley","clear_hstar_cache","complete_split_witness_index_two","dual_support_vertices_cytools","ehrhart_polynomial_latte_vrep","extract_hodge_numbers","format_laurent","hstar_cache_size","pair_dual_faces","stringy_e","stringy_e_cayley","stringy_e_from_cones","stringy_e_from_summands","stringy_hodge","stringy_hodge_from_cones","stringy_hodge_from_summands","validate_gorenstein_cone_data","write_latte_vrep","hodge_diamond_stringy","hodge_diamond_uplift","hodge_diamond_cy4",]
# def get_E_function(F: F_Theory_Uplift,backend: Literal["auto","latte","normaliz","cytools"]="auto",verbose: bool=False,hodge_only: bool=True)->StringyHodgeResult|StringyEResult:
#     """
#     **Description:**
#     Computes the stringy Hodge numbers (or the full stringy E-function) of the Calabi-Yau fourfold of an F-theory uplift from the Cayley polytope `Cayley_M(F)`, its cone and the dual cone, via `stringy_hodge_from_cones` (or `stringy_e_from_cones`).
#     **Arguments:**
#     - `F (F_Theory_Uplift)`: The F-theory uplift.
#     - `backend (Literal["auto","latte","normaliz","cytools"])`: How the Ehrhart h*-polynomials of the faces are computed (passed as `ehrhart_backend`). Defaults to `"auto"`.
#     - `verbose (bool)`: Whether to print progress information. Defaults to `False`.
#     - `hodge_only (bool)`: If `True`, only the Hodge numbers are computed (`stringy_hodge_from_cones`); otherwise the full E-function (`stringy_e_from_cones`). Defaults to `True`.
#     **Returns:**
#     - `StringyHodgeResult|StringyEResult`: A `StringyHodgeResult` if `hodge_only` is `True`, otherwise a `StringyEResult`.
#     """
#     CM=Cayley_M(F)
#     ConeM=Cone(CM.vertices())
#     ConeN=ConeM.dual()
#     GM=is_Gorenstein(ConeM)[1]
#     GN=is_Gorenstein(ConeN)[1]
#     compute=stringy_hodge_from_cones if hodge_only else stringy_e_from_cones
#     return compute(CM,ConeM,ConeN,GM,GN,ehrhart_backend=backend,verbose=verbose)
# ---- End of obsolete block ----
# -----------------------------------------------------------------------------
# Fast stringy Hodge numbers (hodge_diamond_stringy, hodge_diamond_uplift, hodge_diamond_cy4)
# -----------------------------------------------------------------------------
# Hodge-number-only implementation of the Batyrev--Nill formula
#     E_st(u,v) = (uv)^(-r) sum_{F <= P} (-u)^(dim F + 1) S~(F, v/u) S~(F*, uv)
# for a reflexive Gorenstein support P of index r (the Cayley polytope of a nef
# partition of length r) and its dual support P*.  Since p + q = dim F + 1 mod 2
# for every term, the signs cancel and h^{p,q} = sum S~_i(F) S~_j(F*) over the
# faces F and the exponents (i, j) that produce u^p v^q.
# Compared to stringy_hodge / get_E_function above:
# * No Ehrhart computation per face.  A lattice point x of kP lies in the relative
#   interior of exactly one face F, namely the one whose dual face F* is the set
#   of dual vertices w with <x, w> = 0.  Enumerating kP once (with PyNormaliz or
#   CYTools) therefore yields l*(kF) for every face F at the same time, and by
#   Ehrhart reciprocity dilations k <= ceil((dim P - 1)/2) determine every h*.
# * The face order is one boolean matrix, and the g-polynomials are only
#   computed up to the degree (rank - 1)/2 that survives the truncation.
# * Only the (n+1) x (n+1) Hodge diamond is accumulated; E_st is never formed.
# For Calabi-Yau fourfolds from two-part nef partitions, hodge_diamond_cy4 is
# faster still: it evaluates the closed Batyrev--Borisov formulas of
# h11_2_part_l_fast and h21_2_part_fast, which only need l*(F) and l*(2F).
# References: Batyrev--Borisov, https://arxiv.org/abs/alg-geom/9509009 ;
# Batyrev--Nill, https://arxiv.org/abs/math/0703456 .
try:
    from PyNormaliz import Cone as NormalizCone
except ImportError:
    NormalizCone=None

def hodge_diamond_stringy(P: Polytope|ArrayLike,backend: Literal["auto","normaliz","latte","cytools"]="auto")->np.ndarray:
    """
    **Description:**
    Computes the stringy Hodge numbers of the Calabi-Yau complete intersection
    defined by the reflexive Gorenstein support P (e.g. the Cayley polytope of a
    nef partition).  For the Cayley polytope of the Newton polytopes Delta_i this is
    the Hodge diamond of the complete intersection itself; replacing P by its dual
    support (the Cayley polytope of the nabla_i) gives the mirror.
    **Arguments:**
    - `P (Polytope | ArrayLike)`: The support polytope (or its vertices, one per row).
    - `backend (Literal["auto", "normaliz", "latte", "cytools"])`: How the lattice points of the dilated faces are counted.
        "normaliz" (PyNormaliz) or "cytools" enumerate the dilations kP;
        "latte" instead computes the Ehrhart polynomials of all faces of
        dimension >= 3 one by one, which avoids enumerating kP when it is huge.
        "auto" picks the first available of normaliz, latte, cytools. Defaults to `"auto"`.
    **Raises:**
    - `ValueError`: Raised if P is not a reflexive Gorenstein support or the backend is unknown or not installed.
    - `RuntimeError`: Raised if the resulting Hodge numbers violate the Hodge symmetries or are negative.
    **Returns:**
    - `np.ndarray`: The integer Hodge diamond, h[p, q] = h^{p,q}.
    """
    V, W, index, backend=_hd_setup(P,backend)
    faces, duals, dims, leq=_hd_face_lattice(V,W)
    S=_hd_s_tilde(V,W,faces,duals,dims,leq,index,backend)
    S_dual=_hd_s_tilde(W,V,duals,faces,V.shape[1]-2-dims,leq.T,index,backend)
    cy_dim=V.shape[1]-2*index
    h=np.zeros((cy_dim+1,cy_dim+1),dtype=object)
    for a, b, d in zip(S,S_dual,dims):
        for i, j in zip(*np.nonzero(np.outer(a,b))):
            p, q=d+1-i+j-index, i+j-index
            if 0<=p<=cy_dim and 0<=q<=cy_dim:
                h[p,q]+=a[i]*b[j]
    if (h!=h.T).any() or (h!=h[::-1,::-1]).any() or (h<0).any():
        raise RuntimeError(f"inconsistent stringy Hodge numbers:\n{h}")
    return h.astype(int)

def hodge_diamond_uplift(F: F_Theory_Uplift,backend: Literal["auto","normaliz","latte","cytools"]="auto")->np.ndarray:
    """
    **Description:**
    Hodge diamond of the Calabi-Yau of an F_Theory_Uplift, computed from the Cayley
    polytope ``F.Cayley_M()`` of its Newton polytopes.
    **Arguments:**
    - `F (F_Theory_Uplift)`: The F-theory uplift; its Cayley polytope `F.Cayley_M()` is passed to `hodge_diamond_stringy`.
    - `backend (Literal["auto", "normaliz", "latte", "cytools"])`: Lattice-point backend, see `hodge_diamond_stringy`. Defaults to `"auto"`.
    **Returns:**
    - `np.ndarray`: The integer Hodge diamond, h[p, q] = h^{p,q}.
    """
    return hodge_diamond_stringy(F.Cayley_M(),backend)

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
    - `backend (Literal["auto", "normaliz", "latte", "cytools"])`: "normaliz" or "cytools" for enumerating P, 2P, P*, 2P*;
        "latte" also enumerates with CYTools here; "auto" prefers normaliz. Defaults to `"auto"`.
    **Raises:**
    - `ValueError`: Raised if P is not a reflexive Gorenstein support, is not the Cayley polytope of a two-part nef partition in six dimensions, or the backend is unknown or not installed.
    **Returns:**
    - `np.ndarray`: The 5 x 5 integer Hodge diamond, h[p, q] = h^{p,q}.
    """
    V, W, index, backend=_hd_setup(P,backend)
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

def _hd_setup(P: Polytope|ArrayLike,backend: Literal["auto","normaliz","latte","cytools"])->tuple[np.ndarray,np.ndarray,int,Literal["normaliz","latte","cytools"]]:
    """
    **Description:**
    Vertices V of P, vertices W of the dual support P* (the primitive rays of the
    dual cone), the Gorenstein index <m, n>, and the backend to use.  m and n are
    the integral degree vectors with <v, m> = 1 = <w, n>; they exist iff P is a
    reflexive Gorenstein support.
    **Arguments:**
    - `P (Polytope | ArrayLike)`: The support polytope (or its vertices, one per row).
    - `backend (Literal["auto", "normaliz", "latte", "cytools"])`: The requested lattice-point backend; "auto" picks the first available of normaliz, latte, cytools (with a warning if PyNormaliz is missing).
    **Raises:**
    - `ValueError`: Raised if P is not a reflexive Gorenstein support, or if the backend is unknown or not installed.
    **Returns:**
    - `tuple[np.ndarray, np.ndarray, int, Literal["normaliz", "latte", "cytools"]]`: `(V, W, index, backend)`: the integer vertex matrices of P and P*, the Gorenstein index and the resolved backend.
    """
    V=np.asarray(P.vertices() if isinstance(P,Polytope) else P,dtype=int)
    W=np.asarray(Cone(V).dual().extremal_rays(),dtype=int)
    W//=np.gcd.reduce(W,axis=1)[:,None]
    m, n=(np.rint(np.linalg.lstsq(X,np.ones(len(X)),rcond=None)[0]).astype(int) for X in (V,W))
    if np.linalg.matrix_rank(V)!=V.shape[1] or (V@m!=1).any() or (W@n!=1).any():
        raise ValueError("P is not a reflexive Gorenstein support (e.g. the Cayley polytope of a nef partition)")
    available={"normaliz": NormalizCone is not None,"latte": shutil.which("count") is not None,"cytools": True}
    if backend=="auto":
        backend=next(name for name,ok in available.items() if ok)
        if backend!="normaliz":
            warnings.warn(f"PyNormaliz is not installed, so lattice points are counted with {backend}, which may be much slower. " "Install it with `pip install pynormaliz`.")
    elif not available.get(backend,False):
        raise ValueError(f"backend {backend!r} is unknown or not installed; choose from {[name for name,ok in available.items() if ok]}"+(" (install PyNormaliz with `pip install pynormaliz`)" if backend=="normaliz" else ""))
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

def _hd_s_tilde(X: np.ndarray,Y: np.ndarray,faces: Sequence[frozenset[int]],duals: Sequence[frozenset[int]],dims: np.ndarray,leq: np.ndarray,index: int,backend: Literal["normaliz","latte","cytools"])->list[list[int]]:
    """
    **Description:**
    S~(F, t) = sum_{G <= F} (-1)^(dim F - dim G) h*_G(t) g([G, F], t) for every face F
    of conv(X) (dual support conv(Y); faces, duals, dims, leq as in _hd_face_lattice).
    h*_F of a face of dimension d follows from L_F(k) = #(kF cap Z) and
    l*(kF) = #(relint(kF) cap Z) for k <= ceil(d/2) through
        sum_k L_F(k) t^k = h*_F(t) / (1-t)^(d+1),
        sum_k l*(kF) t^k = t^(d+1) h*_F(1/t) / (1-t)^(d+1),
    using l*(kP) = L_P(k - index) for P itself (Gorenstein).  The g-polynomials of
    the intervals [G, F] follow the recursion of
    ``FaceLattice.g_polynomial`` (old code, now commented out).
    **Arguments:**
    - `X (np.ndarray)`: The vertices of P = conv(X), one per row.
    - `Y (np.ndarray)`: The vertices of the dual support, one per row.
    - `faces (Sequence[frozenset[int]])`: The row-index sets of the faces of P, sorted by dimension.
    - `duals (Sequence[frozenset[int]])`: The row-index sets (in Y) of the dual faces.
    - `dims (np.ndarray)`: The face dimensions.
    - `leq (np.ndarray)`: Boolean face order, leq[g, f] = (G <= F).
    - `index (int)`: The Gorenstein index.
    - `backend (Literal["normaliz", "latte", "cytools"])`: "normaliz" or "cytools" enumerate dilations; "latte" computes Ehrhart polynomials of large faces with LattE's `count`.
    **Returns:**
    - `list[list[int]]`: For every face F, the integer coefficients of S~(F, t) (Python or NumPy integers), constant term first.
    """
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
    t_minus_one=[[(-1)**(e-i)*comb(e,i) for i in range(e+1)] for e in range(D+2)]

    @cache
    def g(lo: int,hi: int)->tuple[int,...]:
        """
        **Description:**
        Truncated g-polynomial of the interval [lo, hi] of the face lattice (cached).
        **Arguments:**
        - `lo (int)`: Index of the lower face.
        - `hi (int)`: Index of the upper face.
        **Returns:**
        - `tuple[int, ...]`: Coefficients of g, constant term first, up to degree (rank - 1)//2.
        """
        rank=dims[hi]-dims[lo]
        if rank<=2:
            return (1,)
        n=(rank+1)//2
        h=[0]*n
        for x in np.flatnonzero(leq[lo]&leq[:,hi]).tolist():
            if x!=lo:
                h=[a+b for a,b in zip(h,_hd_mul(t_minus_one[dims[x]-dims[lo]-1],g(x,hi),n))]
        return tuple(_hd_mul([1,-1],h,n))
    S=[]
    for f in range(len(faces)):
        s=[0]*(dims[f]+2)
        for x in np.flatnonzero(leq[:,f]).tolist():
            s=[a+(-1)**(dims[f]-dims[x])*b for a,b in zip(s,_hd_mul(h_star[x],g(x,f),dims[f]+2))]
        S.append(s)
    return S

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
    dim=points.shape[1]
    basis_indices=np.asarray(basis(points),dtype=int)
    B=points[basis_indices]
    bB=b[basis_indices]
    m=_integer_solve(B,bB,atol=atol)
    if m is None:
        return False, None
    if np.array_equal(points@m,b):
        return True, m
    return False, None

def is_partition(points: ArrayLike,divisors: ArrayLike,atol: float=1e-8)->tuple[bool,bool,np.ndarray]:
    """
    **Description:**
    Determine whether divisors L1,...,Lr can be shifted by principal
    divisors to form a partition of the anticanonical divisor.
    This version avoids repeated np.linalg.solve calls by computing
    the inverse of the basis matrix only once.
    **Arguments:**
    - `points (ArrayLike)`: The ray generators of the toric fan, one per row.
    - `divisors (ArrayLike)`: The coefficient vectors of L1,...,Lr in the prime toric divisor basis, one per row.
    - `atol (float)`: Numerical tolerance for the integrality checks. Defaults to `1e-8`.
    **Returns:**
    - `tuple[bool,bool,np.ndarray]`: `(is_partition, sums_to_anticanonical, shifts)`, where `shifts` is the integer (r, dim) array of shift vectors m_a such that `divisors[a] + points@shifts[a]` form the partition (if no partition exists, all shifts are zero except possibly the last row).
    """
    points=np.asarray(points,dtype=int)
    divisors=np.asarray(divisors,dtype=int)
    r, n_rays=divisors.shape
    dim=points.shape[1]
    sta, total_shift=sums_to_anticanonical(points,divisors,atol=atol)
    if not sta:
        return False, False, np.zeros((r,dim),dtype=int)
    basis_indices=np.asarray(basis(points),dtype=int)
    B=points[basis_indices].astype(float)
    try:
        B_inv=np.linalg.inv(B)
    except np.linalg.LinAlgError:
        return False, True, np.zeros((r,dim),dtype=int)
    divisors_B=divisors[:,basis_indices]
    for assignment in product(range(r),repeat=dim):
        assignment=np.asarray(assignment)
        shifts=np.zeros((r,dim),dtype=int)
        valid=True
        for a in range(r-1):
            target=(assignment==a).astype(float)
            rhs=target-divisors_B[a]
            m_float=B_inv@rhs
            m=np.rint(m_float).astype(int)
            if not np.allclose(m_float,m,atol=atol,rtol=0):
                valid=False
                break
            shifts[a]=m
        if not valid:
            continue
        shifts[-1]=(total_shift-np.sum(shifts[:-1],axis=0))
        shifted=divisors+shifts@points.T
        if not np.all((shifted==0)|(shifted==1)):
            continue
        if not np.all(np.sum(shifted,axis=0)==1):
            continue
        return True, True, shifts
    shifts=np.zeros((r,dim),dtype=int)
    shifts[-1]=total_shift
    return False, True, shifts

def chi_fourfold_uplift(F: F_Theory_Uplift)->int:
    """
    **Description:**
    Euler characteristic of the Calabi-Yau fourfold of an F-theory uplift: the toric localization `cicy4_chern_toric_uplift(F,4)` plus 6 times the number of O3-planes `chi_O3`.
    **Arguments:**
    - `F (F_Theory_Uplift)`: The F-theory uplift.
    **Returns:**
    - `int`: The Euler characteristic.
    """
    chi_loc=toric_chern_class_uplift(F,4)
    n_O3=chi_O3(F)
    return chi_loc+n_O3*6

def toric_chern_class_uplift(F: F_Theory_Uplift,n: int)->int|dict[tuple[int,...],int]|dict[int,int]:
    """
    **Description:**
    Apply toric_chern_class directly to an F-theory uplift.
    **Arguments:**
    - `F (F_Theory_Uplift)`: The F-theory uplift; its smooth uplift ambient fan and the line bundles `line_bundle_base_N()`, `line_bundle_weierstrass_N()` are used.
    - `n (int)`: Degree of the Chern class (1 to 4).
    **Returns:**
    - `int|dict[tuple[int,...],int]|dict[int,int]`: The Chern tensor c_n, see `toric_chern_class`.
    """
    fan=F.smooth_uplift_ambient_toric_fan()
    divisors=[F.line_bundle_base_N(),F.line_bundle_weierstrass_N()]
    return toric_chern_class(fan,divisors,n)

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
# def intersecting_points(pts,polytopes):
#     """
#     Lattice points pts whose toric divisors intersect
#     the generic nef-partition CICY with Newton polytopes p1,...,pk.
#     """
#     vertices=[p.vertices() for p in polytopes]
#     out=[]
#     for v in pts[np.any(pts!=0,axis=1)]:
#         faces=[V[V@v==np.min(V@v)] for V in vertices]
#         intersects=True
#         for r in range(1,len(faces)+1):
#             for inds in combinations(range(len(faces)),r):
#                 diffs=[faces[i][1:]-faces[i][0] for i in inds]
#                 diffs=[d for d in diffs if len(d)]
#                 dim=(np.linalg.matrix_rank(np.vstack(diffs))if diffs else 0)
#                 if dim<r:
#                     intersects=False
#                     break
#             if not intersects:
#                 break
#         if intersects:
#             out.append(v)
#     return np.array(out,dtype=int)

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
# def tensor_symm_in_glsm_basis(tensor,Q,basis,tol=1e-12):
#     """Transform a symmetric tensor from a prime-toric basis to the GLSM basis."""
#     if not tensor:
#         return {}
#     basis=np.asarray(basis,int)
#     integer_keys=isinstance(next(iter(tensor)),(int,np.integer))
#     if integer_keys:
#         tensor={(k,):v for k,v in tensor.items()}
#     else:
#         tensor={tuple(sorted(k)):v for k,v in tensor.items()}
#     rank=len(next(iter(tensor)))
#     h=len(basis)
#     A=np.linalg.inv(np.asarray(Q)[:,basis-1])
#     result={}
#     for new in combinations_with_replacement(range(h),rank):
#         value=0
#         for old,c in tensor.items():
#             old=tuple(i-1 for i in old)
#             for p in set(permutations(old)):
#                 value+=c*np.prod([A[i,j] for i,j in zip(p,new)])
#         if abs(value)<tol:
#             continue
#         if abs(value-round(value))<tol:
#             value=int(round(value))
#         key=new[0]+1 if integer_keys else tuple(i+1 for i in new)
#         result[key]=value
#     return result

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
# def tensor_symm_in_glsm_basis(tensor: Mapping[int,int|float]|Mapping[tuple[int,...],int|float],Q: ArrayLike,basis: ArrayLike,tol: float=1e-5,block_bytes: int=2**27)->dict[int,int|float]|dict[tuple[int,...],int|float]:
#     """
#     **Description:**
#     Transform a symmetric tensor from a prime-toric basis to the GLSM basis.
#     The tensor is contracted in every slot with the inverse of the basis columns `Q[:,basis-1]`.
#     The first slot is contracted directly from the sparse input, in blocks of output indices a; the remaining
#     slots then only need the columns b>=a, so the dense h^rank tensor is never formed.
#     **Arguments:**
#     - `tensor (Mapping[int,int|float]|Mapping[tuple[int,...],int|float])`: The symmetric tensor in the prime toric basis, keyed by one-indexed basis positions (rank 1) or tuples of them.
#     - `Q (ArrayLike)`: GLSM charge matrix (rows: GLSM charges, columns: prime toric divisors).
#     - `basis (ArrayLike)`: The one-indexed labels of the prime toric basis divisors.
#     - `tol (float)`: Entries with absolute value below `tol` are dropped and entries within `tol` of an integer are rounded. Defaults to `1e-5`.
#     - `block_bytes (int)`: Approximate memory of one block of partially contracted slices. Defaults to `2**27`.
#     **Returns:**
#     - `dict[int,int|float]|dict[tuple[int,...],int|float]`: The tensor in the GLSM basis, keyed by one-indexed GLSM indices (rank 1) or sorted tuples of them; empty if `tensor` is empty.
#     """
#     if not tensor:
#         return {}
#     basis=np.asarray(basis,int)
#     integer_keys=isinstance(next(iter(tensor)),(int,np.integer))
#     if integer_keys:
#         tensor={(k,):v for k,v in tensor.items()}
#     else:
#         tensor={tuple(sorted(k)):v for k,v in tensor.items()}
#     rank=len(next(iter(tensor)))
#     h=len(basis)
#     A=np.linalg.inv(np.asarray(Q)[:,basis-1])
#     old=np.array(list(tensor),dtype=np.int64).reshape(-1,rank)-1
#     vals=np.array(list(tensor.values()),dtype=float)
#     lin=np.concatenate([np.ravel_multi_index(old[:,p].T,(h,)*rank) for p in permutations(range(rank))])
#     lin,pos=np.unique(lin,return_index=True)
#     vals=np.tile(vals,factorial(rank))[pos]
#     first,rest=np.divmod(lin,h**(rank-1))
#     T=sps.csr_matrix((vals,(first,rest)),shape=(h,h**(rank-1)))
#     blk=max(1,block_bytes//(8*h**(rank-1)))
#     keys,values=[],[]
#     for a0 in range(0,h,blk):
#         a1=min(a0+blk,h)
#         S=np.asarray(A[:,a0:a1].T@T).reshape((a1-a0,)+(h,)*(rank-1))
#         Aa=A[:,a0:]
#         for _ in range(rank-1):
#             S=np.tensordot(S,Aa,axes=([1],[0]))
#         for a in range(a0,a1):
#             idx=_sorted_index_tuples(h-a,rank-1)+a if rank>1 else np.empty((1,0),dtype=np.intp)
#             v=np.atleast_1d(S[a-a0][tuple((idx-a0).T)])
#             keep=np.abs(v)>=tol
#             keys.append(np.column_stack([np.full(keep.sum(),a),idx[keep]]))
#             values.append(v[keep])
#     keys=np.concatenate(keys)+1
#     values=np.concatenate(values)
#     rounded=np.rint(values)
#     is_int=np.abs(values-rounded)<tol
#     out=values.astype(object)
#     out[is_int]=rounded[is_int].astype(np.int64).astype(object)
#     keys=keys[:,0].tolist() if integer_keys else zip(*keys.T.tolist())
#     return dict(zip(keys,out.tolist()))

def mori_cone_in_glsm_basis(Q: ArrayLike,fan: Fan|None=None,H: ArrayLike|None=None,tol: float=1e-10)->Cone:
    """
    **Description:**
    Mori cone of a secondary-fan phase, expressed in the
    basis given by the rows of Q.
    **Arguments:**
    - `Q (ArrayLike)`: Full-rank GLSM charge matrix whose rows define the basis.
    - `fan (Fan|None)`: The toric fan of the phase; its secondary cone hyperplanes are used if `H` is not given. Defaults to `None`.
    - `H (ArrayLike|None)`: The secondary (Kahler) cone hyperplane normals, one per row. Defaults to `None`.
    - `tol (float)`: Numerical tolerance. Defaults to `1e-10`.
    **Raises:**
    - `ValueError`: Raised if neither `fan` nor `H` is given, if Q is not full rank, or if the hyperplane normals are not in the row span of Q.
    **Returns:**
    - `Cone`: The Mori cone, generated by the hyperplane normals in the basis of the rows of Q.
    """
    if H is None:
        if fan is None:
            raise ValueError("Either fan or H has to be provided")
        H=np.asarray(fan.secondary_cone_hyperplanes(),dtype=float)
    else:
        H=np.asarray(H,dtype=float)
    Q=np.asarray(Q,dtype=float)
    if np.linalg.matrix_rank(Q)<Q.shape[0]:
        raise ValueError("Q is not full rank.")
    C=np.linalg.lstsq(Q.T,H.T,rcond=None)[0].T
    if not np.allclose(C@Q,H,atol=tol,rtol=tol):
        raise ValueError("Secondary-cone normals are not in the row span of Q")
    C[np.abs(C)<tol]=0
    return Cone(rays=C)

def cygv_intnums_prime_surfaces(IN4: Mapping[tuple[int,...],int|float],Q: ArrayLike)->tuple[dict[tuple[int,int,int],int],dict[int,tuple[int,int]]]:
    """
    **Description:**
    Intersection numbers of the surfaces D_A*D_B (pairs A < B of prime toric divisors) with pairs of GLSM basis divisors, in the format of the `intnums` input of `compute_gv`: kappa_{s,i,j} = sum_{a,b} Q[a,A] Q[b,B] kappa_{abij}.
    **Arguments:**
    - `IN4 (Mapping[tuple[int,...],int|float])`: Quadruple intersection numbers in the GLSM basis, keyed by sorted one-indexed 4-tuples.
    - `Q (ArrayLike)`: GLSM charge matrix (rows: GLSM basis, columns: prime toric divisors).
    **Returns:**
    - `tuple[dict[tuple[int,int,int],int],dict[int,tuple[int,int]]]`: `(intnums, surface_map)`: the nonzero intersection numbers `{(s, i, j): value}` with zero-based i <= j, and the map from the surface index s to the one-indexed prime toric pair (A, B).
    """
    Q=np.asarray(Q,dtype=object)
    h, n_toric=Q.shape
    prime_pairs=list(combinations(range(n_toric),2))
    intnums={}
    surface_map={}
    for s, (A,B) in enumerate(prime_pairs):
        surface_map[s]=(A+1,B+1)
        for i in range(h):
            for j in range(i,h):
                val=0
                for a in range(h):
                    for b in range(h):
                        key=tuple(sorted((a+1,b+1,i+1,j+1,)))
                        kappa=IN4.get(key,0)
                        val+=Q[a,A]*Q[b,B]*kappa
                if val!=0:
                    intnums[(s,i,j)]=int(val)
    return intnums, surface_map

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
    
# def cicy4_toric_chern_class(fan: Fan,divisors: ArrayLike,n: int)->int|dict[tuple[int,...],int]|dict[int,int]:
#     """
#     **Description:**
#     Chern tensor c_n(X) for a codimension-2 CY fourfold X=L1∩L2 in a 6-dimensional simplicial toric variety, via
#     `toric_chern_class`. Prime-toric divisor indices are 1-based.
#     **Arguments:**
#     - `fan (Fan)`: The simplicial toric fan of the 6-dimensional ambient variety.
#     - `divisors (ArrayLike)`: The coefficient vectors of L1 and L2 in the prime toric divisor basis, one per row.
#     - `n (int)`: Degree of the Chern class (1 to 4).
#     **Raises:**
#     - `ValueError`: Raised if the ambient toric variety is not 6-dimensional.
#     **Returns:**
#     - `int|dict[tuple[int,...],int]|dict[int,int]`: The Chern tensor:
#         n=1 : 0
#         n=2 : {(i,j): value}
#         n=3 : {i: value}
#         n=4 : number
#     """
#     if n==1:
#         return 0
#     if np.asarray(fan.vectors()).shape[1]!=6:
#         raise ValueError("Ambient toric variety must be 6-dimensional.")
#     return toric_chern_class(fan,divisors,n)

def subvariety_intersection_numbers(fan: Fan,divisors: Sequence[ArrayLike]|None=None,basis: ArrayLike|None=None,Q: ArrayLike|None=None,in_basis: bool=False,tol: float|None=1e-5,decimals: int=10)->dict[tuple[int,...],int|float]:
    """
    **Description:**
    Intersection numbers of the restricted toric divisors on the complete intersection X = D^(1) ∩ ... ∩ D^(r) of `divisors` in the toric variety of `fan`, in terms of all prime toric divisors, of a prime toric basis, or of a GLSM basis (see `_subvariety_intersection_numbers_all_prime_torics` and `tensor_symm_in_basis`).
    No Calabi-Yau condition is assumed: the result is the ambient intersection product with the class D^(1)...D^(r), which gives the intersection numbers of X whenever X is a transverse complete intersection (e.g. for generic sections of base-point-free divisors).
    **Arguments:**
    - `fan (Fan)`: The toric fan of the ambient variety.
    - `divisors (Sequence[ArrayLike]|None)`: Coefficient vectors of the divisors cutting out X, ordered by ray label. If `None`, the anticanonical hypersurface is used. Defaults to `None`.
    - `basis (ArrayLike|None)`: One-indexed labels of the prime toric basis divisors. If `None` and `in_basis` is `True` or `Q` is given, `basis_H2_toric_fan(fan)` is used. Defaults to `None`.
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
            basis=basis_H2_toric_fan(fan)
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

def KC_ext(vc: VectorConfiguration,Q: ArrayLike)->Cone:
    """
    **Description:**
    Extended Kahler cone: the cone generated by the Kahler cones of all phases (fine regular triangulations) in the secondary fan of `vc`, expressed in the GLSM basis given by the rows of Q.
    **Arguments:**
    - `vc (VectorConfiguration)`: The vector configuration whose secondary fan is used.
    - `Q (ArrayLike)`: GLSM charge matrix whose rows define the basis.
    **Returns:**
    - `Cone`: The extended Kahler cone.
    """
    H,_=vc.secondary_fan(only_fine=True)
    KC_rays=[]
    for h in H:
        MC=mori_cone_in_glsm_basis(Q,H=h)
        KC=MC.dual()
        KC_rays.append(KC.rays())
    return Cone(np.unique(np.vstack(KC_rays),axis=0))

def divisor_in_zero_one_rep_numpy(points: ArrayLike,a: ArrayLike,atol: float=1e-8)->tuple[np.ndarray,np.ndarray]|tuple[None,None]:
    """
    **Description:**
    Shift divisor a by a principal divisor m (b = a + points@m) so that b has
    coefficients in {0,1}. Returns (b,m) or (None,None) if impossible.
    **Arguments:**
    - `points (ArrayLike)`: The ray generators of the toric fan, one per row.
    - `a (ArrayLike)`: The integer divisor coefficients, one per ray.
    - `atol (float)`: Tolerance for the integrality of the shift. Defaults to `1e-8`.
    **Returns:**
    - `tuple[np.ndarray,np.ndarray]|tuple[None,None]`: `(b, m)` with integer arrays b (coefficients in {0,1}) and m, or `(None, None)`.
    """
    points=np.asarray(points,dtype=int)
    a=np.asarray(a,dtype=int)
    dim=points.shape[1]
    if np.all((a==0)|(a==1)):
        return a.copy(),np.zeros(dim,dtype=int)
    S=np.asarray(basis(points),dtype=int)
    B_inv=np.linalg.inv(points[S].astype(float))
    T=(np.arange(2**dim)[:,None]>>np.arange(dim))&1
    M=(T-a[S])@B_inv.T
    Mi=np.rint(M).astype(int)
    b=a+Mi@points.T
    ok=np.all(np.abs(M-Mi)<atol,axis=1)&np.all((b==0)|(b==1),axis=1)
    i=np.flatnonzero(ok)
    return (b[i[0]],Mi[i[0]]) if len(i) else (None,None)

def divisor_in_zero_one_rep(rays: ArrayLike,a: ArrayLike)->tuple[np.ndarray,np.ndarray]|tuple[None,None]:
    """
    **Description:**
    Shift divisor a by a principal divisor m (b = a + rays@m) so that b has coefficients in {0,1}, using a pivoted QR decomposition to choose a basis of rays.
    **Arguments:**
    - `rays (ArrayLike)`: The ray generators of the toric fan, one per row.
    - `a (ArrayLike)`: The integer divisor coefficients, one per ray.
    **Raises:**
    - `ValueError`: Raised if the shapes of `a` and `rays` are incompatible or the rays do not span the lattice.
    **Returns:**
    - `tuple[np.ndarray,np.ndarray]|tuple[None,None]`: `(b, m)` with integer arrays b (coefficients in {0,1}) and m, or `(None, None)` if impossible.
    """
    a=np.asarray(a,dtype=int)
    V=np.asarray(rays,dtype=int)
    if V.ndim!=2 or a.shape!=(V.shape[0],):
        raise ValueError("Incompatible shapes for a and rays.")
    n=V.shape[1]
    if np.all((a==0)|(a==1)):
        return a.copy(),np.zeros(n,dtype=int)
    _,R,piv=qr(V.T.astype(float),pivoting=True)
    if np.sum(np.abs(np.diag(R))>1e-9)<n:
        raise ValueError("Rays do not span: shift m is not unique.")
    S=piv[:n]
    B=np.array(list(product((0,1),repeat=n)))
    M=(B-a[S])@np.linalg.inv(V[S]).T
    Mi=np.rint(M).astype(int)
    ok=np.all(np.abs(M-Mi)<1e-8,axis=1)
    b=a+Mi@V.T
    ok&=np.all((b==0)|(b==1),axis=1)
    i=np.flatnonzero(ok)
    return (b[i[0]],Mi[i[0]]) if len(i) else (None,None)
    return b,m

def hnf_N(M,cols):
    """Row Hermite form (FLINT) of the lattice spanned by M for the column order cols: rows i.. vanish on the pivot columns of rows <i. Returns (H, pivot columns)."""
    M=np.asarray(M); cols=np.asarray(list(cols),int)
    if M.size==0 or not len(cols): return np.zeros((0,M.shape[-1] if M.ndim==2 else 0),int),np.zeros(0,int)
    perm=np.r_[cols,np.setdiff1d(np.arange(M.shape[1]),cols)]
    H=hnf(M[:,perm]); H=H[H.any(axis=1)]; piv=perm[(H!=0).argmax(axis=1)]; keep=np.isin(piv,cols)
    out=np.empty_like(H); out[:,perm]=H
    return out[keep],piv[keep]

def _reduce_hnf(v,M):
    """Reduce the integer vector v (or each row of the matrix v) modulo the rows of the row-HNF M."""
    v=np.array(v,dtype=int)
    for row in M:
        c=(row!=0).argmax(); v-=np.multiply.outer(v[...,c]//row[c],row)
    return v

def kernel_hnf_N(V,cols):
    """Row Hermite form, pivoting in the order cols, of the saturated relation lattice {q : q@V=0, supp(q) in cols}; same output as hnf_N(rel(cols),cols).
    With N the greedy basis of V[cols] taken from the end and P the other (pivot) columns, the projection of the lattice to P is the kernel of Z^P -> G=Z<V[cols]>/Z<V[N]>.
    Its Hermite form is computed inside the finite group G (pivots >1 only where G grows), the N-entries follow from the coordinates of V[P] in the basis V[N]. Returns (H, pivot columns)."""
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
# def adapted_glsm(V,bulk,u=np.array([0,0,0,0,1,0])):
#     """GLSM of V adapted to the fibration with height V@u; bulk = count of leading bulk points or index array.
#     Rows: [bulk | D7 | fiber-only | tau]; bulk and fiber-only rows are supported on height-zero columns only."""
#     V=np.asarray(V); h=V@u; n=len(V); B=np.arange(bulk) if np.isscalar(bulk) else np.asarray(bulk)
#     N,S=np.flatnonzero(h>0),np.flatnonzero(h<0); tie=lambda i:(V[i,:-2].any(),V[i,-1])
#     p=min(N[h[N]==h[N].min()],key=tie); b0=min(S[h[S]==h[S].max()],key=tie)
#     S=np.r_[N[N!=p],S[S!=b0],b0]; Fl=np.setdiff1d(np.flatnonzero(h==0),B)
#     H,pv=kernel_hnf_N(V,[p,*S,*B,*Fl])
#     Hnb,pvn=kernel_hnf_N(V,[p,*S,*Fl])
#     tau_free=Hnb[0,p]==H[0,p]; tau=Hnb[0] if tau_free else H[0]
#     D7,D7nb=H[np.isin(pv,S)],Hnb[np.isin(pvn,S)]
#     bulkrows,Fo=H[np.isin(pv,B)],H[np.isin(pv,Fl)]
#     Q=np.vstack([bulkrows,D7,Fo,tau])
#     return Q

def adapted_glsm(V,bulk,u=np.array([0,0,0,0,1,0]),return_rows=False):
    """... Rows: [bulk | fiber-only | D7 | tau] ... return_rows=True also returns {'bulk','fiber','d7','tau'} (0-based)."""
    V=np.asarray(V); h=V@u; B=np.arange(bulk) if np.isscalar(bulk) else np.asarray(bulk)
    N,S=np.flatnonzero(h>0),np.flatnonzero(h<0); tie=lambda i:(V[i,:-2].any(),V[i,-1])
    p=min(N[h[N]==h[N].min()],key=tie); b0=min(S[h[S]==h[S].max()],key=tie)
    S=np.r_[N[N!=p],S[S!=b0],b0]; Fl=np.setdiff1d(np.flatnonzero(h==0),B)
    H,pv=kernel_hnf_N(V,[p,*S,*B,*Fl])
    Hnb,_=kernel_hnf_N(V,[p,*S,*Fl])
    tau=Hnb[0] if Hnb[0,p]==H[0,p] else H[0]
    blocks=[H[np.isin(pv,B)],H[np.isin(pv,Fl)],H[np.isin(pv,S)]]
    Q=np.vstack(blocks+[tau])
    if not return_rows: return Q
    c=np.cumsum([0]+[len(b) for b in blocks])
    return Q,{"bulk":np.arange(c[0],c[1]),"fiber":np.arange(c[1],c[2]),"d7":np.arange(c[2],c[3]),"tau":len(Q)-1}

def adapted_glsm_uplift(F,return_rows=False):
    """
    **Description:**
    Applies `adapted_glsm` to an F-theory uplift, using its intersecting M-lattice divisors (in the adapted toric fan ordering) and the number of intersecting base divisors as the bulk size.
    **Arguments:**
    - `F (F_Theory_Uplift)`: The F-theory uplift.
    **Returns:**
    - `np.ndarray`: The adapted GLSM charge matrix.
    """
    V=F.vectors_M_conv_toric_fan()
    n_bulk=len(F.intersecting_divisors_B_M(as_labels=True))+len(F.vertices_B_not_intersecting())
    return adapted_glsm(V,n_bulk,return_rows=return_rows)

_PERMS4=np.array(list(permutations(range(4))))

def _dense_quartic(IN,N)->np.ndarray:
    """
    **Description:**
    Dense symmetric (N,N,N,N) array of quartic intersection numbers from a dict with sorted 1-based keys.
    """
    T=np.zeros((N,)*4)
    items=[(k,v) for k,v in IN.items() if len(k)==4]
    if items:
        k=np.array([k for k,_ in items])-1
        v=np.array([v for _,v in items],dtype=float)
        for p in _PERMS4:
            T[tuple(k[:,p].T)]=v
    return T

def _quartic_dict(T,tol)->dict[tuple[int,...],int|float]:
    """
    **Description:**
    Inverse of `_dense_quartic`: sorted 1-based keys of all entries above `tol`, values rounded to integers where possible.
    """
    idx=np.argwhere(np.abs(T)>tol)
    idx=idx[np.all(np.diff(idx,axis=1)>=0,axis=1)]
    vals=T[tuple(idx.T)]
    r=np.rint(vals)
    vals=[int(ri) if ok else float(v) for v,ri,ok in zip(vals.tolist(),r.tolist(),(np.abs(vals-r)<tol).tolist())]
    return dict(zip(map(tuple,(idx+1).tolist()),vals))

def _form_centre(F,rtol=1e-12)->np.ndarray:
    """
    **Description:**
    Orthonormal basis (shape (c,d,d)) of the centre {X : X^T B = B X for all B in F} of a family of symmetric forms F (shape (r,d,d)). It is the kernel of the positive semidefinite matrix
        G = sum_B L_B^T L_B,   L_B(X) = X^T B - B X,
    which in row-major vec(X) reads G = (sum_B B^2) (x) 1 - sum_B B_{ij} B_{kl} with indices permuted to ((j,k),(l,i)).
    """
    r,d,_=F.shape
    S=F.transpose(1,0,2).reshape(d,r*d)@F.reshape(r*d,d)
    Fm=F.reshape(r,d*d)
    T=(Fm.T@Fm).reshape(d,d,d,d)
    H=T.transpose(1,2,3,0).reshape(d*d,d*d)
    G=np.kron(S,np.eye(d))-0.5*(H+H.T)
    w,V=np.linalg.eigh(G)
    C=V[:,w<=rtol*max(w[-1],1.0)]
    return C.T.reshape(-1,d,d)

def _orthogonal_split(U,F,rng,rtol=1e-12,etol=1e-6)->list[np.ndarray]:
    """
    **Description:**
    Finest splitting of the column span of U (orthonormal) into pieces that are orthogonal for all forms in F, via the (generalized) eigenspaces of a generic centre element, applied recursively.
    """
    d=U.shape[1]
    if d==1:
        return [U]
    FU=U.T@F@U
    nrm=np.linalg.norm(FU.reshape(len(FU),-1),axis=1)
    FU=FU[nrm>1e-9*max(nrm.max(),1.0)] if len(FU) else FU
    C=_form_centre(FU,rtol) if len(FU) else np.eye(d*d).reshape(-1,d,d)
    if len(C)<=1:
        return [U]
    Y=np.einsum('c,cij->ij',rng.standard_normal(len(C)),C)
    ev=np.linalg.eigvals(Y)
    scale=max(np.abs(ev).max(),1.0)
    clusters=[]
    for e in sorted(ev,key=lambda z:(z.real,abs(z.imag))):
        for cl in clusters:
            if min(abs(e-cl[0]),abs(e-np.conj(cl[0])))<etol*scale:
                cl.append(e); break
        else:
            clusters.append([e])
    if len(clusters)==1:
        return [U]
    out=[]
    for cl in clusters:
        lam=cl[0]
        _,Z,sdim=schur(Y,output='real',sort=lambda x,y: min(abs(complex(x,y)-lam),abs(complex(x,y)-np.conj(lam)))<etol*scale)
        out+=_orthogonal_split(np.linalg.qr(U@Z[:,:sdim])[0],F,rng,rtol,etol)
    return out

def _integer_relations(C,d,scale=1e10)->np.ndarray|None:
    """
    **Description:**
    LLL-reduced basis (rows) of the saturated lattice {x in Z^n : x @ C = 0} for an orthonormal float matrix C (n,m) whose orthogonal complement has dimension d, found by LLL on the integer basis [1 | scale*C]. Returns None if the d shortest vectors do not satisfy the relations.
    """
    n=C.shape[0]
    if C.shape[1]==0:
        return np.eye(n,dtype=int)
    B=np.hstack([np.eye(n,dtype=np.int64),np.rint(scale*C).astype(np.int64)])
    L=np.array(fmpz_mat(B.tolist()).lll().tolist(),dtype=object)
    X=L[:d,:n].astype(int)
    if np.abs(X@C).max()>1e-6 or np.linalg.matrix_rank(X)<d:
        return None
    return X

def _is_rational(U)->bool:
    """Whether the column span of U is a rational subspace, i.e. contains a full-rank integer lattice."""
    d=U.shape[1]
    Z=np.linalg.qr(U,mode='complete')[0][:,d:]
    return _integer_relations(Z,d) is not None

def _rational_blocks(blocks)->list[np.ndarray]:
    """
    **Description:**
    Merges real orthogonal pieces that are not rational subspaces (Galois-conjugate pieces of a splitting over a number field) into the smallest groups whose sums are rational.
    """
    good=[U for U in blocks if _is_rational(U)]
    bad=[U for U in blocks if not _is_rational(U)]
    while bad:
        for r in range(2,len(bad)+1):
            S=next((S for S in combinations(range(len(bad)),r) if _is_rational(np.hstack([bad[i] for i in S]))),None)
            if S is not None:
                good.append(np.linalg.qr(np.hstack([bad[i] for i in S]))[0])
                bad=[U for i,U in enumerate(bad) if i not in S]
                break
        else:
            good.append(np.linalg.qr(np.hstack(bad))[0])
            bad=[]
    return good

def _block_tensor(T,rows,W,R=None)->np.ndarray:
    """
    **Description:**
    Dense intersection data of one block: T restricted to the indices rows+W (block first), optionally after the change J'_a = sum_i R[i,a] J_i of the block divisors.
    """
    idx=list(rows)+list(W)
    S=T[np.ix_(idx,idx,idx,idx)]
    if R is not None:
        M=np.eye(len(idx)); M[:len(rows),:len(rows)]=R
        for _ in range(4):
            S=np.tensordot(S,M,axes=([0],[0]))
    return S

def _block_mask(d,w)->np.ndarray:
    """Entries with at least one block index."""
    a=np.arange(d+w)<d
    return a[:,None,None,None]|a[None,:,None,None]|a[None,None,:,None]|a[None,None,None,:]

def _block_covariants(S,d)->np.ndarray:
    """
    **Description:**
    Covectors of a block (columns of a (d,m) array transforming as u -> R^T u under J'=R^T J), from its dense data S (block indices first): the linear invariants, the cubic terms contracted with the inverse of the first invertible bilinear invariant G, and the images of all of these under B G^{-1} for every bilinear invariant B (including the contracted quartic).
    """
    b,w=slice(0,d),slice(d,None)
    U=[S[b,w,w,w].reshape(d,-1)]
    Bs=list(S[b,b,w,w].reshape(d,d,-1).transpose(2,0,1))
    G=next((B for B in Bs if abs(np.linalg.det(B))>1e-9),None)
    if G is not None:
        Gi=np.linalg.inv(G)
        U.append(np.einsum('abcw,bc->aw',S[b,b,b,w],Gi))
        Bs.append(np.einsum('abcd,cd->ab',S[b,b,b,b],Gi))
        U0=np.hstack(U)
        U+=[B@Gi@U0 for B in Bs]
    return np.hstack(U)

def _block_alignment(T,rows_k,rows_r,W,tol=1e-6)->np.ndarray|None:
    """
    **Description:**
    Finds R in GL(d,Z) such that block k in the divisor basis J'=R^T J has the same intersection data as block r, or returns None. For d>1, R is fixed by the covariants of `_block_covariants` when they span; otherwise all R with entries in {-1,0,1} are tried for d<=3.
    """
    d=len(rows_k)
    Sk,Sr=_block_tensor(T,rows_k,W),_block_tensor(T,rows_r,W)
    mask=_block_mask(d,len(W))
    if d==1:
        cands=[np.array([[1]]),np.array([[-1]])]
    else:
        Lk,Lr=_block_covariants(Sk,d),_block_covariants(Sr,d)
        cands=[]
        if np.linalg.matrix_rank(Lk,tol=1e-8)==d:
            RT=np.linalg.lstsq(Lk.T,Lr.T,rcond=None)[0].T
            if np.abs(RT-np.rint(RT)).max()<1e-6:
                cands=[np.rint(RT.T).astype(int)]
        elif d<=3:
            cands=[np.array(e).reshape(d,d) for e in product((-1,0,1),repeat=d*d)]
    for R in cands:
        if abs(round(np.linalg.det(R)))==1 and np.abs(_block_tensor(T,rows_k,W,R)-Sr)[mask].max(initial=0)<tol:
            return R
    return None

def symmetric_d7_basis(Q: np.ndarray,IN: Mapping[tuple[int,...],int|float],d7_rows: Sequence[int],tol: float=1e-6,seed: int=0)->dict[str,Any]:
    """
    **Description:**
    Changes basis within the D7 rows of an adapted GLSM such that the D7 sector of the quartic intersection ring splits into the finest possible sum of mutually non-intersecting blocks,
        K(J_a,J_b,J_s,J_t)=0   for a, b in different blocks and all s, t,
    and brings blocks with isomorphic intersection data to identical form, so that permuting them is a symmetry realised by permutation matrices on the rows. The splitting is read off from the centre {X : X^T B = B X} of the bilinear forms B^{st}_{ab}=K(J_a,J_b,J_s,J_t); it is unique whenever that centre is commutative and semisimple. In the K3 / transverse case each block is a single row (one per D7 brane) and the blocks are permuted by the symmetric group; in general a block collects D7 moduli that cannot be separated. Bulk and tau rows are left unchanged.
    The linear algebra is done numerically (numpy); integral rows are recovered by LLL and the final decoupling is verified to tolerance `tol`.
    **Arguments:**
    - `Q (np.ndarray)`: GLSM charge matrix, e.g. from `adapted_glsm_uplift`.
    - `IN (Mapping[tuple[int,...],int|float])`: Quartic intersection numbers in the divisor basis dual to the rows of `Q` (sorted 1-based keys), e.g. `tensor_symm_in_glsm_basis(F.intersection_numbers_M(),Q,F.basis_homology_M())`.
    - `d7_rows (Sequence[int])`: The 0-based indices of the D7 rows of `Q`.
    - `tol (float)`: Tolerance for comparing intersection numbers. Defaults to `1e-6`.
    - `seed (int)`: Seed for the generic centre elements used in the splitting. Defaults to `0`.
    **Raises:**
    - `RuntimeError`: If no integral rows are found for a block or the blocks do not decouple within `tol`.
    **Returns:**
    - `dict[str,Any]`: With keys
        `Q`: the new GLSM, `Q[d7_rows]` replaced by `transform @ Q[d7_rows]`, blocks contiguous and grouped by class,
        `transform`: the integer matrix A acting on the D7 rows,
        `index`: |det A|; 1 if the new D7 rows generate the same lattice, otherwise the decomposition only holds over Q,
        `blocks`: one array of row indices of `Q` per block,
        `classes`: lists of block numbers with identical intersection data (their permutations are symmetries),
        `sign_symmetric`: per block, whether J -> -J on the block is a symmetry of the ring,
        `rank_one`: per block, `(c,E)` if the block is a single row x entering only as c*((J_x+l)^4-l^4) with E=J_x+l (coefficient vector over all rows), else `None`,
        `intersection_numbers`: the quartic intersection numbers in the new basis.
    """
    Q=np.asarray(Q,dtype=int)
    d7=[int(i) for i in d7_rows]
    n,N=len(d7),len(Q)
    W=[i for i in range(N) if i not in d7]
    T0=_dense_quartic(IN,N)
    F=T0[np.ix_(d7,d7)].reshape(n,n,N*N).transpose(2,0,1)
    F=F[np.abs(F).max(axis=(1,2))>tol]
    blocks=_rational_blocks(_orthogonal_split(np.eye(n),F,np.random.default_rng(seed)))
    A=[]
    for k,U in enumerate(blocks):
        others=[V for l,V in enumerate(blocks) if l!=k]
        Ak=_integer_relations(np.linalg.qr(np.hstack(others))[0],U.shape[1]) if others else np.eye(n,dtype=int)
        if Ak is None:
            raise RuntimeError("no integral rows found for block %d"%k)
        Ak=hnf(Ak)
        Ak=Ak[np.any(Ak!=0,axis=1)]
        _,Tr=fmpz_mat((Ak@Q[d7]).tolist()).lll(transform=True)
        A.append(np.array(Tr.tolist(),dtype=int)@Ak)

    def new_T(A):
        Minv=np.eye(N)
        Minv[np.ix_(d7,d7)]=np.linalg.inv(np.vstack(A).astype(float))
        T=T0
        for _ in range(4):
            T=np.tensordot(T,Minv,axes=([0],[0]))
        return T

    def block_rows(A):
        s=np.cumsum([0]+[len(a) for a in A])
        return [[d7[i] for i in range(s[k],s[k+1])] for k in range(len(A))]
    T,rows=new_T(A),block_rows(A)
    classes=[]
    for k in range(len(A)):
        for cl in classes:
            if len(A[cl[0]])==len(A[k]):
                R=_block_alignment(T,rows[k],rows[cl[0]],W,tol)
                if R is not None:
                    A[k]=np.rint(np.linalg.inv(R)).astype(int)@A[k]
                    cl.append(k)
                    break
        else:
            classes.append([k])
    for cl in classes:
        Rw=np.vstack([A[k] for k in cl])@Q[d7]
        s=int(np.sign(Rw.sum())) or int(np.sign(Rw[0][np.flatnonzero(Rw[0])[0]]))
        for k in cl:
            A[k]=s*A[k]
    for cl in classes:
        cl.sort(key=lambda k: hnf(A[k]@Q[d7]).ravel().tolist())
    classes.sort(key=lambda cl: (-len(cl),-len(A[cl[0]]),hnf(A[cl[0]]@Q[d7]).ravel().tolist()))
    order=[k for cl in classes for k in cl]
    classes=[[order.index(k) for k in cl] for cl in classes]
    A=[A[k] for k in order]
    T,rows=new_T(A),block_rows(A)
    lab=np.full(N,-1);
    for k,b in enumerate(rows):
        lab[b]=k
    l=lab[:,None]
    cross=(l>=0)&(l.T>=0)&(l!=l.T)
    if np.abs(T[cross]).max(initial=0)>tol:
        raise RuntimeError("blocks do not decouple within tol=%g"%tol)
    sign_symmetric,rank_one=[],[]
    for b in rows:
        d=len(b)
        S=_block_tensor(T,b,W)
        nb=(np.arange(d+len(W))<d).astype(int)
        odd=(nb[:,None,None,None]+nb[None,:,None,None]+nb[None,None,:,None]+nb[None,None,None,:])%2==1
        sign_symmetric.append(bool(np.abs(S[odd]).max(initial=0)<tol))
        c=S[0,0,0,0]
        if d==1 and abs(c)>tol:
            lw=S[0,0,0,1:]/c
            ok=np.allclose(S[0,0,1:,1:],c*np.outer(lw,lw),atol=tol) and np.allclose(S[0,1:,1:,1:],c*np.einsum('i,j,k->ijk',lw,lw,lw),atol=tol)
            q=lambda v: (lambda f: int(f) if f.denominator==1 else f)(Fraction(float(v)).limit_denominator(10**6))
            E=np.zeros(N,dtype=object); E[b[0]]=1
            for w,x in zip(W,lw):
                E[w]=q(x)
            rank_one.append((q(c),E) if ok else None)
        else:
            rank_one.append(None)
    Afull=np.vstack(A).astype(int)
    Qn=Q.copy()
    Qn[d7]=Afull@Q[d7]
    return {"Q":Qn,"transform":Afull,"index":abs(int(fmpz_mat(Afull.tolist()).det())),"blocks":[np.array(b) for b in rows],"classes":classes,"sign_symmetric":sign_symmetric,"rank_one":rank_one,"intersection_numbers":_quartic_dict(T,tol)}

def symmetric_glsm_uplift(F: F_Theory_Uplift,tol: float=1e-6)->dict[str,Any]:
    """
    **Description:**
    Applies `symmetric_d7_basis` to the adapted GLSM of an F-theory uplift. The D7 rows are the rows of `adapted_glsm_uplift(F)` between the bulk rows and the final tau row (one per special fiber point other than b0).
    **Arguments:**
    - `F (F_Theory_Uplift)`: The F-theory uplift.
    - `tol (float)`: Tolerance for comparing intersection numbers. Defaults to `1e-6`.
    **Returns:**
    - `dict[str,Any]`: The output of `symmetric_d7_basis`, with the additional key `d7_rows`.
    """
    Q,rows=adapted_glsm_uplift(F,return_rows=True)
    d7=rows["d7"]
    IN=_tensor_symm_in_glsm_basis(F.intersection_numbers_M(),Q,F.basis_homology_M())
    res=symmetric_d7_basis(Q,IN,d7,tol=tol)
    return (res["Q"],res["intersection_numbers"],res["index"])
    # Q=adapted_glsm_uplift(F)
    # V=F.vectors_M_conv_toric_fan()
    # n_d7=int(np.sum(V@np.array([0,0,0,0,1,0])<0))-1
    # d7=np.arange(len(Q)-1-n_d7,len(Q)-1)
    # IN=_tensor_symm_in_glsm_basis(F.intersection_numbers_M(),Q,F.basis_homology_M())
    # res=symmetric_d7_basis(Q,IN,d7,tol=tol)
    # return (res["Q"],res["intersection_numbers"],res["index"])

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
    # column i of P = coordinates of row i of M in the first r rows of H, i.e. P^T H[:r] = M, solved on the pivot columns
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

def intersection_form_H22_vertical(IN_in_basis: Mapping[tuple[int,...],int|float],max_nodes: int=10**3)->tuple[np.ndarray,np.ndarray,dict[int,tuple[int,int]],np.ndarray]:
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
    return C, G, inv_dict,mat

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

def face_fan(p: Polytope,simplicial: bool=False)->Fan:
    """
    **Description:**
    Constructs the face fan of a polytope containing the origin in its interior: its rays are the vertices of `p` and
    its maximal cones are spanned by the vertices of each facet. For a reflexive polytope this is the normal fan of the
    dual polytope, which is never constructed.
    **Arguments:**
    - `p (Polytope)`: A full-dimensional polytope with the origin in its interior.
    **Raises:**
    - `ValueError`: Raised if `p` is not full-dimensional or the origin is not in its interior.
    **Returns:**
    - `Fan`: The (in general non-simplicial) fan over `VectorConfiguration(p.vertices())`.
    """
    V=p.vertices()
    ineqs=p.inequalities()
    if p.dim()!=V.shape[1] or (ineqs[:,-1]<=0).any():
        raise ValueError("The polytope must be full-dimensional with the origin in its interior.")
    on_facet=ineqs[:,:-1]@V.T+ineqs[:,-1:]==0
    if simplicial:
        return make_simplicial(Fan(vc=VectorConfiguration(V),cones=[tuple(np.flatnonzero(row)+1) for row in on_facet]))
    return Fan(vc=VectorConfiguration(V),cones=[tuple(np.flatnonzero(row)+1) for row in on_facet])

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

def is_lattice_newton_polytope(rays,weights):
    """ Checks whether the Newton Polytope of a given divisor will be a lattice polytope"""
    P=ppl.C_Polyhedron(len(rays[0]),"universe")
    for ray, weight in zip(rays,weights,strict=True):
        P.add_constraint(ppl.Linear_Expression(list(ray),weight)>=0)
    return (not P.is_empty() and P.is_bounded() and all(v.divisor()==1 for v in P.minimized_generators()))

def fix_tau_row(Q,IN,d7_rows,invariant_rows=None,tol=1e-4,decimals=10):
    """
    **Description:**
    Fixes the ambiguity q_tau -> q_tau + n^mu q_mu of the tau row (last row) of an adapted GLSM by Weyl invariance of the tau curve class, which reduces to
        kappa_{ijk mu} = n_mu kappa_{ijk tau}   for all invariant triples (i,j,k) and D7 directions mu.
    With K_{(ijk),mu}=kappa_{ijk mu}, k3_{ijk}=kappa_{ijk tau}, n is determined on V' = ker(|k3|^2 K - k3 k3^T K) (all of span(C_mu) iff the enhancement is
    semisimple) by n(v) = k3.Kv/|k3|^2; on the complement (Weyl-invariant, abelian directions) the shift is a convention, fixed here by integrality:
    n takes values in Z/gcd(k3) on the saturated lattice V'_Z, k is the minimal integer with k n(V'_Z) in Z, and k n is an integer extension (it exists
    since V'_Z is a direct summand), shortened along the free directions. The new tau row is k*q_tau + sum_mu k n_mu q_mu, so that
    z'^tau = k z^tau + sum_mu k n_mu z^mu is the axio-dilaton. In the dual basis J'^mu = J^mu - n_mu J^tau, J'^tau = J^tau/k (others unchanged) one has kappa'_{ijk mu}=0, kappa'_{ijk tau'} = kappa_{ijk tau}/k.
    The intersection numbers are updated sparsely: substituting x_tau = y_tau/k - n.y_D7, an entry kappa_c with c=(rest,tau^t) contributes to
        kappa'_{sort(rest+M)} += kappa_c prod_e binom(#_e rest + #_e M, #_e M) k^{-#_tau M} prod_{mu in M}(-n_mu)
    for every multiset M of size t from {tau} u {mu : n_mu != 0}; entries without tau are unchanged.
    **Arguments:**
    - `Q (ArrayLike)`: GLSM charge matrix [bulk | D7 | (fiber-only) | tau].
    - `IN (Mapping[tuple[int,...],int|float])`: Integral intersection numbers in the basis dual to the rows of Q (sorted 1-based keys); all key lengths are transformed.
    - `d7_rows (Sequence[int])`: 0-based indices of the D7 rows.
    - `invariant_rows (Sequence[int]|None)`: 0-based rows with Weyl-invariant flat coordinates. Defaults to all rows except `d7_rows` and the last.
    - `tol (float)`: New entries are rounded to `decimals`, values within `tol` of an integer are set to it, and entries with |value| <= tol are dropped
      (as in `subvariety_intersection_numbers`); near-integral values are returned as int. Defaults to `1e-5`.
    - `decimals (int)`: Number of decimals to round to. Defaults to `10`.
    **Returns:**
    - `tuple`: (Q_new, IN_new, k, n, free) with IN_new values int (near-integral) or float, n a float array, free an integer basis (rows) of the shifts of n not fixed by the
      condition (empty iff the fix is unique).
    """
    Q=np.asarray(Q,dtype=int); N=len(Q); d7=[int(i)+1 for i in d7_rows]; m=len(d7)
    inv=set(range(1,N))-set(d7) if invariant_rows is None else {int(i)+1 for i in invariant_rows}
    pos={a:i for i,a in enumerate(d7+[N])}; R={}
    for key,v in IN.items():
        x=[a for a in key if a in pos]
        if len(key)==4 and len(x)==1:
            s=list(key); s.remove(x[0])
            if inv.issuperset(s): R.setdefault(tuple(s),np.zeros(m+1))[pos[x[0]]]=v
    R=np.rint(np.array(list(R.values())).reshape(-1,m+1)).astype(np.int64); K,k3=R[:,:m],R[:,m]; s=int(k3@k3)
    if s==0: raise ValueError("kappa_{ijk tau} vanishes on the invariant rows")
    V=integral_kernel(s*K-np.outer(k3,k3@K)); r=len(V)
    free=integral_kernel(V) if r else np.eye(m,dtype=int)
    g=np.gcd.reduce(k3); kh=k3//g
    c,rem=np.divmod(V@K.T@kh,kh@kh); assert not rem.any()
    d=np.gcd.reduce(np.r_[g,c]); k=int(g//d)
    nk=np.zeros(m,dtype=int)
    if r:
        H,U=hnf(V.T,return_transform=True); assert (H[:r]==np.eye(r)).all()
        nk=U[:r].T@(c//d)
        if len(free): nk-=free.T@np.rint(np.linalg.lstsq(free.T,nk,rcond=None)[0]).astype(int)
    n=nk/k
    Qn=Q.copy(); Qn[-1]=k*Q[-1]+nk@Q[np.array(d7,dtype=int)-1]
    w={N:1/k,**{a:-x for a,x in zip(d7,n.tolist()) if x}}
    acc=defaultdict(float)
    for key,v in IN.items():
        r=len(key); t=key.count(N); rest=key[:r-t]; C=Counter(rest)
        for M in combinations_with_replacement(w,t):
            acc[tuple(sorted(rest+M))]+=v*prod(comb(C[e]+j,j)*w[e]**j for e,j in Counter(M).items())
    v=np.round(np.fromiter(acc.values(),float,len(acc)),decimals); r=np.rint(v); isint=np.abs(v-r)<=tol
    return Qn,{b:(int(ri) if ii else x) for b,x,ri,ii in zip(acc,v.tolist(),r.tolist(),isint.tolist()) if abs(x)>tol},k,n,free

def fix_tau_row_uplift(F,symmetric=True,tol=1e-6):
    """`fix_tau_row` applied to the adapted GLSM of an F-theory uplift (optionally in the symmetric D7 basis). Returns its output plus the D7 rows."""
    Q,rows=adapted_glsm_uplift(F,return_rows=True); d7=rows["d7"]
    IN=_tensor_symm_in_glsm_basis(F.intersection_numbers_M(),Q,F.basis_homology_M())
    if symmetric:
        res=symmetric_d7_basis(Q,IN,d7,tol=tol); Q,IN=res["Q"],res["intersection_numbers"]
    return (*fix_tau_row(Q,IN,d7),d7)

def check_sen_scaling(F: "F_Theory_Uplift",Q: ArrayLike)->tuple[bool,np.ndarray]:
    """
    **Description:**
    Checks the normalization of the tau row (last row) of a GLSM against Sen's weak-coupling limit a3 -> eps a3, a4 -> eps a4, a6 -> eps^2 a6 (3.29),
    in which e^{2 pi i tau} ~ 1/j ~ eps^2 while the bulk and D7 moduli stay finite. A point (rho~,t1,t2) of the Weierstrass polytope is the Tate monomial
    a_k with z-degree k = t2+1 (y^2, x^3: k=0; points of the base polytope have t2=0, i.e. k=1); its coefficient scales as eps^s with s = 1 for a3, a4,
    s = 2 for a6 and s = 0 otherwise, and the origins do not scale. The GKZ coordinate Psi_a of row a therefore scales as eps^{(Q s)_a}, and by the leading
    mirror map so does e^{2 pi i z^a}. Expected: (0,...,0,2).
    **Arguments:**
    - `F (F_Theory_Uplift)`: The F-theory uplift.
    - `Q (ArrayLike)`: A GLSM of `F` with columns ordered as `F.vectors_M_conv_toric_fan()` and the tau row last, e.g. from `adapted_glsm_uplift`,
      `symmetric_glsm_uplift` or `fix_tau_row`.
    **Returns:**
    - `tuple[bool,np.ndarray]`: Whether the tau row scales as eps^2 and all other rows as eps^0, and the eps-exponents of all rows.
    """
    V=np.asarray(F.vectors_M_conv_toric_fan()); Q=np.asarray(Q)
    assert {(0,0,0,0,0,-1),(0,0,0,0,1,-1)}<=set(map(tuple,V.tolist()))
    k=V[:,-1]+1                                                      
    w=Q@np.where(k>2,(k-1)//2,0)                                 
    return bool(w[-1]==2 and not w[:-1].any()),w

def d7_mori_face(G: ArrayLike,d7_rows: Sequence[int])->tuple[np.ndarray,np.ndarray]:
    """
    **Description:**
    Face of the Mori cone spanned by the curves that shrink on the D7 face of the Kahler cone. Curve classes are written in the basis of the GLSM rows, so t_a is the volume of the
    row curve l^a and the D7 face is K_D7 = K cap {t_mu = 0 : mu in d7_rows}, K = {t : G t >= 0}. A point t* in the relative interior of K_D7 is found by the LP
        max sum_i s_i   s.t.   G t >= s,   0 <= s <= 1,   t_mu = 0,
    whose optimum has s_i = 1 exactly for the generators that are not identically zero on K_D7. The curves shrinking at t* span the dual face MC cap t*^perp, generated by the G_i with s_i = 0.
    The face is extremal, so every splitting of a class in it into effective classes stays in it: the HKTY series truncated to the face is closed and its GV invariants are exact.
    **Arguments:**
    - `G (ArrayLike)`: Generators (rays or Hilbert basis) of the Mori cone in the GLSM basis, one per row.
    - `d7_rows (Sequence[int])`: 0-based indices of the D7 rows.
    **Returns:**
    - `tuple[np.ndarray,np.ndarray]`: Boolean mask of the generators lying in the face, and t*.
    """
    G=np.asarray(G,dtype=float); n,h=G.shape
    bounds=[(None,None)]*h+[(0,1)]*n
    for mu in d7_rows: bounds[mu]=(0,0)
    res=linprog(np.r_[np.zeros(h),-np.ones(n)],A_ub=np.hstack([-G,np.eye(n)]),b_ub=np.zeros(n),bounds=bounds,method="highs")
    if res.status: raise RuntimeError(res.message)
    return res.x[h:]<0.5,res.x[:h]

def compute_d7_gv_invariants(F,q=None,d7_rows=None,max_deg=5,min_points=None,hilbert_basis=False):
    """
    **Description:**
    Genus-zero GV invariants of the curve classes in the face of the Mori cone dual to the D7 face of the Kahler cone (all t_mu = 0 for the D7 rows, all other directions generic),
    see `d7_mori_face`. Only classes in that face are enumerated, which is exact since the face is extremal.
    Warns if the face is not spanned by the D7 rows alone, i.e. if curves with non-D7 components shrink as well.
    **Arguments:**
    - `F (F_Theory_Uplift)`: The F-theory uplift.
    - `q (ArrayLike|None)`: GLSM charge matrix. Defaults to `adapted_glsm_uplift(F)`.
    - `d7_rows (Sequence[int]|None)`: 0-based D7 rows of `q`; required if `q` is given.
    - `max_deg (int|None)`: Maximal degree w.r.t. the grading vector of the Mori cone (takes precedence over `min_points` in cygv). Defaults to `5`.
    - `min_points (int|None)`: Minimal number of points, used if `max_deg` is `None`.
    - `hilbert_basis (bool)`: Generate the face by the Hilbert basis instead of the rays; needed to reach all lattice points of the face if the ray semigroup is not saturated.
    **Returns:**
    - `tuple`: (gv, q, surface_dict) with gv the output of `compute_gv` (all classes lie in the face).
    """
    if q is None: q,rows=adapted_glsm_uplift(F,return_rows=True); d7_rows=rows["d7"]
    elif d7_rows is None: raise ValueError("d7_rows is required if q is given")
    q=np.asarray(q,dtype=int)
    MC=mori_cone_in_glsm_basis(q,F.M_conv_toric_fan())
    G=np.rint(np.asarray(MC.hilbert_basis() if hilbert_basis else MC.rays(),dtype=float)).astype(int)
    face,_=d7_mori_face(G,d7_rows)
    if not face.any() or face.all(): raise ValueError("The D7 locus t_mu=0 is not a proper face of the Kahler cone: "+("no curve shrinks." if not face.any() else "it is the origin."))
    Gf=G[face]; r=np.linalg.matrix_rank(Gf)
    if r!=len(d7_rows) or np.delete(Gf,d7_rows,axis=1).any():
        warnings.warn(f"The shrinking face has rank {r} for {len(d7_rows)} D7 rows and/or contains curves with non-D7 components; they shrink on the D7 face too.")
    IN_surfaces,surface_dict=intersection_number_surfaces(tensor_symm_in_basis(F.intersection_numbers_M(),F.basis_homology_M(),q,in_prime_basis=True))
    gv=compute_gv(generators=Gf,grading_vector=MC.find_grading_vector(),q=q,intnums=IN_surfaces,max_deg=max_deg,min_points=min_points,nefpart=F.nef_partition_M(zero_based=True,toric_fan_adapted=True))
    return gv,q,surface_dict

def minimal_mori_face(G: ArrayLike,zero_rows: Sequence[int])->np.ndarray:
    """
    **Description:**
    Minimal face of the Mori cone MC = cone(G) containing MC cap L, L = {d : d_a = 0 for a in zero_rows}. A generator g_i lies in it iff it appears with positive coefficient in some
    lambda >= 0 with (sum_j lambda_j g_j)_a = 0 for all a in zero_rows, since faces are extremal and a point of relint(MC cap L) has the minimal face as its own face. The maximal such support
    is found by the LP
        max sum_i s_i   s.t.   sum_j lambda_j G_ja = 0 (a in zero_rows),   s <= lambda,   lambda >= 0,   0 <= s <= 1,
    whose optimum has s_i = 1 exactly on the support. No assumption is made on whether the unit vectors of zero_rows are generators of MC.
    **Arguments:**
    - `G (ArrayLike)`: Generators (rays or Hilbert basis) of the Mori cone in the GLSM basis, one per row.
    - `zero_rows (Sequence[int])`: 0-based GLSM rows on which the classes of interest have degree zero.
    **Returns:**
    - `np.ndarray`: Boolean mask of the generators lying in the face.
    """
    G=np.asarray(G,dtype=float); n=len(G); z=list(zero_rows)
    A_eq=np.hstack([G[:,z].T,np.zeros((len(z),n))])
    res=linprog(np.r_[np.zeros(n),-np.ones(n)],A_ub=np.hstack([-np.eye(n),np.eye(n)]),b_ub=np.zeros(n),A_eq=A_eq,b_eq=np.zeros(len(z)),
                bounds=[(0,None)]*n+[(0,1)]*n,method="highs")
    if res.status: raise RuntimeError(res.message)
    return res.x[n:]>0.5

def period_sector(q: ArrayLike,nefpart: Sequence[Sequence[int]],d: ArrayLike)->int:
    """
    **Description:**
    Order of vanishing at rho = 0 of the I-function coefficient c_d(rho) = prod_j Gamma(1+L_j.(d+rho))/Gamma(1+L_j.rho) prod_I Gamma(1+D_I.rho)/Gamma(1+D_I.(d+rho))
    of the curve class d, i.e. the lowest cohomological sector H^{2m} of the periods to which d contributes: m = #{I : D_I.d < 0} - #{j : L_j.d < 0}, with D_I.d = (q^T d)_I
    and L_j = sum of the columns of nef part j. m = 0: fundamental period, m = 1: mirror map, m = 2: genus-zero (GV) sector, m >= 3: sectors fixed by the lower ones.
    **Arguments:**
    - `q (ArrayLike)`: GLSM charge matrix.
    - `nefpart (Sequence[Sequence[int]])`: 0-based column indices of the nef partition parts.
    - `d (ArrayLike)`: Curve class in the basis dual to the rows of `q`.
    **Returns:**
    - `int`: The sector m.
    """
    c=np.asarray(q,dtype=int).T@np.asarray(d,dtype=int)
    return int((c<0).sum())-sum(int(c[list(part)].sum()<0) for part in nefpart)

def compute_d7_gv_invariants_weak_coupling(F,q=None,zero_rows=None,directions=None,max_deg=5,min_points=None,face_hilbert_basis=True,strict=True,return_diagnostics=False):
    """
    **Description:**
    Genus-zero GV invariants that survive the limit t = t0 + s v, s -> infinity, with v a generic point of the cone Gamma spanned by the rows of `directions` (supported on
    `zero_rows`): large complex structure of the base and weak coupling (e^{2 pi i tau} -> 0) at finite, generic D7 moduli, in the phase of `F.M_conv_toric_fan()`.
    A term N_d Li_2(q^d) scales as exp(-2 pi s d.v): it survives iff d.r = 0 for every generator r of Gamma (for Gamma of full rank in the zero_rows coordinates this is
    d_a = 0 for all a in zero_rows), is suppressed iff d.r >= 0, and diverges if some d.r < 0. The function
      1. finds the minimal Mori face containing {d : d_a = 0 for a in zero_rows} (`minimal_mori_face` on the Mori rays), generates it by the Hilbert basis of this
         low-dimensional face (so that cygv reaches all its lattice points), enumerates its GV invariants (exact, since a face is closed under effective splitting) and keeps
         the classes with d.r = 0 for all r;
      2. runs cheap validity tests of the limit in this phase: a class with d.r < 0 for some r must not enter the periods. Tested are the Mori rays with d.r < 0 (the
         obstructing generators) and the enumerated face classes: `period_sector` m <= 1 (fundamental period or mirror map; the flat D7 coordinates then diverge) is fatal,
         m = 2 is fatal iff the GV invariant is nonzero (cygv `target_points` for the rays), m >= 3 is harmless. If no ray obstructs, Gamma lies in the ambient Kahler cone and
         the limit is valid unconditionally; otherwise it is only tested on these classes, which is reported in a warning.
    Warns if the Sen scaling vector of `q` (`check_sen_scaling`) has components outside zero_rows (the weak-coupling limit then moves D7 or fiber moduli), and raises if the
    rows of `q` do not span the saturated lattice (cygv then misses curve classes; e.g. `fix_tau_row` with k > 1).
    **Arguments:**
    - `F (F_Theory_Uplift)`: The F-theory uplift.
    - `q (ArrayLike|None)`: GLSM charge matrix with the row structure [bulk | fiber-only | D7 | tau] of `adapted_glsm_uplift` (e.g. from `symmetric_glsm_uplift`). Defaults to `adapted_glsm_uplift(F)`.
    - `zero_rows (Sequence[int]|None)`: 0-based rows sent to infinity. Defaults to the bulk rows and the tau row of `adapted_glsm_uplift(F)`.
    - `directions (ArrayLike|str|None)`: Generators of Gamma, one per row, in the Kahler-parameter basis dual to the rows of `q`, supported on `zero_rows`. `None`: the unit
      vectors of `zero_rows` (every modulus independently). `"nef"`: the extremal rays of the projection of the ambient Kahler cone onto the non-tau zero_rows (the nef cone of
      the pure-bulk curves, so that the base LCS limit is taken inside it) together with the unit vector of the tau row. Defaults to `None`.
    - `max_deg (int|None)`: Maximal degree w.r.t. the grading vector of the Mori cone for the face enumeration (takes precedence over `min_points` in cygv). Defaults to `5`.
    - `min_points (int|None)`: Minimal number of points for the face enumeration, used if `max_deg` is `None`.
    - `face_hilbert_basis (bool)`: Generate the face by its Hilbert basis (cheap, the face is low-dimensional) instead of its rays. Defaults to `True`.
    - `strict (bool)`: Raise `ValueError` if a validity test fails; otherwise warn and return the restricted invariants anyway. Defaults to `True`.
    - `return_diagnostics (bool)`: Also return a dict with keys `directions` (R), `ambient_nef` (bool), `obstructing_generators` (Mori rays g with some g.r < 0, with their
      sector), `face_generators`, `gv_face` (unrestricted face invariants), `bad` (classes that invalidate the limit, with sector and GV invariants), `sen_scaling`. Defaults to `False`.
    **Returns:**
    - `tuple`: (gv, q, surface_dict) with gv the restricted output of `compute_gv`; with `return_diagnostics`, (gv, q, surface_dict, diagnostics).
    """
    Q0,rows=adapted_glsm_uplift(F,return_rows=True)
    q=Q0 if q is None else np.asarray(q,dtype=int)
    if (idx:=abs(int(np.prod(np.diag(snf(q))[:len(q)]))))!=1: raise ValueError(f"The rows of q span a sublattice of index {idx}; use a basis of the saturated lattice for GV computations.")
    z=[int(a) for a in ([*rows["bulk"],rows["tau"]] if zero_rows is None else zero_rows)]; nz=[a for a in range(len(q)) if a not in z]
    try:
        _,w=check_sen_scaling(F,q)
        if w[nz].any(): warnings.warn(f"The Sen scaling vector {w.tolist()} has components outside zero_rows: the weak-coupling limit moves D7/fiber moduli; fix the tau row first (fix_tau_row).")
    except AssertionError: w=None
    fan=F.M_conv_toric_fan(); nefpart=F.nef_partition_M(zero_based=True,toric_fan_adapted=True)
    MC=mori_cone_in_glsm_basis(q,fan); gvec=MC.find_grading_vector()
    G=np.rint(np.asarray(MC.rays(),dtype=float)).astype(int)
    # limit-direction cone Gamma, generated by the rows of R
    if directions is None: R=np.eye(len(q),dtype=int)[z]
    elif isinstance(directions,str) and directions=="nef":
        tau=int(rows["tau"]); zb=[a for a in z if a!=tau]
        Kr=np.rint(np.asarray(MC.dual().rays(),dtype=float)).astype(int)[:,zb]; Kr=Kr[Kr.any(axis=1)]
        R=np.zeros((0,len(q)),dtype=int)
        if len(zb): Kr=np.rint(np.asarray(Cone(Kr).extremal_rays(),dtype=float)).astype(int); R=np.zeros((len(Kr),len(q)),dtype=int); R[:,zb]=Kr
        if tau in z: R=np.vstack([R,np.eye(len(q),dtype=int)[[tau]]])
    else:
        R=np.asarray(directions,dtype=int)
        if R[:,nz].any(): raise ValueError("directions must be supported on zero_rows.")
    face=minimal_mori_face(G,z)
    if not face.any(): raise ValueError("No effective class has zero degree along zero_rows.")
    Gf=np.rint(np.asarray(Cone(G[face]).hilbert_basis() if face_hilbert_basis else G[face],dtype=float)).astype(int)
    IN_surfaces,surface_dict=intersection_number_surfaces(tensor_symm_in_basis(F.intersection_numbers_M(),F.basis_homology_M(),q,in_prime_basis=True))
    gv_face=compute_gv(generators=Gf,grading_vector=gvec,q=q,intnums=IN_surfaces,max_deg=max_deg,min_points=min_points,nefpart=nefpart)
    gv=[e for e in gv_face if not (R@np.asarray(e[0][0])).any()]
    # validity of the limit on the obstructing rays and the face classes: negative volume growth must not enter the periods
    obstructing={tuple(int(x) for x in g):period_sector(q,nefpart,g) for g in G if (R@g<0).any()}; ambient_nef=not obstructing; bad={}
    if not ambient_nef:
        bad={d:(m,None) for d,m in obstructing.items() if m<=1}
        targets=[d for d,m in obstructing.items() if m==2]
        gv_t=list(compute_gv(generators=G,grading_vector=gvec,q=q,intnums=IN_surfaces,nefpart=nefpart,target_points=np.array(targets))) if targets else []
        for d in targets:
            if inv:=[e for e in gv_t if e[0][0]==d and e[1]!=0]: bad[d]=(2,inv)
        for e in gv_face:
            if e[1]!=0 and (R@np.asarray(e[0][0])<0).any(): bad.setdefault(e[0][0],(period_sector(q,nefpart,e[0][0]),[]))[1].append(e)
        if bad:
            msg=("The limit along the given directions leaves the convergence region of this phase: classes with negative volume growth enter the periods: "+
                 ", ".join(f"{d} (sector {m}{'' if inv is None else ', GV '+str(inv)})" for d,(m,inv) in bad.items())+".")
            if strict: raise ValueError(msg)
            warnings.warn(msg)
        else: warnings.warn(f"The directions {R.tolist()} are not nef on the ambient toric variety (obstructing rays {list(obstructing)}); the limit was verified to be valid only on these rays and on the face classes up to degree {max_deg}.")
    if not return_diagnostics: return gv,q,surface_dict
    diag={"directions":R,"ambient_nef":ambient_nef,"obstructing_generators":obstructing,"face_generators":Gf,"gv_face":gv_face,"bad":bad,"sen_scaling":w}
    return gv,q,surface_dict,diag
