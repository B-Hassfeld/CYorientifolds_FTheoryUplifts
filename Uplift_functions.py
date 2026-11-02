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
from fractions import Fraction
from functools import cache
from math import comb, factorial
import warnings
from typing import TYPE_CHECKING, Any, Literal
import math
# 3rd party imports
import numpy as np
import shutil
import tempfile
import subprocess
import re
from pathlib import Path
from flint import fmpz, fmpz_mat
from scipy.linalg import schur, qr
import scipy.sparse as sps
# from scipy.optimize import linprog
from scipy.spatial import ConvexHull
# CYTools imports
from cytools import Polytope, h_polytope, Cone
from cytools.utils import lll_reduce
from cytools.vector_config.fan import Fan
from cytools.vector_config import VectorConfiguration
from cygv import compute_gv
from numpy.typing import ArrayLike
from .Utility_functions.matrix_utilities import *
from .Utility_functions.fan_utilities import *
from .Utility_functions.toric_geometry_utilities import *
if TYPE_CHECKING:
    from .FT_CY import CY_orientifold, F_Theory_Uplift

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
# def compute_partition(divisors: Sequence[np.ndarray],rays: np.ndarray)->tuple[bool,np.ndarray|None]:
#     """
#     **Description:**
#     Uses linear equivalence to shift a collection of toric divisors into a representation whose coefficients form a partition of the anticanonical divisor, if such a representation exists. The shift is found by solving a mixed-integer linear program; the current implementation supports exactly two divisors.
#     **Arguments:**
#     - `divisors (Sequence[np.ndarray])`: The divisor coefficient vectors (one integer array per divisor, in the toric prime divisor basis).
#     - `rays (np.ndarray)`: The primitive ray generators of the toric fan, one per row.
#     **Returns:**
#     - `tuple[bool, np.ndarray | None]`: A pair `(exists, shifted_divisors)`, where `exists` is a boolean and `shifted_divisors` is the integer array of shifted divisors (one per row) if it exists, otherwise `None`.
#     """
#     null=np.zeros([rays.shape[0],rays.shape[1]],dtype=int)
#     linear1=np.tensordot(np.identity(2,dtype=int),rays,axes=0)
#     linear1=linear1.transpose(0,2,1,3).reshape(len(divisors)*rays.shape[0],len(divisors)*rays.shape[1])
#     linear2=np.hstack([rays]*len(divisors))
#     affine1=np.concatenate(divisors)
#     affine2=sum(divisors)-1
#     lower1=-affine1
#     upper1=np.full_like(affine1,np.inf,dtype=float)
#     lower2=-affine2
#     upper2=-affine2
#     linear=np.vstack([linear1,linear2])
#     lower=np.concatenate([lower1,lower2])
#     upper=np.concatenate([upper1,upper2])
#     c=np.zeros(linear.shape[1])
#     integrality=np.ones_like(c)
#     constraints=LinearConstraint(linear,lower,upper)
#     res=milp(c=np.zeros(linear.shape[1]),constraints=constraints,integrality=np.ones(linear.shape[1]),bounds=(-np.inf,np.inf))
#     if not res.success or res.x is None:
#         return (False,None)
#     sol=np.rint(res.x.reshape(len(divisors),len(rays[0]))@(rays.T)+np.array(divisors)).astype(int)
#     return (True,sol)


###Make lowercase Newton_Polytope->newton_polytope

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

# def solve_over_integers(M: ArrayLike,b: ArrayLike)->tuple[bool,np.ndarray|None]:
#     """
#     **Description:**
#     Uses Smith normal form to determine whether the equation `Mx=b` has an integral solution, and returns one if it exists.
#     **Arguments:**
#     - `M (ArrayLike)`: Integer matrix.
#     - `b (ArrayLike)`: Integer inhomogeneous term.
#     **Returns:**
#     - `tuple[bool,np.ndarray|None]`: A pair `(has_solution, x)`, where `x` is an integral solution if one exists, otherwise `None`.
#     """
#     a,s,t=snf(M,return_decomposition=True)
#     c=s@np.asarray(b,dtype=int)
#     y=np.zeros(a.shape[1],dtype=int)
#     for ii in range(np.min(a.shape)):
#         if a[ii,ii]!=0:
#             if c[ii]%a[ii,ii]!=0:
#                 return (False,None)
#             y[ii]=c[ii]//a[ii,ii]
#     if np.all(a@y==c):
#         return (True,t@y)
#     else:
#         return (False,None)

# def refine_fan_old(fan,blowups_or_all_vectors=None,tol=1e-10):
#     """
#     **Description:**
#     Refines a fan by successive stellar subdivisions at rays that are vectors of its configuration but not used by any cone.
#     If `blowups_or_all_vectors` is given, its vectors that are not yet in the configuration are appended first (vectors already
#     present are ignored, so either the new rays alone or the full list of vectors may be passed). Each unused ray is then
#     located in the fan: the minimal cone containing it is the set of generators with positive coefficient in its (unique, for
#     simplicial fans) expansion, and every cone containing that face is replaced by the joins of the ray with the facets of the
#     face. For simplicial fans the point location uses the precomputed inverses of the cone matrices (fast path, with the
#     tolerance `tol`); for non-simplicial fans `find_cone` (full-dimensional) or `find_cone_general` is used. Fans that do not
#     span the ambient space are handled in coordinates of their linear span.
#     **Arguments:**
#     - `fan (Fan)`: The fan to refine. Its unused vectors are all inserted.
#     - `blowups_or_all_vectors (ArrayLike|None)`: Vectors to add as rays before refining, one per row. Defaults to `None`.
#     - `tol (float)`: Tolerance for the containment test and for the positivity of the expansion coefficients. Defaults to `1e-10`.
#     **Raises:**
#     - `ValueError`: Raised if a new ray does not lie in the linear span of the fan or is not contained in its support.
#     **Returns:**
#     - `Fan`: The refined fan over the enlarged vector configuration, every vector of which is now a ray. If there is nothing to
#       insert, the fan over the (possibly enlarged) configuration is returned unchanged.
#     """
#     if blowups_or_all_vectors is not None:
#         blowups=row_difference(blowups_or_all_vectors,fan.vc.vectors())
#         all_vectors=np.concatenate((fan.vc.vectors(),blowups),axis=0)
#         vc=VectorConfiguration(all_vectors)
#         new_fan=Fan(vc,cones=fan.cones())
#     else:
#         new_fan=fan
#         vc=fan.vc
#     vec_diff=row_difference(vc.vectors(),new_fan.vectors())
#     to_be_refined=set(vc.vectors_to_labels(vec_diff))
#     if len(to_be_refined)==0:
#         return new_fan
#     vectors=vc.vectors()
#     dim=fan.dim
#     full_dim=dim==vectors.shape[1]
#     if vc.dim!=dim:
#         raise ValueError("The new rays do not lie in the span of the fan.")
#     cones=set(new_fan.cones())
#     l2c={}
#     for c in cones:
#         for l in c:
#             l2c.setdefault(l,set()).add(c)
#     fast=all(len(c)==dim for c in cones)
#     if fast:
#         coords=vectors.astype(float) if full_dim else vectors@np.linalg.svd(vectors.astype(float))[2][:dim].T
#         cone_list=list(cones)
#         cone_idx={c:i for i,c in enumerate(cone_list)}
#         try:
#             inv=np.linalg.inv(coords[np.array(cone_list)-1].transpose(0,2,1))
#         except np.linalg.LinAlgError:
#             fast=False
#         alive=np.ones(len(cone_list),dtype=bool)
#     for label in to_be_refined:
#         ray=vc.vectors(label)
#         link_base=None
#         if fast:
#             alive_idx=np.flatnonzero(alive)
#             hit=np.flatnonzero(np.all(inv[alive_idx]@coords[label-1]>=-tol,axis=1))
#             if len(hit)>0:
#                 c=cone_list[alive_idx[hit[0]]]
#                 coeffs=np.linalg.solve(coords[np.array(c)-1].T,coords[label-1])
#                 link_base=tuple(l for l,x in zip(c,coeffs) if x>tol)
#         elif full_dim:
#             link_base=find_cone(ray,cones,vectors)
#         else:
#             link_base=find_cone_general(ray,cones,vectors)
#         if link_base is None:
#             raise ValueError(f"The ray {ray} is not contained in the support of the fan.")
#         link_base=tuple(link_base)
#         star=set.intersection(*(l2c[l] for l in link_base))
#         new_cones=[]
#         for c_old in star:
#             cones.discard(c_old)
#             for l in c_old:
#                 l2c[l].discard(c_old)
#             rest=tuple(l for l in c_old if l not in link_base)
#             for comb in combinations(link_base,len(link_base)-1):
#                 new_cones.append(tuple(sorted(comb+rest+(label,))))
#         for c_new in new_cones:
#             cones.add(c_new)
#             for l in c_new:
#                 l2c.setdefault(l,set()).add(c_new)
#         if fast:
#             alive[[cone_idx.pop(c_old) for c_old in star]]=False
#             for c_new in new_cones:
#                 cone_idx[c_new]=len(cone_list)
#                 cone_list.append(c_new)
#             try:
#                 inv=np.concatenate((inv,np.linalg.inv(coords[np.array(new_cones)-1].transpose(0,2,1))))
#                 alive=np.concatenate((alive,np.ones(len(new_cones),dtype=bool)))
#             except np.linalg.LinAlgError:
#                 fast=False
#     return Fan(vc=vc,cones=cones)


# def find_cone_general(new_ray: ArrayLike,current_cones: Iterable[tuple[int,...]],all_vectors: np.ndarray)->frozenset[int]|None:
#     """
#     **Description:**
#     Searches the current cones (of any dimension, in any ambient space) for the first one containing `new_ray`, and returns the labels of its generators with strictly positive coefficient. Cones with linearly independent generators are tested all at once by a batched least-squares solve, cones with linearly dependent generators by non-negative least squares.
#     For a simplicial cone the coefficients are unique, so the result is the carrier face of `new_ray`. For a non-simplicial cone the coefficients are not unique and the returned set can be a proper subset of the carrier face (e.g. a diagonal of a square face), which is then not a face of the fan.
#     **Arguments:**
#     - `new_ray (ArrayLike)`: The ray to locate.
#     - `current_cones (Iterable[tuple[int,...]])`: Current cones, represented by tuples of one-indexed ray labels.
#     - `all_vectors (np.ndarray)`: Full ray matrix.
#     **Returns:**
#     - `frozenset[int]|None`: The carrier face labels, or `None` if no carrier cone is found.
#     """

#     def nnls_face(cone):
#         cone_list=list(cone)
#         A=all_vectors[[idx-1 for idx in cone_list]].T
#         x, residual=nnls(A,new_ray)
#         if residual<1e-10:
#             return frozenset(cone_list[i] for i,coeff in enumerate(x) if coeff>1e-10)
#         return None
#     cones=list(current_cones)
#     for cone in cones[:20]:
#         face=nnls_face(cone)
#         if face is not None:
#             return face
#     cones=cones[20:]
#     if len(cones)==0:
#         return None
#     b=np.asarray(new_ray,dtype=float)
#     lengths=np.fromiter(map(len,cones),dtype=int,count=len(cones))
#     independent=np.zeros(len(cones),dtype=bool)
#     hit=np.zeros(len(cones),dtype=bool)
#     coeffs={}
#     for k in np.unique(lengths):
#         if k>len(b):
#             continue
#         idx=np.flatnonzero(lengths==k)
#         C=np.array(cones) if len(idx)==len(cones) else np.array([cones[i] for i in idx])
#         A=all_vectors[C-1].transpose(0,2,1).astype(float)
#         G=A.transpose(0,2,1)@A
#         ind=np.linalg.det(G)>1e-9*np.prod(np.diagonal(G,axis1=1,axis2=2),axis=1)
#         if not np.any(ind):
#             continue
#         idx,A,G=idx[ind],A[ind],G[ind]
#         independent[idx]=True
#         x=np.linalg.solve(G,(b@A)[...,None])[...,0]
#         r=b-np.einsum('nij,nj->ni',A,x)
#         cand=np.flatnonzero((np.linalg.norm(r,axis=1)<1e-6)&np.all(x>=-1e-6,axis=1))
#         if len(cand)==0:
#             continue
#         A,G,x=A[cand],G[cand],x[cand]+np.linalg.solve(G[cand],np.einsum('nij,ni->nj',A[cand],r[cand])[...,None])[...,0]
#         residual=np.linalg.norm(np.einsum('nij,nj->ni',A,x)-b,axis=1)
#         ok=(residual<1e-10)&np.all(x>=-1e-10,axis=1)
#         hit[idx[cand[ok]]]=True
#         coeffs.update(zip(idx[cand[ok]],x[ok]))
#     for i in np.flatnonzero(hit|~independent):
#         if independent[i]:
#             return frozenset(l for l,coeff in zip(cones[i],coeffs[i]) if coeff>1e-10)
#         face=nnls_face(cones[i])
#         if face is not None:
#             return face
#     return None

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

# def chi_hypersurface(V: ArrayLike,cones_by_dim: list[np.ndarray],divisor: ArrayLike)->int:
#     """
#     **Description:**
#     Euler characteristic of the generic hypersurface in the class `divisor` of a complete simplicial toric variety, as the
#     sum of the Kouchnirenko Euler characteristics of its intersections with all torus orbits (Danilov-Khovanskii).
#     **Arguments:**
#     - `V (ArrayLike)`: The rays of the fan, one per row.
#     - `cones_by_dim (list[np.ndarray])`: All cones of the fan of dimension 1,...,dim, as returned by `_cones_by_dimension`.
#     - `divisor (ArrayLike)`: Divisor coefficients in the basis of the rays.
#     **Returns:**
#     - `int`: The Euler characteristic.
#     """
#     V=np.asarray(V,dtype=int)
#     _,W,E=_newton_data(V,np.asarray(divisor,dtype=int))
#     cache={}
#     return int(sum(_orbit_pieces(V,cones,W,E,cache)[0].sum() for cones in [np.zeros((1,0),dtype=int)]+cones_by_dim))

# def find_O7_planes(O: CY_orientifold|F_Theory_Uplift)->np.ndarray:
#     """
#     **Description:**
#     Finds the divisorial O7-planes from the orbifold rays alone (no fan): the rays of `vectors_orbifold()` that are odd
#     under `orientifold_mod2_character` (their divisors are pointwise fixed) and whose divisor meets the orientifold B,
#     i.e. at least two monomials of the orientifold line bundle survive on it. Non-divisorial O7-planes (fixed 2-cones
#     whose stratum lies in B) need the fan, see `orientifold_fixed_locus`.
#     **Arguments:**
#     - `O (CY_orientifold|F_Theory_Uplift)`: The Calabi-Yau orientifold or an F-theory uplift.
#     **Raises:**
#     - `ValueError`: Raised if a pointwise fixed divisor is contained in B (B reducible).
#     **Returns:**
#     - `np.ndarray`: Sorted one-indexed labels of `vectors_orbifold()`.
#     """
#     O=O.orientifold() if hasattr(O,"orientifold") else O
#     if O.line_bundle() is None:
#         raise ValueError("The orientifold is not regular (no projected-in monomial), so B and its O7-planes are not defined.")
#     V=np.asarray(O.vectors_orbifold(),dtype=int)
#     _,_,E=_newton_data(V,np.asarray(O.line_bundle(),dtype=int))
#     n_face=E.sum(axis=1)
#     odd=(V@orientifold_mod2_character(O.xi()))%2==1
#     if np.any(odd&(n_face==0)):
#         raise ValueError(f"The fixed divisors {np.flatnonzero(odd&(n_face==0))+1} are contained in B.")
#     return np.flatnonzero(odd&(n_face>=2))+1

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

# def chi_I(O: CY_orientifold|F_Theory_Uplift)->int:
#     """
#     **Description:**
#     Euler characteristic of the fixed locus of the orientifold involution, chi_O7 plus the number of O3-planes
#     (see `orientifold_fixed_locus`).
#     **Arguments:**
#     - `O (CY_orientifold|F_Theory_Uplift)`: The Calabi-Yau orientifold or F-theory uplift.
#     **Returns:**
#     - `int`: The Euler characteristic of the fixed locus.
#     """
#     return _fixed_locus(O)["chi_I"]

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

# def hodge_diamond_uplift(F: F_Theory_Uplift,backend: Literal["auto","normaliz","latte","cytools"]="auto")->np.ndarray:
#     """
#     **Description:**
#     Hodge diamond of the Calabi-Yau of an F_Theory_Uplift, computed from the Cayley
#     polytope ``F.Cayley_M()`` of its Newton polytopes.
#     **Arguments:**
#     - `F (F_Theory_Uplift)`: The F-theory uplift; its Cayley polytope `F.Cayley_M()` is passed to `hodge_diamond_stringy`.
#     - `backend (Literal["auto", "normaliz", "latte", "cytools"])`: Lattice-point backend, see `hodge_diamond_stringy`. Defaults to `"auto"`.
#     **Returns:**
#     - `np.ndarray`: The integer Hodge diamond, h[p, q] = h^{p,q}.
#     """
#     return hodge_diamond_stringy(F.Cayley_M(),backend)

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

# def is_partition(points: ArrayLike,divisors: ArrayLike,atol: float=1e-8)->tuple[bool,bool,np.ndarray]:
#     """
#     **Description:**
#     Determine whether divisors L1,...,Lr can be shifted by principal
#     divisors to form a partition of the anticanonical divisor.
#     This version avoids repeated np.linalg.solve calls by computing
#     the inverse of the basis matrix only once.
#     **Arguments:**
#     - `points (ArrayLike)`: The ray generators of the toric fan, one per row.
#     - `divisors (ArrayLike)`: The coefficient vectors of L1,...,Lr in the prime toric divisor basis, one per row.
#     - `atol (float)`: Numerical tolerance for the integrality checks. Defaults to `1e-8`.
#     **Returns:**
#     - `tuple[bool,bool,np.ndarray]`: `(is_partition, sums_to_anticanonical, shifts)`, where `shifts` is the integer (r, dim) array of shift vectors m_a such that `divisors[a] + points@shifts[a]` form the partition (if no partition exists, all shifts are zero except possibly the last row).
#     """
#     points=np.asarray(points,dtype=int)
#     divisors=np.asarray(divisors,dtype=int)
#     r, n_rays=divisors.shape
#     dim=points.shape[1]
#     sta, total_shift=sums_to_anticanonical(points,divisors,atol=atol)
#     if not sta:
#         return False, False, np.zeros((r,dim),dtype=int)
#     basis_indices=np.asarray(basis(points),dtype=int)
#     B=points[basis_indices].astype(float)
#     try:
#         B_inv=np.linalg.inv(B)
#     except np.linalg.LinAlgError:
#         return False, True, np.zeros((r,dim),dtype=int)
#     divisors_B=divisors[:,basis_indices]
#     for assignment in product(range(r),repeat=dim):
#         assignment=np.asarray(assignment)
#         shifts=np.zeros((r,dim),dtype=int)
#         valid=True
#         for a in range(r-1):
#             target=(assignment==a).astype(float)
#             rhs=target-divisors_B[a]
#             m_float=B_inv@rhs
#             m=np.rint(m_float).astype(int)
#             if not np.allclose(m_float,m,atol=atol,rtol=0):
#                 valid=False
#                 break
#             shifts[a]=m
#         if not valid:
#             continue
#         shifts[-1]=(total_shift-np.sum(shifts[:-1],axis=0))
#         shifted=divisors+shifts@points.T
#         if not np.all((shifted==0)|(shifted==1)):
#             continue
#         if not np.all(np.sum(shifted,axis=0)==1):
#             continue
#         return True, True, shifts
#     shifts=np.zeros((r,dim),dtype=int)
#     shifts[-1]=total_shift
#     return False, True, shifts

# def chi_fourfold_uplift(F: F_Theory_Uplift)->int:
#     """
#     **Description:**
#     Euler characteristic of the Calabi-Yau fourfold of an F-theory uplift: the toric localization `cicy4_chern_toric_uplift(F,4)` plus 6 times the number of O3-planes `chi_O3`.
#     **Arguments:**
#     - `F (F_Theory_Uplift)`: The F-theory uplift.
#     **Returns:**
#     - `int`: The Euler characteristic.
#     """
#     chi_loc=toric_chern_class_uplift(F,4)
#     n_O3=chi_O3(F)
#     return chi_loc+n_O3*6

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

# def divisor_in_zero_one_rep_numpy(points: ArrayLike,a: ArrayLike,atol: float=1e-8)->tuple[np.ndarray,np.ndarray]|tuple[None,None]:
#     """
#     **Description:**
#     Shift divisor a by a principal divisor m (b = a + points@m) so that b has
#     coefficients in {0,1}. Returns (b,m) or (None,None) if impossible.
#     **Arguments:**
#     - `points (ArrayLike)`: The ray generators of the toric fan, one per row.
#     - `a (ArrayLike)`: The integer divisor coefficients, one per ray.
#     - `atol (float)`: Tolerance for the integrality of the shift. Defaults to `1e-8`.
#     **Returns:**
#     - `tuple[np.ndarray,np.ndarray]|tuple[None,None]`: `(b, m)` with integer arrays b (coefficients in {0,1}) and m, or `(None, None)`.
#     """
#     points=np.asarray(points,dtype=int)
#     a=np.asarray(a,dtype=int)
#     dim=points.shape[1]
#     if np.all((a==0)|(a==1)):
#         return a.copy(),np.zeros(dim,dtype=int)
#     S=np.asarray(basis(points),dtype=int)
#     B_inv=np.linalg.inv(points[S].astype(float))
#     T=(np.arange(2**dim)[:,None]>>np.arange(dim))&1
#     M=(T-a[S])@B_inv.T
#     Mi=np.rint(M).astype(int)
#     b=a+Mi@points.T
#     ok=np.all(np.abs(M-Mi)<atol,axis=1)&np.all((b==0)|(b==1),axis=1)
#     i=np.flatnonzero(ok)
#     return (b[i[0]],Mi[i[0]]) if len(i) else (None,None)

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
    # return b,m


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





# def symmetric_glsm_uplift(F: F_Theory_Uplift,tol: float=1e-6)->dict[str,Any]:
#     """
#     **Description:**
#     Applies `symmetric_d7_basis` to the adapted GLSM of an F-theory uplift. The D7 rows are the rows of `adapted_glsm_uplift(F)` between the bulk rows and the final tau row (one per special fiber point other than b0).
#     **Arguments:**
#     - `F (F_Theory_Uplift)`: The F-theory uplift.
#     - `tol (float)`: Tolerance for comparing intersection numbers. Defaults to `1e-6`.
#     **Returns:**
#     - `dict[str,Any]`: The output of `symmetric_d7_basis`, with the additional key `d7_rows`.
#     """
#     Q,rows=adapted_glsm_uplift(F,return_rows=True)
#     d7=rows["d7"]
#     IN=_tensor_symm_in_glsm_basis(F.intersection_numbers_M(),Q,F.basis_homology_M())
#     res=symmetric_d7_basis(Q,IN,d7,tol=tol)
#     return (res["Q"],res["intersection_numbers"],res["index"])

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

def nullspace_mod2(M):
    """
    **Description:**
    Basis of the left kernel of an integer matrix over the field with two elements, i.e. of the vectors y in {0,1}^r with
    y @ M = 0 mod 2, by Gaussian elimination on the augmented matrix [M mod 2 | 1].
    **Arguments:**
    - `M (ArrayLike)`: Integer matrix of shape (r,n).
    **Returns:**
    - `np.ndarray`: 0/1 integer matrix whose rows form a basis of the left kernel modulo two (shape (r - rank_2(M), r)).
    """
    A=np.asarray(M)%2; r,n=A.shape; E=np.hstack([A,np.eye(r,dtype=int)]); piv=0
    for c in range(n):
        rows=np.flatnonzero(E[piv:,c])+piv
        if not len(rows): continue
        E[[piv,rows[0]]]=E[[rows[0],piv]]
        for i in np.flatnonzero(E[:,c]):
            if i!=piv: E[i]^=E[piv]
        piv+=1
        if piv==r: break
    return E[piv:,n:]

# def solve_integer(M, b):
#     """
#     **Description:**
#     Integer solution x of the linear system M @ x = b for a matrix with linearly independent columns: the (unique) real
#     least-squares solution is rounded and verified exactly in arbitrary-precision integer arithmetic.
#     **Arguments:**
#     - `M (ArrayLike)`: Integer matrix with linearly independent columns.
#     - `b (ArrayLike)`: Integer right-hand side.
#     **Raises:**
#     - `ValueError`: Raised if the columns of `M` are linearly dependent or the rounded solution does not solve the system exactly
#       (no integer solution exists).
#     **Returns:**
#     - `np.ndarray`: The integer solution x.
#     """
#     M, b = np.asarray(M), np.asarray(b)
#     x, _, rank, _ = np.linalg.lstsq(M, b, rcond=None)
#     if rank < M.shape[1]: raise ValueError("M must have linearly independent columns.")
#     x = np.rint(x).astype(np.int64)
#     if not np.array_equal(M.astype(object) @ x, b.astype(object)): raise ValueError("Rounded solution failed the exact check.")
#     return x

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
    return (Q,(len(Qb_intersecting),len(Qd7_intersecting),1,len(Qd7_not_intersecting),len(Qb_not_intersecting)))

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
    combination of protected rows alone: the protected rows stay within themselves. For the matrix of `glsm_from_refinement`
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
            new=N[:len(pivots)].astype(object)@Q
            if (new%p).any(): raise RuntimeError("inexact kernel combination in saturate_glsm")
            Q[pivots]=new//p; T[pivots]=(N[:len(pivots)].astype(object)@T)/p
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

def compute_d7_gv(F,Q=None,INQ=None,IN_surfaces=None,hilbert_basis=False,min_points=600,max_deg=2,target_points=None):
    Q,sizes=glsm_from_refinement(F)
    d7_rows=np.arange(sizes[0],sizes[0]+sizes[1])
    q,order_dict=saturate_glsm(Q,protected_rows=d7_rows,return_combinations=True)
    INq=tensor_symm_in_basis(F.intersection_numbers_M(),F.basis_homology_M(),q,in_prime_basis=True)
    MC_face=mori_cone_in_glsm_basis(Q=q,fan=F.M_conv_toric_fan(),sub_rows=d7_rows,as_face_in_higher_dim_ambient=True)
    if hilbert_basis:
        gens=MC_face.hilbert_basis()
    else:
        gens=MC_face.rays()
    gvec_face=MC_face.find_grading_vector()
    IN_surf,surf_dict=intersection_number_surfaces(INq,q.shape[0])
    nef_part_GV=F.nef_partition_M(toric_fan_adapted=True,zero_based=True,as_divisors=False)
    gv=compute_gv(generators=gens,min_points=min_points,grading_vector=gvec_face,q=q,intnums=IN_surf,max_deg=max_deg,nefpart=nef_part_GV,target_points=target_points)
    return gv

##################SYMM D7 BASIS#######################
#######################################################
#######################################################
#######################################################
#######################################################
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


def _dense_quartic(IN,N)->np.ndarray:
    """
    **Description:**
    Dense symmetric (N,N,N,N) array of quartic intersection numbers from a dict with sorted 1-based keys. Keys of length
    other than four are ignored; every entry is copied to all permutations of its indices.
    **Arguments:**
    - `IN (Mapping[tuple[int,...],int|float])`: Intersection numbers with sorted 1-based index tuples as keys.
    - `N (int)`: Number of divisors, i.e. the size of each axis.
    **Returns:**
    - `np.ndarray`: The float array `T` with `T[i,j,k,l] = IN[(i+1,j+1,k+1,l+1)]` for sorted indices.
    """
    _PERMS4=np.array(list(permutations(range(4))))
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
    **Arguments:**
    - `T (np.ndarray)`: Dense symmetric quartic tensor.
    - `tol (float)`: Entries with absolute value at most `tol` are dropped; values within `tol` of an integer are returned as `int`.
    **Returns:**
    - `dict[tuple[int,...],int|float]`: The sparse intersection numbers.
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
    **Arguments:**
    - `F (np.ndarray)`: The symmetric forms, shape (r,d,d).
    - `rtol (float)`: Eigenvalues of G at most `rtol` times the largest one count as zero. Defaults to `1e-12`.
    **Returns:**
    - `np.ndarray`: Orthonormal basis of the centre, shape (c,d,d) with c its dimension.
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
    A random linear combination of the centre of the restricted forms is formed; its eigenvalues are clustered (complex
    conjugates together), the corresponding real invariant subspaces are split off by a sorted real Schur decomposition, and
    each piece is split further in the same way until its centre is trivial or the eigenvalues no longer separate.
    **Arguments:**
    - `U (np.ndarray)`: Orthonormal basis of the subspace to split, shape (n,d).
    - `F (np.ndarray)`: Symmetric forms on the ambient space, shape (r,n,n).
    - `rng (np.random.Generator)`: Random number generator for the generic centre element.
    - `rtol (float)`: Relative threshold for the centre computation, see `_form_centre`. Defaults to `1e-12`.
    - `etol (float)`: Eigenvalues closer than `etol` times the spectral radius are clustered. Defaults to `1e-6`.
    **Returns:**
    - `list[np.ndarray]`: Orthonormal bases of the pieces (shapes (n,d_i) with sum of d_i equal to d), mutually orthogonal for every form in F.
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
    **Arguments:**
    - `C (np.ndarray)`: Orthonormal float matrix of shape (n,m).
    - `d (int)`: Expected rank of the relation lattice, i.e. n minus the rank of C.
    - `scale (float)`: Factor by which C is scaled before rounding to integers. Defaults to `1e10`.
    **Returns:**
    - `np.ndarray | None`: Integer matrix of shape (d,n) whose rows span the relations, or `None` if no such lattice of rank d
      is found (the span of C is then not a rational subspace). The identity is returned if C has no columns.
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
    """
    **Description:**
    Whether the column span of U is a rational subspace, i.e. contains a full-rank integer lattice. This holds iff the
    orthogonal complement of the span admits as many integer relations as the dimension of the span, which is tested with
    `_integer_relations`.
    **Arguments:**
    - `U (np.ndarray)`: Orthonormal basis of the subspace, shape (n,d).
    **Returns:**
    - `bool`: `True` if the subspace is rational.
    """
    d=U.shape[1]
    Z=np.linalg.qr(U,mode='complete')[0][:,d:]
    return _integer_relations(Z,d) is not None

def _rational_blocks(blocks)->list[np.ndarray]:
    """
    **Description:**
    Merges real orthogonal pieces that are not rational subspaces (Galois-conjugate pieces of a splitting over a number field) into the smallest groups whose sums are rational.
    Rational pieces are kept; among the others, groups of increasing size are tried until their sum is rational, and all
    remaining pieces are merged into one block if no proper group is found.
    **Arguments:**
    - `blocks (list[np.ndarray])`: Orthonormal bases of the pieces, e.g. from `_orthogonal_split`.
    **Returns:**
    - `list[np.ndarray]`: Orthonormal bases of the merged, rational pieces.
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
    **Arguments:**
    - `T (np.ndarray)`: Dense symmetric quartic intersection tensor.
    - `rows (Sequence[int])`: 0-based indices of the block divisors.
    - `W (Sequence[int])`: 0-based indices of the remaining divisors to keep (the other blocks and the invariant divisors).
    - `R (np.ndarray|None)`: Integer change of basis of the block, shape (d,d). Defaults to `None` (no change).
    **Returns:**
    - `np.ndarray`: The restricted tensor of shape (d+w,)*4 with the block indices first.
    """
    idx=list(rows)+list(W)
    S=T[np.ix_(idx,idx,idx,idx)]
    if R is not None:
        M=np.eye(len(idx)); M[:len(rows),:len(rows)]=R
        for _ in range(4):
            S=np.tensordot(S,M,axes=([0],[0]))
    return S

def _block_mask(d,w)->np.ndarray:
    """
    **Description:**
    Boolean mask on a tensor of shape (d+w,)*4 (block indices first) selecting the entries with at least one block index,
    i.e. the entries that change under a change of basis of the block.
    **Arguments:**
    - `d (int)`: Size of the block.
    - `w (int)`: Number of remaining indices.
    **Returns:**
    - `np.ndarray`: Boolean array of shape (d+w,)*4.
    """
    a=np.arange(d+w)<d
    return a[:,None,None,None]|a[None,:,None,None]|a[None,None,:,None]|a[None,None,None,:]

def _block_covariants(S,d)->np.ndarray:
    """
    **Description:**
    Covectors of a block (columns of a (d,m) array transforming as u -> R^T u under J'=R^T J), from its dense data S (block indices first): the linear invariants, the cubic terms contracted with the inverse of the first invertible bilinear invariant G, and the images of all of these under B G^{-1} for every bilinear invariant B (including the contracted quartic).
    If no bilinear invariant is invertible, only the linear invariants are returned.
    **Arguments:**
    - `S (np.ndarray)`: Dense intersection data of the block from `_block_tensor`, shape (d+w,)*4.
    - `d (int)`: Size of the block.
    **Returns:**
    - `np.ndarray`: Array of shape (d,m) whose columns are the covectors.
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
    For d=1 only the sign is tried. Every candidate is verified by comparing all entries with at least one block index to tolerance `tol`.
    **Arguments:**
    - `T (np.ndarray)`: Dense symmetric quartic intersection tensor.
    - `rows_k (Sequence[int])`: 0-based divisor indices of the block to transform.
    - `rows_r (Sequence[int])`: 0-based divisor indices of the reference block, of the same size.
    - `W (Sequence[int])`: 0-based indices of the remaining divisors entering the comparison.
    - `tol (float)`: Tolerance for the comparison of the intersection data. Defaults to `1e-6`.
    **Returns:**
    - `np.ndarray | None`: The unimodular integer matrix R of shape (d,d), or `None` if no alignment is found.
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

######################END SYMMETRIC D7#########################