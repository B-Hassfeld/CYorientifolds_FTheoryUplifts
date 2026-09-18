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
# 'standard' imports
from itertools import combinations, combinations_with_replacement, permutations, product
from collections.abc import Iterable, Mapping, Sequence
from collections import defaultdict, Counter
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from math import comb,factorial
from time import perf_counter
from typing import Any, Dict, FrozenSet, List, Optional, Tuple
import numpy as np
import math
# 3rd party imports
import numpy as np
import math
import shutil
import tempfile
import subprocess
import re
from pathlib import Path
from flint import fmpz_mat
from sympy import Matrix, ZZ, lcm, fraction
from sympy.matrices.normalforms import smith_normal_decomp
from scipy.optimize import nnls
from scipy.optimize import milp, LinearConstraint
# CYTools imports
from cytools import Polytope, h_polytope, Cone
from cytools.polytope import poly_v_to_h
from cytools.utils import integral_nullspace, lll_reduce
from cytools.vector_config.fan import Fan
from cytools.vector_config import VectorConfiguration

def compute_partition(divisors,rays):
    """
    **Description:**
    Uses linear equivalence to shift a collection of toric divisors into a representation whose coefficients form a partition of the anticanonical divisor, if such a representation exists.
    **Arguments:**
    - `divisors (array-like)`: The divisor coefficient vectors.
    - `rays (numpy.ndarray)`: The primitive ray generators of the toric fan.
    **Returns:**
    - `tuple`: A pair `(exists, shifted_divisors)`, where `exists` is a boolean and `shifted_divisors` is the shifted divisor array if it exists, otherwise `None`.
    """
    null = np.zeros([rays.shape[0],rays.shape[1]],dtype=int)
    linear1 = np.tensordot(np.identity(2,dtype=int),rays,axes=0)
    linear1 = linear1.transpose(0, 2, 1, 3).reshape(len(divisors)*rays.shape[0], len(divisors)*rays.shape[1])
    linear2 = np.hstack([rays]*len(divisors))
    affine1 = np.concatenate(divisors)
    affine2 = sum(divisors)-1
    lower1 = -affine1
    upper1 = np.full_like(affine1,np.inf,dtype=float)
    lower2 = -affine2
    upper2 = -affine2
    linear = np.vstack([linear1,linear2])
    lower = np.concatenate([lower1,lower2])
    upper = np.concatenate([upper1,upper2])
    c = np.zeros(linear.shape[1])
    integrality = np.ones_like(c)
    constraints = LinearConstraint(linear, lower, upper)
    res = milp(c=np.zeros(linear.shape[1]), constraints=constraints, integrality=np.ones(linear.shape[1]), bounds=(-np.inf, np.inf))
    if not res.success or res.x is None:
        return (False, None)
    sol = np.rint(res.x.reshape(len(divisors),len(rays[0]))@(rays.T)+np.array(divisors)).astype(int)
    return (True,sol)

def contains_row(arr: np.array, target: np.array):
    """
    **Description:**
    Determines whether the one-dimensional array `target` occurs as a row of the two-dimensional array `arr`.
    **Arguments:**
    - `arr (numpy.ndarray)`: The two-dimensional array to search.
    - `target (numpy.ndarray)`: The one-dimensional row to find.
    **Returns:**
    - `bool`: `True` if `target` is a row of `arr`, otherwise `False`.
    """
    return np.any(np.all(arr == target, axis=1))

def contains_rows(arr: np.array,targets: np.array):
    """
    **Description:**
    Determines whether every row of `targets` occurs as a row of `arr`.
    **Arguments:**
    - `arr (numpy.ndarray)`: The two-dimensional array to search.
    - `targets (numpy.ndarray)`: The target rows.
    **Returns:**
    - `bool`: `True` if every target row occurs in `arr`, otherwise `False`.
    """
    return np.any((targets[:, None, :] == arr[None, :, :]).all(axis=2), axis=1).all()

def get_same_rows(A: np.array, B: np.array):
    """
    **Description:**
    Computes the rows of `A` that also occur as rows of `B`.
    **Arguments:**
    - `A (numpy.ndarray)`: First two-dimensional array.
    - `B (numpy.ndarray)`: Second two-dimensional array.
    **Returns:**
    - `numpy.ndarray`: The rows of `A` that also occur in `B`.
    """
    return A[np.where((A[:, None, :] == B[None, :, :]).all(axis=2))[0]]

def same_rows(A, B):
    """
    **Description:**
    Determines whether `A` and `B` contain the same rows with the same multiplicities, independent of row order.
    **Arguments:**
    - `A (numpy.ndarray)`: First two-dimensional array.
    - `B (numpy.ndarray)`: Second two-dimensional array.
    **Returns:**
    - `bool`: `True` if `A` and `B` have the same rows with the same multiplicities, otherwise `False`.
    """
    rowsA, countsA = np.unique(A, axis=0, return_counts=True)
    rowsB, countsB = np.unique(B, axis=0, return_counts=True)
    return np.array_equal(rowsA, rowsB) and np.array_equal(countsA, countsB)

def dual_face_Cayley_polytope(Cdvert: np.array,f):
    """
    **Description:**
    Computes the face of the dual Cayley polytope whose vertices pair trivially with all vertices of the input face `f`.
    **Arguments:**
    - `Cdvert (numpy.ndarray)`: Vertices of the dual Cayley polytope.
    - `f`: A face of the Cayley polytope.
    **Returns:**
    - `Polytope`: The dual face.
    """
    return Polytope(Cdvert[np.all(Cdvert@f.vertices().T == 0, axis=1)])

def h11_2_part(Cay: Polytope,Cayd: Polytope,det=False):
    """
    **Description:**
    Computes the Hodge number `h^{1,1}` of a complete intersection Calabi-Yau described by a two-part nef partition using the associated Cayley polytope and its dual.
    **Arguments:**
    - `Cay (Polytope)`: The Cayley polytope.
    - `Cayd (Polytope)`: The dual Cayley polytope.
    - `det (bool)`: Whether to print intermediate contributions. Defaults to `False`.
    **Returns:**
    - `int`: The Hodge number `h^{1,1}`.
    """
    Cdvert=Cayd.vertices()
    n=Cay.dim()-1
    h11_ret=len(Cayd.points())-n-2
    if det:
        print("Trivial term: ",h11_ret)
    for f in Cayd.faces(n):
        h11_ret=h11_ret-len(Polytope(2*(f.vertices())).interior_points())
    if det:
        print("After 2*dual facets: ",h11_ret)
    for f in Cayd.faces(n-1):
        h11_ret=h11_ret+len(f.interior_points())
    if det:
        print("After dual codim-2: ",h11_ret)
    for f in Cay.faces(1):
        k=len(f.interior_points())
        if k>0:
            h11_ret=h11_ret+(k*(len(Polytope(2*(dual_face_Cayley_polytope(Cdvert,f).vertices())).interior_points())))
    if det: 
        print("After 1-face/2*codim-2 dual face: ",h11_ret)
    for f in Cay.faces(2):
        k=len(Polytope(2*(f.vertices())).interior_points())
        if k>0:
            h11_ret=h11_ret-(k*len(dual_face_Cayley_polytope(Cdvert,f).interior_points()))
    if det:
        print("After 2*(2-face)/codim-3 dual face: ",h11_ret)
    for f in Cay.faces(3):
        k=len(Polytope(2*(f.vertices())).interior_points())
        if k>0:
            h11_ret=h11_ret+(k*len(dual_face_Cayley_polytope(Cdvert,f).interior_points()))
        k=len(dual_face_Cayley_polytope(Cdvert,f).interior_points())
        if k>0:
            for g in f.faces(2):
                h11_ret=h11_ret-(len(g.interior_points())*k)
    return h11_ret

def h21_2_part(Cay: Polytope,Cayd: Polytope,det=False):
    """
    **Description:**
    Computes the Hodge number `h^{2,1}` of a complete intersection Calabi-Yau described by a two-part nef partition in a six-dimensional ambient variety.
    **Arguments:**
    - `Cay (Polytope)`: The Cayley polytope.
    - `Cayd (Polytope)`: The dual Cayley polytope.
    - `det (bool)`: Whether to print intermediate contributions. Defaults to `False`.
    **Returns:**
    - `int`: The Hodge number `h^{2,1}`.
    """
    Cdvert=Cayd.vertices()
    n=Cay.dim()-1
    h21_ret=0
    for f in Cay.faces(2):
        h21_ret=h21_ret+len(Polytope(2*(dual_face_Cayley_polytope(Cdvert,f).vertices())).interior_points())*len(f.interior_points())
    if det:
        print(h21_ret)
    for f in Cay.faces(4):
        h21_ret=h21_ret+len(Polytope(2*(f.vertices())).interior_points())*len(dual_face_Cayley_polytope(Cdvert,f).interior_points())
    if det:
        print(h21_ret)
    for f in Cay.faces(3):
        k=len(dual_face_Cayley_polytope(Cdvert,f).interior_points())
        if k>0:
            for g in f.faces(2):
                h21_ret=h21_ret-len(g.interior_points())*k
    if det: 
        print(h21_ret)
    for f in Cay.faces(4):
        k=len(dual_face_Cayley_polytope(Cdvert,f).interior_points())
        if k>0:
            for g in f.faces(3):
                h21_ret=h21_ret-len(g.interior_points())*k
    return h21_ret

def get_indices(arr: np.array,targets: np.array):
    """
    **Description:**
    Finds the indices of rows of `arr` that also occur among the rows of `targets`.
    **Arguments:**
    - `arr (numpy.ndarray)`: The two-dimensional array to search.
    - `targets (numpy.ndarray)`: The target row or rows.
    **Returns:**
    - `numpy.ndarray`: The indices of matching rows in `arr`.
    """
    return  np.where(np.any((arr[:, None, :] == targets).all(axis=2), axis=1))[0]

def get_index(arr: np.array,target: np.array):
    """
    **Description:**
    Finds the indices of rows of `arr` equal to the one-dimensional array `target`.
    **Arguments:**
    - `arr (numpy.ndarray)`: The two-dimensional array to search.
    - `target (numpy.ndarray)`: The target row.
    **Returns:**
    - `numpy.ndarray`: The indices of rows of `arr` equal to `target`.
    """
    return  np.where(np.all(arr == target, axis=1))[0]

def glsm_from_points(pts):
    """
    **Description:**
    Computes an integral basis of linear relations among the input toric points using Smith normal form.
    **Arguments:**
    - `pts (array-like)`: The toric point configuration.
    **Returns:**
    - `numpy.ndarray`: A GLSM charge matrix for the point configuration.
    """
    a,s,t=smith_normal_decomp(Matrix(np.array(pts).T),domain=ZZ)
    aa=np.array(a,dtype=int)
    ss=np.array(s,dtype=int)
    tt=np.array(t,dtype=int)
    rank_a=np.linalg.matrix_rank(aa)
    return tt.T[rank_a:]

def points_from_glsm(glsm):
    """
    **Description:**
    Computes an integral point configuration whose relations are described by the input GLSM charge matrix.
    **Arguments:**
    - `glsm (array-like)`: The GLSM charge matrix.
    **Returns:**
    - `numpy.ndarray`: A toric point configuration.
    """
    D,U,V=smith_normal_decomp(Matrix(np.array(glsm).T),domain=ZZ)
    DD=np.array(D,dtype=int)
    UU=np.array(U,dtype=int)
    VV=np.array(V,dtype=int)
    rank_D=np.linalg.matrix_rank(DD)
    return (UU[rank_D:].astype(int)).T

def find_trilayer_vertex_polytope(p,as_index=False):
    """
    **Description:**
    Uses the GLSM charge matrix of the vertices to identify the vertex corresponding to half the anticanonical class.
    **Arguments:**
    - `p (Polytope)`: The trilayer polytope.
    - `as_index (bool)`: Whether to return the vertex index rather than the vertex coordinates. Defaults to `False`.
    **Returns:**
    - `numpy.ndarray` or `int`: The distinguished vertex, or its index if `as_index=True`.
    """
    glsm_vert=glsm_from_points(p.vertices())
    half_anticanon = np.sum(glsm_vert, axis=1)//2
    index=get_indices(glsm_vert.T,np.array([half_anticanon]))[0]
    if as_index:
        return index
    else:
        return p.vertices()[index]

def find_trilayer_vertex_vertices(V,as_vertex_index=False):
    """
    **Description:**
    Uses the GLSM charge matrix of the vertex set to identify the vertex corresponding to half the anticanonical class.
    **Arguments:**
    - `V (array-like)`: The vertices of a trilayer polytope.
    - `as_vertex_index (bool)`: Whether to return the vertex index rather than the vertex coordinates. Defaults to `False`.
    **Returns:**
    - `numpy.ndarray` or `int`: The distinguished vertex, or its index if `as_vertex_index=True`.
    """
    glsm_vert=glsm_from_points(V)
    half_anticanon = np.sum(glsm_vert, axis=1)//2
    index=get_indices(glsm_vert.T,np.array([half_anticanon]))[0]
    if as_vertex_index:
        return index
    else:
        return V[index]

def trilayer_normal_form(p):
    """
    **Description:**
    Applies an integral change of basis that moves the distinguished trilayer vertex into a standard position.
    **Arguments:**
    - `p (Polytope)`: The trilayer polytope.
    **Returns:**
    - `Polytope`: The polytope in trilayer normal form.
    """
    verts=p.vertices()
    index=find_trilayer_vertex_vertices(verts,as_vertex_index=True)
    verts[[0,index]]=verts[[index,0]]
    aa,ss,tt=smith_normal_decomp(Matrix(verts),domain=ZZ)
    a=np.array(aa,dtype=int)
    s=np.array(ss,dtype=int)
    t=np.array(tt,dtype=int)
    b=np.ones(len(verts),dtype=int)
    b[0]=-1
    c=s@b
    y=np.zeros(a.shape[1],dtype=int)
    for ii in range(len(y)):
        if a[ii][ii]!=0:
            y[ii]=c[ii]/a[ii][ii]
    r=t@y
    aa2,ss2,tt2=smith_normal_decomp(Matrix(r[:,None]),domain=ZZ)
    a2=np.array(aa2,dtype=int)
    s2=np.array(ss2,dtype=int)
    t2=np.array(tt2,dtype=int)
    U0=np.round(np.linalg.inv(s2).T).astype(int)
    M=U0@verts.T
    for i in range(1,p.ambient_dim()):
        U0[i]=U0[i]+M[i,0]*U0[0]
    return Polytope((U0@verts.T).T)

def Newton_Polytope(pts,weights):
    """
    **Description:**
    Constructs the Newton polytope associated with a toric divisor with coefficient vector `weights` on the fan with rays `pts`.
    **Arguments:**
    - `pts (numpy.ndarray)`: Matrix whose rows are the rays of the toric fan.
    - `weights (array-like)`: Divisor coefficients in the toric prime divisor basis.
    **Returns:**
    - `Polytope`: The Newton polytope of the divisor.
    """
    return h_polytope.HPolytope(np.column_stack([pts, weights]).astype(int))

def row_difference(A: np.array, B: np.array):
    """
    **Description:**
    Computes all rows of `A` that do not occur as rows of `B`.
    **Arguments:**
    - `A (numpy.ndarray)`: First two-dimensional array.
    - `B (numpy.ndarray)`: Second two-dimensional array.
    **Returns:**
    - `numpy.ndarray`: Rows of `A` that are not rows of `B`.
    """
    return A[~np.any((A[:, None, :] == B[None, :, :]).all(axis=2), axis=1)]

def points_not_interior_to_facets_and_codim2_faces(p: Polytope):
    """
    **Description:**
    Computes the lattice points of a polytope after removing points interior to facets and codimension-two faces.
    **Arguments:**
    - `p (Polytope)`: The polytope.
    **Returns:**
    - `numpy.ndarray`: The selected lattice points.
    """
    pts=p.points_not_interior_to_facets()
    for f in p.faces(p.dim()-2):
        if len(f.interior_points())>0:
            pts=np.delete(pts,get_indices(pts,f.interior_points()),axis=0)
    return pts

def get_lower_dimensional_cones(cones,d):
    """
    **Description:**
    Computes all `d`-element faces of the given maximal cones.
    **Arguments:**
    - `cones (iterable)`: Collection of cones, each represented by a tuple of one-indexed ray labels.
    - `d (int)`: Number of rays in the lower-dimensional cones to extract.
    **Returns:**
    - `list`: The list of distinct `d`-ray cones.
    """
    return list({combo for row in cones for combo in combinations(row, d)})

def lattice_refinement(q, denominator = 2):
    """
    **Description:**
    Returns the smallest integral embedding of the unit lattice into a refined lattice in which `q/denominator` becomes integral.
    **Arguments:**
    - `q (array-like)`: Integer vector defining the fractional lattice refinement.
    - `denominator (int)`: Denominator of the fractional vector. Defaults to `2`.
    **Returns:**
    - `numpy.ndarray`: The lattice refinement map.
    """
    lattice_basis = denominator*np.identity(len(q)).astype(int)
    lattice_basis = np.vstack([denominator*np.identity(len(q)).astype(int),[q]])
    A = fmpz_mat(lattice_basis.tolist())
    A_lll = A.lll()
    scaled_up_basis = np.array(A_lll.tolist()).astype(int)
    vanishing_pos = np.where(np.all(scaled_up_basis == 0,axis=1))[0]
    scaled_up_basis = np.delete(scaled_up_basis,vanishing_pos,axis=0)
    Lambda = np.linalg.inv(scaled_up_basis.T)
    return np.rint(Lambda*denominator).astype(int)

def toric_orbifold(pts_CY_ambient,q,denominator=2):
    """
    **Description:**
    Applies the lattice refinement defined by `q/denominator` to the ambient toric rays and returns the primitive orbifold rays together with the ray rescalings.
    **Arguments:**
    - `pts_CY_ambient (numpy.ndarray)`: Rays of the original Calabi-Yau ambient toric fan.
    - `q (array-like)`: Integer vector defining the fractional lattice refinement.
    - `denominator (int)`: Denominator of the fractional vector. Defaults to `2`.
    **Returns:**
    - `tuple`: A pair `(orbifold_points, rescalings)` consisting of primitive orbifold rays and the corresponding edge rescalings.
    """
    Lambda = lattice_refinement(q,denominator)
    orbifold_points = pts_CY_ambient@(Lambda.T)
    rescalings = np.array([math.gcd(*list(i)) for i in orbifold_points])
    orbifold_points = np.rint((orbifold_points.T/rescalings).T).astype(int)
    return (orbifold_points,rescalings,Lambda)

def O3O7_line_bundle(pts_CY_ambient,q,rescalings):
    """
    **Description:**
    Determines the divisor coefficients of the orientifold line bundle by selecting a projected-in monomial of the Calabi-Yau hypersurface Newton polytope and rescaling divisor classes under the orbifold map.
    **Arguments:**
    - `pts_CY_ambient (numpy.ndarray)`: Rays of the original Calabi-Yau ambient toric fan.
    - `q (array-like)`: Integer vector defining the `Z_2` action.
    - `rescalings (array-like)`: Rescalings of toric divisor classes under the orbifold map.
    **Returns:**
    - `numpy.ndarray` or `None`: The O3/O7 line-bundle coefficients, or `None` if no projected-in monomial is found.
    """
    CY3_equation_newton_polytope = Newton_Polytope(pts_CY_ambient,[1]*len(pts_CY_ambient))
    projected_in_monomial_indices = np.where(np.mod(CY3_equation_newton_polytope.points()@q,2)==1)[0]
    if len(projected_in_monomial_indices)==0:
        return None
    arbitrary_monomial_point = CY3_equation_newton_polytope.points()[projected_in_monomial_indices[0]]
    line_bundle_weights_CYhypersurface = pts_CY_ambient@arbitrary_monomial_point+1
    line_bundle_weights_FtheoryBase = np.rint(line_bundle_weights_CYhypersurface/rescalings).astype(int)
    return line_bundle_weights_FtheoryBase

def Z2_fixed_locus(vc_triangulation,q,cone_dimension=None,denominator=2):
    """
    **Description:**
    Finds cones whose associated toric strata are fixed by the lattice refinement defined by `q/denominator`. Optionally restricts to cones of a specified dimension.
    **Arguments:**
    - `vc_triangulation (Fan)`: The toric fan of the ambient variety.
    - `q (array-like)`: Integer vector defining the fractional lattice refinement.
    - `cone_dimension (int or None)`: If specified, only cones with this number of rays are considered. Defaults to `None`.
    - `denominator (int)`: Denominator of the fractional vector. Defaults to `2`.
    **Returns:**
    - `list`: Fixed-locus cones, represented as tuples of one-indexed ray labels.
    """
    if type(cone_dimension)==type(None):
        all_cones = {j for c in vc_triangulation.cones() for i in range(1,len(c))  for j in combinations(c,i)}
        all_cones = [c for c in all_cones]
    else:
        all_cones = list(get_lower_dimensional_cones(vc_triangulation.cones(),cone_dimension))
    fixed_locus_cones = [all_cones[i] for i in np.where([np.all(np.mod(sum(vc_triangulation.vectors()[np.array(c)-1])+q,denominator)==0) for c in all_cones])[0]]
    return fixed_locus_cones

def inequivalent_Z2_actions(lattice_symmetries):
    """
    **Description:**
    Enumerates half-integer lattice points defining `Z_2` torus actions modulo the action of the supplied lattice symmetry group.
    **Arguments:**
    - `lattice_symmetries (array-like)`: List or array of square integer matrices acting from the left.
    **Returns:**
    - `numpy.ndarray`: Inequivalent integer representatives `q` such that `q/2` defines a `Z_2` action.
    """
    dim = lattice_symmetries[0].shape[0]
    t_possibilities = {t for t0 in combinations_with_replacement([0,1],dim) for t in permutations(t0)}
    t_possibilities = [t for t in t_possibilities]
    t_possibilities = np.delete(t_possibilities,t_possibilities.index(tuple([0]*dim)),0)
    inequivalent_t_possibilities = {frozenset([tuple(y) for y in x]) for x in  np.transpose(np.array([np.mod(s@(t_possibilities.T),2) for s in lattice_symmetries]),[2,0,1])}
    inequivalent_t_possibilities = np.array([[y for y in x][0] for x in inequivalent_t_possibilities])
    return inequivalent_t_possibilities

def linebundle_weights_from_Newton_Polytope(vectors,Newton_polytope: Polytope):
    """
    **Description:**
    Recovers the toric divisor coefficients whose Newton polytope is `Newton_polytope` by maximizing the corresponding inequalities over its lattice points.
    **Arguments:**
    - `vectors (numpy.ndarray)`: Rays of the toric fan.
    - `Newton_polytope (Polytope)`: The Newton polytope.
    **Returns:**
    - `numpy.ndarray`: The divisor coefficient vector.
    """
    return np.max(-(vectors@Newton_polytope.points().T),axis=1)

def is_Gorenstein_full_dim(cone):
    """
    Tests whether the cone is Gorenstein and returns its Gorenstein vector.
    """
    M = np.asarray(cone.extremal_rays(), dtype=np.int64)
    n = np.rint(np.linalg.lstsq(M, np.ones(len(M)), rcond=None)[0]).astype(np.int64)
    return (True, n) if np.all(M @ n == 1) else (False, None)

def is_Gorenstein(cone):
    """Tests whether a cone is Gorenstein and returns a Gorenstein vector."""
    M = np.asarray(cone.extremal_rays(), dtype=int)
    n = np.zeros(M.shape[1], dtype=int)
    B = np.eye(M.shape[1], dtype=int)
    for a in M:
        c, b = a @ B, int(1 - a @ n)
        if not np.any(c):
            if b:
                return False, None
            continue
        for j in range(1, len(c)):
            if c[j]:
                x, y = int(c[0]), int(c[j])
                g = math.gcd(x, y)
                u = pow(x // g, -1, abs(y // g))
                v = (g - u*x) // y
                T = np.array([[u, -y//g], [v, x//g]], dtype=int)
                B[:, [0, j]] = B[:, [0, j]] @ T
                c[0], c[j] = g, 0
        g = int(c[0])
        if b % g:
            return False, None
        n += B[:, 0] * (b // g)
        B = B[:, 1:]
    return True, n

def is_reflexive_Gorenstein(cone):
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

def Gorenstein_index(cone):
    """
    **Description:**
    Computes the pairing of the Gorenstein generators of a reflexive Gorenstein cone and its dual.
    **Arguments:**
    - `cone (Cone)`: The cone to test.
    **Raises:**
    - `ValueError`: Raised if the cone is not reflexive Gorenstein.
    **Returns:**
    - `int`: The Gorenstein index.
    """
    if is_reflexive_Gorenstein(cone):
        dual_cone=cone.dual()
        return is_Gorenstein(cone)[1]@is_Gorenstein(dual_cone)[1]
    raise ValueError("Cone is not reflexive Gorenstein")

def Cartier_index(toric_fan, weights):
    """
    Computes the Cartier index of a Q-Cartier toric divisor.
    Returns None if the divisor is not Q-Cartier.
    """
    weights = np.asarray(weights)
    indices = []
    for c in toric_fan.cones():
        A = toric_fan.vectors(c)
        b = weights[np.asarray(c)-1]
        y = -np.linalg.lstsq(A, b, rcond=None)[0]
        if not np.allclose(A @ y, -b, rtol=0, atol=1e-10):
            return None
        k = 1
        while not np.allclose(k*y, np.rint(k*y), rtol=0, atol=1e-10):
            k += 1
        indices.append(k)
    return math.lcm(*indices)

def is_Cartier(toric_fan, weights, return_Q_Cartier_data=False, decimals=10):
    """
    Determines whether a toric divisor is Cartier and optionally returns
    local Q-Cartier data.
    """
    weights = np.array(weights)
    cartier_data = []
    is_cartier = True
    for c in toric_fan.cones():
        arr = toric_fan.vectors(c)
        cone_gen_weights = weights[np.array(c)-1]
        y = -np.linalg.lstsq(arr, cone_gen_weights, rcond=None)[0]
        if not np.allclose(arr @ y, -cone_gen_weights, rtol=0, atol=1e-10):
            if return_Q_Cartier_data:
                cartier_data.append(None)
                is_cartier = False
                continue
            return False, None
        if np.allclose(y, np.round(y), rtol=0, atol=1e-10):
            cartier_data.append(np.round(y).astype(int))
        else:
            is_cartier = False
            if return_Q_Cartier_data:
                cartier_data.append(np.round(y, decimals=decimals))
            else:
                return False, None
    return is_cartier, cartier_data

def is_nef(toric_fan,weights):
    """
    **Description:**
    Tests whether the divisor coefficient vector lies in the nef cone by pairing with the secondary-cone hyperplanes.
    **Arguments:**
    - `toric_fan (Fan)`: The toric fan.
    - `weights (array-like)`: Divisor coefficients in the toric prime divisor basis.
    **Returns:**
    - `bool`: `True` if the divisor is nef, otherwise `False`.
    """
    return np.all(np.array(toric_fan.secondary_cone_hyperplanes())@weights>=0)

def is_ample(toric_fan,weights):
    """
    **Description:**
    Tests whether the divisor coefficient vector lies in the interior of the nef cone by strict pairing with the secondary-cone hyperplanes.
    **Arguments:**
    - `toric_fan (Fan)`: The toric fan.
    - `weights (array-like)`: Divisor coefficients in the toric prime divisor basis.
    **Returns:**
    - `bool`: `True` if the divisor is ample, otherwise `False`.
    """
    return np.all(np.array(toric_fan.secondary_cone_hyperplanes())@weights>0)

def is_effective(points,weights):
    """
    **Description:**
    Determines whether the Newton polytope of a divisor has at least one lattice point.
    **Arguments:**
    - `points (numpy.ndarray)`: Rays of the toric fan.
    - `weights (array-like)`: Divisor coefficients in the toric prime divisor basis.
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

def moving_cone(toric_variety):
    """
    **Description:**
    Computes the moving cone from the GLSM charges by intersecting the cones obtained after deleting each toric ray.
    **Arguments:**
    - `toric_variety (Fan)`: The toric fan or toric variety object.
    **Returns:**
    - `Cone`: The moving cone.
    """
    rays = toric_variety.vectors()
    glsm = np.array(integral_nullspace(np.asarray(rays.T, dtype=int))).T
    h_planes = np.array([h for i in range(len(rays)) for h in Cone(np.delete(glsm.T,i,0)).hyperplanes()])@glsm
    mov = Cone(hyperplanes = h_planes)
    return mov

def generic_section_factorizes(points,linebundle_weights):
    """
    **Description:**
    Determines whether the generic section of a toric divisor factorizes by testing whether every toric coordinate appears nontrivially in some section.
    **Arguments:**
    - `points (array-like)`: Rays of the toric fan.
    - `linebundle_weights (array-like)`: Divisor coefficients in the toric prime divisor basis.
    **Returns:**
    - `bool`: `True` if the generic section factorizes, otherwise `False`.
    """
    try:
        NP=Newton_Polytope(points,linebundle_weights)
    except ValueError:
        return True
    return ~np.all(np.any(points@NP.points().T+linebundle_weights[:,None]!=0,axis=1))

def attempt_to_make_nef(toric_variety,line_bundle,epsilon=1e-5):
    """
    **Description:**
    Perturbs the triangulation heights in the direction of the given line bundle in order to find a triangulation for which the line bundle is nef.
    **Arguments:**
    - `toric_variety`: A triangulated vector configuration or toric fan with a vector configuration.
    - `line_bundle (array-like)`: Divisor coefficients in the toric prime divisor basis.
    - `epsilon (float)`: Magnitude of the perturbation used to obtain a triangulation rather than a subdivision. Defaults to `1e-5`.
    **Returns:**
    - `Fan`: A triangulation of the vector configuration.
    """
    line_bundle = np.array(line_bundle)
    hts0 = toric_variety.heights()+epsilon*np.array([(np.sin(i+1)+1) for i in range(len(line_bundle))])
    hts1 = line_bundle
    return toric_variety.vc.triangulate(heights=hts1/epsilon+hts0)

def basis(points):
    """
    **Description:**
    Returns indices of rows of `points` forming a basis for the row span.
    **Arguments:**
    - `points (numpy.ndarray)`: Matrix whose rows are candidate basis vectors.
    **Raises:**
    - `ValueError`: Raised if no basis can be found.
    **Returns:**
    - `list`: Indices of basis rows.
    """
    n=points.shape[0]
    d=np.linalg.matrix_rank(points)
    basis_indices = []
    for i in range(n):
        test_indices = basis_indices + [i]
        if np.linalg.matrix_rank(points[test_indices]) == len(test_indices):
            basis_indices.append(i)
        if len(basis_indices) == d:
            return basis_indices
    raise ValueError("No basis could be found")
# def sums_to_anticanonical(pts,L1,L2):
#     """
#     **Description:**
#     Determines whether `L1 + L2` is linearly equivalent to the anticanonical divisor and, if so, returns the character implementing the equivalence.
#     **Arguments:**
#     - `pts (numpy.ndarray)`: Rays of the toric fan.
#     - `L1 (array-like)`: First divisor coefficient vector.
#     - `L2 (array-like)`: Second divisor coefficient vector.
#     **Returns:**
#     - `tuple`: A pair `(sums_to_anticanonical, character)`, where `character` is the linear-equivalence shift if it exists, otherwise `None`.
#     """
#     pts_float = np.array(pts, dtype=float)
#     b_float = (1 - np.array(L1) - np.array(L2)).astype(float)
#     try:
#         x_float, residuals, rank, s = np.linalg.lstsq(pts_float, b_float, rcond=None)
#         x_int = np.round(x_float).astype(int)
#         pts_exact = np.array(pts, dtype=object)
#         b_exact = 1 - np.array(L1) - np.array(L2)
#         if np.array_equal(pts_exact @ x_int, b_exact):
#             return True, x_int
#     except np.linalg.LinAlgError:
#         pass 
#     return False, None
# def is_partition(points, L1,L2):
#     """
#     **Description:**
#     Determines whether two toric divisors can be shifted by principal divisors so that their coefficients are in `{0,1}` and their sum is the anticanonical divisor.
#     **Arguments:**
#     - `points (numpy.ndarray)`: Rays of the toric fan.
#     - `L1 (array-like)`: First divisor coefficient vector.
#     - `L2 (array-like)`: Second divisor coefficient vector.
#     **Returns:**
#     - `tuple`: A tuple `(is_partition, sums_to_anticanonical, shift_L1, shift_L2)`.
#     """
#     sta=sums_to_anticanonical(points,L1,L2)
#     if sta[0]==False:
#         return (False,False,np.zeros(points.shape[1],dtype=int),np.zeros(points.shape[1],dtype=int))
#     basis_indices=basis(points)
#     basis_vectors = points[basis_indices]
#     basis_L2 = L2[basis_indices]
#     possible_targets = [[-w, 1 - w] for w in basis_L2]
#     if len(basis_indices)==points.shape[1]:
#         for target_comb in product(*possible_targets):
#             target_vec = np.array(target_comb)
#             m2_float = np.linalg.solve(basis_vectors, target_vec)
#             m2_int = np.round(m2_float).astype(int)
#             if not np.allclose(m2_float, m2_int):
#                 continue
#             dot_products = points @ m2_int
#             valid_lower = (dot_products == -L2)
#             valid_upper = (dot_products == 1 - L2)
#             if np.all(valid_lower | valid_upper):
#                 return (True,True,sta[1]-m2_int,m2_int)
#     else:
#         basis_vecs_pseudo=basis_vectors@basis_vectors.T
#         for target_comb in product(*possible_targets):
#             target_vec = np.array(target_comb)
#             m2_float = basis_vectors.T@np.linalg.solve(basis_vecs_pseudo, target_vec)
#             m2_int = np.round(m2_float).astype(int)
#             if not np.allclose(m2_float, m2_int):
#                 continue
#             dot_products = points @ m2_int
#             valid_lower = (dot_products == -L2)
#             valid_upper = (dot_products == 1 - L2)
#             if np.all(valid_lower | valid_upper):
#                 return (True,True,sta[1]-m2_int,m2_int)
#     return (False,True,np.zeros(points.shape[1],dtype=int),sta[1])

def attempt_to_make_Cartier(tri,D):
    """
    **Description:**
    If the divisor is not Cartier on the given fan, this function adds rays from the Newton-polytope inequalities and attempts to triangulate the refined configuration so that the divisor becomes Cartier and nef.
    **Arguments:**
    - `tri (Fan)`: The initial toric fan.
    - `D (array-like)`: Divisor coefficients in the toric prime divisor basis.
    **Returns:**
    - `tuple`: A pair `(new_fan, new_D)` consisting of the refined fan and the updated divisor coefficients.
    """
    if is_Cartier(tri,D)[0]:
        return (tri,D)
    else:
        NP=Newton_Polytope(tri.vectors(),D)
        inequalities = NP.inequalities()
        new_points = row_difference(inequalities[:,:-1],tri.vectors())
        indices_new_points = get_indices(inequalities[:,:-1],new_points)
        new_vectors = np.concatenate((tri.vectors(),new_points),axis=0)
        new_vc=VectorConfiguration(new_vectors)
        new_D=np.concatenate((D,inequalities[:,-1][indices_new_points]))
        tri_Cartier = attempt_to_make_nef(new_vc.triangulate(),new_D)
        return (tri_Cartier,new_D)

def BL(fan,lb):
    """
    **Description:**
    Computes sections of the divisor with weights `lb` and returns the corresponding base locus in the given fan.
    **Arguments:**
    - `fan (Fan)`: The toric fan.
    - `lb (array-like)`: Divisor coefficients in the toric prime divisor basis.
    **Returns:**
    - `list`: Cones defining the base locus.
    """
    return base_locus(sections(fan.vectors(),lb),cones=fan.cones())

def base_locus(sections,cones=None,dim=4):
    """
    **Description:**
    Computes minimal coordinate strata on which all sections vanish. If cones are provided, the search is restricted to strata of the corresponding toric fan.
    **Arguments:**
    - `sections (numpy.ndarray)`: Section exponent matrix.
    - `cones (iterable or None)`: Cones of the toric fan. If `None`, all coordinate strata up to dimension `dim` are considered. Defaults to `None`.
    - `dim (int)`: Dimension used when `cones=None`. Defaults to `4`.
    **Returns:**
    - `list`: Cones defining the base locus.
    """
    num_coords,num_sections=sections.shape
    B=sections > 0 
    minimal_hitting_sets=[]
    if type(cones)==type(None):
        for codim in range(1,dim + 1):
            for combo in combinations(range(num_coords), codim):
                combo_set = set(combo)
                if any(set(mhs).issubset(combo_set) for mhs in minimal_hitting_sets):
                    continue
                if B[list(combo), :].any(axis=0).all():
                    minimal_hitting_sets.append(combo)            
        return [tuple(x + 1 for x in mhs) for mhs in minimal_hitting_sets]
    else:
        for codim in range(1,len(cones[0])+1):
            for combo in get_lower_dimensional_cones(cones,codim):
                combo_set = set(combo)
                if any(set(mhs).issubset(combo_set) for mhs in minimal_hitting_sets):
                    continue
                if B[np.array(combo)-1, :].any(axis=0).all():
                    minimal_hitting_sets.append(combo)            
        return [mhs for mhs in minimal_hitting_sets]

def normal_fan(polytopes,inequalities=None,maximal_refinement=False,triangulate_refinement=False,return_unrefined_fan=False):
    """
    **Description:**
    Constructs the normal fan of a lattice polytope, or of the Minkowski sum of a list of lattice polytopes. Optionally constructs a maximal refinement subject to the specified inequalities.
    **Arguments:**
    - `polytopes (Polytope or list[Polytope])`: A lattice polytope, or a list of lattice polytopes whose Minkowski sum is used.
    - `inequalities (array-like or None)`: Inequality data used for maximal refinement. Required if `maximal_refinement=True`. Defaults to `None`.
    - `maximal_refinement (bool)`: Whether to construct the maximal refinement. Defaults to `False`.
    - `triangulate_refinement (bool)`: Whether to triangulate the refined vector configuration. Defaults to `False`.
    - `return_unrefined_fan (bool)`: Whether to also return the unrefined normal fan. Defaults to `False`.
    **Raises:**
    - `Exception`: Raised if `maximal_refinement=True` but no inequalities are provided.
    **Returns:**
    - `tuple`: The normal fan or refined vector data, together with line-bundle weights and optionally the unrefined normal fan.
    """
    if type(polytopes)==type([]):
        msum_vertices = nested_sum([p.vertices() for p in polytopes])
        p = Polytope(np.unique(flatten(msum_vertices,len(polytopes)-1),axis=0))
        vertex_split = np.array([np.array(np.where(np.all(np.array(msum_vertices)-v==0,axis=-1))).T[0] for v in p.vertices()])
    else:
        p = polytopes
        weights = p.inequalities().T[-1]
    hyperplane_saturations = [p.inequalities()[np.where(x==0)[0]] for x in (np.vstack([p.vertices().T,[1]*len(p.vertices())]).T@(p.inequalities().T))]
    normal_fan_edges = np.delete(p.inequalities().T,-1,0).T
    normal_fan_vc = VectorConfiguration(normal_fan_edges)
    cones = [[int(np.where(np.all(normal_fan_edges-x==0,axis=1))[0][0])+1 for x in np.delete(s.T,-1,0).T] for s in hyperplane_saturations]
    n_fan = Fan(vc=normal_fan_vc,cones=cones)
    if type(polytopes)==type([]):
        vertices_to_vertices_map = [vertex_split[np.where([i in c for c in cones])[0][0]] for i in range(1,len(normal_fan_edges)+1)]
        weights = np.array([-np.array([polytopes[j].vertices()[x] for j,x in enumerate(pointers)])@(normal_fan_edges[i]) for i,pointers in enumerate(vertices_to_vertices_map)])
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
    maximal_blow_ups = [h_polytope.HPolytope(np.vstack([[np.concatenate([np.delete(inequalities,-1,0)@np.array([polytopes[j].vertices()[x] for j,x in enumerate(vertex_split[i])]),[inequalities[-1]]])],np.vstack([(p.vertices()-m).T, [0]*len(p.vertices())]).T ])).points() for i,m in enumerate(p.vertices())]
    maximal_blow_ups = [np.unique([np.rint(x/np.gcd.reduce(x)).astype(int) for x in np.delete(b,np.where(np.all(b==0,axis=1))[0][0],0)],axis=0) for b in maximal_blow_ups]
    all_vectors = np.unique(np.array([y for x in maximal_blow_ups for y in x]),axis=0)
    all_weights = np.array([-np.array([(pol.vertices()[vertex_split[np.where([np.any(np.all(y-x==0,axis=-1)) for y in maximal_blow_ups])[0][0]][j]])@x for x in all_vectors]) for j,pol in enumerate(polytopes)])
    old_indices = np.where([type(n_fan.vc.vectors_to_labels(v))!=type(None) for v in all_vectors])[0]
    blow_up_weights = np.delete(all_weights.T,old_indices,0)
    blow_up_vectors = np.delete(all_vectors,old_indices,0)
    all_vectors = np.vstack([n_fan.vectors(),blow_up_vectors])
    all_weights = np.vstack([weights,blow_up_weights])
    if not triangulate_refinement:
        if return_unrefined_fan:
            return (all_vectors,all_weights,n_fan)
        else:
            return (all_vectors,all_weights)
    if return_unrefined_fan:
        return (refine_fan(make_simplicial(n_fan),all_vectors),all_weights,n_fan)
    else:
        return (refine_fan(make_simplicial(n_fan),all_vectors),all_weights)

def nested_sum(lists, depth=0, acc=0):
    """
    **Description:**
    Recursively forms all sums obtained by choosing one element from each list in `lists`.
    **Arguments:**
    - `lists (list)`: A list of lists or arrays whose elements are to be summed.
    - `depth (int)`: Recursion depth. Defaults to `0`.
    - `acc`: Accumulated partial sum. Defaults to `0`.
    **Returns:**
    - `list`: The nested list of sums.
    """
    if depth == len(lists):
        return acc
    return [nested_sum(lists, depth + 1, acc + x) for x in lists[depth]]

def flatten(lst, depth):
    """
    **Description:**
    Recursively flattens a nested list by the specified number of levels.
    **Arguments:**
    - `lst (list)`: The nested list to flatten.
    - `depth (int)`: Number of nesting levels to flatten.
    **Returns:**
    - `list`: The flattened list.
    """
    if depth == 0:
        return lst
    result = []
    for x in lst:
        if isinstance(x, list):
            result.extend(flatten(x, depth - 1))
        else:
            result.append(x)
    return result

def O7_cones(vc_orbifold,O7_labels,d):
    """
    **Description:**
    Computes the `d`-ray cones of a toric fan whose rays are all contained in the set of O7-plane labels.
    **Arguments:**
    - `vc_orbifold (Fan)`: The toric fan of the orbifold.
    - `O7_labels (array-like)`: One-indexed labels of the O7 divisors.
    - `d (int)`: Number of rays in the cones to consider.
    **Returns:**
    - `list`: Cones whose rays are contained in `O7_labels`.
    """
    d_cones= get_lower_dimensional_cones(vc_orbifold.cones(),d)
    relevant_d_cones = [t for t in d_cones if set(t).issubset(O7_labels)]
    return relevant_d_cones

def basis_H2_toric_fan(toric_fan):
    """
    **Description:**
    Finds a smooth maximal cone and returns the complementary ray labels, giving a convenient GLSM or curve-homology basis.
    **Arguments:**
    - `toric_fan (Fan)`: The toric fan.
    **Raises:**
    - `ValueError`: Raised if no smooth maximal cone is found.
    **Returns:**
    - `numpy.ndarray`: One-indexed ray labels forming the basis.
    """
    for c in toric_fan.cones():
        if Cone(toric_fan.vectors(c)).is_smooth():
            mask = np.ones(len(toric_fan.vectors()),dtype=bool)
            mask[np.array(c)-1]=False
            basis = np.arange(1,len(mask)+1)[mask]
            return basis
    raise ValueError("No basis could be found")

def trilayer_5d_Ftheory_uplift(p,verbosity=1):
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
    p = trilayer_normal_form(p)
    mid_layer_points = np.where(p.points().T[0]==0)[0]
    mid_layer_points = np.array([i for i in set(mid_layer_points).intersection(set(p.points_to_indices(p.points_not_interior_to_facets())))])
    if len(mid_layer_points)>1 and verbosity>0:
        print(f"{len(mid_layer_points)-1} exceptional divisors have been blown down.")
    p_dim = p.dim()
    p3 = Polytope(np.delete(p.points()[np.where(p.points().T[0]==1)[0]].T,0,0).T)
    p2KB = Newton_Polytope(p3.points(),len(p3.points())*[2])
    n_fan,wts,cns = normal_fan(p2KB)
    cone_hyperplanes = [Cone(n_fan.vectors(c)).dual().rays() for c in cns]
    blown_up = [h_polytope.HPolytope(np.vstack([np.vstack([h.T,[0]*len(h)]).T,[np.concatenate([p2KB.vertices()[i],[2]]),np.concatenate([-p2KB.vertices()[i],[-1]])]])).points() for i,h in enumerate(cone_hyperplanes)]
    blown_up = [[np.rint(x/np.gcd.reduce(x)).astype(int) for x in b] for b in blown_up]
    blown_up = [np.unique(b,axis=0) for b in blown_up]
    all_vecs = np.unique([i for b in blown_up for i in b],axis=0)
    full_vc = VectorConfiguration(all_vecs)
    monomials = Newton_Polytope(full_vc.vectors(),[2]*len(full_vc.vectors())).points()@(full_vc.vectors().T)+np.array([2]*len(full_vc.vectors()))
    O7pos = np.where(np.array([min(m) for m in monomials.T])==1)[0]
    vx = np.concatenate([[0]*(p_dim-1),[3,1]])
    vy = np.concatenate([[0]*(p_dim-1),[-2,-1]])
    vz = np.concatenate([[0]*(p_dim-1),[0,1]])
    uplift_vecs_singular = np.vstack([np.vstack([full_vc.vectors().T,[[0]*len(full_vc.vectors()),[1]*len(full_vc.vectors())]]).T,[vx,vy,vz]])
    D4res1 = uplift_vecs_singular[O7pos]+vx+2*vy
    D4res2 = 2*uplift_vecs_singular[O7pos]+2*vx+3*vy
    uplift_vecs = np.vstack([uplift_vecs_singular,D4res1,D4res2])
    p5 = Polytope(uplift_vecs)
    return p5

def sections(points,weights):
    """
    **Description:**
    Computes the exponent vectors of all monomial sections of the divisor with coefficient vector `weights` on the fan with rays `points`.
    **Arguments:**
    - `points (numpy.ndarray)`: Rays of the toric fan.
    - `weights (array-like)`: Divisor coefficients in the toric prime divisor basis.
    **Returns:**
    - `numpy.ndarray`: Matrix of section exponents, or an empty array if there are no sections.
    """
    NP=Newton_Polytope(points,weights)
    if len(NP.points())==0:
        return np.array([])
    else:
        return points@NP.points().T+weights[:,None]

def solve_over_integers(M,b):
    """
    **Description:**
    Uses Smith normal form to determine whether the equation `Mx=b` has an integral solution, and returns one if it exists.
    **Arguments:**
    - `M (array-like)`: Integer matrix.
    - `b (array-like)`: Integer inhomogeneous term.
    **Returns:**
    - `tuple`: A pair `(has_solution, x)`, where `x` is an integral solution if one exists, otherwise `None`.
    """
    A,S,T = smith_normal_decomp(Matrix(M),domain=ZZ)
    a=np.array(A,dtype=int)
    s=np.array(S,dtype=int)
    t=np.array(T,dtype=int)
    c=-s@b
    y=np.zeros(a.shape[1],dtype=int)
    for ii in range(np.min(a.shape)):
        if a[ii,ii]!=0:
            y[ii]=c[ii]/a[ii,ii]
        else:
            if c[ii]!=0:
                print("PROBLEM")
    if np.all(a@y==c):
        return (True,t@y)
    else:
        return (False,None)

def make_simplicial(fan):
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

def refine_fan(fan,blowups_or_all_vectors=None):
    """
    **Description:**
    Adds new rays to a fan and star-subdivides the cones containing them. If no new vectors are given, the vector configuration of the fan is used to detect rays not already present in the fan.
    **Arguments:**
    - `fan (Fan)`: The toric fan to refine.
    - `blowups_or_all_vectors (numpy.ndarray or None)`: Blowup rays or a full vector configuration containing the old rays and new rays. Defaults to `None`.
    **Returns:**
    - `Fan`: The refined toric fan.
    """
    if blowups_or_all_vectors is not None:
        blowups=row_difference(blowups_or_all_vectors,fan.vc.vectors())
        all_vectors=np.concatenate((fan.vc.vectors(),blowups),axis=0)
        vc_all=VectorConfiguration(all_vectors)
        new_fan=Fan(vc_all,cones=fan.cones())
    else: 
        new_fan=fan
    vec_diff=row_difference(new_fan.vc.vectors(),new_fan.vectors())
    to_be_refined = set(new_fan.vc.vectors_to_labels(vec_diff))
    if fan.dim==len(new_fan.vc.vectors()[0]):
        for label in to_be_refined:
            all_cones=set(new_fan.cones())
            link_base=tuple(find_cone(new_fan.vc.vectors(label),all_cones,new_fan.vc.vectors()))
            link_base_len=len(link_base)
            for c in new_fan.link(link_base):
                all_cones.discard(tuple(sorted(link_base + c)))
                for comb in combinations(link_base, link_base_len - 1):
                    all_cones.add(tuple(sorted(comb + c + (label,))))
            new_fan=Fan(vc=new_fan.vc,cones=all_cones)
        return new_fan
    else:
        for label in to_be_refined:
            all_cones=set(new_fan.cones())
            link_base=tuple(find_cone_general(new_fan.vc.vectors(label),all_cones,all_vectors))
            link_base_len=len(link_base)
            for c in new_fan.link(link_base):
                all_cones.discard(tuple(sorted(link_base + c)))
                for comb in combinations(link_base, link_base_len - 1):
                    all_cones.add(tuple(sorted(comb + c + (label,))))
            new_fan=Fan(vc=new_fan.vc,cones=all_cones)
        return new_fan

def find_cone_general(new_ray, current_cones, all_vectors):
    """
    **Description:**
    Searches the current cones for a minimal set of one-indexed ray labels whose strictly positive linear combination gives `new_ray`.
    **Arguments:**
    - `new_ray (numpy.ndarray)`: The ray to locate.
    - `current_cones (iterable)`: Current cones, represented by tuples of one-indexed ray labels.
    - `all_vectors (numpy.ndarray)`: Full ray matrix.
    **Returns:**
    - `frozenset` or `None`: The carrier face labels, or `None` if no carrier cone is found.
    """
    for cone in current_cones:
        cone_list = list(cone)
        numpy_indices = [idx - 1 for idx in cone_list]
        A = all_vectors[numpy_indices].T
        x, residual = nnls(A, new_ray)
        if residual < 1e-10:
            carrier_face = frozenset(cone_list[i] for i, coeff in enumerate(x) if coeff > 1e-10)
            return carrier_face
    return None

def array_to_latex(arr):
    """
    **Description:**
    Converts a two-dimensional NumPy array into a LaTeX `pmatrix` string.
    **Arguments:**
    - `arr (numpy.ndarray)`: The two-dimensional array to convert.
    **Raises:**
    - `ValueError`: Raised if `arr` is not two-dimensional.
    **Returns:**
    - `str`: A LaTeX `pmatrix` representation of the array.
    """
    if len(arr.shape) > 2:
        raise ValueError("Only 2D matrices are supported.")
    lines = ["            " + " & ".join(map(str, row)) + r" \\" for row in arr]
    return "\\begin{pmatrix}\n" + "\n".join(lines) + "\n        \\end{pmatrix}"

def integral_gale_transform(points):
    """
    **Description:**
    Lifts the input points by appending a column of ones, computes the rational nullspace exactly using SymPy, and clears denominators to obtain an integral Gale transform.
    **Arguments:**
    - `points (array-like)`: Point configuration with shape `(n, d)`.
    **Raises:**
    - `ValueError`: Raised if the number of points is not greater than `d+1`.
    **Returns:**
    - `numpy.ndarray`: The integral Gale transform.
    """
    points = np.array(points)
    n, d = points.shape
    if n <= d + 1:
        raise ValueError(f"Need strictly more points (n={n}) than dimensions + 1 (d+1={d+1}).")
    lifted_points = np.hstack((points, np.ones((n, 1))))
    A = Matrix(lifted_points.T)
    null_basis_vectors = A.nullspace()
    if not null_basis_vectors:
        return np.array([])
    B = Matrix.hstack(*null_basis_vectors)
    for j in range(B.cols):
        LCM = lcm([fraction(B[i, j])[1] for i in range(B.rows)])
        B[:, j] = B[:, j] * LCM
    gale_points = np.array(B.T).astype(int)
    return gale_points

def find_cone(new_ray, current_cones, all_vectors, tol=1e-10):
    """
    **Description:**
    Lifts the input points by appending a column of ones, computes the rational nullspace exactly using SymPy, and clears denominators to obtain an integral Gale transform.
    **Arguments:**
    - `points (array-like)`: Point configuration with shape `(n, d)`.
    **Raises:**
    - `ValueError`: Raised if the number of points is not greater than `d+1`.
    **Returns:**
    - `numpy.ndarray`: The integral Gale transform.
    """
    for cone in current_cones:
        numpy_indices = np.array(cone) - 1
        A = all_vectors[numpy_indices].T
        coeffs = np.linalg.solve(A, new_ray)
        if np.all(coeffs >= -tol):
            return frozenset(idx for idx, coeff in zip(cone, coeffs) if coeff > tol)
    return None

def divisor_intersections(fan, intersection_dict,divisors, basis_set,as_LLL=True):
    """
    **Description:**
    Computes the curve classes obtained by intersecting a list of divisors with toric strata, expressed in a chosen basis of curve homology.
    **Arguments:**
    - `fan (Fan)`: The toric fan.
    - `intersection_dict (dict)`: Dictionary of toric intersection numbers.
    - `divisors (list)`: List of divisor coefficient vectors.
    - `basis_set (set)`: Set of one-indexed ray labels used as the homology basis.
    - `as_LLL (bool)`: Whether to LLL-reduce the resulting lattice basis. Defaults to `True`.
    **Returns:**
    - `numpy.ndarray`: The divisor-intersection curve classes, optionally LLL-reduced.
    """
    codim_cicy=len(divisors)
    simplices = get_lower_dimensional_cones(fan.cones(), fan.dim - codim_cicy-1)
    divisor_nonvanishing_sets = []
    for div in divisors:
        divisor_nonvanishing_sets.append(set(np.where(div != 0)[0] + 1))
    curves_homology_in_basis = np.zeros((len(simplices), len(basis_set)), dtype=int)
    basis_idx_map = {b: idx for idx, b in enumerate(basis_set)}
    for s_idx, s in enumerate(simplices):
        star_s=fan.star(s)
        link_rays = {item for sub_tuple in star_s for item in sub_tuple}
        valid_intersections = []
        for div_set in divisor_nonvanishing_sets:
            valid_intersections.append(div_set.intersection(link_rays))
        valid_i = basis_set.intersection(link_rays)
        if not (all(valid_intersections) and valid_i):
            continue
        for i in valid_i:
            i_idx = basis_idx_map.get(i)
            total_intersection = 0
            for ts in product(*valid_intersections):
                key = tuple(sorted(s + ts+ (i,)))
                coefficient = np.prod([divisors[a][ray - 1]for a, ray in enumerate(ts)])
                total_intersection += coefficient * intersection_dict.get(key, 0)
            curves_homology_in_basis[s_idx, i_idx] = total_intersection
    if as_LLL:
        reduced = np.array(lll_reduce(np.asarray(curves_homology_in_basis, dtype=int).T)).T
        return reduced[np.any(reduced != 0, axis=1)]
    return curves_homology_in_basis

def make_rows_integer(arr,den=2):
    arr = np.asarray(arr).copy()
    integer = np.all(np.isclose(arr, np.rint(arr)), axis=1)
    den_integer = np.all(np.isclose(den * arr, np.rint(den * arr)), axis=1)
    mask = ~integer & den_integer
    arr[mask] *= den
    return np.rint(arr).astype(int)

def hodge_numbers_weighted(O):
    chiO3=chi_O3_uplift(O)
    chiO7=chi_O7(O)
    pol=O.polytope()
    if not pol.is_reflexive():
        raise ValueError("Weighted Hodge Numbers can only be computed if the original CY is defined by a reflexive polytope")
    hodge_numbers={}
    if pol.is_favorable(lattice="N"):
        hodge_numbers["h11-"]=0
        hodge_numbers["h11+"]=pol.h11()
    else:
        h11m=0
        for f in pol.faces(2):
            cf=len(f.interior_points())
            if cf > 0:
                fd=f.dual_face()
                cfd=len(fd.interior_points())
                if cfd > 0:
                    if cfd%2==1:
                        fdverts=fd.vertices()
                        dq=np.round((fdverts[1]-fdverts[0])/(cfd+1)).astype(int)
                        if (dq@O.xi())%2==1:
                            h11m+=np.round(cf*(cfd+1)/2).astype(int)
        hodge_numbers["h11+"]=pol.h11()-h11m
        hodge_numbers["h11-"]=h11m
    hodge_numbers["h21-"]=hodge_numbers.get("h11-")+np.round((chi_I(O)-pol.chi(lattice="N"))/4-1).astype(int)
    hodge_numbers["h21+"]=pol.h21()-hodge_numbers.get("h21-")
    return hodge_numbers

def chi_I(O):
    return chi_O7(O)+chi_O3_uplift(O)

def chi_O7(O):
    O7s_CY_ambient=np.array(find_O7_planes(O))
    IN_CY=O.CY_ambient_toric_fan().intersection_numbers(pushed_down=True)
    c2=O.CY_ambient_toric_fan().c2()
    chi=0
    for o7 in O7s_CY_ambient:
        t=(o7,o7,o7)
        chi+=IN_CY.get(t,0)+c2[o7-1]
    return chi

def chi_O3(fan,fixed_cones):
    chi=0
    IN=fan.intersection_numbers()
    for c in fixed_cones:
        for d in range(1,len(fan.vectors())+1):
            t=tuple(sorted(c+(d,)))
            chi+=IN.get(t,0)
    return chi

def chi_O3_uplift(O):
    fixed_cones=Z2_fixed_locus(O.CY_ambient_toric_fan(),O.xi(),O.dim()-1)
    return chi_O3(O.CY_ambient_toric_fan(),fixed_cones)

def intersection_test(fan, intersection_dict,divisors,target_divisors):
    codim_cicy=len(divisors)
    divisor_nonvanishing_sets = []
    for div in divisors:
        divisor_nonvanishing_sets.append(set(np.where(div != 0)[0] + 1))
    curves_homology_in_basis = np.zeros( len(target_divisors),dtype=bool)
    for i, d in enumerate(target_divisors):
        stari=fan.star((d,))
        stari_set={item for sub_tuple in stari for item in sub_tuple}
        simps=get_lower_dimensional_cones(stari,3)
        valid_intersections = []
        for div_set in divisor_nonvanishing_sets:
            valid_intersections.append(div_set.intersection(stari_set))
        for s in simps:
            total_intersection = 0
            for ts in product(*valid_intersections):
                key = tuple(sorted(s + ts+ (d,)))
                coefficient = np.prod([divisors[a][ray - 1]for a, ray in enumerate(ts)])
                total_intersection += coefficient * intersection_dict.get(key, 0)
            if total_intersection!=0:
                curves_homology_in_basis[i]=True
                break
    return curves_homology_in_basis

def intersection_test_uplift(F,target_divisors):
    LBW=np.zeros(len(F.line_bundle_base_N()),dtype=int)
    xlab=len(F.vectors_singular_uplift_ambient())-3
    LBW[xlab]=3
    return intersection_test(F.smooth_uplift_ambient_toric_fan(),F.intersection_numbers_smooth_uplift_ambient(),[F.line_bundle_base_N(),LBW],target_divisors)

def find_O7_planes(O):
    O7_tup=Z2_fixed_locus(O.CY_ambient_toric_fan(),O.xi(),1)
    O7tot=np.unique([t[0] for t in O7_tup])
    return O7tot

def _vertex_key(vertices: np.ndarray) -> tuple:
    """
    Hashable key for a face, independent of the ordering of its vertices.
    """
    vertices = np.asarray(vertices, dtype=np.int64)
    return tuple(sorted(map(tuple, vertices.tolist())))

def h11_2_part_l_fast(Cay: Polytope, Cayd: Polytope, det: bool = False) -> int:
    """
    Faster implementation of h11_2_part_l.
    The implementation avoids constructing a new Polytope for every face.
    It assumes the Cayley dual-face relation used in the original code:
        <v_dual, v_face> = 0
    for every vertex v_face of the original face.
    """
    dim = Cay.dim()
    if Cayd.dim() != dim:
        raise ValueError("Cay and Cayd must have the same dimension, got "f"{dim} and {Cayd.dim()}.")
    n = dim - 1
    if n < 3:
        raise ValueError("This implementation expects Cayley polytopes for which 1-, 2-, and 3-faces occur.")
    cay_faces_1 = Cay.faces(1)
    cay_faces_2 = Cay.faces(2)
    cay_faces_3 = Cay.faces(3)
    dual_dimensions = {n - 1, n - 2, n - 3}
    cayd_faces = {d: Cayd.faces(d) for d in dual_dimensions}
    cayd_face_lookup = {d: {_vertex_key(face.vertices()): face for face in faces} for d, faces in cayd_faces.items()}
    cay_vertices = np.asarray(Cay.vertices(), dtype=int)
    cayd_vertices = np.asarray(Cayd.vertices(), dtype=int)
    zero_pairing = (cayd_vertices @ cay_vertices.T == 0)
    point_to_vertex_index = np.full(len(Cay.points()),-1,dtype=int)
    cay_vertex_point_indices = np.asarray(Cay.vertices(as_indices=True),dtype=int)
    point_to_vertex_index[cay_vertex_point_indices] = np.arange(len(cay_vertices),dtype=int)

    def dual_face_data(face):
        """
        Return the existing Cayd PolytopeFace dual to `face`, together
        with its dual vertices.
        """
        face_point_indices = np.asarray(face.vertices(as_indices=True),dtype=np.intp)
        face_vertex_indices = point_to_vertex_index[face_point_indices]
        if np.any(face_vertex_indices < 0):
            raise RuntimeError("A face vertex could not be matched to Cay.vertices().")
        mask = np.all(zero_pairing[:, face_vertex_indices],axis=1)
        dual_vertices = cayd_vertices[mask]
        if len(dual_vertices) == 0:
            raise RuntimeError("The zero-pairing condition produced no dual vertices.")
        dual_dimension = n - face.dim()
        key = _vertex_key(dual_vertices)
        try:
            dual_face = cayd_face_lookup[dual_dimension][key]
        except KeyError as exc:
            raise RuntimeError("The zero-pairing vertex set was not found among the "+f"{dual_dimension}-faces of Cayd.") from exc
        return dual_face, dual_vertices
    doubled_cayd = Polytope(2 * cayd_vertices)
    doubled_dual_codim2_lookup = {_vertex_key(face.vertices()): face for face in doubled_cayd.faces(n - 1)}
    trivial_term = len(Cayd.points()) - n - 2
    doubled_dual_facet_term = -sum(len(face.labels_int) for face in doubled_cayd.faces(n))
    dual_codim2_term = sum(len(face.labels_int) for face in cayd_faces[n - 1])
    edge_term = 0
    nonzero_edges = 0
    for face in cay_faces_1:
        lstar_face = len(face.labels_int)
        if lstar_face == 0:
            continue
        _, dual_vertices = dual_face_data(face)
        doubled_dual_key = _vertex_key(2 * dual_vertices)
        try:
            doubled_dual_face = doubled_dual_codim2_lookup[doubled_dual_key]
        except KeyError as exc:
            raise RuntimeError("Could not match a dual face with its face in 2*Cayd.") from exc
        edge_term += lstar_face*len(doubled_dual_face.labels_int)
        nonzero_edges += 1
    face_2_term = 0
    nonzero_2faces = 0
    for face in cay_faces_2:
        dual_face, _ = dual_face_data(face)
        lstar_dual = len(dual_face.labels_int)
        if lstar_dual == 0:
            continue
        lstar_face = len(face.labels_int)
        lattice_points_face = len(face.labels)
        lstar_doubled_face = 3 * lstar_face + lattice_points_face - 3
        face_2_term -= lstar_dual * lstar_doubled_face
        nonzero_2faces += 1
    face_3_term = 0
    nonzero_3faces = 0
    for face in cay_faces_3:
        dual_face, _ = dual_face_data(face)
        lstar_dual = len(dual_face.labels_int)
        if lstar_dual == 0:
            continue
        lstar_face = len(face.labels_int)
        lattice_points_face = len(face.labels)
        normalized_volume_raw = face.as_polytope().volume()
        normalized_volume = int(round(float(normalized_volume_raw)))
        if not np.isclose(float(normalized_volume_raw),normalized_volume):
            raise RuntimeError("The normalized volume of a lattice 3-face was not integral: "+f"{normalized_volume_raw}.")
        lstar_doubled_face = normalized_volume - lattice_points_face + 3 * lstar_face + 3
        subface_correction = sum(len(subface.labels_int) for subface in face.faces(2))
        face_3_term += lstar_dual * (lstar_doubled_face - subface_correction)
        nonzero_3faces += 1
    result = trivial_term + doubled_dual_facet_term + dual_codim2_term + edge_term + face_2_term + face_3_term
    if det:
        print("Trivial term:                         ", trivial_term)
        print("Doubled dual facets:                 ", doubled_dual_facet_term)
        print("Dual codimension-2 faces:            ", dual_codim2_term)
        print("1-face / doubled dual-face term:     ", edge_term)
        print("2-face / dual-face term:             ", face_2_term)
        print("3-face and subface correction term:  ", face_3_term)
        print("Nonzero contributing edges:          ", nonzero_edges)
        print("Nonzero contributing 2-faces:        ", nonzero_2faces)
        print("Nonzero contributing 3-faces:        ", nonzero_3faces)
        print("h11:                                  ", result)
    return int(result)

def h21_2_part_fast(Cay: Polytope, Cayd: Polytope, det: bool = False) -> int:
    """
    Fast implementation of h21_2_part.
    This is algebraically equivalent to
        sum_{dim f=2} l*(f) l*(2 f*)
      + sum_{dim f=4} l*(2f) l*(f*)
      - sum_{dim f=3} l*(f*) sum_{g< f, dim g=2} l*(g)
      - sum_{dim f=4} l*(f*) sum_{g< f, dim g=3} l*(g),
    where f* is the dual Cayley face.
    Optimizations
    -------------
    1. Dual faces are found using the zero-pairing matrix rather than by
       repeatedly calling dual_face_Cayley_polytope.
    2. 2*Cay and 2*Cayd are constructed only once.
    3. Interior lattice-point counts use ``labels_int`` directly.
    4. The two contributions involving 4-faces are evaluated in the same loop.
    Parameters
    ----------
    Cay : Polytope
        Cayley polytope.
    Cayd : Polytope
        Dual Cayley polytope.
    det : bool
        Print individual contributions.
    Returns
    -------
    int
        h^{2,1}.
    """
    dim = Cay.dim()
    if Cayd.dim() != dim:
        raise ValueError("Cay and Cayd must have the same dimension, got "f"{dim} and {Cayd.dim()}.")
    n = dim - 1
    if n < 4:
        raise ValueError("This implementation expects Cayley polytopes with 4-faces.")
    cay_faces_2 = Cay.faces(2)
    cay_faces_3 = Cay.faces(3)
    cay_faces_4 = Cay.faces(4)
    dual_dimensions = {n - 2, n - 3, n - 4}
    cay_vertices = np.asarray(Cay.vertices(), dtype=np.int64)
    cayd_vertices = np.asarray(Cayd.vertices(), dtype=np.int64)
    zero_pairing = (cayd_vertices @ cay_vertices.T == 0)
    cay_vertex_lookup = {tuple(map(int, v)): i for i, v in enumerate(cay_vertices)}

    def vertex_key(vertices):
        return tuple(sorted(tuple(map(int, row)) for row in np.asarray(vertices)))
    cayd_lstar_lookup = {d: {vertex_key(face.vertices()): len(face.labels_int) for face in Cayd.faces(d)} for d in dual_dimensions}

    def dual_face_key_and_lstar(face):
        fv = np.asarray(face.vertices(), dtype=np.int64)
        try:
            face_vertex_indices = np.fromiter((cay_vertex_lookup[tuple(map(int, v))] for v in fv), dtype=np.intp, count=len(fv))
        except KeyError as exc:
            raise RuntimeError("A face vertex could not be matched to Cay.vertices().") from exc
        mask = np.all(zero_pairing[:, face_vertex_indices], axis=1)
        dual_vertices = cayd_vertices[mask]
        if len(dual_vertices) == 0:
            raise RuntimeError("The zero-pairing condition produced no dual vertices.")
        dual_dim = n - face.dim()
        key = vertex_key(dual_vertices)
        try:
            lstar = cayd_lstar_lookup[dual_dim][key]
        except KeyError as exc:
            raise RuntimeError("The zero-pairing vertex set was not found among the " f"{dual_dim}-faces of Cayd.") from exc
        return key, lstar
    doubled_cay = Polytope(2 * cay_vertices)
    doubled_cayd = Polytope(2 * cayd_vertices)
    doubled_cay_4_lstar = {}
    for face in doubled_cay.faces(4):
        doubled_vertices = np.asarray(face.vertices(), dtype=np.int64)
        if np.any(doubled_vertices % 2):
            raise RuntimeError("A vertex of 2*Cay was unexpectedly not divisible by 2.")
        original_key = vertex_key(doubled_vertices // 2)
        doubled_cay_4_lstar[original_key] = len(face.labels_int)
    doubled_dual_dim = n - 2
    doubled_cayd_lstar = {}
    for face in doubled_cayd.faces(doubled_dual_dim):
        doubled_vertices = np.asarray(face.vertices(), dtype=np.int64)
        if np.any(doubled_vertices % 2):
            raise RuntimeError("A vertex of 2*Cayd was unexpectedly not divisible by 2.")
        original_key = vertex_key(doubled_vertices // 2)
        doubled_cayd_lstar[original_key] = len(face.labels_int)
    face_2_term = 0
    nonzero_2faces = 0
    for face in cay_faces_2:
        lstar_face = len(face.labels_int)
        if lstar_face == 0:
            continue
        dual_key, _ = dual_face_key_and_lstar(face)
        try:
            lstar_doubled_dual = doubled_cayd_lstar[dual_key]
        except KeyError as exc:
            raise RuntimeError("Could not match a dual face with its corresponding face in 2*Cayd.") from exc
        if lstar_doubled_dual == 0:
            continue
        face_2_term += lstar_face * lstar_doubled_dual
        nonzero_2faces += 1
    face_3_term = 0
    nonzero_3faces = 0
    for face in cay_faces_3:
        _, lstar_dual = dual_face_key_and_lstar(face)
        if lstar_dual == 0:
            continue
        subface_correction = sum(len(subface.labels_int) for subface in face.faces(2))
        if subface_correction == 0:
            continue
        face_3_term -= lstar_dual * subface_correction
        nonzero_3faces += 1
    face_4_term = 0
    nonzero_4faces = 0
    for face in cay_faces_4:
        _, lstar_dual = dual_face_key_and_lstar(face)
        if lstar_dual == 0:
            continue
        face_key = vertex_key(face.vertices())
        try:
            lstar_doubled_face = doubled_cay_4_lstar[face_key]
        except KeyError as exc:
            raise RuntimeError("Could not match a Cay 4-face with its corresponding face in 2*Cay.") from exc
        subface_correction = sum(len(subface.labels_int) for subface in face.faces(3))
        face_4_term += lstar_dual * (lstar_doubled_face - subface_correction)
        nonzero_4faces += 1
    result = face_2_term + face_3_term + face_4_term
    if det:
        print("2-face / doubled dual-face term:      ", face_2_term)
        print("3-face subface correction term:       ", face_3_term)
        print("4-face combined term:                  ", face_4_term)
        print("Nonzero contributing 2-faces:          ", nonzero_2faces)
        print("Nonzero contributing 3-faces:          ", nonzero_3faces)
        print("Nonzero contributing 4-faces:          ", nonzero_4faces)
        print("h21:                                     ", result)
    return int(result)

def count_interior_lattice_points_hrep(A, b, count_bin="count",latte_options=("--redundancy-check=none",)):
    """
    Count interior lattice points of
        P = {x : A x <= b}
    assuming A and b are primitive integer facet inequalities.
    Interior lattice points satisfy
        A x <= b - 1.
    """
    A = np.asarray(A, dtype=object)
    b = np.asarray(b, dtype=object)
    return count_lattice_points_hrep(A, b - 1,count_bin=count_bin,latte_options=latte_options)

def parse_latte_output(stdout, stderr=""):
    """
    Robustly parse LattE's output.
    In LattE integrale 1.7.6, stdout is often simply
    77
    while stderr contains all diagnostic information.
    """
    stdout_clean = stdout.strip()
    if re.fullmatch(r"[+-]?\d+", stdout_clean):
        return int(stdout_clean)
    output = stdout + "\n" + stderr
    m = re.search(r"Total number of lattice points(?:\s+is)?\s*:?\s*([+-]?\d+)",output)
    if m:
        return int(m.group(1))
    m = re.search(r"The number of lattice points is.*?\n\s*([+-]?\d+)\s*(?:\n|$)",output,flags=re.DOTALL)
    if m:
        return int(m.group(1))
    candidates = []
    for line in stdout.splitlines():
        line = line.strip()
        if re.fullmatch(r"[+-]?\d+", line):
            candidates.append(int(line))
    if candidates:
        return candidates[-1]
    raise RuntimeError(f"Could not parse LattE output.\n\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}")

def count_lattice_points_hrep(A, b, count_bin="count",latte_options=("--redundancy-check=none",),linearity_rows=None,keep_file=False,filename="polytope.hrep.latte"):
    """
    Count lattice points in
        P = {x in Z^d : A x <= b}
    using LattE.
    Parameters
    ----------
    A:
        numpy array of shape (n_ineqs, dim)
    b:
        numpy array of shape (n_ineqs,)
    count_bin:
        either "count" if it is on PATH, or full path to LattE's count binary
    latte_options:
        tuple/list of extra LattE options
    linearity_rows:
        optional list of 1-indexed rows treated as equalities by LattE
    keep_file:
        if True, write the LattE input file to `filename` and keep it
    filename:
        output filename used if keep_file=True
    Returns
    -------
    int
        number of lattice points
    """
    if shutil.which(count_bin) is None and not Path(count_bin).exists():
        raise FileNotFoundError(f"Could not find LattE executable '{count_bin}'. Use command-v count in the shell, or pass the full path as count_bin.")
    if keep_file:
        hrep_file = Path(filename)
        write_latte_hrep(A, b, hrep_file, linearity_rows=linearity_rows)
        cmd = [count_bin, *latte_options, str(hrep_file)]
        proc = subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,)
        if proc.returncode != 0:
            raise RuntimeError(f"LattE failed.\nCommand: {' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}")
        return parse_latte_output(proc.stdout, proc.stderr)
    with tempfile.TemporaryDirectory() as tmpdir:
        hrep_file = Path(tmpdir) / "polytope.hrep.latte"
        write_latte_hrep(A, b, hrep_file, linearity_rows=linearity_rows)
        cmd = [count_bin, *latte_options, str(hrep_file)]
        proc = subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,)
        if proc.returncode != 0:
            raise RuntimeError("LattE failed.\n"f"Command: {' '.join(cmd)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}")
        return parse_latte_output(proc.stdout, proc.stderr)

def write_latte_hrep(A, b, filename, linearity_rows=None):
    """
    Write the H-representation
        A x <= b
    to a LattE .hrep.latte file.
    LattE row convention:
        b_i  -A_i1  -A_i2  ...  -A_id
    Parameters
    ----------
    A:
        numpy array of shape (n_ineqs, dim)
    b:
        numpy array of shape (n_ineqs,)
    filename:
        output filename
    linearity_rows:
        optional list of 1-indexed row numbers that should be treated
        as equalities by LattE. Usually leave as None.
    """
    A = np.asarray(A, dtype=object)
    b = np.asarray(b, dtype=object)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array.")
    if b.ndim != 1:
        raise ValueError("b must be a 1D array.")
    if A.shape[0] != b.shape[0]:
        raise ValueError("A and b have incompatible shapes.")
    n_ineqs, dim = A.shape
    with open(filename, "w") as f:
        f.write(f"{n_ineqs} {dim + 1}\n")
        for Ai, bi in zip(A, b):
            row = [int(bi)] + [-int(x) for x in Ai]
            f.write(" ".join(map(str, row)) + "\n")
        if linearity_rows is not None and len(linearity_rows) > 0:
            linearity_rows = list(map(int, linearity_rows))
            f.write("linearity "+ str(len(linearity_rows))+ " "+ " ".join(map(str, linearity_rows))+ "\n")

def count_interior_lattice_points_vrep(vertices,backend='ppl',count_bin="count",latte_options=("--redundancy-check=none",)):
    out = poly_v_to_h(vertices,backend=backend)[0]
    A=out[:,:-1]
    b=out[:,-1]
    return count_interior_lattice_points_hrep(A, b, count_bin=count_bin,latte_options=latte_options)

def count_lattice_points_vrep(vertices,count_bin="count",latte_options=("--redundancy-check=none",),keep_file=False,filename="polytope.vrep.latte",):
    """
    Count lattice points in the convex hull of `vertices`
    using LattE directly in V-representation.
    Returns
    -------
    int
        Number of lattice points in P, including the boundary.
    """
    vertices = np.asarray(vertices, dtype=object)
    if shutil.which(count_bin) is None and not Path(count_bin).exists():
        raise FileNotFoundError(f"Could not find LattE executable '{count_bin}'. Use `command -v count` in the shell, or pass the full path as count_bin.")
    if keep_file:
        vrep_file = Path(filename)
        write_latte_vrep(vertices, vrep_file)
        cmd = [count_bin,"--vrep",*latte_options,str(vrep_file),]
        proc = subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,)
        if proc.returncode != 0:
            raise RuntimeError("LattE failed.\n"f"Command: {' '.join(cmd)}\n\n"f"STDOUT:\n{proc.stdout}\n\n"f"STDERR:\n{proc.stderr}")
        return parse_latte_output(proc.stdout, proc.stderr)
    with tempfile.TemporaryDirectory() as tmpdir:
        vrep_file = Path(tmpdir) / "polytope.vrep.latte"
        write_latte_vrep(vertices, vrep_file)
        cmd = [count_bin,"--vrep",*latte_options,str(vrep_file),]
        proc = subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,)
        if proc.returncode != 0:
            raise RuntimeError("LattE failed.\n"f"Command: {' '.join(cmd)}\n\n"f"STDOUT:\n{proc.stdout}\n\n"f"STDERR:\n{proc.stderr}")
        return parse_latte_output(proc.stdout, proc.stderr)

def count_interior_lattice_points_vrep_ehrhart(vertices,count_bin="count",latte_options=("--redundancy-check=none",),keep_file=False,filename="polytope.vrep.latte",):
    """
    Count interior lattice points of a full-dimensional lattice polytope
    given by its vertices, using Ehrhart-Macdonald reciprocity.
    If L_P(t) = #(tP ∩ Z^d),
    then #(int(P) ∩ Z^d) = (-1)^d L_P(-1).
    """
    vertices = np.asarray(vertices, dtype=object)
    if vertices.ndim != 2:
        raise ValueError("vertices must be a 2D array.")
    dim = vertices.shape[1]
    if shutil.which(count_bin) is None and not Path(count_bin).exists():
        raise FileNotFoundError(f"Could not find LattE executable '{count_bin}'.")

    def run(vrep_file):
        write_latte_vrep(vertices, vrep_file)
        cmd = [count_bin,"--ehrhart-polynomial","--vrep",*latte_options,str(vrep_file),]
        proc = subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,)
        if proc.returncode != 0:
            raise RuntimeError("LattE failed.\n"f"Command: {' '.join(cmd)}\n\n"f"STDOUT:\n{proc.stdout}\n\n"f"STDERR:\n{proc.stderr}")
        poly = _extract_latte_ehrhart_polynomial(proc.stdout)
        L_minus_one = _evaluate_latte_polynomial(poly, -1)
        result = ((-1) ** dim) * L_minus_one
        if result.denominator != 1:
            raise RuntimeError("Ehrhart reciprocity produced a non-integral result: "f"{result}\n"f"Ehrhart polynomial: {poly}")
        return int(result)
    if keep_file:
        return run(Path(filename))
    with tempfile.TemporaryDirectory() as tmpdir:
        return run(Path(tmpdir) / "polytope.vrep.latte")

def _extract_latte_ehrhart_polynomial(stdout):
    """
    Extract the Ehrhart polynomial from LattE stdout.
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if re.search(r"\bt(?:\^?\d+)?\b", line):
            return line
    for line in reversed(lines):
        if re.fullmatch(r"[+-]?\d+(?:/\d+)?", line):
            return line
    raise RuntimeError("Could not extract Ehrhart polynomial from LattE output.\n\n"f"STDOUT:\n{stdout}")

def _evaluate_latte_polynomial(poly, x):
    """
    Evaluate a univariate LattE Ehrhart polynomial exactly.
    """
    poly = poly.replace(" ", "")
    poly = poly.replace("*", "")
    poly = poly.replace("T", "t")
    if not poly.startswith(("+", "-")):
        poly = "+" + poly
    terms = re.findall(r"([+-])([^+-]+)", poly)
    value = Fraction(0)
    for sign, term in terms:
        sgn = 1 if sign == "+" else -1
        if "t" in term:
            coeff_part, power_part = term.split("t", 1)
            if coeff_part == "":
                coeff = Fraction(1)
            else:
                coeff = Fraction(coeff_part.strip("()"))
            if power_part == "":
                power = 1
            elif power_part.startswith("^"):
                power = int(power_part[1:])
            else:
                raise RuntimeError(f"Could not parse polynomial term: {term}")
        else:
            coeff = Fraction(term.strip("()"))
            power = 0
        value += sgn * coeff * (x ** power)
    return value

def Cayley_N(F):
    return Polytope(np.concatenate((np.column_stack([F.pol_W_N().points(), np.ones(len(F.pol_W_N().points()), dtype=int), np.zeros(len(F.pol_W_N().points()), dtype=int)]),np.column_stack([F.pol_B_N().points(), np.zeros(len(F.pol_B_N().points()), dtype=int), np.ones(len(F.pol_B_N().points()), dtype=int)])), axis=0))

def Cayley_M(F):
    return Polytope(np.concatenate((np.column_stack([F.pol_W_M().points(), np.ones(len(F.pol_W_M().points()), dtype=int), np.zeros(len(F.pol_W_M().points()), dtype=int)]),np.column_stack([F.pol_B_M().points(), np.zeros(len(F.pol_B_M().points()), dtype=int), np.ones(len(F.pol_B_M().points()), dtype=int)])), axis=0))
"""Batyrev--Nill stringy E-functions for Gorenstein support polytopes.
Specialized to reflexive Gorenstein Cayley cones.  The primary entry point
accepts an already-computed Cayley support P, its cone C, the dual cone C_dual,
and the two Gorenstein degree vectors.
Input may be a CYTools Polytope or an integer array whose rows are points.
NumPy arrays are converted to CYTools Polytope objects internally.  All cone
duality, face-lattice, and lattice-point geometry is performed by CYTools.
The *dual support* is required: its vertices must be in the dual lattice and
all primal/dual pairings must be nonnegative.  This is cone-support duality,
not ordinary polar duality after translating the degree-two slice.
Example
-------
    from stringy_e_cayley import stringy_e_from_cones
    ans = stringy_e_from_cones(P, C, C_dual, degree_C, degree_C_dual)
    print(ans.e_polynomial())
    print(ans.hodge_numbers)
    print(ans.primal_completely_split, ans.dual_completely_split)
    print(ans.interpretation())
If only the primal cone is completely split, ``ans.hodge_numbers`` applies to
that complete-intersection side.  The dual polynomial is still the canonical
Gorenstein-cone stringy E-invariant, but it must not be advertised as the
Hodge diamond of an ordinary Batyrev--Borisov mirror complete intersection.
The expensive part is the facewise Ehrhart/h* computation.  When LattE's
``count`` executable is available, the default backend computes one exact
Ehrhart polynomial per nontrivial face instead of enumerating every dilation
separately.  Cache the returned result (or the FaceLattice objects) for large
examples.
"""
Poly1 = Tuple[int, ...] 
Laurent2 = Dict[Tuple[int, int], int]
_HSTAR_CACHE: Dict[Tuple[int, Tuple[Tuple[int, ...], ...]], Poly1] = {}

def clear_hstar_cache() -> None:
    """Clear the process-wide exact h* cache used across repeated notebook runs."""
    _HSTAR_CACHE.clear()

def hstar_cache_size() -> int:
    """Return the number of face h* polynomials held in the global cache."""
    return len(_HSTAR_CACHE)

def _trim(a: Iterable[int]) -> Poly1:
    z = list(map(int, a))
    while len(z) > 1 and z[-1] == 0:
        z.pop()
    return tuple(z or [0])

def _add(a: Poly1, b: Poly1) -> Poly1:
    return _trim((a[i] if i < len(a) else 0) + (b[i] if i < len(b) else 0) for i in range(max(len(a), len(b))))

def _scale(a: Poly1, c: int) -> Poly1:
    return _trim(c * x for x in a)

def _mul(a: Poly1, b: Poly1) -> Poly1:
    z = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            z[i + j] += x * y
    return _trim(z)

def _t_minus_one_pow(k: int) -> Poly1:
    return tuple(comb(k, i) * (-1) ** (k - i) for i in range(k + 1))

def _h_star_from_counts(counts: Sequence[int], d: int) -> Poly1:
    """Recover h* from L_P(0),...,L_P(d) exactly."""
    if len(counts) < d + 1:
        raise ValueError(f"need at least {d + 1} Ehrhart values")
    return _trim(sum((-1) ** (j - i) * comb(d + 1, j - i) * int(counts[i]) for i in range(j + 1)) for j in range(d + 1))

def write_latte_vrep(vertices: Any, filename: Any) -> None:
    """Write integral vertices in LattE's native homogenized V-representation.
    Each vertex ``v`` is written as ``1 v_1 ... v_d``.  Native LattE V-input
    must be full-dimensional in the supplied coordinates; lower-dimensional
    faces are first put into intrinsic lattice coordinates by
    :func:`_intrinsic_lattice_vertices`.
    """
    vertices = _integer_points(vertices)
    n_vertices, dim = vertices.shape
    with open(filename, "w", encoding="utf-8") as stream:
        stream.write(f"{n_vertices} {dim + 1}\n")
        for row in vertices:
            stream.write("1 " + " ".join(map(str, map(int, row))) + "\n")

def _intrinsic_lattice_vertices(vertices: Any, d: int) -> np.ndarray:
    """Put a d-dimensional lattice face into full-dimensional Z^d coordinates.
    LattE's native V-representation does not handle a polytope that is
    lower-dimensional in its supplied ambient coordinates.  CYTools already
    uses an exact unimodular LLL transformation for this situation.  Reusing
    that transformation preserves the affine lattice and hence the Ehrhart
    polynomial, while avoiding construction/enumeration of a new Polytope.
    """
    verts = _integer_points(vertices)
    if d < 0:
        raise ValueError("intrinsic dimension must be nonnegative")
    if d == 0:
        return np.zeros((len(verts), 0), dtype=np.int64)
    if d > verts.shape[1]:
        raise ValueError("intrinsic dimension exceeds ambient dimension")
    translated = np.asarray(verts - verts[0], dtype=np.int64)
    if d == verts.shape[1] and _affine_dim(translated) == d:
        return translated
    try:
        from cytools.utils import lll_reduce
    except ImportError as e:
        raise ImportError("lower-dimensional LattE V-input requires cytools.utils.lll_reduce") from e
    reduced = np.asarray(lll_reduce(translated), dtype=np.int64)
    if reduced.ndim != 2 or reduced.shape != translated.shape:
        raise RuntimeError("CYTools lll_reduce returned an unexpected shape")
    intrinsic = reduced[:, reduced.shape[1] - d :]
    if intrinsic.shape[1] != d or _affine_dim(intrinsic) != d:
        raise RuntimeError("failed to obtain full-dimensional intrinsic lattice coordinates")
    return np.unique(intrinsic, axis=0)

def _cytools_work_is_small(vertices: np.ndarray, d: int, cutoff: int) -> bool:
    """
    Cheap upper-bound proxy for the total lattice-point enumeration problem
    CYTools faces when computing L_P(1), ..., L_P(d).
    The face must already be in intrinsic Z^d coordinates.
    Returns True as soon as the total size of the axis-aligned bounding boxes
    is known to be <= cutoff, and exits early once the cutoff is exceeded.
    """
    if d <= 0:
        return True
    widths = [int(w) for w in np.ptp(vertices, axis=0)]
    total = 0
    for k in range(1, d + 1):
        box_points = 1
        for w in widths:
            box_points *= k * w + 1
            if box_points > cutoff:
                return False
        total += box_points
        if total > cutoff:
            return False
    return True

def _parse_latte_ehrhart_polynomial(stdout: str, d: int) -> Tuple[Fraction, ...]:
    """Parse LattE's exact univariate Ehrhart polynomial into coefficients.
    LattE normally prints, for example,
        + 1 * t^0 + 10/3 * t^1 + 8 * t^2 + 20/3 * t^3
    The returned tuple is ``(c_0,...,c_d)`` for ``sum c_i t^i``.
    """
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    candidates = [line for line in lines if re.search(r"\bt(?:\s*\^\s*\d+)?\b", line)]
    if not candidates:
        raise RuntimeError(f"could not find an Ehrhart polynomial in LattE output\n\nSTDOUT:\n{stdout}")
    expr = candidates[-1].replace("T", "t")
    if not expr.startswith(("+", "-")):
        expr = "+" + expr
    terms = re.findall(r"([+-])\s*([^+-]+)", expr)
    coeffs = [Fraction(0) for _ in range(d + 1)]
    seen = False
    for sign, body in terms:
        body = re.sub(r"\s+", "", body)
        if not body:
            continue
        sgn = 1 if sign == "+" else -1
        if "t" in body:
            m = re.fullmatch(r"(?:(\d+(?:/\d+)?)\*?)?t(?:\^(\d+))?", body,)
            if m is None:
                continue
            coefficient = Fraction(m.group(1) or "1") * sgn
            power = int(m.group(2) or "1")
        else:
            try:
                coefficient = Fraction(body) * sgn
            except ValueError:
                continue
            power = 0
        if power > d:
            raise RuntimeError(f"LattE returned Ehrhart degree {power}, expected at most {d}: {expr}")
        coeffs[power] += coefficient
        seen = True
    if not seen:
        raise RuntimeError(f"could not parse LattE Ehrhart polynomial: {expr!r}")
    if coeffs[0] != 1:
        raise RuntimeError(f"unexpected Ehrhart constant coefficient {coeffs[0]} (expected 1): {expr}")
    if d > 0 and coeffs[d] == 0:
        raise RuntimeError(f"LattE returned a polynomial of degree below face dimension {d}: {expr}")
    return tuple(coeffs)

def _ehrhart_values_from_coefficients(coefficients: Sequence[Fraction], d: int) -> Tuple[int, ...]:
    """Evaluate an exact Ehrhart polynomial at 0,...,d."""
    values: List[int] = []
    for k in range(d + 1):
        value = sum(c * (k**j) for j, c in enumerate(coefficients))
        if value.denominator != 1:
            raise RuntimeError(f"Ehrhart polynomial evaluated non-integrally at k={k}: {value}")
        values.append(int(value))
    return tuple(values)

def ehrhart_polynomial_latte_vrep(vertices: Any, *, intrinsic_dim: Optional[int] = None, count_bin: str = "count", latte_options: Sequence[str] = ("--redundancy-check=none",),) -> Tuple[Fraction, ...]:
    """Compute an exact Ehrhart polynomial with LattE from vertex data.
    Returns coefficients ``(c_0,...,c_d)`` of ``L_P(t)``.  If the vertices are
    embedded in a larger ambient lattice (as every proper face of a support
    polytope is), pass its intrinsic dimension; the face is then moved to
    full-dimensional intrinsic lattice coordinates before LattE is called.
    """
    verts = _integer_points(vertices)
    d = _affine_dim(verts) if intrinsic_dim is None else int(intrinsic_dim)
    if d < 0:
        raise ValueError("intrinsic_dim must be nonnegative")
    if d == 0:
        return (Fraction(1),)
    executable = shutil.which(count_bin)
    if executable is None and Path(count_bin).exists():
        executable = str(Path(count_bin).resolve())
    if executable is None:
        raise FileNotFoundError(f"Could not find LattE executable {count_bin!r}. Pass its full path " "as count_bin or put 'count' on PATH.")
    intrinsic = _intrinsic_lattice_vertices(verts, d)
    with tempfile.TemporaryDirectory(prefix="stringy_e_latte_") as tmp:
        path = Path(tmp) / "face.vrep.latte"
        write_latte_vrep(intrinsic, path)
        cmd = [executable, "--vrep", "--ehrhart-polynomial", *tuple(latte_options), str(path),]
        process = subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, check=False,)
        if process.returncode != 0:
            raise RuntimeError("LattE failed to compute an Ehrhart polynomial.\n" f"Command: {' '.join(cmd)}\n\n" f"STDOUT:\n{process.stdout}\n\nSTDERR:\n{process.stderr}")
        return _parse_latte_ehrhart_polynomial(process.stdout, d)

def _integer_points(a: Any) -> np.ndarray:
    """Extract integer vertices/points from an array or CYTools-like object."""
    if isinstance(a, np.ndarray) or isinstance(a, (list, tuple)):
        x = np.asarray(a)
    elif hasattr(a, "vertices"):
        x = np.asarray(a.vertices())
    else:
        raise TypeError("expected an integer array or a CYTools Polytope-like object")
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("points must be a nonempty two-dimensional array")
    if not np.all(np.equal(x, np.rint(x))):
        raise ValueError("all coordinates must be integers")
    return np.asarray(x, dtype=object).astype(np.int64)

def _integer_vector(a: Any, *, name: str, length: Optional[int] = None) -> np.ndarray:
    x = np.asarray(a)
    if x.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if length is not None and len(x) != length:
        raise ValueError(f"{name} must have length {length}")
    if not np.all(np.equal(x, np.rint(x))):
        raise ValueError(f"{name} must be integral")
    return np.asarray(x, dtype=object).astype(np.int64)

def cayley_support(delta1: Any, delta2: Any, *, as_cytools: bool = False, backend: Optional[str] = None) -> Any:
    """Return conv((Delta1,1,0) union (Delta2,0,1)).
    ``delta1`` and ``delta2`` must use the same lattice coordinates (normally
    six columns).  By default an integer NumPy array is returned; request a
    CYTools Polytope with ``as_cytools=True``.
    """
    a, b = _integer_points(delta1), _integer_points(delta2)
    if a.shape[1] != b.shape[1]:
        raise ValueError("Delta_1 and Delta_2 must have the same ambient dimension")
    ca = np.hstack((a, np.tile([[1, 0]], (len(a), 1))))
    cb = np.hstack((b, np.tile([[0, 1]], (len(b), 1))))
    out = np.unique(np.vstack((ca, cb)), axis=0)
    if not as_cytools:
        return out
    try:
        from cytools import Polytope
    except ImportError as e:
        raise ImportError("CYTools is required for as_cytools=True") from e
    return Polytope(out, **({"backend": backend} if backend else {}))

def cayley_support_2part(delta1: Any, delta2: Any, **kwargs: Any) -> Any:
    """Descriptive alias for :func:`cayley_support`."""
    return cayley_support(delta1, delta2, **kwargs)

def _primitive_rows(rays: Any) -> np.ndarray:
    """Primitive, unique integral ray generators, with their signs preserved."""
    r = _integer_points(rays)
    if np.any(np.all(r == 0, axis=1)):
        raise ValueError("a cone ray cannot be zero")
    out = []
    for row in r:
        g = int(np.gcd.reduce(np.abs(row)))
        out.append(tuple(map(int, row // g)))
    return np.asarray(sorted(set(out)), dtype=np.int64)

@dataclass
class CayleyGorensteinData:
    """CYTools objects for a two-part Cayley cone and its Gorenstein dual."""
    primal_cone: Any
    dual_cone: Any
    primal_support: Any
    dual_support: Any
    primal_vertices: np.ndarray
    dual_vertices: np.ndarray
    degree: np.ndarray

@dataclass
class GorensteinConeData:
    """Validated precomputed reflexive Gorenstein cone data.
    ``degree_C`` lies in the lattice dual to ``C`` and has value one on
    primitive extremal rays of ``C``.  ``degree_C_dual`` satisfies the
    analogous condition on ``C_dual``.
    """
    primal_cone: Any
    dual_cone: Any
    primal_support: Any
    dual_support: Any
    primal_vertices: np.ndarray
    dual_vertices: np.ndarray
    degree_C: np.ndarray
    degree_C_dual: np.ndarray
    index: int

def cayley_gorenstein_data(delta1: Any, delta2: Any, *, interior_point: Optional[Sequence[int]] = None, poly_backend: Optional[str] = None, check: bool = True,) -> CayleyGorensteinData:
    """Build primal/dual cones and degree-one supports using CYTools only.
    CYTools performs cone dualization lazily; the expensive conversion from the
    dual H-representation to rays occurs at ``dual_cone.rays()``.  For a
    reflexive Gorenstein cone, its primitive extremal rays all have degree one
    and therefore are precisely the vertices of the dual support.
    ``interior_point`` is the unique interior lattice point of Delta1+Delta2.
    By default it is inferred with CYTools.  The primal degree element is
    ``(interior_point,1,1)``.
    """
    try:
        from cytools import Cone, Polytope
    except ImportError as e:
        raise ImportError("cayley_gorenstein_data requires CYTools") from e

    def cy_poly(obj: Any) -> Any:
        if hasattr(obj, "vertices") and hasattr(obj, "faces"):
            return obj
        kw = {"backend": poly_backend} if poly_backend else {}
        return Polytope(_integer_points(obj), **kw)
    poly1, poly2 = cy_poly(delta1), cy_poly(delta2)
    d1, d2 = _integer_points(poly1), _integer_points(poly2)
    primal_vertices = cayley_support(d1, d2)
    base_dim = d1.shape[1]
    if d2.shape[1] != base_dim:
        raise ValueError("Delta_1 and Delta_2 must have the same ambient dimension")
    minkowski_sum = poly1.minkowski_sum(poly2)
    if int(minkowski_sum.dimension()) != base_dim:
        raise ValueError("Delta_1 + Delta_2 is not full-dimensional in the base lattice")
    if interior_point is None:
        if minkowski_sum.is_reflexive():
            p0 = np.zeros(base_dim, dtype=np.int64)
        else:
            interior = np.asarray(minkowski_sum.interior_points(), dtype=np.int64)
            if interior.ndim != 2:
                interior = interior.reshape((-1, base_dim))
            if len(interior) != 1:
                same = np.array_equal(np.asarray(sorted(map(tuple, d1))), np.asarray(sorted(map(tuple, d2))),)
                hint = (" The two inputs are identical; stringy_e_cayley expects the " "two Cayley summands, not a primal/dual pair or two copies of " "the anticanonical polytope." if same else "")
                raise ValueError("Delta_1 + Delta_2 must have exactly one interior lattice " f"point, but CYTools found {len(interior)}.{hint}")
            p0 = interior[0]
    else:
        p0 = np.asarray(interior_point)
    if p0.shape != (base_dim,) or not np.all(np.equal(p0, np.rint(p0))):
        raise ValueError(f"interior_point must be an integral vector of length {base_dim}")
    if check:
        centered_sum = Polytope(_integer_points(minkowski_sum) - np.asarray(p0, dtype=np.int64), **({"backend": poly_backend} if poly_backend else {}),)
        if not centered_sum.is_reflexive():
            raise ValueError("Delta_1 + Delta_2, translated by its interior point, is not " "reflexive; the inputs do not define the required Gorenstein " "Cayley construction")
    degree = np.concatenate((np.asarray(p0, dtype=np.int64), [1, 1]))
    primal_cone = Cone(rays=primal_vertices, check=check)
    dual_cone = primal_cone.dual_cone()
    dual_vertices = _primitive_rows(dual_cone.extremal_rays())
    pair = primal_vertices @ dual_vertices.T
    for j in range(pair.shape[1]):
        if np.all(pair[:, j] <= 0) and np.any(pair[:, j] < 0):
            dual_vertices[j] *= -1
            pair[:, j] *= -1
    if np.any(pair < 0):
        i, j = map(int, np.argwhere(pair < 0)[0])
        raise RuntimeError("CYTools cone dualization returned an invalid ray orientation: " f"<primal_ray[{i}], dual_ray[{j}]> = {int(pair[i, j])}")
    degrees = dual_vertices @ degree
    if np.any(degrees != 1):
        values = sorted(set(map(int, degrees)))
        raise ValueError("the dual cone is not Gorenstein for degree " f"{degree.tolist()}; dual-ray degrees are {values}")
    kw = {"backend": poly_backend} if poly_backend else {}
    primal_support = Polytope(primal_vertices, **kw)
    dual_support = Polytope(dual_vertices, **kw)
    if check:
        expected_dim = base_dim + 1
        if int(primal_cone.dimension()) != base_dim + 2:
            raise ValueError("the Cayley cone is not full-dimensional")
        if int(primal_support.dimension()) != expected_dim:
            raise ValueError(f"primal support must have affine dimension {expected_dim}")
        if int(dual_support.dimension()) != expected_dim:
            raise ValueError(f"dual support must have affine dimension {expected_dim}")
        if np.any(~np.any(pair == 0, axis=0)) or np.any(~np.any(pair == 0, axis=1)):
            raise ValueError("some support vertex lies on no dual facet")
    return CayleyGorensteinData(primal_cone=primal_cone, dual_cone=dual_cone, primal_support=primal_support, dual_support=dual_support, primal_vertices=primal_vertices, dual_vertices=dual_vertices, degree=degree,)

def dual_support_vertices_cytools(delta1: Any, delta2: Any, *, interior_point: Optional[Sequence[int]] = None, check: bool = True,) -> np.ndarray:
    """Fast CYTools computation of dual degree-one support vertices."""
    return cayley_gorenstein_data(delta1, delta2, interior_point=interior_point, check=check).dual_vertices

def validate_gorenstein_cone_data(P: Any, C: Any, C_dual: Any, degree_C: Any, degree_C_dual: Any, *, pairing: Optional[np.ndarray] = None, expected_index: Optional[int] = None, poly_backend: Optional[str] = None, check: bool = True,) -> GorensteinConeData:
    """Validate precomputed cone data and construct the dual support polytope.
    Convention: ``degree_C`` grades ``C`` and therefore lies in the lattice of
    ``C_dual``; ``degree_C_dual`` grades ``C_dual`` and lies in the lattice of
    ``C``.  If the two supplied vectors satisfy the equations only after being
    swapped, they are swapped automatically.
    """
    try:
        from cytools import Polytope
    except ImportError as e:
        raise ImportError("validate_gorenstein_cone_data requires CYTools") from e
    for name, cone in (("C", C), ("C_dual", C_dual)):
        if not hasattr(cone, "extremal_rays") or not hasattr(cone, "dimension"):
            raise TypeError(f"{name} must be a CYTools Cone")
    primal_rays = _primitive_rows(C.extremal_rays())
    dual_rays = _primitive_rows(C_dual.extremal_rays())
    ambient = primal_rays.shape[1]
    if dual_rays.shape[1] != ambient:
        raise ValueError("C and C_dual have different ambient dimensions")
    B = np.eye(ambient, dtype=np.int64) if pairing is None else np.asarray(pairing)
    if B.shape != (ambient, ambient):
        raise ValueError(f"pairing must have shape {(ambient, ambient)}")
    if not np.all(np.equal(B, np.rint(B))):
        raise ValueError("pairing matrix must be integral")
    B = np.asarray(B, dtype=np.int64)
    dC = _integer_vector(degree_C, name="degree_C", length=ambient)
    dD = _integer_vector(degree_C_dual, name="degree_C_dual", length=ambient)

    def heights_on_C(v: np.ndarray) -> np.ndarray:
        return primal_rays @ B @ v

    def heights_on_dual(v: np.ndarray) -> np.ndarray:
        return dual_rays @ B.T @ v
    direct = np.all(heights_on_C(dC) == 1) and np.all(heights_on_dual(dD) == 1)
    swapped = np.all(heights_on_C(dD) == 1) and np.all(heights_on_dual(dC) == 1)
    if not direct and swapped:
        dC, dD = dD, dC
        direct = True
    if not direct:
        raise ValueError("the supplied vectors are not Gorenstein degree vectors for these " "cones. Required: every primitive extremal ray of C has degree 1 " "under degree_C, and every primitive extremal ray of C_dual has " "degree 1 under degree_C_dual. Observed degree sets: " f"C={sorted(set(map(int, heights_on_C(dC))))}, " f"C_dual={sorted(set(map(int, heights_on_dual(dD))))}")
    cross = primal_rays @ B @ dual_rays.T
    if np.any(cross < 0):
        i, j = map(int, np.argwhere(cross < 0)[0])
        raise ValueError("C_dual is not dual to C for the supplied pairing: " f"pairing of extremal rays ({i},{j}) is {int(cross[i, j])}")
    index = int(dD @ B @ dC)
    if index < 1:
        raise ValueError(f"the Gorenstein index must be positive, got {index}")
    if expected_index is not None and index != int(expected_index):
        raise ValueError(f"expected Gorenstein index {expected_index}, got {index}")
    kw = {"backend": poly_backend} if poly_backend else {}
    primal_support = P if hasattr(P, "faces") else Polytope(_integer_points(P), **kw)
    support_vertices = _integer_points(primal_support)
    if set(map(tuple, support_vertices)) != set(map(tuple, primal_rays)):
        raise ValueError("vertices of P are not exactly the primitive extremal rays of C")
    dual_support = Polytope(dual_rays, **kw)
    if check:
        cone_dim = int(C.dimension())
        if cone_dim != ambient or int(C_dual.dimension()) != ambient:
            raise ValueError("C and C_dual must both be full-dimensional")
        expected_support_dim = ambient - 1
        if int(primal_support.dimension()) != expected_support_dim:
            raise ValueError(f"P must have affine dimension {expected_support_dim}")
        if int(dual_support.dimension()) != expected_support_dim:
            raise ValueError(f"the dual degree-one support must have dimension {expected_support_dim}")
        if np.any(~np.any(cross == 0, axis=0)) or np.any(~np.any(cross == 0, axis=1)):
            raise ValueError("some extremal ray lies on no facet of the dual cone")
    return GorensteinConeData(primal_cone=C, dual_cone=C_dual, primal_support=primal_support, dual_support=dual_support, primal_vertices=support_vertices, dual_vertices=dual_rays, degree_C=dC, degree_C_dual=dD, index=index,)

def complete_split_witness_index_two(opposite_support: Any, degree: Any,) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return a two-point complete-splitting witness, if one exists.
    For a reflexive Gorenstein cone ``C`` of index two, the Batyrev--Nill
    criterion says that ``C`` is completely split exactly when its Gorenstein
    degree element is a sum of two lattice points of the degree-one support of
    ``C.dual()``.  Consequently, to test ``C`` pass the *opposite* support and
    the degree vector that grades ``C``.
    The result is ``(e1, e2)`` with ``e1 + e2 == degree``, or ``None``.  This
    is an exact lattice test; it does not try to recognize a Cayley embedding
    from floating-point affine geometry.
    """
    if hasattr(opposite_support, "points"):
        points = np.asarray(opposite_support.points())
    else:
        try:
            from cytools import Polytope
        except ImportError as e:
            raise ImportError("array input to complete_split_witness_index_two requires CYTools") from e
        points = np.asarray(Polytope(_integer_points(opposite_support)).points())
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("opposite support has no lattice points")
    if not np.all(np.equal(points, np.rint(points))):
        raise ValueError("opposite support returned nonintegral lattice points")
    points = np.asarray(points, dtype=np.int64)
    target = _integer_vector(degree, name="degree", length=points.shape[1])
    point_set = {tuple(map(int, p)) for p in points}
    for first in sorted(point_set):
        second = tuple(int(target[i]) - first[i] for i in range(len(target)))
        if second in point_set and second != first:
            return (np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64),)
    return None

def _affine_dim(v: np.ndarray) -> int:
    if len(v) <= 1:
        return 0
    return int(np.linalg.matrix_rank(np.asarray(v[1:] - v[0], dtype=float)))

@dataclass(frozen=True)
class Face:
    id: int
    vertices: FrozenSet[int]
    dim: int

class FaceLattice:
    """Face poset plus exact Ehrhart and Stanley-polynomial operations."""

    def __init__(self, poly: Any, *, ehrhart_backend: str = "auto", latte_count_bin: str = "count", latte_options: Sequence[str] = ("--redundancy-check=none",),cytools_work_cutoff=100_0):
        if not hasattr(poly, "faces"):
            try:
                from cytools import Polytope
            except ImportError as e:
                raise ImportError("FaceLattice requires CYTools") from e
            poly = Polytope(_integer_points(poly))
        self.source = poly
        self.vertices = _integer_points(poly)
        self.dim = int(poly.dimension())
        if ehrhart_backend not in ("auto", "latte", "normaliz", "cytools"):
            raise ValueError("ehrhart_backend must be 'auto', 'latte', 'normaliz', or 'cytools'")
        self.ehrhart_backend = ehrhart_backend
        self._latte_count_bin = latte_count_bin
        self._latte_options = tuple(latte_options)
        self._latte_executable = shutil.which(latte_count_bin)
        if self._latte_executable is None and Path(latte_count_bin).exists():
            self._latte_executable = str(Path(latte_count_bin).resolve())
        self._normaliz_cone = None
        self._normaliz_executable = None
        if ehrhart_backend == "latte":
            if self._latte_executable is None:
                raise FileNotFoundError(f"ehrhart_backend='latte' requires LattE's " f"{latte_count_bin!r} executable")
        elif ehrhart_backend == "normaliz":
            try:
                from PyNormaliz import Cone as NormalizCone
                self._normaliz_cone = NormalizCone
            except ImportError:
                self._normaliz_executable = shutil.which("normaliz")
                if self._normaliz_executable is None:
                    raise ImportError("ehrhart_backend='normaliz' requires " "PyNormaliz or the normaliz executable")
        self._cytools_work_cutoff = cytools_work_cutoff
        self._ehrhart_usage = {"analytic": 0, "cytools": 0, "latte": 0, "normaliz": 0, "normaliz-cli": 0,}
        face_dims = self._face_sets_cytools(poly)
        face_dims[frozenset()] = -1
        face_dims[frozenset(range(len(self.vertices)))] = self.dim
        raw = sorted(face_dims, key=lambda s: (face_dims[s], tuple(sorted(s))))
        self.faces = tuple(Face(i, s, face_dims[s]) for i, s in enumerate(raw))
        self.by_vertices = {f.vertices: f.id for f in self.faces}
        self.by_dim: Dict[int, Tuple[int, ...]] = {d: tuple(f.id for f in self.faces if f.dim == d) for d in range(-1, self.dim + 1)}
        self.empty = self.by_vertices[frozenset()]
        self.full = self.by_vertices[frozenset(range(len(self.vertices)))]

    def _face_sets_cytools(self, poly: Any) -> Dict[FrozenSet[int], int]:
        lookup = {tuple(map(int, x)): i for i, x in enumerate(self.vertices)}
        ans: Dict[FrozenSet[int], int] = {}
        for dim, level in enumerate(poly.faces()):
            for face in level:
                key = frozenset(lookup[tuple(map(int, x))] for x in np.asarray(face.vertices()))
                ans[key] = dim
        return ans

    @lru_cache(maxsize=None)
    def intrinsic_vertices(self, face_id: int) -> np.ndarray:
        """Vertices of a face in its intrinsic lattice Z^d."""
        face = self.faces[face_id]
        if face.dim <= 0:
            return np.zeros((len(face.vertices), max(face.dim, 0)), dtype=np.int64,)
        verts = self.vertices[list(face.vertices)]
        return _intrinsic_lattice_vertices(verts, face.dim,)

    def _ehrhart_backend_for_face(self, face_id: int, intrinsic: np.ndarray,) -> str:
        """
        Select the Ehrhart backend for one individual face.
        """
        if self.ehrhart_backend != "auto":
            return self.ehrhart_backend
        d = self.faces[face_id].dim
        if self._latte_executable is None:
            return "cytools"
        if _cytools_work_is_small(intrinsic, d, self._cytools_work_cutoff,):
            return "cytools"
        return "latte"

    @lru_cache(maxsize=None)
    def interval(self, lo: int, hi: int) -> Tuple[int, ...]:
        a, b = self.faces[lo].vertices, self.faces[hi].vertices
        if not a.issubset(b):
            raise ValueError("not a face interval")
        return tuple(f.id for f in self.faces if a.issubset(f.vertices) and f.vertices.issubset(b))

    @lru_cache(maxsize=None)
    def g_polynomial(self, lo: int, hi: int) -> Poly1:
        """Stanley's g-polynomial of the Eulerian interval [lo,hi]."""
        rank = self.faces[hi].dim - self.faces[lo].dim
        if rank == 0:
            return (1,)
        h: Poly1 = (0,)
        for x in self.interval(lo, hi):
            if x == lo:
                continue
            exponent = self.faces[x].dim - self.faces[lo].dim - 1
            h = _add(h, _mul(_t_minus_one_pow(exponent), self.g_polynomial(x, hi)))
        one_minus_t_h = _mul((1, -1), h)
        cutoff = (rank - 1) // 2
        return _trim(one_minus_t_h[: cutoff + 1])

    def _count_dilate_cytools(self, verts: np.ndarray, k: int) -> int:
        try:
            from cytools import Polytope
        except ImportError as e:
            raise ImportError("lattice-point enumeration requires CYTools") from e
        try:
            return int(len(Polytope(k * verts).points()))
        except Exception as e:
            raise RuntimeError("CYTools failed while enumerating lattice points of a dilated face") from e

    @staticmethod
    def _h_star_from_series(series: Any, d: int) -> Poly1:
        """Decode a Normaliz series and recover h* exactly through degree d."""
        if not isinstance(series, (list, tuple)) or len(series) != 3:
            raise RuntimeError(f"unexpected Normaliz Ehrhart-series output: {series!r}")
        numerator, denominator_exponents, shift = series
        numerator = [int(x) for x in numerator]
        denominator_exponents = [int(x) for x in denominator_exponents]
        shift = int(shift)
        inverse_denominator = [0] * (d + 1)
        inverse_denominator[0] = 1
        for exponent in denominator_exponents:
            if exponent <= 0:
                raise RuntimeError("Normaliz returned a nonpositive denominator exponent")
            for k in range(exponent, d + 1):
                inverse_denominator[k] += inverse_denominator[k - exponent]
        counts = []
        for k in range(d + 1):
            counts.append(sum(coefficient * inverse_denominator[k - shift - i] for i, coefficient in enumerate(numerator) if 0 <= k - shift - i <= d))
        return _h_star_from_counts(counts, d)

    def _h_star_latte(self, verts: np.ndarray, d: int) -> Poly1:
        """Compute h* from one exact LattE Ehrhart-polynomial call."""
        coefficients = ehrhart_polynomial_latte_vrep(verts, intrinsic_dim=d, count_bin=self._latte_executable or self._latte_count_bin, latte_options=self._latte_options,)
        return _h_star_from_counts(_ehrhart_values_from_coefficients(coefficients, d), d)

    def _h_star_normaliz(self, verts: np.ndarray, d: int) -> Poly1:
        """Compute h* with the in-process PyNormaliz binding."""
        if self._normaliz_cone is None:
            raise RuntimeError("PyNormaliz backend was not initialized")
        cone = self._normaliz_cone(polytope=np.asarray(verts, dtype=int).tolist())
        series = (cone.EhrhartSeries() if hasattr(cone, "EhrhartSeries") else cone.HilbertSeries())
        return self._h_star_from_series(series, d)

    def _h_star_normaliz_cli(self, verts: np.ndarray, d: int) -> Poly1:
        """Compute h* with the Normaliz executable when PyNormaliz is absent."""
        if self._normaliz_executable is None:
            raise RuntimeError("Normaliz executable backend was not initialized")
        vertices = np.asarray(verts, dtype=np.int64)
        with tempfile.TemporaryDirectory(prefix="stringy_e_nmz_") as tmp:
            input_path = f"{tmp}/face.in"
            output_path = f"{tmp}/face.out"
            with open(input_path, "w", encoding="utf-8") as stream:
                stream.write(f"amb_space {vertices.shape[1] + 1}\n")
                stream.write(f"polytope {len(vertices)}\n")
                for row in vertices:
                    stream.write(" ".join(map(str, map(int, row))) + "\n")
            process = subprocess.run([self._normaliz_executable, "-q", input_path], cwd=tmp, capture_output=True, text=True, check=False,)
            if process.returncode != 0 or not Path(output_path).exists():
                raise RuntimeError("Normaliz failed to compute an Ehrhart series: " + (process.stderr.strip() or process.stdout.strip()))
            with open(output_path, encoding="utf-8") as stream:
                lines = stream.readlines()
        start = next((i for i, line in enumerate(lines) if "Hilbert series:" in line or "Ehrhart series:" in line), None,)
        if start is None:
            raise RuntimeError("Normaliz output contains no Ehrhart/Hilbert series")
        numerator = [int(x) for x in lines[start + 1].split()]
        denominator_exponents: List[int] = []
        shift = next((int(line.split("=", 1)[1].strip()) for line in lines[start + 2 :] if "shift" in line and "=" in line), 0,)
        for line in lines[start + 2 :]:
            for exponent, multiplicity in re.findall(r"(-?\d+)\s*:\s*(\d+)", line):
                denominator_exponents.extend([int(exponent)] * int(multiplicity))
            if denominator_exponents and not line.strip():
                break
        if not denominator_exponents:
            raise RuntimeError("could not parse the Normaliz series denominator")
        return self._h_star_from_series([numerator, denominator_exponents, shift], d)

    @lru_cache(maxsize=None)
    def lattice_count(self, face_id: int, k: int) -> int:
        if k < 0:
            raise ValueError("dilation must be nonnegative")
        face = self.faces[face_id]
        if face.dim == -1:
            return 0
        if k == 0:
            return 1
        verts = self.vertices[list(face.vertices)]
        return self._count_dilate_cytools(verts, k)

    @lru_cache(maxsize=None)
    def h_star(self, face_id: int) -> Poly1:
        """Ehrhart h*-polynomial; h*_empty=1."""
        face = self.faces[face_id]
        d = face.dim
        if d == -1:
            return (1,)
        if d == 0:
            return (1,)
        verts = self.vertices[list(face.vertices)]
        cache_key = (d, tuple(sorted(tuple(map(int, row)) for row in verts)),)
        if cache_key in _HSTAR_CACHE:
            return _HSTAR_CACHE[cache_key]
        if d == 1:
            delta = np.asarray(verts[1] - verts[0], dtype=np.int64,)
            lattice_length = int(np.gcd.reduce(np.abs(delta)))
            answer = _trim((1, lattice_length - 1))
            self._ehrhart_usage["analytic"] += 1
        else:
            intrinsic = self.intrinsic_vertices(face_id)
            backend = self._ehrhart_backend_for_face(face_id, intrinsic,)
            if backend == "latte":
                answer = self._h_star_latte(intrinsic, d,)
            elif backend == "normaliz":
                if self._normaliz_cone is not None:
                    answer = self._h_star_normaliz(intrinsic, d,)
                else:
                    answer = self._h_star_normaliz_cli(intrinsic, d,)
                    backend = "normaliz-cli"
            elif backend == "cytools":
                counts = [1]
                for k in range(1, d + 1):
                    counts.append(self._count_dilate_cytools(intrinsic, k,))
                answer = _h_star_from_counts(counts, d,)
            else:
                raise RuntimeError(f"unknown Ehrhart backend {backend!r}")
            self._ehrhart_usage[backend] += 1
        _HSTAR_CACHE[cache_key] = answer
        return answer

    def precompute_s_tilde(self, *, verbose: bool = False, label: str = "") -> None:
        """Populate all h*, g, and S-tilde caches, optionally showing progress."""
        total = len(self.faces)
        report_every = max(1, total // 20)
        for i, face in enumerate(self.faces, 1):
            self.s_tilde(face.id)
            if verbose and (i == 1 or i == total or i % report_every == 0):
                prefix = f"{label}: " if label else ""
                print(f"{prefix}{i}/{total} faces", flush=True)

    @lru_cache(maxsize=None)
    def s_tilde(self, face_id: int) -> Poly1:
        """Borisov--Mavlyutov/Batyrev--Nill S-tilde polynomial."""
        p = self.faces[face_id]
        total: Poly1 = (0,)
        for g in self.interval(self.empty, face_id):
            fg = self.faces[g]
            sign = (-1) ** (p.dim - fg.dim)
            total = _add(total, _scale(_mul(self.h_star(g), self.g_polynomial(g, face_id)), sign))
        return total

def _translated_hstar_key(verts, d):
    """Cheap translation-invariant cache key."""
    v = np.asarray(verts, dtype=np.int64)
    v = v - v[0]
    return (d, tuple(sorted(tuple(map(int, row)) for row in v)),)

def pair_dual_faces(primal: FaceLattice, dual: FaceLattice, pairing: Optional[np.ndarray] = None, *, check: bool = True,) -> Dict[int, int]:
    """Pair faces by annihilation: F* has vertices y with <x,y>=0 for all x in F."""
    if pairing is None:
        if primal.vertices.shape[1] != dual.vertices.shape[1]:
            raise ValueError("coordinate dimensions differ; supply an explicit pairing matrix")
        pair = primal.vertices @ dual.vertices.T
    else:
        b = np.asarray(pairing)
        pair = primal.vertices @ b @ dual.vertices.T
    if not np.all(np.equal(pair, np.rint(pair))):
        raise ValueError("pairings must be integral")
    pair = np.asarray(pair, dtype=np.int64)
    if check and np.any(pair < 0):
        ij = tuple(map(int, np.argwhere(pair < 0)[0]))
        raise ValueError(f"negative primal/dual support pairing at vertex pair {ij}")
    out: Dict[int, int] = {}
    all_dual = frozenset(range(len(dual.vertices)))
    for f in primal.faces:
        if f.dim == -1:
            star = all_dual
        else:
            rows = list(f.vertices)
            star = frozenset(np.flatnonzero(np.all(pair[rows, :] == 0, axis=0)).tolist())
        if star not in dual.by_vertices:
            raise ValueError(f"zero set for primal face {f.id} is not a dual face")
        out[f.id] = dual.by_vertices[star]
        if check and f.dim + dual.faces[out[f.id]].dim != primal.dim - 1:
            raise ValueError(f"dimension-reversing duality fails at primal face {f.id}")
    if check and len(set(out.values())) != len(dual.faces):
        raise ValueError("face pairing is not a bijection; supports/pairing are inconsistent")
    return out

@dataclass
class StringyEResult:
    coefficients: Laurent2
    hodge_numbers: Dict[Tuple[int, int], int]
    index: int
    polytope_dim: int
    cy_dim: int
    primal: FaceLattice
    dual: FaceLattice
    dual_faces: Dict[int, int]
    primal_completely_split: Optional[bool] = None
    dual_completely_split: Optional[bool] = None
    primal_split_witness: Optional[Tuple[np.ndarray, np.ndarray]] = None
    dual_split_witness: Optional[Tuple[np.ndarray, np.ndarray]] = None
    cayley_data: Optional[CayleyGorensteinData] = None
    cone_data: Optional[GorensteinConeData] = None

    def e_polynomial(self) -> str:
        return format_laurent(self.coefficients)

    def hodge_diamond(self, fill: int = 0) -> List[List[int]]:
        return [[self.hodge_numbers.get((p, q), fill) for q in range(self.cy_dim + 1)] for p in range(self.cy_dim + 1)]

    @property
    def formal_hodge_numbers(self) -> Dict[Tuple[int, int], int]:
        """Alias emphasizing that a geometric realization needs a split cone."""
        return self.hodge_numbers

    def interpretation(self) -> str:
        """Describe what the two cone supports geometrically justify."""
        p, q = self.primal_completely_split, self.dual_completely_split
        if p is True and q is True:
            return ("both cones are completely split: this is the usual " "Batyrev--Borisov complete-intersection mirror setting")
        if p is True and q is False:
            return ("the primal cone gives a complete intersection, but the dual " "cone has no complete-intersection realization from a splitting; " "on the dual side the result is the Gorenstein-cone/generalized-CY " "stringy E-invariant")
        if p is False and q is True:
            return ("the dual cone gives a complete intersection, but the primal " "cone has no complete-intersection realization from a splitting")
        if p is False and q is False:
            return ("neither cone is completely split: the result is the formal " "Batyrev--Nill stringy E-invariant of the Gorenstein polytope")
        return "complete-splitting was not tested for this result"

@dataclass
class StringyHodgeResult:
    """Hodge-only result that does not materialize the stringy E-polynomial."""
    hodge_numbers: Dict[Tuple[int, int], int]
    index: int
    polytope_dim: int
    cy_dim: int
    primal: FaceLattice
    dual: FaceLattice
    dual_faces: Dict[int, int]

    def hodge_diamond(self, fill: int = 0) -> List[List[int]]:
        return [[self.hodge_numbers.get((p, q), fill) for q in range(self.cy_dim + 1)] for p in range(self.cy_dim + 1)]

def _accumulate_term(out: Laurent2, a: Poly1, b: Poly1, power_u: int, sign: int, shift: int) -> None:
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            key = (power_u - i + j - shift, i + j - shift)
            out[key] = out.get(key, 0) + sign * x * y
            if out[key] == 0:
                del out[key]

def _accumulate_hodge_term(out: Laurent2, a: Poly1, b: Poly1, power_u: int, shift: int) -> None:
    """Accumulate Hodge numbers directly, without first storing E coefficients.
    For a face of rank ``power_u``, the E-term sign and the extraction sign
    cancel because the parity of the two resulting exponents is
    ``power_u``. Thus each product ``a_i*b_j`` contributes directly.
    """
    for i, x in enumerate(a):
        if not x:
            continue
        for j, y in enumerate(b):
            if not y:
                continue
            key = (power_u - i + j - shift, i + j - shift)
            out[key] = out.get(key, 0) + x * y
            if out[key] == 0:
                del out[key]

def stringy_hodge(primal_poly: Any, dual_poly: Any, *, index: int = 2, pairing: Optional[np.ndarray] = None, check: bool = True, require_nonnegative_hodge: bool = True, ehrhart_backend: str = "auto", latte_count_bin: str = "count", latte_options: Sequence[str] = ("--redundancy-check=none",), verbose: bool = False,cytools_work_cutoff: int = 100_0) -> StringyHodgeResult:
    """Compute only the stringy Hodge numbers/Hodge diamond.
    This uses the same exact facewise S-tilde formula as :func:`stringy_e`,
    but avoids allocating the E-polynomial and avoids assembling the dual
    E-polynomial solely for the optional mirror-reciprocity check.
    """
    if index < 1:
        raise ValueError("index must be positive")
    started = perf_counter()
    p = FaceLattice(primal_poly, ehrhart_backend=ehrhart_backend, latte_count_bin=latte_count_bin, latte_options=latte_options,cytools_work_cutoff=cytools_work_cutoff)
    q = FaceLattice(dual_poly, ehrhart_backend=ehrhart_backend, latte_count_bin=latte_count_bin, latte_options=latte_options,cytools_work_cutoff=cytools_work_cutoff)
    if p.dim != q.dim:
        raise ValueError("dual Gorenstein support polytopes must have equal dimension")
    star = pair_dual_faces(p, q, pairing, check=check)
    if verbose:
        print(f"face lattices: {len(p.faces)} primal, {len(q.faces)} dual; Ehrhart backend: {p.ehrhart_backend}", flush=True)
    cy_dim = p.dim + 1 - 2 * index
    hodge: Laurent2 = {(p_, q_): 0 for p_ in range(cy_dim + 1) for q_ in range(cy_dim + 1)}
    for face in p.faces:
        rank = face.dim + 1
        _accumulate_hodge_term(hodge, p.s_tilde(face.id), q.s_tilde(star[face.id]), rank, index,)
    result = StringyHodgeResult(hodge, index, p.dim, cy_dim, p, q, star)
    if check:
        _check_hodge_result(result, require_nonnegative_hodge=require_nonnegative_hodge)
    if verbose:
        print(f"completed Hodge-only computation in {perf_counter() - started:.2f} seconds", flush=True)
    return result

def stringy_e(primal_poly: Any, dual_poly: Any, *, index: int = 2, pairing: Optional[np.ndarray] = None, check: bool = True, require_nonnegative_hodge: bool = True, ehrhart_backend: str = "auto", latte_count_bin: str = "count", latte_options: Sequence[str] = ("--redundancy-check=none",), verbose: bool = False,cytools_work_cutoff: int = 100_0) -> StringyEResult:
    """Compute the Batyrev--Nill stringy E-function and stringy Hodge numbers."""
    if index < 1:
        raise ValueError("index must be positive")
    started = perf_counter()
    p = FaceLattice(primal_poly, ehrhart_backend=ehrhart_backend, latte_count_bin=latte_count_bin, latte_options=latte_options,cytools_work_cutoff=cytools_work_cutoff)
    q = FaceLattice(dual_poly, ehrhart_backend=ehrhart_backend, latte_count_bin=latte_count_bin, latte_options=latte_options,cytools_work_cutoff=cytools_work_cutoff)
    if p.dim != q.dim:
        raise ValueError("dual Gorenstein support polytopes must have equal dimension")
    star = pair_dual_faces(p, q, pairing, check=check)
    if verbose:
        print(f"face lattices: {len(p.faces)} primal, {len(q.faces)} dual; " f"Ehrhart backend: {p.ehrhart_backend}", flush=True,)
    p.precompute_s_tilde(verbose=verbose, label="primal")
    q.precompute_s_tilde(verbose=verbose, label="dual")
    e: Laurent2 = {}
    for f in p.faces:
        k = f.dim + 1
        _accumulate_term(e, p.s_tilde(f.id), q.s_tilde(star[f.id]), k, (-1) ** k, index)
    cy_dim = p.dim + 1 - 2 * index
    hodge = {(a, b): (-1) ** (a + b) * c for (a, b), c in e.items()}
    result = StringyEResult(e, hodge, index, p.dim, cy_dim, p, q, star)
    if check:
        _check_result(result, require_nonnegative_hodge=require_nonnegative_hodge)
    if verbose:
        print(f"completed in {perf_counter() - started:.2f} seconds", flush=True)
    return result

def stringy_e_from_summands(delta1: Any, delta2: Any, *, interior_point: Optional[Sequence[int]] = None, index: int = 2, poly_backend: Optional[str] = None, check: bool = True, require_nonnegative_hodge: bool = True, ehrhart_backend: str = "auto", latte_count_bin: str = "count", latte_options: Sequence[str] = ("--redundancy-check=none",), verbose: bool = False,) -> StringyEResult:
    """Compatibility entry point that constructs cones from two summands.
    This constructs and dualizes the Cayley cone natively, verifies that the
    dual rays form a degree-one Gorenstein support, and computes its stringy
    E-function.  ``index=2`` is retained as an argument only for explicitness
    and testing; a two-part Cayley Gorenstein construction should use 2.
    """
    if check and index != 2:
        raise ValueError("a two-part Cayley support has Gorenstein index 2")
    data = cayley_gorenstein_data(delta1, delta2, interior_point=interior_point, poly_backend=poly_backend, check=check,)
    result = stringy_e(data.primal_support, data.dual_support, index=index, check=check, require_nonnegative_hodge=require_nonnegative_hodge, ehrhart_backend=ehrhart_backend, latte_count_bin=latte_count_bin, latte_options=latte_options, verbose=verbose,)
    result.cayley_data = data
    return result

def stringy_hodge_from_summands(delta1: Any, delta2: Any, *, interior_point: Optional[Sequence[int]] = None, index: int = 2, poly_backend: Optional[str] = None, check: bool = True, require_nonnegative_hodge: bool = True, ehrhart_backend: str = "auto", latte_count_bin: str = "count", latte_options: Sequence[str] = ("--redundancy-check=none",), verbose: bool = False,) -> StringyHodgeResult:
    """Hodge-only counterpart of :func:`stringy_e_from_summands`."""
    if check and index != 2:
        raise ValueError("a two-part Cayley support has Gorenstein index 2")
    data = cayley_gorenstein_data(delta1, delta2, interior_point=interior_point, poly_backend=poly_backend, check=check,)
    return stringy_hodge(data.primal_support, data.dual_support, index=index, check=check, require_nonnegative_hodge=require_nonnegative_hodge, ehrhart_backend=ehrhart_backend, latte_count_bin=latte_count_bin, latte_options=latte_options, verbose=verbose,)

def stringy_e_from_cones(P: Any, C: Any, C_dual: Any, degree_C: Any, degree_C_dual: Any, *, pairing: Optional[np.ndarray] = None, expected_index: Optional[int] = None, poly_backend: Optional[str] = None, check: bool = True, require_nonnegative_hodge: bool = True, ehrhart_backend: str = "auto", latte_count_bin: str = "count", latte_options: Sequence[str] = ("--redundancy-check=none",), verbose: bool = False,) -> StringyEResult:
    """Compute the stringy E-function from fully precomputed Cayley cone data.
    No Cayley cone, dual cone, Gorenstein degree vector, or index is
    recomputed.  Only primitive extremal rays are requested from the supplied
    cones, since these are needed to construct and validate the two support
    polytopes.
    """
    data = validate_gorenstein_cone_data(P, C, C_dual, degree_C, degree_C_dual, pairing=pairing, expected_index=expected_index, poly_backend=poly_backend, check=check,)
    result = stringy_e(data.primal_support, data.dual_support, index=data.index, pairing=pairing, check=check, require_nonnegative_hodge=require_nonnegative_hodge, ehrhart_backend=ehrhart_backend, latte_count_bin=latte_count_bin, latte_options=latte_options, verbose=verbose,)
    result.cone_data = data
    if data.index == 2:
        result.primal_split_witness = complete_split_witness_index_two(data.dual_support, data.degree_C)
        result.dual_split_witness = complete_split_witness_index_two(data.primal_support, data.degree_C_dual)
        result.primal_completely_split = result.primal_split_witness is not None
        result.dual_completely_split = result.dual_split_witness is not None
        if verbose:
            print("complete splitting: " f"C={result.primal_completely_split}, " f"C_dual={result.dual_completely_split}", flush=True,)
            print(result.interpretation(), flush=True)
    return result

def stringy_hodge_from_cones(P: Any, C: Any, C_dual: Any, degree_C: Any, degree_C_dual: Any, *, pairing: Optional[np.ndarray] = None, expected_index: Optional[int] = None, poly_backend: Optional[str] = None, check: bool = True, require_nonnegative_hodge: bool = True, ehrhart_backend: str = "auto", latte_count_bin: str = "count", latte_options: Sequence[str] = ("--redundancy-check=none",), verbose: bool = False,) -> StringyHodgeResult:
    """Compute only the Hodge diamond from precomputed Cayley cone data."""
    data = validate_gorenstein_cone_data(P, C, C_dual, degree_C, degree_C_dual, pairing=pairing, expected_index=expected_index, poly_backend=poly_backend, check=check,)
    return stringy_hodge(data.primal_support, data.dual_support, index=data.index, pairing=pairing, check=check, require_nonnegative_hodge=require_nonnegative_hodge, ehrhart_backend=ehrhart_backend, latte_count_bin=latte_count_bin, latte_options=latte_options, verbose=verbose,)
stringy_e_cayley = stringy_e_from_cones

def _check_hodge_result(r: StringyHodgeResult, *, require_nonnegative_hodge: bool) -> None:
    if r.cy_dim < 0:
        raise ValueError("negative Calabi--Yau dimension")
    bad = [key for key in r.hodge_numbers if min(key) < 0 or max(key) > r.cy_dim]
    if bad:
        raise ValueError(f"Hodge numbers fall outside the CY range: {bad[:5]}")
    for (p, q), value in r.hodge_numbers.items():
        if r.hodge_numbers.get((q, p), 0) != value:
            raise ValueError("Hodge symmetry check failed")
        if r.hodge_numbers.get((r.cy_dim - p, r.cy_dim - q), 0) != value:
            raise ValueError("Poincare duality check failed")
    if require_nonnegative_hodge and any(value < 0 for value in r.hodge_numbers.values()):
        raise ValueError("negative extracted stringy Hodge numbers")

def _check_result(r: StringyEResult, *, require_nonnegative_hodge: bool) -> None:
    if r.cy_dim < 0:
        raise ValueError("negative Calabi--Yau dimension")
    bad = [k for k in r.coefficients if min(k) < 0 or max(k) > r.cy_dim]
    if bad:
        raise ValueError(f"E-function did not become a CY-range polynomial; bad exponents: {bad[:5]}")
    for (a, b), c in r.coefficients.items():
        if r.coefficients.get((b, a), 0) != c:
            raise ValueError("u-v symmetry check failed")
        if r.coefficients.get((r.cy_dim - a, r.cy_dim - b), 0) != c:
            raise ValueError("Poincare duality check failed")
    inverse_star = {dual_id: primal_id for primal_id, dual_id in r.dual_faces.items()}
    if len(inverse_star) != len(r.dual.faces):
        raise ValueError("cannot verify mirror reciprocity: face duality is not bijective")
    dual_e: Laurent2 = {}
    for face in r.dual.faces:
        k = face.dim + 1
        _accumulate_term(dual_e, r.dual.s_tilde(face.id), r.primal.s_tilde(inverse_star[face.id]), k, (-1) ** k, r.index,)
    expected_from_dual = {(r.cy_dim - a, b): (-1) ** r.cy_dim * coefficient for (a, b), coefficient in dual_e.items() if coefficient}
    if expected_from_dual != r.coefficients:
        raise ValueError("primal/dual mirror-reciprocity check failed")
    if require_nonnegative_hodge:
        bad_h = {k: v for k, v in r.hodge_numbers.items() if v < 0}
        if bad_h:
            raise ValueError(f"negative extracted stringy Hodge numbers: {bad_h}")

def extract_hodge_numbers(e: Mapping[Tuple[int, int], int], *, check_nonnegative: bool = True) -> Dict[Tuple[int, int], int]:
    """Extract h^{p,q} from E=sum (-1)^(p+q) h^{p,q} u^p v^q."""
    h = {(p, q): (-1) ** (p + q) * int(c) for (p, q), c in e.items() if c}
    if check_nonnegative and any(x < 0 for x in h.values()):
        raise ValueError("coefficients do not define nonnegative stringy Hodge numbers")
    return h

def format_laurent(poly: Mapping[Tuple[int, int], int]) -> str:
    """Human-readable exact two-variable Laurent polynomial."""
    if not poly:
        return "0"
    terms: List[str] = []
    for (a, b), c in sorted(poly.items(), key=lambda z: (sum(z[0]), z[0]), reverse=True):
        factors = (["u" if a == 1 else f"u^{a}"] if a else []) + (["v" if b == 1 else f"v^{b}"] if b else [])
        mon = "*".join(factors)
        mag = abs(c)
        body = mon if mag == 1 and mon else str(mag) + (("*" + mon) if mon else "")
        terms.append(("- " if c < 0 else "+ ") + body)
    s = " ".join(terms)
    return s[2:] if s.startswith("+ ") else "-" + s[2:] if s.startswith("- ") else s

def check_two_part_cayley(points: Any, *, base_dim: int = 6) -> None:
    """Strong shape/height check for the convention (Delta_i,e_i)."""
    v = _integer_points(points)
    if v.shape[1] != base_dim + 2:
        raise ValueError(f"expected {base_dim + 2} coordinates")
    heights = {tuple(map(int, x[-2:])) for x in v}
    if not heights.issubset({(1, 0), (0, 1)}) or heights != {(1, 0), (0, 1)}:
        raise ValueError("Cayley heights must be exactly (1,0) and (0,1), both present")
    if _affine_dim(v) != base_dim + 1:
        raise ValueError(f"expected affine dimension {base_dim + 1}")
__all__ = ["CayleyGorensteinData", "GorensteinConeData", "Face", "FaceLattice", "StringyEResult", "StringyHodgeResult", "cayley_support", "cayley_support_2part", "cayley_gorenstein_data", "check_two_part_cayley", "clear_hstar_cache", "complete_split_witness_index_two", "dual_support_vertices_cytools", "ehrhart_polynomial_latte_vrep", "extract_hodge_numbers", "format_laurent", "hstar_cache_size", "pair_dual_faces", "stringy_e", "stringy_e_cayley", "stringy_e_from_cones", "stringy_e_from_summands", "stringy_hodge", "stringy_hodge_from_cones", "stringy_hodge_from_summands", "validate_gorenstein_cone_data", "write_latte_vrep",]

def get_E_function(F, backend="auto", verbose=False, hodge_only=True):
    CM = Cayley_M(F)
    ConeM = Cone(CM.vertices())
    ConeN = ConeM.dual()
    GM = is_Gorenstein(ConeM)[1]
    GN = is_Gorenstein(ConeN)[1]
    compute = stringy_hodge_from_cones if hodge_only else stringy_e_from_cones
    return compute(CM, ConeM, ConeN, GM, GN, ehrhart_backend=backend, verbose=verbose)
import numpy as np
from itertools import product

def _integer_solve(B, b, atol=1e-8):
    """
    Solve B @ x = b numerically and return x if it is sufficiently
    close to an integer vector.
    """
    B = np.asarray(B, dtype=float)
    b = np.asarray(b, dtype=float)
    try:
        x = np.linalg.solve(B, b)
    except np.linalg.LinAlgError:
        return None
    x_int = np.rint(x).astype(int)
    if not np.allclose(x, x_int, atol=atol, rtol=0):
        return None
    if not np.allclose(B @ x_int, b, atol=atol, rtol=0):
        return None
    return x_int

def sums_to_anticanonical(points, divisors, atol=1e-8):
    """
    Determine whether L1 + ... + Lr is linearly equivalent to
    the anticanonical divisor.
    """
    points = np.asarray(points, dtype=int)
    divisors = np.asarray(divisors, dtype=int)
    if divisors.ndim != 2:
        raise ValueError("divisors must have shape (r, n_rays).")
    if divisors.shape[1] != points.shape[0]:
        raise ValueError("Each divisor must have one coefficient for every ray.")
    total = np.sum(divisors, axis=0)
    b = 1 - total
    if np.all(b == 0):
        return True, np.zeros(points.shape[1], dtype=int)
    dim = points.shape[1]
    basis_indices = np.asarray(basis(points), dtype=int)
    if len(basis_indices) != dim:
        raise ValueError("The rays do not span the full ambient lattice.")
    B = points[basis_indices]
    bB = b[basis_indices]
    m = _integer_solve(B, bB, atol=atol)
    if m is None:
        return False, None
    if np.array_equal(points @ m, b):
        return True, m
    return False, None

def is_partition(points, divisors, atol=1e-8):
    """
    Determine whether divisors L1,...,Lr can be shifted by principal
    divisors to form a partition of the anticanonical divisor.
    This version avoids repeated np.linalg.solve calls by computing
    the inverse of the basis matrix only once.
    """
    points = np.asarray(points, dtype=int)
    divisors = np.asarray(divisors, dtype=int)
    if divisors.ndim != 2:
        raise ValueError("divisors must have shape (r, n_rays).")
    r, n_rays = divisors.shape
    dim = points.shape[1]
    if n_rays != points.shape[0]:
        raise ValueError("Each divisor must have one coefficient for every ray.")
    sta, total_shift = sums_to_anticanonical(points, divisors, atol=atol)
    if not sta:
        return False, False, np.zeros((r, dim), dtype=int)
    basis_indices = np.asarray(basis(points), dtype=int)
    if len(basis_indices) != dim:
        raise ValueError("The rays do not span the full ambient lattice.")
    B = points[basis_indices].astype(float)
    try:
        B_inv = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        return False, True, np.zeros((r, dim), dtype=int)
    divisors_B = divisors[:, basis_indices]
    for assignment in product(range(r), repeat=dim):
        assignment = np.asarray(assignment)
        shifts = np.zeros((r, dim), dtype=int)
        valid = True
        for a in range(r - 1):
            target = (assignment == a).astype(float)
            rhs = target - divisors_B[a]
            m_float = B_inv @ rhs
            m = np.rint(m_float).astype(int)
            if not np.allclose(m_float, m, atol=atol, rtol=0):
                valid = False
                break
            shifts[a] = m
        if not valid:
            continue
        shifts[-1] = (total_shift - np.sum(shifts[:-1], axis=0))
        shifted = divisors + shifts @ points.T
        if not np.all((shifted == 0) | (shifted == 1)):
            continue
        if not np.all(np.sum(shifted, axis=0) == 1):
            continue
        return True, True, shifts
    shifts = np.zeros((r, dim), dtype=int)
    shifts[-1] = total_shift
    return False, True, shifts

def three_cone_intersections(dual_poly, fan, cones=None):
    verts = dual_poly.vertices()
    max_cones = [tuple(sorted(c)) for c in fan.cones()]
    if cones is None:
        cones = get_lower_dimensional_cones(max_cones, 3)
    cones = [tuple(sorted(c)) for c in cones]
    adjacent = defaultdict(list)
    for mc in max_cones:
        for c in combinations(mc, 3):
            adjacent[tuple(sorted(c))].append(mc)
    tot_intersection = 0
    endpoint_O3s = set()
    for c in cones:
        rays = fan.vectors(c)
        face_verts = verts[np.all(verts @ rays.T == -1, axis=1)]
        if len(face_verts) == 2:
            tot_intersection += int(np.gcd.reduce(np.abs(face_verts[1] - face_verts[0])))
        elif len(face_verts) == 0:
            raise ValueError(f"3-cone {c} is contained in the CY; " "this is not an isolated O3 intersection.")
        for mc in adjacent[c]:
            extra = [r for r in mc if r not in c]
            if len(extra) != 1:
                raise ValueError("Expected a simplicial 4-dimensional fan.")
            v = fan.vectors(extra)[0]
            survives = np.any(face_verts @ v == -1)
            if not survives:
                endpoint_O3s.add(mc)
    return tot_intersection + len(endpoint_O3s)

def chi_fourfold_uplift(F):
    chi_loc = cicy4_chern_localization_uplift(F,4)
    n_O3=three_cone_intersections(F.polytope().dual(),F.CY_ambient_toric_fan(),Z2_fixed_locus(F.CY_ambient_toric_fan(),F.xi(),3))
    return chi_loc + n_O3*6

def cicy4_chern_localization_divisors(fan, divisors, n, labels=None, seed=12345, n_checks=3):
    """
    Compute selected pure-toric Chern localization data for a codimension-two
    complete-intersection fourfold in a six-dimensional toric variety.
    Parameters
    ----------
    fan :
        Six-dimensional simplicial ambient toric fan.
    divisors :
        Two divisor coefficient arrays [D1, D2].
    n : int or iterable of int
        Requested Chern degrees, chosen from 2, 3, and 4.
    labels : iterable of int or None
        Optional one-based ray labels for the c2 and c3 pairings.
        None uses every prime toric divisor.
    seed : int
        Seed for the generic localization vectors.
    n_checks : int
        Number of localization vectors used to check stability.
    Returns
    -------
    For scalar n:
        n=2: {(i,j): integral_Y c2(Y) D_i D_j}
        n=3: {i: integral_Y c3(Y) D_i}
        n=4: integral_Y c4(Y)
    For iterable n:
        Dictionary keyed by the requested Chern degrees.
    Notes
    -----
    Ray labels are one-based. Missing keys in the sparse c2 and c3
    dictionaries have value zero. No O3-plane correction is included.
    """
    scalar_input = np.isscalar(n) or (isinstance(n, np.ndarray) and n.ndim == 0)
    raw_degrees = [n.item() if isinstance(n, np.ndarray) else n] if scalar_input else list(n)
    if not raw_degrees:
        raise ValueError("n must contain at least one Chern degree.")
    degree_list = []
    for value in raw_degrees:
        degree = int(value)
        if float(value) != degree:
            raise ValueError("Chern degrees must be integers.")
        if degree not in (2, 3, 4):
            raise ValueError("Only Chern degrees 2, 3, and 4 are supported.")
        if degree not in degree_list:
            degree_list.append(degree)
    degrees = tuple(degree_list)
    if not isinstance(n_checks, (int, np.integer)) or n_checks < 1:
        raise ValueError("n_checks must be a positive integer.")
    vectors = np.asarray(fan.vectors(), dtype=int)
    if vectors.ndim != 2:
        raise ValueError("fan.vectors() must return a 2d array.")
    n_vectors, dim = vectors.shape
    if dim != 6:
        raise ValueError(f"Expected a 6-dimensional fan, got dimension {dim}.")
    if len(divisors) != 2:
        raise ValueError("divisors must be [D1,D2].")
    D1 = np.asarray(divisors[0], dtype=float)
    D2 = np.asarray(divisors[1], dtype=float)
    if D1.shape != (n_vectors,):
        raise ValueError(f"D1 has shape {D1.shape}, but expected ({n_vectors},).")
    if D2.shape != (n_vectors,):
        raise ValueError(f"D2 has shape {D2.shape}, but expected ({n_vectors},).")
    if labels is None:
        selected_labels = tuple(range(1, n_vectors + 1))
    else:
        converted_labels = []
        for value in labels:
            label = int(value)
            if float(value) != label:
                raise ValueError("Toric ray labels must be integers.")
            converted_labels.append(label)
        selected_labels = tuple(dict.fromkeys(converted_labels))
        if not selected_labels:
            raise ValueError("labels must not be empty.")
        if min(selected_labels) < 1 or max(selected_labels) > n_vectors:
            raise ValueError(f"labels must lie in 1,...,{n_vectors}.")
    selected_set = set(selected_labels)
    cones = [tuple(int(i) for i in cone) for cone in fan.cones() if len(cone) == dim]
    if not cones:
        raise ValueError("No 6-dimensional cones found.")
    n_cones = len(cones)
    cone_labels = np.asarray(cones, dtype=int)
    if cone_labels.min() < 1 or cone_labels.max() > n_vectors:
        raise ValueError(f"Unexpected cone vector labels. Found range [{cone_labels.min()}, {cone_labels.max()}], but expected 1,...,{n_vectors}.")
    cone_rays = np.asarray([fan.vectors(cone) for cone in cones], dtype=float)
    if cone_rays.shape != (n_cones, dim, dim):
        raise ValueError(f"Unexpected shape returned by fan.vectors(c): {cone_rays.shape}")
    V = np.transpose(cone_rays, (0, 2, 1))
    cone_indices = cone_labels - 1
    d1_sigma = D1[cone_indices]
    d2_sigma = D2[cone_indices]
    dets = np.abs(np.linalg.det(V))
    multiplicities_float = np.rint(dets)
    if not np.allclose(dets, multiplicities_float, rtol=1e-8, atol=1e-8):
        raise RuntimeError("Could not reliably determine integral cone multiplicities.")
    multiplicities = multiplicities_float.astype(int)
    if np.any(multiplicities == 0):
        raise RuntimeError("Found a degenerate maximal cone.")
    rng = np.random.default_rng(seed)
    answers = {degree: [] for degree in degrees}
    max_degree = max(degrees)
    attempts = 0
    while len(answers[degrees[0]]) < n_checks:
        attempts += 1
        if attempts > 100:
            raise RuntimeError("Could not find sufficiently generic localization vectors.")
        lam = rng.integers(-50, 51, size=dim).astype(float)
        if np.all(lam == 0):
            continue
        rhs = np.broadcast_to(lam, (n_cones, dim))[..., None]
        try:
            w = np.linalg.solve(V, rhs)[..., 0]
        except np.linalg.LinAlgError:
            continue
        if np.min(np.abs(w)) < 1e-10:
            continue
        A = np.sum(d1_sigma * w, axis=1)
        B = np.sum(d2_sigma * w, axis=1)
        chern = np.zeros((n_cones, max_degree + 1), dtype=float)
        chern[:, 0] = 1.0
        for j in range(dim):
            for k in range(max_degree, 0, -1):
                chern[:, k] += w[:, j] * chern[:, k - 1]
        for root in (A, B):
            quotient = np.empty_like(chern)
            quotient[:, 0] = chern[:, 0]
            for k in range(1, max_degree + 1):
                quotient[:, k] = chern[:, k] - root * quotient[:, k - 1]
            chern = quotient
        fundamental_prefactor = A * B / (multiplicities * np.prod(w, axis=1))
        if 2 in answers:
            c2_prefactor = fundamental_prefactor * chern[:, 2]
            c2_terms = defaultdict(list)
            for cone_number in range(n_cones):
                local = [(position, int(label)) for position, label in enumerate(cone_labels[cone_number]) if int(label) in selected_set]
                for p, (position_i, label_i) in enumerate(local):
                    for position_j, label_j in local[p:]:
                        key = tuple(sorted((label_i, label_j)))
                        value = c2_prefactor[cone_number] * w[cone_number, position_i] * w[cone_number, position_j]
                        c2_terms[key].append(value)
            answers[2].append({key: math.fsum(values) for key, values in c2_terms.items()})
        if 3 in answers:
            c3_prefactor = fundamental_prefactor * chern[:, 3]
            c3_terms = defaultdict(list)
            for cone_number in range(n_cones):
                for position, label in enumerate(cone_labels[cone_number]):
                    label = int(label)
                    if label in selected_set:
                        value = c3_prefactor[cone_number] * w[cone_number, position]
                        c3_terms[label].append(value)
            answers[3].append({key: math.fsum(values) for key, values in c3_terms.items()})
        if 4 in answers:
            answers[4].append(math.fsum((fundamental_prefactor * chern[:, 4]).tolist()))
    results = {}
    spreads = {}
    for degree in degrees:
        if degree == 4:
            mean = math.fsum(answers[4]) / n_checks
            spreads[4] = max(abs(value - mean) for value in answers[4])
            nearest_integer = round(mean)
            results[4] = int(nearest_integer) if abs(mean - nearest_integer) < 1e-6 else float(mean)
            continue
        keys = set().union(*(answer.keys() for answer in answers[degree]))
        means = {key: math.fsum(answer.get(key, 0.0) for answer in answers[degree]) / n_checks for key in sorted(keys)}
        spreads[degree] = max((abs(answer.get(key, 0.0) - means[key]) for answer in answers[degree] for key in keys), default=0.0)
        cleaned = {}
        for key, value in means.items():
            if abs(value) < 1e-8:
                continue
            nearest_integer = round(value)
            cleaned[key] = int(nearest_integer) if abs(value - nearest_integer) < 1e-6 else float(value)
        results[degree] = cleaned
    spread = max(spreads.values(), default=0.0)
    if spread > 1e-5:
        print("WARNING: result depends noticeably on the localization vector.")
    return results[degrees[0]] if scalar_input else results

def cicy4_chern_localization_uplift(F, n, labels=None, seed=12345, n_checks=3,):
    """
    Apply cicy4_chern_localization_divisors directly to an F-theory uplift.
    """
    fan = F.smooth_uplift_ambient_toric_fan()
    divisors = [F.line_bundle_base_N(), F.line_bundle_weierstrass_N()]
    return cicy4_chern_localization_divisors(fan, divisors, n, labels=labels, seed=seed, n_checks=n_checks)

def intersecting_points(pts, polytopes):
    """
    Lattice points pts whose toric divisors intersect
    the generic nef-partition CICY with Newton polytopes p1,...,pk.
    """
    vertices = [p.vertices() for p in polytopes]
    out = []
    for v in pts[np.any(pts != 0, axis=1)]:
        faces = [V[V @ v == np.min(V @ v)] for V in vertices]
        intersects = True
        for r in range(1, len(faces) + 1):
            for inds in combinations(range(len(faces)), r):
                diffs = [faces[i][1:] - faces[i][0] for i in inds]
                diffs = [d for d in diffs if len(d)]
                dim = (np.linalg.matrix_rank(np.vstack(diffs))if diffs else 0)
                if dim < r:
                    intersects = False
                    break
            if not intersects:
                break
        if intersects:
            out.append(v)
    return np.array(out, dtype=int)

def cicy_intersection_numbers_in_basis(fan, L1, L2, basis=None, tol=0.00001,decimals=10):
    """
    Intersection numbers of the codim-2 CICY
        X = {L1 = 0} ∩ {L2 = 0}
    in a supplied prime-toric-divisor basis.
    Parameters
    ----------
    fan : CYTools Fan
        Ambient toric fan. Ray labels must be 1,...,N.
    L1, L2 : array-like, shape (N,)
        Coefficients of the defining divisors in the prime divisor basis.
    basis : array-like
        Labels of the prime toric divisors forming the divisor basis,
        e.g. [2, 5, 7, 9, 11, 13].
    tol : float or None
        If given, discard output entries with absolute value <= tol.
    Returns
    -------
    dict
        CICY intersection tensor in the supplied divisor basis.
        Output indices are 1,...,h. Thus, for
            basis = [2,5,7,9,11,13],
        the key
            (1,1,2,6)
        means
            ∫_X D_2 D_2 D_5 D_13.
    """
    if basis is None:
        basis = basis_H2_toric_fan(fan)
    labels = np.asarray(fan.vc.labels, dtype=int)
    N = len(labels)
    if not np.array_equal(labels, np.arange(1, N + 1)):
        raise ValueError("Fan vectors must be labelled 1,...,N.")
    if set(fan.used_labels) != set(labels):
        raise ValueError("All VectorConfiguration vectors must be rays of the fan.")
    L1 = np.asarray(L1)
    L2 = np.asarray(L2)
    basis = np.asarray(basis, dtype=int)
    if L1.shape != (N,) or L2.shape != (N,):
        raise ValueError("L1 and L2 must have one entry per toric divisor.")
    if len(np.unique(basis)) != len(basis):
        raise ValueError("Basis labels must be distinct.")
    if np.any((basis < 1) | (basis > N)):
        raise ValueError("Basis contains invalid divisor labels.")
    K = fan.intersection_numbers(symmetrize=False, eps=0.0, digits=None,)
    in_basis = np.zeros(N + 1, dtype=bool)
    in_basis[basis] = True
    basis_ind = np.zeros(N + 1, dtype=int)
    basis_ind[basis] = np.arange(1, len(basis) + 1)
    c1 = np.concatenate(([0], L1))
    c2 = np.concatenate(([0], L2))
    if np.count_nonzero(c2) < np.count_nonzero(c1):
        c1, c2 = c2, c1
    out = defaultdict(float)
    for key, val in K.items():
        if 0 in key:
            continue
        n_nonbasis = sum(not in_basis[i] for i in key)
        if n_nonbasis > 2:
            continue
        previous = None
        for p, i in enumerate(key):
            if i == previous:
                continue
            previous = i
            ci = c1[i]
            if ci == 0:
                continue
            n_nonbasis_1 = n_nonbasis - int(not in_basis[i])
            if n_nonbasis_1 > 1:
                continue
            rem1 = key[:p] + key[p + 1:]
            previous2 = None
            for q, j in enumerate(rem1):
                if j == previous2:
                    continue
                previous2 = j
                cj = c2[j]
                if cj == 0:
                    continue
                if n_nonbasis_1 - int(not in_basis[j]) != 0:
                    continue
                rem2 = rem1[:q] + rem1[q + 1:]
                newkey = tuple(sorted(basis_ind[k] for k in rem2))
                out[newkey] += val * ci * cj
    if tol is None:
        return dict(out)
    return {k: np.round(v,decimals) for k, v in out.items() if abs(v) > tol}

def intersection_number_surfaces(IN,basis_len):
    new_IN={}
    in_basis_tuple=tuple(range(1,basis_len+1))
    ct=-1
    surface_dict={}
    for c in combinations(in_basis_tuple,2):
        ct+=1
        surface_dict[ct]=tuple(np.array(c)-1)
        for prod in product(in_basis_tuple,in_basis_tuple):
            tu=tuple(sorted(c+prod))
            if tu in IN.keys():
                tt=tuple((ct,)+tuple(np.array(sorted(prod))-1))
                new_IN[tt]=IN.get(tu)
    return new_IN,surface_dict

def transform_intersections(IN, Q,basis, tol=1e-12):
    """
    Transform intersection numbers under
        t_old = Minv @ u_new.
    Parameters
    ----------
    IN : dict
        Symmetric intersection-number dictionary.
        Keys are sorted tuples with indices 1,...,h, e.g.
            (1, 1, 2, 5): value
    Minv : (h,h) array
        Change-of-basis matrix such that
            t_old = Minv @ u_new.
    Returns
    -------
    dict
        Intersection numbers in the new basis,
        again with sorted keys labelled 1,...,h.
    """
    M=Q[:,basis-1]
    Minv=np.round(np.linalg.inv(M)).astype(int)
    A = np.asarray(Minv)
    h = A.shape[0]
    if A.shape != (h, h):
        raise ValueError("Minv must be square.")
    if not IN:
        return {}
    d = len(next(iter(IN)))
    rows = [[(j + 1, A[i, j]) for j in range(h) if abs(A[i, j]) > tol] for i in range(h)]
    out = defaultdict(float)
    for key, kappa in IN.items():
        counts = Counter(key)
        mult = factorial(d)
        for n in counts.values():
            mult //= factorial(n)
        poly = {(): kappa * mult}
        for i in key:
            newpoly = defaultdict(float)
            for mon, coeff in poly.items():
                for j, a in rows[i - 1]:
                    newpoly[tuple(sorted(mon + (j,)))] += coeff * a
            poly = newpoly
        for mon, coeff in poly.items():
            out[mon] += coeff
    result = {}
    for key, coeff in out.items():
        counts = Counter(key)
        mult = factorial(d)
        for n in counts.values():
            mult //= factorial(n)
        val = coeff / mult
        if abs(val) > tol:
            if np.isclose(val, round(val), atol=tol):
                val = int(round(val))
            result[key] = val
    return result

def mori_cone_in_glsm_basis(fan, Q, tol=1e-10):
    """
    Mori cone of a secondary-fan phase, expressed in the
    basis given by the rows of Q.
    """
    Q = np.asarray(Q, dtype=float)
    H = np.asarray(fan.secondary_cone_hyperplanes(), dtype=float)
    if Q.ndim != 2 or H.ndim != 2:
        raise ValueError("Q and H must be matrices.")
    if Q.shape[1] != H.shape[1]:
        raise ValueError("Q and secondary-cone normals have incompatible sizes.")
    if np.linalg.matrix_rank(Q, tol) != Q.shape[0]:
        raise ValueError("The rows of Q are not linearly independent.")
    C = np.linalg.lstsq(Q.T, H.T, rcond=None)[0].T
    if not np.allclose(C @ Q, H, atol=tol, rtol=tol):
        err = np.max(np.abs(C @ Q - H))
        raise ValueError(f"Secondary-cone normals are not in the row span of Q " f"(max residual {err:.2e}).")
    C[np.abs(C) < tol] = 0
    return Cone(rays=C)