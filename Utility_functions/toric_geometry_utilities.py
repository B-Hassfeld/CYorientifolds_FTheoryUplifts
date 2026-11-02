import math
import warnings
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations, product
from typing import Any, Literal, overload
import numpy as np
import ppl
from scipy.optimize import linprog
from numpy.typing import ArrayLike
from cytools import Cone, Polytope, h_polytope
from cytools.vector_config import VectorConfiguration
from cytools.vector_config.fan import Fan
from .fan_utilities import get_lower_dimensional_cones, make_simplicial, refine_fan
# Functions that live in Uplift_functions (which imports this module) are imported inside the functions using them.

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

def is_lattice_newton_polytope(rays,weights):
    """
    **Description:**
    Checks whether the Newton polytope {m : <m, v_i> + a_i >= 0 for all i} of the toric divisor D = sum_i a_i D_i, with D_i the
    divisors of the rays v_i, is a non-empty bounded lattice polytope, i.e. whether all its vertices are integral (PPL).
    **Arguments:**
    - `rays (Iterable[ArrayLike])`: The rays v_i of the fan, one integer vector each.
    - `weights (Iterable[int])`: The coefficients a_i of the divisor, in the same order as `rays`.
    **Returns:**
    - `bool`: `True` if the Newton polytope is non-empty, bounded and has only lattice vertices.
    """
    P=ppl.C_Polyhedron(len(rays[0]),"universe")
    for ray, weight in zip(rays,weights,strict=True):
        P.add_constraint(ppl.Linear_Expression(list(ray),weight)>=0)
    return (not P.is_empty() and P.is_bounded() and all(v.divisor()==1 for v in P.minimized_generators()))

def face_fan(p: Polytope,simplicial: bool=False)->Fan:
    """
    **Description:**
    Constructs the face fan of a polytope containing the origin in its relative interior: its rays are the vertices of
    `p` and its maximal cones are spanned by the vertices of each facet. If `p` is not full-dimensional, the facets are
    taken relative to its affine hull, so the fan is supported on the linear span of `p` and is not full-dimensional.
    For a reflexive polytope this is the normal fan of the dual polytope, which is never constructed.
    **Arguments:**
    - `p (Polytope)`: A polytope of dimension at least one with the origin in its relative interior.
    - `simplicial (bool)`: Whether to refine the non-simplicial cones via `make_simplicial`. Defaults to `False`.
    **Raises:**
    - `ValueError`: Raised if `p` is a point or the origin is not in its relative interior.
    **Returns:**
    - `Fan`: The (in general non-simplicial) fan over `VectorConfiguration(p.vertices())`.
    """
    V=p.vertices()
    ineqs=p.inequalities()
    if p.dim()<1 or np.linalg.matrix_rank(V)!=p.dim() or (ineqs[:,-1]<=0).any():
        raise ValueError("The polytope must have dimension at least one and the origin in its relative interior.")
    on_facet=ineqs[:,:-1]@V.T+ineqs[:,-1:]==0
    fan=Fan(vc=VectorConfiguration(V),cones=[tuple(np.flatnonzero(row)+1) for row in on_facet])
    return make_simplicial(fan) if simplicial else fan

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

def mori_cone_in_glsm_basis(Q: ArrayLike,fan: Fan|None=None,sub_rows: ArrayLike|None=None,as_face_in_higher_dim_ambient: bool=False,H: ArrayLike|None=None,tol: float=1e-10,check_face: bool=True)->Cone:
    """
    **Description:**
    Mori cone of a secondary-fan phase, expressed in the basis given by the rows of Q, or, if `sub_rows` is given, the face of it spanned by the curve classes
    in the row span of the rows `Q[sub_rows]`, without constructing the full Mori cone. The Mori cone is generated by the secondary-cone hyperplane normals
    (one wall curve per wall of the fan), each of which has a unique expansion in the rows of the full-rank matrix Q. The face is generated by those normals
    whose coefficients vanish along all rows outside `sub_rows`, i.e. the Mori generators in the row span of `Q[sub_rows]`. With `check_face=True` an LP
    certifies that this subcone is a face of the Mori cone (it is one iff some Kahler class is orthogonal to exactly these generators); otherwise a warning
    is issued, as the subcone may then be strictly smaller than the intersection of the Mori cone with the row span. Mori generators that are combinations
    of several rows (some outside `sub_rows`) are never found this way, so `sub_rows` has to contain all rows on which the curves of interest have nonzero degree.
    **Arguments:**
    - `Q (ArrayLike)`: Full-rank GLSM charge matrix whose rows define the basis.
    - `fan (Fan|None)`: The toric fan of the phase; its secondary cone hyperplanes are used if `H` is not given. Defaults to `None`.
    - `sub_rows (ArrayLike|None)`: 0-based indices of the rows of Q (e.g. the D7 rows) whose row span contains the face. If `None`, the full Mori cone is returned. Defaults to `None`.
    - `as_face_in_higher_dim_ambient (bool)`: Whether to return the face as a lower-dimensional cone in the basis of all rows of Q (the ambient space of the full Mori cone) instead of in the basis of the rows `Q[sub_rows]`. Ignored if `sub_rows` is `None`. Defaults to `False`.
    - `H (ArrayLike|None)`: The secondary (Kahler) cone hyperplane normals, one per row. Defaults to `None`.
    - `tol (float)`: Numerical tolerance. Defaults to `1e-10`.
    - `check_face (bool)`: Whether to verify by an LP that the subcone is a face of the Mori cone and warn otherwise. Only used if `sub_rows` is given. Defaults to `True`.
    **Raises:**
    - `ValueError`: Raised if neither `fan` nor `H` is given, if Q is not full rank, if the hyperplane normals are not in the row span of Q, or if no hyperplane normal lies in the row span of `Q[sub_rows]`.
    **Returns:**
    - `Cone`: The Mori cone, generated by the hyperplane normals in the basis of the rows of Q; or its face generated by the hyperplane normals in the row span of `Q[sub_rows]`, in the basis of the rows `Q[sub_rows]` (or of all rows of Q if `as_face_in_higher_dim_ambient` is `True`).
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
    if sub_rows is None:
        return Cone(rays=C)
    sub_rows=np.asarray(sub_rows,dtype=int).ravel()
    other=np.setdiff1d(np.arange(Q.shape[0]),sub_rows)
    in_span=(C[:,other]==0).all(axis=1)
    if not in_span.any():
        raise ValueError("No secondary-cone normal lies in the row span of Q[sub_rows]")
    if check_face and not in_span.all():
        res=linprog(np.zeros(H.shape[1]),A_ub=-H[~in_span],b_ub=-np.ones((~in_span).sum()),A_eq=H[in_span],b_eq=np.zeros(in_span.sum()),bounds=(None,None),method="highs")
        if res.status!=0:
            warnings.warn("The Mori generators in the row span of Q[sub_rows] do not span a face of the Mori cone; the returned cone may be smaller than the intersection of the Mori cone with the row span.")
    C=C[in_span]
    if not as_face_in_higher_dim_ambient:
        C=C[:,sub_rows]
    return Cone(rays=C)