from cytools import config
from cytools import Polytope, h_polytope
import numpy as np 
from cytools.vector_config import VectorConfiguration
from cytools.vector_config.fan import Fan
from . import Uplift_functions as UF

class CY_orientifold():
    """
    Class representing a Calabi-Yau orientifold constructed from toric data.
    """
    def __init__(self, fan_polytope_or_points=None, xi=None, resolve_A1_singularities=True, triangulate_points=False, construct_nef_decomposition=True):
        """
        Initializes the Calabi-Yau orientifold.

        Args:
            fan_polytope_or_points: The input toric data (Polytope, VectorConfiguration, Fan, array, or list).
            xi: The orientifold action/vector.
            resolve_A1_singularities (bool): Whether to resolve A1 singularities in the geometry.
            triangulate_points (bool): Whether to triangulate the fan points upon initialization.
            construct_nef_decomposition (bool): Whether to enforce Cartier and nef conditions on the line bundles.
        """
        self.__p = None
        self.__toric_fan = None
        self.__xi = None
        self.__orbifold_toric_fan = None
        self.__orbifold_line_bundle = None
        self.__yields_nef_decomposition = None
        self.__regular = None
        self.__orbifold_pts = None
        self.__CY_ambient_pts=None
        self.__normal_fan = None
        self.__Newton_Polytope = None
        self.__triangulated = False
        self._multiplier = 6
        match fan_polytope_or_points:
            case Polytope():
                if xi is None:
                    if fan_polytope_or_points.is_trilayer():
                        xi = -UF.find_trilayer_vertex_polytope(fan_polytope_or_points)
                    else:
                        raise ValueError("If polytope is not trilayer, xi has to be defined")
                self.__p = fan_polytope_or_points
                ind = UF.get_index(fan_polytope_or_points.points(), np.zeros(len(fan_polytope_or_points.vertices()[0])))
                if len(ind) > 0:
                    pts = np.delete(fan_polytope_or_points.points(), ind[0], axis=0)
                else:
                    pts = fan_polytope_or_points.points()
            case VectorConfiguration():
                if xi is None:
                    raise ValueError("xi has to be defined")
                pts = fan_polytope_or_points.vectors()
                # toric_fan = fan_polytope_or_points
            case Fan():
                if xi is None:
                    raise ValueError("xi has to be defined")
                pts = fan_polytope_or_points.vectors()
                self.__toric_fan = fan_polytope_or_points
                self.__triangulated = True
                triangulate_points=True

            case np.ndarray():
                if xi is None:
                    raise ValueError("xi has to be defined")
                pts = fan_polytope_or_points
            case list():
                if xi is None:
                    raise ValueError("xi has to be defined")
                pts = np.array(fan_polytope_or_points, dtype=int)
            case _:
                raise TypeError("Unsupported input type")
        self.__CY_ambient_pts = pts
        self.__dim = np.linalg.matrix_rank(pts,tol=1e-6)
        # self.__toric_fan = toric_fan
        self.__xi = xi          
        
        orbifold_pts, rescalings = UF.toric_orbifold_NEW(pts_CY_ambient=pts, q=xi)
        orbifold_line_bundle = UF.O3O7_line_bundle_NEW(pts_CY_ambient=pts, q=xi, rescalings=rescalings)
        if orbifold_line_bundle is None:
            self.__regular = False
        else:
            self.__regular = not bool(UF.generic_section_factorizes(orbifold_pts, orbifold_line_bundle))
        
        if self.__regular:
            if construct_nef_decomposition:
                self.__yields_nef_decomposition=False
                multiplier = 6
                KB_multiplier = UF.Newton_Polytope(orbifold_pts, self._multiplier * (np.ones(len(orbifold_line_bundle), dtype=int) - orbifold_line_bundle))
                if len(KB_multiplier.points()) > 0:
                    NP_orbifold = UF.Newton_Polytope(orbifold_pts, orbifold_line_bundle)
                    self.__Newton_Polytope = NP_orbifold
                    if NP_orbifold.minkowski_sum(KB_multiplier).dim() == self.__dim:
                        orbifold_pts_refined, line_bundles, normal_fan = UF.normal_fan_NEW([NP_orbifold, KB_multiplier], np.array([self._multiplier, 1, self._multiplier]), maximal_refinement=True, triangulate_refinement=False, return_unrefined_fan=True)
                        if orbifold_pts_refined is not None:
                            self.__yields_nef_decomposition = True
                            self.__normal_fan = normal_fan
                            orbifold_pts = orbifold_pts_refined
                            orbifold_line_bundle = line_bundles[:, 0]
                            self.__triangulated=False
                            # triangulate_points=False
            
        else:
            self.__yields_nef_decomposition=False



        self.__orbifold_pts = orbifold_pts
        self.__orbifold_line_bundle = orbifold_line_bundle
        if self.__triangulated:
            self.__orbifold_toric_fan=Fan(vc=VectorConfiguration(orbifold_pts),cones=self.__toric_fan.cones())
            if resolve_A1_singularities:
                self.__resolve_A1_singularities

        

        if triangulate_points:
            self.triangulate()


    def __resolve_A1_singularities(self):
        # fan=self.orbifold_toric_fan()
        singular_two_cones = UF.Z2_fixed_locus(self.CY_ambient_toric_fan(),self.xi(),cone_dimension=2,denominator=2)
        orbifold_blowups=[]
        if len(singular_two_cones)>0:
            for c in singular_two_cones:
                vec=np.sum(fan.vectors(c),axis=0)
                vec_add=np.rint(vec/np.gcd.reduce(vec)).astype(int)
                orbifold_blowups.append(vec_add.tolist())
            orbifold_blowups=np.array(orbifold_blowups)
            fan=UF.refine_fan_2(self.orbifold_toric_fan(),orbifold_blowups)
            self.__orbifold_pts=fan.vectors()
            blow_up_weights = np.rint(np.array([sum(self.line_bundle()[np.array(c)-1])-1 for c in singular_two_cones])/2).astype(int)
            self.__orbifold_line_bundle=np.concatenate((self.line_bundle(),blow_up_weights),axis=0)
            self.__orbifold_toric_fan=fan
        return None
    def orbifold_toric_fan(self):
        """Returns the toric fan of the toric orbifold."""
        if self.__orbifold_toric_fan is None:
            self.__orbifold_toric_fan = VectorConfiguration(self.__orbifold_pts)
        return self.__orbifold_toric_fan

    def CY_ambient_toric_fan(self):
        """Returns the underlying base toric fan before orbifolding."""
        if self.__toric_fan is None:
            self.__toric_fan = VectorConfiguration(self.__CY_ambient_pts)
        return self.__toric_fan

    def points_orbifold(self, c=None):
        """Returns the vectors (points) of the orbifold toric fan."""
        if c is not None:
            return self.__orbifold_pts[np.array(c)-1]
        else:
            return self.__orbifold_pts

    def points_CY_ambient(self, c=None):
        """Returns the vectors (points) of the base toric fan."""
        if c is not None:
            return self.__CY_ambient_pts[np.array(c)-1]
        else:
            return self.__CY_ambient_pts

    def xi(self):
        """Returns the orientifold action/vector xi."""
        return self.__xi

    def is_regular(self):
        """Returns True if the CY orientifold is regular."""
        return self.__regular

    def line_bundle(self):
        """Returns the O3/O7 orbifold line bundle."""
        return self.__orbifold_line_bundle

    def dim(self):
        """Returns the dimension of the orbifold toric fan."""
        return self.__dim

    def normal_fan(self):
        """Computes (if not cached) and returns the normal fan of the Newton Polytope."""
        if self.__normal_fan is None:
            self.__normal_fan = UF.normal_fan([self.Newton_Polytope(), UF.Newton_Polytope(self.points_orbifold(), self._multiplier*(1-self.line_bundle()))])[0]
        return self.__normal_fan

    def ambient_dim(self):
        """Returns the ambient dimension of the orbifold toric fan."""
        return len(self.points_orbifold()[0])

    def Newton_Polytope(self):
        """Computes and caches the Newton Polytope for the orbifold configuration."""
        if self.__Newton_Polytope is None:
            self.__Newton_Polytope = UF.Newton_Polytope(self.points_orbifold(), self.line_bundle())
        return self.__Newton_Polytope

    def is_triangulated(self):
        """Checks if the internal fan geometry has been triangulated."""
        return self.__triangulated

    def yields_nef_decomposition(self):
        """Checks if the geometry was successfully forced to be Cartier and nef."""
        if self.__yields_nef_decomposition is None:
            self.__yields_nef_decomposition = UF.contains_rows(self.points_orbifold(),self.normal_fan().vectors())
        return self.__yields_nef_decomposition

    def triangulate(self):
        """Triangulates the base fan and refines the orbifold fan accordingly."""
        if not self.is_triangulated():
            if self.yields_nef_decomposition():
                nf=UF.make_simplicial(self.normal_fan())
                VC=VectorConfiguration(self.points_orbifold())
                inds=VC.vectors_to_labels(nf.vectors())
                new_cones=set()
                for c in nf.cones():
                    nc=tuple(sorted([inds[x-1] for x in c]))
                    new_cones.add(nc)
                new_fan=Fan(vc=VC,cones=new_cones)
                self.__orbifold_toric_fan = UF.refine_fan_2(new_fan)
            else:
                self.__toric_fan = self.CY_ambient_toric_fan().triangulate()
                prelim_fan = Fan(self.orbifold_toric_fan(), cones=self.__toric_fan.cones())
                self.__orbifold_toric_fan = UF.refine_fan(prelim_fan, self.points_orbifold())
            self.__triangulated = True
    def polytope(self):
        return self.__p

class F_Theory_Uplift():
    """
    Class handling the uplifting of an orientifold to an F-Theory geometric model.
    Manages properties related to partitions, nef-partitions, and Hodge numbers.
    """
    def __init__(self, orientifold_or_points=None, xi=None, resolve_A1_singularities=False, construct_nef_decomposition=True, triangulate_points=False):
        """
        Initializes the F-Theory Uplift model.

        Args:
            orientifold_or_points: Can be an existing CY_orientifold instance or raw point data.
            xi: The orientifold action vector.
            resolve_A1_singularities (bool): Resolution flag for A1 singularities.
            construct_nef_decomposition (bool): Attempts to construct a nef decomposition.
            triangulate_points (bool): Auto-triangulate singular and smooth uplifts.
        """
        self.__blowups = None
        self.__Cayley_M = None
        self.__Cayley_N = None
        self.__pol_N_sum = None
        self.__pol_N_conv = None
        self.__pol_M_conv = None
        self.__pol_M_sum = None
        self.__pts_not_interior_to_facets_and_codim_2_faces_N_conv = None
        self.__pts_not_interior_to_facets_and_codim_2_faces_M_conv = None
        self.__pol_B_M = None
        self.__pol_W_M = None
        self.__pol_W_N = None
        self.__pol_B_N = None
        self.__NHC_labels = None
        self.__M_conv_toric_fan = None
        self.__intersection_numbers_M_conv = None
        self.__intersection_numbers_singular_uplift = None
        self.__intersection_numbers_smooth_uplift = None
        self.__LBW_N = None
        self.__LBB_N = None
        self.__LBW_M = None
        self.__LBB_M = None
        self.__basis_homology_M = None
        self.__basis_homology_N = None
        self.__normal_fan = None
        self.__singular_uplift_toric_fan = None
        self.__pts_singular_uplift=None
        self.__pts_smooth_uplift=None
        self.__smooth_uplift_toric_fan = None
        self.__orbifold_toric_fan = None
        self.x = None
        self.y = None
        self.z = None
        self.__h11 = None
        self.__h31 = None
        self.__h21 = None
        self.__chi = None
        self.__is_nef_partition = None
        self.__is_partition = None
        self.__is_nef_decomposition = None
        self.__triangulated = None
        self.__CY_orientifold = None

        match orientifold_or_points:
            case CY_orientifold():
                self.__CY_orientifold = orientifold_or_points
            case _:
                self.__CY_orientifold = CY_orientifold(orientifold_or_points, xi=xi, resolve_A1_singularities=resolve_A1_singularities, triangulate_points=triangulate_points, construct_nef_decomposition=construct_nef_decomposition)
        self.__blowups_labels = None
        self.__triangulated = self.__CY_orientifold.triangulated()
        if self.is_regular():
            if self.__triangulated:
                self.__triangulate_singular_uplift()
                self.__triangulate_smooth_uplift()
        else:
            self.__is_partition=False
            self.__is_nef_partition=False

    def __set_singular_toric_fan(self):
        """Constructs the fan of the ambient space of the generically singular F-theory uplift."""
        self.x = np.concatenate((np.zeros(self.ambient_dim_base(), dtype=int), np.array([3, 1])))
        self.y = np.concatenate((np.zeros(self.ambient_dim_base(), dtype=int), np.array([-2, -1])))
        self.z = np.concatenate((np.zeros(self.ambient_dim_base(), dtype=int), np.array([0, 1])))
        
        coordinates231 = (1 - self.line_bundle_orbifold())[:, None] @ self.z[-2:][None, :]
        pts6 = np.concatenate((self.points_orbifold(), coordinates231), axis=1)
        pts6 = np.concatenate((pts6, np.array([self.x, self.y, self.z])), axis=0)
        self.__pts_singular_uplift=pts6
        self.__singular_uplift_toric_fan = VectorConfiguration(pts6)

    def __set_divisor_representations(self):
        """Sets up the Base and Weierstrass line bundle arrays and verifies the partition status."""
        n_sing = len(self.points_orbifold()) + 3
        n_smooth = len(self.points_smooth_uplift())
        LBB_N = np.zeros(n_smooth, dtype=int)
        LBW_N = np.zeros(n_smooth, dtype=int)
        LBB_N[:n_sing-3] = self.line_bundle_orbifold()
        LBW_N[n_sing-3] = 3
        NHC = self.NHC(as_labels=True) - 1
        if len(NHC) > 0:
            cont = np.where(self.line_bundle_orbifold() != 0)[0]
            nhc_cont = np.where(np.isin(NHC, cont))[0]
            for n in nhc_cont:
                contribution_n = self.line_bundle_orbifold()[NHC[n]]
                LBB_N[2*n + n_sing] = 1 * contribution_n
                LBB_N[2*n + 1 + n_sing] = 2 * contribution_n
        is_part=UF.compute_partition([LBB_N, LBW_N],self.points_smooth_uplift())
        if is_part[0]:
            self.__LBB_N = is_part[1][0]
            self.__LBW_N = is_part[1][1]
        else:
            sta=UF.sums_to_anticanonical(self.points_smooth_uplift(),LBB_N,LBW_N)
            self.__LBB_N = LBB_N
            self.__LBW_N = self.points_smooth_uplift()@sta[1]+LBW_N
        self.__is_partition = is_part[0]
    # def __set_divisor_representations(self):
    #     """Sets up the Base and Weierstrass line bundle arrays and verifies the partition status."""
    #     n_sing = len(self.points_orbifold()) + 3
    #     n_smooth = len(self.points_smooth_uplift())
    #     LBB_N = np.zeros(n_smooth, dtype=int)
    #     LBW_N = np.zeros(n_smooth, dtype=int)
        
    #     LBB_N[:n_sing-3] = self.line_bundle_orbifold()
    #     LBW_N[n_sing-3] = 3
        
    #     NHC = self.NHC(as_labels=True) - 1
    #     if len(NHC) > 0:
    #         cont = np.where(self.line_bundle_orbifold() != 0)[0]
    #         nhc_cont = np.where(np.isin(NHC, cont))[0]
    #         for n in nhc_cont:
    #             contribution_n = self.line_bundle_orbifold()[NHC[n]]
    #             LBB_N[2*n + n_sing] = 1 * contribution_n
    #             LBB_N[2*n + 1 + n_sing] = 2 * contribution_n
                
    #     is_partition = UF.is_partition(self.points_smooth_uplift(), LBB_N, LBW_N)
    #     self.__LBB_N = LBB_N + self.points_smooth_uplift() @ is_partition[2]
    #     self.__LBW_N = LBW_N + self.points_smooth_uplift() @ is_partition[3]
    #     self.__is_partition = is_partition[0]
        
    def orientifold(self):
        """Returns the underlying CY_orientifold instance."""
        return self.__CY_orientifold

    def is_nef_partition(self):
        """Checks if the geometric structure represents a nef-partition."""
        if self.__is_nef_partition is None:
            if self.is_nef_decomposition():
                if self.is_partition():
                    self.__is_nef_partition = True
                    return True
            self.__is_nef_partition = False        
        return self.__is_nef_partition
        
    def is_partition(self):
        """Returns boolean indicating whether the uplift defines a valid partition."""
        if self.__is_partition is None:
            self.__set_divisor_representations()
        return self.__is_partition

    def smooth_uplift_toric_fan(self):
        """Constructs and returns the smooth uplift toric fan, including blowups for Non-Higgsable Clusters (NHCs)."""
        if self.__smooth_uplift_toric_fan is None:
            if len(self.NHC()) > 0:
                new_vc = VectorConfiguration(np.concatenate((self.points_singular_uplift(), np.vstack(self.blowups())), axis=0))
            else: 
                new_vc = VectorConfiguration(self.points_singular_uplift())
            self.__smooth_uplift_toric_fan = new_vc
        return self.__smooth_uplift_toric_fan

    def singular_uplift_toric_fan(self):            
        """Returns the previously computed singular uplift toric fan."""
        if self.__singular_uplift_toric_fan is None:
            self.__set_singular_toric_fan()
        return self.__singular_uplift_toric_fan
        
    def Cayley_M(self):
        """Constructs and caches the Cayley Polytope M (dual side). Warns if not a nef-partition."""
        if self.__Cayley_M is None:
            self.__Cayley_M = Polytope(np.concatenate((
                np.column_stack([self.pol_W_M().points(), np.ones(len(self.pol_W_M().points()), dtype=int), np.zeros(len(self.pol_W_M().points()), dtype=int)]),
                np.column_stack([self.pol_B_M().points(), np.zeros(len(self.pol_B_M().points()), dtype=int), np.ones(len(self.pol_B_M().points()), dtype=int)])
            ), axis=0))
        if not self.is_nef_partition():
            print("Careful, uplift is not a nef-partition")
        return self.__Cayley_M
        
    def Cayley_N(self):
        """Constructs and caches the Cayley Polytope N."""
        if self.is_partition():
            if self.__Cayley_N is None:
                self.__Cayley_N = Polytope(np.concatenate((
                    np.column_stack([self.pol_W_N().points(), np.ones(len(self.pol_W_N().points()), dtype=int), np.zeros(len(self.pol_W_N().points()), dtype=int)]),
                    np.column_stack([self.pol_B_N().points(), np.zeros(len(self.pol_B_N().points()), dtype=int), np.ones(len(self.pol_B_N().points()), dtype=int)])
                ), axis=0))
        else:
            print("Uplift is not a partition")
        return self.__Cayley_N

    def pol_W_M(self):
        """Returns the Newton Polytope for the Weierstrass divisor in M-lattice."""
        if self.__pol_W_M is None:
            self.__pol_W_M = UF.Newton_Polytope(self.points_smooth_uplift(), self.line_bundle_weierstrass_N())
        return self.__pol_W_M
        
    def pol_B_M(self):
        """Returns the Newton Polytope for the Base divisor in M-lattice."""
        if self.__pol_B_M is None:
            self.__pol_B_M = UF.Newton_Polytope(self.points_smooth_uplift(), self.line_bundle_base_N())
        return self.__pol_B_M

    def pol_M_conv(self):
        """Returns the convex hull polytope of the Base and Weierstrass Newton-polytopes."""
        if self.__pol_M_conv is None:
            self.__pol_M_conv = Polytope(np.unique(np.concatenate((self.pol_B_M().vertices(), self.pol_W_M().vertices()), axis=0), axis=0))
        return self.__pol_M_conv
        
    def pol_M_sum(self):
        """Returns the Minkowski sum of the Base and Weierstrass Newton-polytopes."""
        if self.__pol_M_sum is None:
            self.__pol_M_sum = self.pol_B_M().minkowski_sum(self.pol_W_M())
        return self.__pol_M_sum

    def pol_W_N(self):
        """Returns the Weierstrass N-lattice Newton polytope of the dual nef partition."""
        if self.is_partition():
            if self.__pol_W_N is None:
                self.__pol_W_N = Polytope(np.concatenate((np.zeros((1, self.ambient_dim_base()+2), dtype=int), self.points_smooth_uplift()[np.where(self.line_bundle_weierstrass_N()==1)[0]]), axis=0))
        else:
            print("Uplift is not a partition")
        return self.__pol_W_N

    def pol_B_N(self):
        """Returns the Base N-lattice Newton polytope of the dual nef partition."""
        if self.is_partition():
            if self.__pol_B_N is None:
                self.__pol_B_N = Polytope(np.concatenate((np.zeros((1, self.ambient_dim_base()+2), dtype=int), self.points_smooth_uplift()[np.where(self.line_bundle_base_N()==1)[0]]), axis=0))
        else:
            print("Uplift does not yield a partition")
        return self.__pol_B_N
    
    def pol_N_conv(self):
        """Returns the convex N-polytope defined by the smooth uplift points."""
        if self.__pol_N_conv is None:
            self.__pol_N_conv = Polytope(self.points_smooth_uplift())
        return self.__pol_N_conv
        
    def pol_N_sum(self):
        """Returns the Minkowski sum of the Base and Weierstrass N-polytopes."""
        if self.is_partition():
            if self.__pol_N_sum is None:
                self.__pol_N_sum = self.pol_B_N().minkowski_sum(self.pol_W_N())
        else:
            print("Uplift is not a partition")
        return self.__pol_N_sum
        
    def points_singular_uplift(self, labels=None):
        """Returns points of the singular uplift fan, optionally filtered by labels."""
        if self.__pts_singular_uplift is None:
            self.__set_singular_toric_fan()
        if labels is None:
            return self.__pts_singular_uplift
        return self.__pts_singular_uplift[np.array(labels)-1]

    def blowups(self, as_labels=False):
        """Generates and returns the blowup coordinates required to resolve NHC singularities."""
        if self.__blowups is None:
            bus = []
            for xxx in self.NHC_singular_uplift(as_labels=True):
                bus.append(np.array([
                    self.singular_uplift_toric_fan().vectors()[xxx-1] + self.x + 2*self.y,
                    2*self.singular_uplift_toric_fan().vectors()[xxx-1] + 2*self.x + 3*self.y
                ]))
            self.__blowups = bus
        
        if as_labels:
            if getattr(self, '_F_Theory_Uplift__blowups_labels', None) is None:
                bu_labels = []
                for entry in self.__blowups:
                    if self.__triangulated:
                        bu_labels.append(self.smooth_uplift_toric_fan().vc.vectors_to_labels(entry))
                    else:
                        bu_labels.append(self.smooth_uplift_toric_fan().vectors_to_labels(entry))
                self.__blowups_labels = bu_labels
            return self.__blowups_labels
        return self.__blowups
        
    def points_not_interior_to_codim_1_and_2_face_M(self):
        """Returns points not interior to codim 1 and 2 faces of the M-conv polytope."""
        if self.__pts_not_interior_to_facets_and_codim_2_faces_M_conv is None:
            self.__pts_not_interior_to_facets_and_codim_2_faces_M_conv = UF.points_not_interior_to_facets_and_codim2_faces(self.pol_M_conv())
        return self.__pts_not_interior_to_facets_and_codim_2_faces_M_conv

    def points_not_interior_to_codim_1_and_2_face_N(self):
        """Returns points not interior to codim 1 and 2 faces of the N-conv polytope."""
        if self.__pts_not_interior_to_facets_and_codim_2_faces_N_conv is None:
            self.__pts_not_interior_to_facets_and_codim_2_faces_N_conv = UF.points_not_interior_to_facets_and_codim2_faces(self.pol_N_conv())
        return self.__pts_not_interior_to_facets_and_codim_2_faces_N_conv
        
    def polytope(self):
        """Returns the  polytope if one was defined, otherwise returns None."""
        # if self.orientifold()._CY_orientifold__p is not None:
        return self.orientifold().polytope()
        # else:
            # raise ValueError("The uplift is not defined by a polytope")
        
    def intersection_numbers_M_conv(self, check=True):
        """Returns the intersection numbers of the M-conv toric fan."""
        if self.__intersection_numbers_M_conv is None:
            self.__intersection_numbers_M_conv = self.M_conv_toric_fan().intersection_numbers()
        return self.__intersection_numbers_M_conv
        
    def intersection_numbers_smooth_uplift(self):
        """Returns the intersection numbers of the smooth uplift toric fan."""
        if self.__intersection_numbers_smooth_uplift is None:
            self.__intersection_numbers_smooth_uplift = self.smooth_uplift_toric_fan().intersection_numbers()
        return self.__intersection_numbers_smooth_uplift
        
    def intersection_numbers_singular_uplift(self, check=True):
        """Returns the intersection numbers of the singular uplift toric fan."""
        if self.__intersection_numbers_singular_uplift is None:
            self.__intersection_numbers_singular_uplift = self.singular_uplift_toric_fan().intersection_numbers()
        return self.__intersection_numbers_singular_uplift
        
    def h11(self):
        """Computes the Hodge number h11 using the Batyrev formulas for nef-partitions."""
        if self.is_nef_partition():
            if self.__h11 is None:
                self.__h11 = UF.h11_2_part(self.Cayley_M(), self.Cayley_N())
            return self.__h11 
        else: 
            raise ValueError("Hodge Numbers can only be computed for nef-partitions")
        
    def h31(self):
        """Computes the Hodge number h31 using the Batyrev formulas for nef-partitions."""
        if self.is_nef_partition():
            if self.__h31 is None:
                self.__h31 = UF.h11_2_part(self.Cayley_N(), self.Cayley_M())
            return self.__h31
        else: 
            raise ValueError("Hodge Numbers can only be computed for nef-partitions")
            
    def h31_trivial(self):
        """Returns a trivial fast calculation of a pseudo-h31 based on Cayley points."""
        return len(self.Cayley_M().points()) - 1 - (self.Cayley_N().dim())

    def line_bundle_weierstrass_M(self):
        """Returns the dual of the Weierstrass line bundle."""
        if self.is_nef_partition():
            if self.__LBW_M is None:
                LB_W = np.zeros(len(self.points_not_interior_to_codim_1_and_2_face_M()), dtype=int)
                LB_W[UF.get_indices(self.points_not_interior_to_codim_1_and_2_face_M(), self.pol_W_M().points())] = 1
                self.__LBW_M = LB_W[1:]
        else:
            print("Uplift is not a nef-partition")
        return self.__LBW_M

    def line_bundle_base_M(self):
        """Returns the dual of the base line bundle."""
        if self.is_nef_partition():
            if self.__LBB_M is None:
                LB_B = np.zeros(len(self.points_not_interior_to_codim_1_and_2_face_M()), dtype=int)
                LB_B[UF.get_indices(self.points_not_interior_to_codim_1_and_2_face_M(), self.pol_B_M().points())] = 1
                self.__LBB_M = LB_B[1:]
        else:
            print("Uplift is not a nef-partition")
        return self.__LBB_M

    def line_bundle_weierstrass_N(self):
        """Returns the Weierstrass line bundle."""
        if self.__LBW_N is None:
            self.__set_divisor_representations()
        return self.__LBW_N

    def line_bundle_base_N(self):
        """Returns the Base line bundle."""
        if self.__LBB_N is None:
            self.__set_divisor_representations()
        return self.__LBB_N

    def line_bundle_orbifold(self):
        """Returns the orientifold line bundle."""
        return self.orientifold().line_bundle()
        
    def M_conv_toric_fan(self):
        if self.__M_conv_toric_fan is None:
            index0 = UF.get_index(self.points_not_interior_to_codim_1_and_2_face_M(),np.zeros(len(self.points_not_interior_to_codim_1_and_2_face_M()[0]),dtype=int))[0]
            pts=np.delete(self.points_not_interior_to_codim_1_and_2_face_M(),index0,axis=0)
            vc = VectorConfiguration(pts)
            tri = vc.triangulate()
            self.__M_conv_toric_fan = Fan(vc=vc,cones=tri.cones())
        return self.__M_conv_toric_fan

    def divisor_intersection_M(self, as_LLL=True):
        """Computes the curve homology intersections in the N basis."""
            
        fan = self.M_conv_toric_fan()
        intersection_dict = self.intersection_numbers_M_conv()
        three_simplices = UF.get_lower_dimensional_cones(fan.cones(), fan.dim - 3)
        basis = self.basis_homology_M()
        
        LBB_set = set(np.where(self.line_bundle_base_M() == 1)[0] + 1)
        LBW_set = set(np.where(self.line_bundle_weierstrass_M() == 1)[0] + 1)
        basis_set = set(basis)
        
        curves_homology_in_basis = np.zeros((len(three_simplices), len(basis_set)), dtype=int)
        
        basis_idx_map = {b: idx for idx, b in enumerate(basis)}

        for s_idx, s in enumerate(three_simplices):
            link_rays = {item for sub_tuple in fan.star(s) for item in sub_tuple}
            
            valid_x = LBW_set.intersection(link_rays)
            valid_y = LBB_set.intersection(link_rays)
            valid_i = basis_set.intersection(link_rays)
            
            if not (valid_x and valid_y and valid_i):
                continue
                
            for i in valid_i:
                i_idx = basis_idx_map.get(i)
                total_intersection = 0
                
                for x in valid_x:
                    for y in valid_y:
                        if x == y or x == i or y == i or x in s or y in s or i in s:
                            continue
                        key = tuple(sorted(s + (x, y, i)))
                        total_intersection += intersection_dict.get(key, 0)
                
                curves_homology_in_basis[s_idx, i_idx] = total_intersection

        if as_LLL:
            return (basis, UF.LLL_wrapper(curves_homology_in_basis))
        return (basis, curves_homology_in_basis)
 
    def divisor_intersection_N(self, as_LLL=True):
        """Computes the curve homology intersections in the N basis."""
        if not self.is_triangulated():
            print("Can't compute intersections without a triangulation")
            return []
            
        fan = self.smooth_uplift_toric_fan()
        intersection_dict = self.intersection_numbers_smooth_uplift()
        three_simplices = UF.get_lower_dimensional_cones(fan.cones(), fan.dim - 3)
        basis = self.basis_homology_N()
        
        LBB_set = set(np.where(self.line_bundle_base_N() == 1)[0] + 1)
        LBW_set = set(np.where(self.line_bundle_weierstrass_N() == 1)[0] + 1)
        basis_set = set(basis)
        

        curves_homology_in_basis = np.zeros((len(three_simplices), len(basis_set)), dtype=int)
        basis_idx_map = {b: idx for idx, b in enumerate(basis)}

        for s_idx, s in enumerate(three_simplices):
            link_rays = {item for sub_tuple in fan.star(s) for item in sub_tuple}
            
            valid_x = LBW_set.intersection(link_rays)
            valid_y = LBB_set.intersection(link_rays)
            valid_i = basis_set.intersection(link_rays)
            
            if not (valid_x and valid_y and valid_i):
                continue
                
            for i in valid_i:
                i_idx = basis_idx_map.get(i)
                total_intersection = 0
                
                for x in valid_x:
                    for y in valid_y:
                        if x == y or x == i or y == i or x in s or y in s or i in s:
                            continue
                        key = tuple(sorted(s + (x, y, i)))
                        total_intersection += intersection_dict.get(key, 0)
                
                curves_homology_in_basis[s_idx, i_idx] = total_intersection

        if as_LLL:
            return (basis, UF.LLL_wrapper(curves_homology_in_basis))
        return (basis, curves_homology_in_basis)  
    
    def NHC(self, as_labels=False):
        """Returns the Non-Higgsable Clusters (NHC) in the base space."""
        if self.__NHC_labels is None:
            sections_NP4 = UF.sections(self.points_orbifold(), 4 * (1 - self.line_bundle_orbifold()))
            if len(sections_NP4) > 0:
                self.__NHC_labels = np.where(np.min(sections_NP4, axis=1) >= 2)[0] + 1
            else:
                self.__NHC_labels = np.array([])
                
        if as_labels:
            return self.__NHC_labels
        else:
            if len(self.__NHC_labels) > 0:
                return self.points_orbifold(self.__NHC_labels)
            return self.__NHC_labels

    def NHC_singular_uplift(self, as_labels=False):
        """Returns the Non-Higgsable Clusters of the singular uplift fan."""
        if as_labels:
            return self.NHC(as_labels=True)
        else:
            labels = self.NHC(as_labels=True)
            if len(labels) > 0:
                return self.points_singular_uplift(labels)
        
    def points_orbifold(self, labels=None):
        """Returns the vectors of the orbifold fan."""
        return self.orientifold().points_orbifold(labels)
    
    def points_CY_ambient(self, labels=None):
        """Returns the vectors of the initial fan."""
        return self.orientifold().points_CY_ambient(labels)
    
    def dim_base(self):
        """Returns the dimension of the base orientifold."""
        return self.orientifold().dim()
        
    def ambient_dim_base(self):
        """Returns the ambient dimension of the base orientifold."""
        return self.orientifold().ambient_dim()

    def CY_ambient_toric_fan(self):
        """Returns the fan of the initial toric variety."""
        return self.orientifold().CY_ambient_toric_fan()

    def orbifold_toric_fan(self):
        """Returns the orbifold toric fan."""
        return self.orientifold().orbifold_toric_fan()

    def points_smooth_uplift(self, labels=None):
        """Returns the vectors of the smooth uplift fan."""
        if self.__pts_smooth_uplift is None:
            if len(self.NHC()) > 0:
                self.__pts_smooth_uplift = np.concatenate((self.points_singular_uplift(), np.vstack(self.blowups())), axis=0)
            else: 
                self.__pts_smooth_uplift = self.points_singular_uplift()
        if labels is None:
            return self.__pts_smooth_uplift
        return self.__pts_smooth_uplift[np.array(labels)-1]

    def xi(self):
        """Returns the xi vector of the orientifold."""
        return self.__CY_orientifold.xi()

    def h21(self):
        """Computes the Hodge number h21 via Cayley polytopes."""
        if self.__h21 is None:
            self.__h21 = UF.h21_2_part(self.Cayley_N(), self.Cayley_M())
        return self.__h21

    def chi(self):
        """Computes the Euler characteristic of the uplift geometry based on Hodge numbers."""
        if self.__chi is None:
            self.__chi = 48 + 6 * (self.h11() + self.h31() - self.h21())
        return self.__chi

    def is_triangulated(self):
        """Checks if the uplift geometries have been triangulated."""
        return self.__triangulated

    def triangulate(self):
        """ 
        Triangulates the orbifold toric fan, the singular uplift and the smooth uplift 
        toric fan compatibly with the fibration structure. Enforces nef and Cartier geometry if applicable.
        """
        if not self.is_triangulated():
            self.__CY_orientifold.triangulate()    
            self.__triangulate_singular_uplift()
            self.__triangulate_smooth_uplift()
            self.__triangulated = True

    def __triangulate_singular_uplift(self):
        """Performs the internal triangulation logic for the singular F-theory uplift fan."""
        cones6 = set()
        x_pts6 = len(self.points_orbifold()) + 1
        y_pts6 = x_pts6 + 1
        z_pts6 = x_pts6 + 2
        
        for c in self.orbifold_toric_fan().cones():
            cones6.add(c + (x_pts6, y_pts6))
            cones6.add(c + (x_pts6, z_pts6))
            cones6.add(c + (y_pts6, z_pts6))
            
        self.__singular_uplift_toric_fan = Fan(vc=self.singular_uplift_toric_fan(), cones=cones6)
        
    def __triangulate_smooth_uplift(self):
        """Performs internal triangulation by inserting exceptional blowup divisors recursively."""
        n_sing_uplift = len(self.points_singular_uplift())
        xlabel = n_sing_uplift - 2
        ylabel = n_sing_uplift - 1
        ct = 0
        vc_smooth = self.smooth_uplift_toric_fan()
        
        all_cones = set(self.singular_uplift_toric_fan().cones())
        
        for xxx in self.NHC_singular_uplift(as_labels=True):
            cone_target = set((xxx, xlabel, ylabel))
            link_list = []
            for c in list(all_cones):
                if cone_target.issubset(c):
                    link = tuple(v for v in c if v not in cone_target)
                    link_list.append((c, link))
            
            for original_c, link in link_list:
                all_cones.discard(original_c)
                all_cones.add(tuple(sorted(link + (xxx, n_sing_uplift + 2*ct + 1, n_sing_uplift + 2*ct + 2))))
                all_cones.add(tuple(sorted(link + (xxx, xlabel, n_sing_uplift + 2*ct + 2))))
                all_cones.add(tuple(sorted(link + (xxx, ylabel, n_sing_uplift + 2*ct + 1))))
                all_cones.add(tuple(sorted(link + (xlabel, ylabel, n_sing_uplift + 2*ct + 1))))
                all_cones.add(tuple(sorted(link + (xlabel, n_sing_uplift + 2*ct + 1, n_sing_uplift + 2*ct + 2))))
            ct += 1
            
        self.__smooth_uplift_toric_fan = Fan(vc=vc_smooth, cones=all_cones)
        
    def is_regular(self):
        """Checks if the underlying orientifold geometry is regular."""
        return self.orientifold().is_regular()

    def normal_fan(self):
        """Returns the normal fan of the underlying base structure."""
        return self.orientifold().normal_fan()
    
    def basis_homology_M(self, as_LLL=True):
        """Returns the basis of the homology of M."""
        if self.__basis_homology_M is None:
            self.__basis_homology_M = UF.basis_H2_toric_fan(self.M_conv_toric_fan())
        return self.__basis_homology_M
    
    def basis_homology_N(self, as_LLL=True):
        """Returns the basis of the homology of N."""
        if self.__basis_homology_N is None:
            self.__basis_homology_N = UF.basis_H2_toric_fan(self.smooth_uplift_toric_fan())
        return self.__basis_homology_N
    
    def nef_partition_N(self):
        """Returns the nef partition of N."""
        if self.is_nef_partition():
            return (tuple(np.where(self.line_bundle_base_N()==1)[0]),tuple(np.where(self.line_bundle_weierstrass_N()==1)[0]))
        else:
            return ((),())
    def nef_partition_M(self):
        """Returns the nef partition of M."""
        if self.is_nef_partition():
            return (tuple(np.where(self.line_bundle_base_M()==1)[0]),tuple(np.where(self.line_bundle_weierstrass_M()==1)[0]))
        else:
            return ((),())
    def is_nef_decomposition(self):
        if self.__is_nef_decomposition is None:
            if self.is_triangulated():
                self.__is_nef_decomposition = np.all([UF.is_Cartier(self.smooth_uplift_toric_fan(),self.line_bundle_base_N())[0],UF.is_Cartier(self.smooth_uplift_toric_fan(),self.line_bundle_weierstrass_N())[0],UF.is_nef(self.smooth_uplift_toric_fan(),self.line_bundle_base_N()),UF.is_nef(self.smooth_uplift_toric_fan(),self.line_bundle_weierstrass_N())])
            else:
                self.__is_nef_decomposition = self.orientifold().yields_nef_decomposition()
        return self.__is_nef_decomposition
            