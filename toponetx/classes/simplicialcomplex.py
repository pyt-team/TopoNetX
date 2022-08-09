# --------------------------------------------------------
# Constructing (co)boundary matrices, Hodge Laplacians, higher order
# (co)adjacency matrices for Simplicial Complexes
# Normalization of these matrices are also implemented.
#
# Date: Dec 2021
# --------------------------------------------------------

import sys
from functools import lru_cache
from itertools import combinations
from warnings import warn
from collections import OrderedDict
import numpy as np
import scipy.sparse.linalg as spl
from scipy.linalg import fractional_matrix_power
from scipy.sparse import coo_matrix, csr_matrix, diags, dok_matrix, eye
from sklearn.preprocessing import normalize
from hypernetx import Hypergraph
import networkx as nx
from rankedentity import RankedEntity, RankedEntitySet
from combinatorialcomplex import CombinatorialComplex

try:
    from gudhi import SimplexTree
except ImportError:
    warn(
        "gudhi library is not installed."
        + " Default computing protocol will be set for 'normal'.\n"
        + " gudhi can be installed using: 'pip install gudhi'",
        stacklevel=2,
    )


__all__ = ["SimplicialComplex"]


class SimplicialComplex:
    """Class for construction boundary operators, Hodge Laplacians,
    higher order (co)adjacency operators from collection of
    simplices.

    Parameters
    ----------
    simplices : list of maximal simplices that define the simplicial complex
    name : hashable, optional, default: None
        If None then a placeholder '_'  will be inserted as name
    maxdimension: constrain the max dimension of the input inserted simplices    
    
    mode : computational mode, available options are "normal" or "gudhi". 
    default is normal
    """

    def __init__(self, simplices = None , name = None, maxdimension = None, mode = "normal"):
        self.mode = mode
        if name is None:
            self.name = name
        else:
            self.name = "_"
        self._simplex_set = set()
        if maxdimension!= None:
            self.constrain_dim = maxdimension
        else:
            self.constrain_dim = -1
        
        if simplices is None:
            self.maxdim = -1
            if self.mode== "normal":
                pass
            elif self.mode == "gudhi":
                self.st = self.get_simplex_tree()
                self.faces_dict = SimplicialComplex.extract_simplices(self.st)
            else:
                raise ValueError(f" Import modes must be 'normal' and 'gudhi', got {mode}")                
                
        else:
            if not isinstance(simplices, (list, tuple)):
                raise TypeError(f"Input simplices must be given as a list or tuple, got {type(simplices)}.")
    
            max_simplex_size = len(max(simplices, key=lambda el: len(el)))

    
            try:
                from gudhi import SimplexTree
            except ImportError:
                warn(
                    "gudhi library is not installed."+
                    "normal mode will be used for computations",
                    stacklevel=2,
                )
                self.mode = "normal"
            if self.mode == "normal":
    
                self._import_simplices(simplices = simplices)
  
                if maxdimension is None:
                    self.maxdim = max_simplex_size - 1
                else:
    
                    if maxdimension > max_simplex_size - 1:
                        warn(
                            f"Maximal simplex in the collection has size {max_simplex_size}."+
                            "\n maxdimension is set to {max_simplex_size-1}",
                            stacklevel=2,
                        )
                        self.maxdim = max_simplex_size - 1
                    elif maxdimension < 0:
                        raise ValueError(f"maxdimension should be a positive integer, got {maxdimension}.")
                    else:
                        self.constrain_dim = maxdimension
            elif self.mode == "gudhi":
                self.st = self.get_simplex_tree()
                for s in simplices:
                    self.st.insert(s)
                self.faces_dict = SimplicialComplex.extract_simplices(self.st)
                self._import_simplices(simplices=simplices)
                max_simplex_size = self.st.dimension() + 1
                if maxdimension is None:
                    self.maxdim = max_simplex_size - 1
                else:
    
                    if maxdimension > max_simplex_size - 1:
                        warn(
                            f"Maximal simplex in the collection has size {max_simplex_size}."+
                            f" \n {maxdimension} is set to {max_simplex_size-1}",
                            stacklevel=2,
                        )
                        self.maxdim = max_simplex_size - 1
                    elif maxdimension < 0:
                        raise ValueError(f"maxdimension should be a positive integer, got {maxdimension}.")
                    else:
                        self.constrain_dim = maxdimension
            else:
                raise ValueError(f" Import modes must be 'normal' and 'gudhi', got {mode}")
    @property
    def shape(self):
        """
        (number of simplices[i], for i in range(0,dim(Sc))  )

        Returns
        -------
        tuple

        """
        if len(self._simplex_set)== 0:
            print("Simplicial Complex is empty.")
        else:    
            return [len(self._n_faces(i)) for i in range(self.maxdim+1) ]

    @property
    def dim(self):
        """
        dimension of the simplicial complex is the highest dimension of any simplex in the complex
        """
        return self.maxdim
    @property
    def simplices(self):
        """
        set of all simplices
        """
        return self._simplex_set
   
    def skeleton(self, n):
        """
        Returns
        -------
        set of simplices of dimesnsion n
        """        
        return self.dic_order_faces(n)
    
    def __str__(self):
        """
        String representation of SC

        Returns
        -------
        str

        """
        return f"simplicial Complex with shape {self.shape()} and dimension {self.maxdim}"

    def __repr__(self):
        """
        String representation of simplicial complex

        Returns
        -------
        str

        """
        return f"SimplicialComplex(name={self.name})"

    def __len__(self):
        """
        Number of simplices

        Returns
        -------
        int

        """

        return len(self._simplex_set)

    def __iter__(self):
        """
        Iterate over all faces of the simplicial complex 

        Returns
        -------
        dict_keyiterator

        """
        return iter(self._simplex_set)

    def __contains__(self, item):
        """
        Returns boolean indicating if item is in self.face_set

        Parameters
        ----------
        item : tuple, list

        """
        return tuple(item) in self._simplex_set



    @staticmethod
    def _faces(simplices,max_dim=None):
        """
        Parameters
        ----------
        simplices : list
            DESCRIPTION. list or of simplices, typically integers.
        max_dim : constrain the max dimension of faces    
        Returns
        -------
        faceset : set
            DESCRIPTION. list of tuples or all faces of the input list of simplices
        """
        
        # valid in normal mode and can be used as a static method on any face
        # TODO, do for gudhi mode as well.
        if not isinstance(simplices, (list, tuple)):
            raise TypeError(f"Input simplices must be given as a list or tuple, got {type(simplices)}.")
  
        faceset = set()
        for simplex in simplices:
            numnodes = len(simplex)
            for r in range(numnodes, 0, -1):
                for face in combinations(simplex, r):
                    if max_dim is None:
                        faceset.add(tuple(sorted(face))) 
                    elif len(face)<=max_dim+1:    
                        faceset.add(tuple(sorted(face)))
        return faceset


    def _import_simplices(self, simplices = []):
        if self.mode == "normal":
            ordered_simplices = tuple(
                map(lambda simplex: tuple(sorted(simplex)), simplices)
            )
            if self.constrain_dim==-1:
                self._simplex_set = SimplicialComplex._faces(ordered_simplices)
            else:
                self._simplex_set = SimplicialComplex._faces(ordered_simplices,self.constrain_dim)

        elif self.mode == "gudhi":
            lst = []
            for i in range(0, len(self.faces_dict)):
                lst = lst + list(self.faces_dict[i].keys())
            self._simplex_set = lst



    def _n_faces(self, n):
        if n >= 0 and n <= self.maxdim:
            if self.mode == "normal":
                return tuple(filter(lambda face: len(face) == n + 1, self._simplex_set))
            elif self.mode == "gudhi":
                d=self.faces_dict[n]
                return tuple(OrderedDict(sorted(d.items(), key=lambda t: tuple(t[0]) )).keys())
        else:
            raise ValueError(
                f"dimension n should be larger than zero and not greater than {self.maxdim}\n"
                f"(maximal simplices dimension), got {n}"
            )


    def dic_order_faces(self, n):
        return sorted(self._n_faces(n))

    def get_sorted_entire_simplex_set(self):
        simpleces_lst = list(self._simplex_set)
        simpleces_lst.sort(key=lambda a: len(a))
        return simpleces_lst

    def get_simplex_tree(self):  # requires gudhi
        """
        get an empty simplex tree.
        """
        st = SimplexTree()
        return st

    def add_simplex(self,simplex):
        assert(isinstance(simplex,tuple) or isinstance(simplex,list) )
        simplex= tuple(sorted(simplex)) # put the simplex in cananical order
        if simplex is self._simplex_set:
            warn(f" simplex  {simplex} is already inserted.")
            return
        else:
            if self.mode == "gudhi": 
                raise NotImplementedError(" insertion is only implemented for normal mode, "+
                                          "gudhi mode only supports defining complexes from constructors.")

            elif self.mode == 'normal':
                    
                if self.constrain_dim == -1:
                    s_tree=SimplicialComplex._faces([simplex])
                else:
                    s_tree=SimplicialComplex._faces([simplex],self.constrain_dim)
                self._simplex_set = self._simplex_set.union(set(s_tree))
                if len(simplex) -1> self.maxdim:  
                    if self.constrain_dim==-1:    
                        self.maxdim = len(simplex) -1          
                    else:
                        self.maxdim = self.constrain_dim
    def insert_simplices_from(self,simplices):
        for s in simplices:
            self.insert_simplex(s)
    
    @staticmethod    
    def extract_simplices(simplex_tree ):
        """
        extract skeletons from gudhi simples tree
        
        Remark
            faces_dict[i] = X^i where X^i is the ith skeleton of the input SC X.
        
        """
        faces_dict = [OrderedDict() for _ in range(simplex_tree.dimension() + 1)]
        for simplex, _ in simplex_tree.get_skeleton( simplex_tree.dimension() ):
            k = len(simplex)
            faces_dict[k - 1][frozenset(simplex)] = len(faces_dict[k - 1])
        return faces_dict

    @staticmethod
    def get_edges_from_matrix(matrix):
        """
        Parameters
        ----------
        matrix : numpy or scipy array

        Returns
        -------
        edges : list of indices where the operator is not zero

        Rational:
        -------
         Most operaters (e.g. adjacencies/(co)boundary maps) that describe 
         connectivity of the simplicial complex
         can be described as a graph whose nodes are the simplices used to 
         construct the operator and whose edges correspond to the entries
         in the matrix where the operator is not zero.
         
         This property implies that many computations on simplicial complexes
         can be reduced to graph computations.
         
        """
        rows, cols = np.where(np.sign(np.abs(matrix)) == 1)
        edges = zip(rows.tolist(), cols.tolist())
        return edges

    # ---------- operators ---------------#

    def boundary_matrix_gudhi(self, d, signed = True, index =False):
        """
        get the boundary map using gudhi
        """
   
        if d == 0:
            boundary = dok_matrix(
                (1, len(self.faces_dict[d].items())), dtype=np.float32
            )
            boundary[0, 0 : len(self.faces_dict[d].items())] = 1
            return boundary.tocsr()
        idx_simplices, idx_faces, values = [], [], []
        for simplex, idx_simplex in self.faces_dict[d].items():
            for i, left_out in enumerate(np.sort(list(simplex))):
                idx_simplices.append(idx_simplex)
                values.append((-1) ** i)
                face = simplex.difference({left_out})
                idx_faces.append(self.faces_dict[d - 1][face])
        assert len(values) == (d + 1) * len(self.faces_dict[d])
        boundary = coo_matrix(
            (values, (idx_faces, idx_simplices)),
            dtype=np.float32,
            shape=(len(self.faces_dict[d - 1]), len(self.faces_dict[d])),
        )
        if index:
            if signed:
                return self.faces_dict[d],self.faces_dict[d-1], boundary
            else:
                return self.faces_dict[d],self.faces_dict[d-1], abs(boundary)            
        else:
            if signed:
                return boundary
            else:
                return abs(boundary)

    def boundary_matrix_normal(self, d, signed = True,index=False):
        source_simplices = self.dic_order_faces(d)
        target_simplices = self.dic_order_faces(d - 1)

        if len(target_simplices) == 0:
            S = dok_matrix((1, len(source_simplices)), dtype=np.float32)
            S[0, 0 : len(source_simplices)] = 1
        else:
            source_simplices_dict = {
                source_simplices[j]: j for j in range(len(source_simplices))
            }
            target_simplices_dict = {
                target_simplices[i]: i for i in range(len(target_simplices))
            }

            S = dok_matrix(
                (len(target_simplices), len(source_simplices)), dtype=np.float32
            )
            for source_simplex in source_simplices:
                for a in range(len(source_simplex)):
                    target_simplex = source_simplex[:a] + source_simplex[(a + 1) :]
                    i = target_simplices_dict[target_simplex]
                    j = source_simplices_dict[source_simplex]
                    if signed:
                        S[i, j] = -1 if a % 2 == 1 else 1
                    else:
                        S[i, j] = 1
        if index:
            return source_simplices, target_simplices, S
        else:
            return S

    def boundary_matrix(self, d, signed = True,index=False):

        if d >= 0 and d <= self.maxdim:
            if self.mode == "normal":
                return self.boundary_matrix_normal(d, signed,index)
            elif self.mode == "gudhi":
                return self.boundary_matrix_gudhi(d, signed,index)
        else:
            raise ValueError(
                f"d should be larget than zero and not greater than {self.maxdim} (maximal allowed dimension for simplices), got {d}"
            )

    def coboundary_matrix(self, d, signed = True,index=False):
        return self.boundary_matrix(d, signed,index).T

    def hodge_laplacian_matrix(self, d, signed = True,index=False):
        if d == 0:
            B_next = self.boundary_matrix(d + 1)
            L = B_next @ B_next.transpose()
        elif d < self.maxdim:
            B_next = self.boundary_matrix(d + 1)
            B = self.boundary_matrix(d)
            L = B_next @ B_next.transpose() + B.transpose() @ B
        elif d == self.maxdim:
            B = self.boundary_matrix(d)
            L = B.transpose() @ B
        else:
            raise ValueError(
                f"d should be larger than 0 and <= {self.maxdim} (maximal dimension simplices), got {d}"
            )
        if signed:
            return L
        else:
            return abs(L)

    def up_laplacian_matrix(self, d, signed = True):
        if d == 0:
            B_next = self.boundary_matrix(d + 1)
            L_up = B_next @ B_next.transpose()
        elif d < self.maxdim:
            B_next = self.boundary_matrix(d + 1)
            L_up = B_next @ B_next.transpose()
        else:

            raise ValueError(
                f"d should larger than 0 and <= {self.maxdim-1} (maximal dimension simplices-1), got {d}"
            )
        if signed:
            return L_up
        else:
            return abs(L_up)

    def down_laplacian_matrix(self, d, signed = True):
        if d <= self.maxdim and d > 0:
            B = self.boundary_matrix(d)
            L_down = B.transpose() @ B
        else:
            raise ValueError(
                f"d should be larger than 1 and <= {self.maxdim} (maximal dimension simplices), got {d}."
            )
        if signed:
            return L_down
        else:
            return abs(L_down)

    def adjacency_matrix(self, d, signed = False):

        L_up = self.up_laplacian_matrix(d, signed)
        L_up.setdiag(0)

        if signed:
            return L_up
        else:
            return abs(L_up)

    def coadjacency_matrix(self, d, signed = False):

        L_down = self.down_laplacian_matrix(d, signed)
        L_down.setdiag(0)
        if signed:
            return L_down
        else:
            return abs(L_down)
        
    def k_hop_boundary_matrix(self, d,k):
        Bd = self.boundary_matrix(d , signed = True)
        if d < self.maxdim and d >= 0:
            Ad = self.adjacency_matrix(d, signed = True)
        if d <= self.maxdim and d > 0:
            coAd = self.coadjacency_matrix(d , signed = True)
        if d == self.maxdim:
            return Bd @ np.power(coAd,k)
        elif d == 0 :
            return Bd @ np.power(Ad,k)
        else:            
            return Bd @ np.power(Ad,k)+ Bd @ np.power(coAd,k) 

    def k_hop_coboundary_matrix(self, d,k):
        BTd = self.coboundary_matrix(d , signed = True)
        if d < self.maxdim and d >= 0:
            Ad = self.adjacency_matrix(d, signed = True)
        if d <= self.maxdim and d > 0:
            coAd = self.coadjacency_matrix(d , signed = True)
        if d == self.maxdim:
            return np.power(coAd,k) @ BTd
        elif d == 0 :
            return np.power(Ad,k) @ BTd
        else:
            return  np.power(Ad,k) @ BTd + np.power(coAd,k) @ BTd 

    def is_connected(self):
        g = nx.Graph()
        for e in self.skeleton(1):
            g.add_edge(e[0],e[1])
        return nx.is_connected(g)
    
    def to_hypergraph(self):    
        graph = []
        for i in range(1,self.maxdim+1):
            edge = [ list(j) for j in  self._n_faces(i) ]
            graph = graph+edge
        return Hypergraph(graph, static = True)   
        
    def to_combinatorialcomplex(self):
        graph = []
        for i in range(1,self.maxdim+1):
            edge = [RankedEntity(uid=tuple(j),elements=list(j), rank=len(j)-1)  for j in  self._n_faces(i) ]
            graph = graph+edge     
        RES = RankedEntitySet("", graph) 
        return CombinatorialComplex(RES)
    

    #  -----------normalized operators------------------#

    def normalized_hodge_laplacian_matrix(self, d, signed = True):

        return SimplicialComplex.normalize_laplacian(
            self.hodge_laplacian_matrix(d=d), signed=signed
        )

    def normalized_down_laplacian_matrix(self, d):
        Ld = self.hodge_laplacian_matrix(d=d) 
        Ldown = self.down_laplacian_matrix(d=d) 
        out = SimplicialComplex.normalize_x_laplacian(Ld,Ldown)
        return out
    def normalized_up_laplacian_matrix(self, d):
        Ld = self.hodge_laplacian_matrix(d=d) 
        Lup = self.up_laplacian_matrix(d=d) 
        out = SimplicialComplex.normalize_x_laplacian(Ld,Lup)
        return out
    def normalized_coboundary_operator_matrix(self, d, signed = True, normalization="xu"):

        CoBd = self.coboundary_matrix(d, signed)

        if normalization == "row":
            return normalize(CoBd, norm="l1", axis=1)
        elif normalization == "kipf":
            return SimplicialComplex.asymmetric_kipf_normalization(CoBd)
        elif normalization == "xu":
            return SimplicialComplex.asymmetric_xu_normalization(CoBd)
        else:
            raise Exception("invalid normalization method entered.")

    def normalized_k_hop_coboundary_matrix(self, d, k, normalization="xu"):

        CoBd = self.k_hop_coboundary_matrix(d, k)

        if normalization == "row":
            return normalize(CoBd, norm="l1", axis=1)
        elif normalization == "kipf":
            return SimplicialComplex.asymmetric_kipf_normalization(CoBd)
        elif normalization == "xu":
            return SimplicialComplex.asymmetric_xu_normalization(CoBd)
        else:
            raise Exception("invalid normalization method entered.")

    def normalized_boundary_matrix(self, d, signed = True, normalization="xu"):

        Bd = self.boundary_matrix(d, signed)
        if normalization == "row":
            return normalize(Bd, norm="l1", axis=1)
        elif normalization == "kipf":
            return SimplicialComplex.asymmetric_kipf_normalization(Bd)
        elif normalization == "xu":
            return SimplicialComplex.asymmetric_xu_normalization(Bd)
        else:
            raise Exception("invalid normalization method entered.")

    def normalized_k_hop_boundary(self, d,k, normalization="xu"):

        Bd = self.k_hop_boundary_matrix(d,k)
        if normalization == "row":
            return normalize(Bd, norm="l1", axis=1)
        elif normalization == "kipf":
            return SimplicialComplex.asymmetric_kipf_normalization(Bd)
        elif normalization == "xu":
            return SimplicialComplex.asymmetric_xu_normalization(Bd)
        else:
            raise Exception("invalid normalization method entered.")


    def normalized_adjacency(self, d, signed = False, normalization="kipf"):
        """
        Args:
            d: dimenion of the higher order adjacency matrix
            signed: Boolean determines if the adj matrix is signed or not.

        return:
             D^{-0.5}* (adj(d)+Id)* D^{-0.5}.
        """
        # A_adjacency is an opt that maps a j-cochain to a k-cochain.
        #   shape [num_of_k_simplices num_of_j_simplices]
        A_adjacency = self.adjacency_matrix(d, signed=signed)
        A_adjacency = A_adjacency + eye(A_adjacency.shape[0])
        if normalization == "row":
            return normalize(A_adjacency, norm="l1", axis=1)
        elif normalization == "kipf":
            return SimplicialComplex.normalize_adjacency(A_adjacency)
        elif normalization == "xu":
            return SimplicialComplex.asymmetric_xu_normalization(A_adjacency)
        else:
            raise Exception("invalid normalization method entered.")

    def normalized_coadjacency(self, d, signed = False, normalization="xu"):
        """
        Args:
            d: dimenion of the higher order adjacency matrix
            signed: Boolean determines if the adj matrix is signed or not.

        return:
             D^{-0.5}* (co-adj(d)+Id)* D^{-0.5}.
        """
        # A_adjacency is an opt that maps a j-cochain to a k-cochain.
        #   shape [num_of_k_simplices num_of_j_simplices]
        A_coadjacency = self.coadjacency_matrix(d, signed=signed)
        A_coadjacency = A_coadjacency + eye(A_coadjacency.shape[0])
        if normalization == "row":
            return normalize(A_coadjacency, norm="l1", axis=1)
        elif normalization == "kipf":
            return SimplicialComplex.normalize_adjacency(A_coadjacency)
        elif normalization == "xu":
            return SimplicialComplex.asymmetric_xu_normalization(A_coadjacency)
        else:
            raise Exception("invalid normalization method entered.")

    @staticmethod
    def normalize_laplacian(L, signed=True):

        topeigen_val = spl.eigsh(L, k=1, which="LM", return_eigenvectors = False)[0]
        out = L.copy()
        out *= 1.0 / topeigen_val
        if signed:
            return out
        else:
            return abs(out)
    @staticmethod        
    def normalize_x_laplacian(L,Lx): # used to normalize the up or the down Laplacians
        assert(L.shape[0] == L.shape[1])
        topeig = spl.eigsh(L, k=1, which="LM", return_eigenvectors = False)[0]
        out = Lx.copy()
        out *= 1.0/topeig
        return out        

    @staticmethod
    def normalize_adjacency(A_opt):
        """
        Args:
            A_opt is an opt that maps a j-cochain to a k-cochain.
            shape [num_of_k_simplices num_of_j_simplices]

        return:
             D^{-0.5}* (A_opt)* D^{-0.5}.
        """
        rowsum = np.array(np.abs(A_opt).sum(1))
        r_inv_sqrt = np.power(rowsum, -0.5).flatten()
        r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
        r_mat_inv_sqrt = diags(r_inv_sqrt)
        A_opt_to = A_opt.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

        return coo_matrix(A_opt_to)

    @staticmethod
    def asymmetric_kipf_normalization(A_opt, is_sparse=True):
        """
        This version works for asymmetric matrices such as
        the coboundary matrices, as well as symmetric ones
        such as higher order adjacency.

        Args:
            A_opt is an opt that maps a j-cochain to a k-cochain.
            shape [num_of_k_simplices num_of_j_simplices]

        return:
            a normalized version of the operator A_opt:
                D_{i}^{-0.5}* (A_opt)* D_{j}^{-0.5}
                where Di = np.sum(A_opt, axis=1)
                and Dj = np.sum(A_opt, axis=0)
        """
        if is_sparse:
            rowsum = np.array(np.abs(A_opt).sum(1))
            colsum = np.array(np.abs(A_opt).sum(0))
            degree_mat_inv_sqrt_row = diags(np.power(rowsum, -0.5).flatten())
            degree_mat_inv_sqrt_col = diags(np.power(colsum, -0.5).flatten())
            degree_mat_inv_sqrt_row = degree_mat_inv_sqrt_row.toarray()
            degree_mat_inv_sqrt_col = degree_mat_inv_sqrt_col.toarray()
            degree_mat_inv_sqrt_row[np.isinf(degree_mat_inv_sqrt_row)] = 0.0
            degree_mat_inv_sqrt_col[np.isinf(degree_mat_inv_sqrt_col)] = 0.0
            degree_mat_inv_sqrt_row = coo_matrix(degree_mat_inv_sqrt_row)
            degree_mat_inv_sqrt_col = coo_matrix(degree_mat_inv_sqrt_col)

            normalized_operator = (
                A_opt.dot(degree_mat_inv_sqrt_col)
                .transpose()
                .dot(degree_mat_inv_sqrt_row)
            ).T.tocoo()
            return normalized_operator

        else:
            Di = np.sum(np.abs(A_opt), axis = 1)
            Dj = np.sum(np.abs(A_opt), axis = 0)
            inv_Dj = np.mat(np.diag(np.power(Dj, -0.5)))
            inv_Dj[np.isinf(inv_Dj)] = 0.0
            Di2 = np.mat(np.diag(np.power(Di, -0.5)))
            Di2[np.isinf(Di2)] = 0.0
            A_opt = np.mat(A_opt)
            G = Di2 * A_opt * inv_Dj
            return G

    @staticmethod
    def asymmetric_xu_normalization(A_opt, is_sparse = True):
        """
        This version works for asymmetric matrices such as
        the coboundary matrices, as well as symmetric ones
        such as higher order adjacency.

        Args:
            A_opt is an opt that maps a j-cochain to a k-cochain.
            shape [num_of_k_simplices num_of_j_simplices]

        return:
            a normalized version of the operator A_opt:
                D_{i}^{-1}* (A_opt)
                where Di = np.sum(A_opt, axis=1)
        """
        if is_sparse:
            rowsum = np.array(np.abs(A_opt).sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.0
            r_mat_inv = diags(r_inv)
            normalized_operator = r_mat_inv.dot(A_opt)
            return normalized_operator

        else:
            Di = np.sum(np.abs(A_opt), axis=1)
            Di2 = np.mat(np.diag(np.power(Di, -1)))
            Di2[np.isinf(Di2)] = 0.0
            A_opt = np.mat(A_opt)
            normalized_operator = Di2 * A_opt
            return normalized_operator
