"""
Integer-matrix and row utilities: row matching between arrays (sort-based and memory-bounded), Smith and Hermite
normal forms, saturated integral kernels, GLSM charge matrices and point configurations. Requires NumPy >= 1.24.
"""
from fractions import Fraction
from typing import overload, Literal
import numpy as np
from numpy.typing import ArrayLike
from flint import fmpz_mat
from sympy import Matrix
from sympy.matrices.normalforms import smith_normal_decomp
from cytools.utils import integral_nullspace

_ROW_BUFFER = 1 << 20   # maximal number of elements of a temporary row-comparison array

def _row_counts(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    **Description:**
    Counts, for every row of `A`, the rows of `B` equal to it, with NumPy's elementwise equality (so `1 == 1.0` and
    `-0.0 == 0.0`). For large numeric inputs of equal width each row is viewed as a single value: the rows of `B` are
    sorted once and the rows of `A` are found by binary search, in O((len(A)+len(B)) log len(B)) time without a pairwise
    comparison array. Otherwise the rows are compared pairwise in blocks of at most `_ROW_BUFFER` elements.
    **Arguments:**
    - `A (np.ndarray)`: Two-dimensional array whose rows are counted.
    - `B (np.ndarray)`: Two-dimensional array of the rows to match against.
    **Returns:**
    - `np.ndarray`: Integer array of length `len(A)` with the number of rows of `B` equal to each row of `A`.
    """
    A, B = np.asarray(A), np.asarray(B)
    counts = np.zeros(len(A), dtype=np.intp)
    width = np.broadcast_shapes(A.shape[1:], B.shape[1:])[0]
    if not len(A) or not len(B):
        return counts
    if width == 0:
        counts.fill(len(B))
        return counts
    numeric = A.dtype.kind in "biufc" and B.dtype.kind in "biufc"
    if numeric:
        dtype_a, dtype_b, _ = np.equal.resolve_dtypes((A.dtype, B.dtype, None))
        numeric = dtype_a == dtype_b
    if not numeric or A.shape[1] != B.shape[1] or min(len(A), len(B)) == 1 or len(A) * len(B) * width <= _ROW_BUFFER:
        nb = min(len(B), max(1, _ROW_BUFFER // width))
        na = max(1, _ROW_BUFFER // (nb * width))
        for i in range(0, len(A), na):
            for j in range(0, len(B), nb):
                counts[i:i + na] += (A[i:i + na, None, :] == B[None, j:j + nb, :]).all(axis=2).sum(axis=1)
        return counts
    if width == 1:
        row_dtype = dtype_a
    elif dtype_a.kind in "iu":
        row_dtype = np.dtype((np.void, width * dtype_a.itemsize))
    else:
        row_dtype = np.dtype([(f"f{i}", dtype_a) for i in range(width)])
    a = np.ascontiguousarray(A, dtype=dtype_a).view(row_dtype).ravel()
    b = np.ascontiguousarray(B, dtype=dtype_a).view(row_dtype).ravel()
    rows, frequencies = np.unique(b, return_counts=True)
    positions = np.minimum(rows.searchsorted(a), len(rows) - 1)
    matches = a == rows[positions]
    counts[matches] = frequencies[positions[matches]]
    return counts

def contains_row(arr: np.ndarray, target: ArrayLike) -> np.bool_:
    """
    **Description:**
    Determines whether the one-dimensional array `target` occurs as a row of `arr`. The rows are compared in blocks of
    bounded size, stopping at the first match.
    **Arguments:**
    - `arr (np.ndarray)`: The two-dimensional array to search.
    - `target (ArrayLike)`: The one-dimensional row to find.
    **Returns:**
    - `np.bool_`: `True` if `target` is a row of `arr`, otherwise `False`.
    """
    arr, target = np.asarray(arr), np.asarray(target)
    step = max(1, _ROW_BUFFER // max(arr.shape[1], target.size, 1))
    for i in range(0, len(arr), step):
        if (arr[i:i + step] == target).all(axis=1).any():
            return np.bool_(True)
    return np.bool_(False)

def contains_rows(arr: np.ndarray, targets: np.ndarray) -> np.bool_:
    """
    **Description:**
    Determines whether every row of `targets` occurs as a row of `arr`. Repeated target rows need to occur only once in
    `arr`.
    **Arguments:**
    - `arr (np.ndarray)`: The two-dimensional array to search.
    - `targets (np.ndarray)`: The two-dimensional array of target rows.
    **Returns:**
    - `np.bool_`: `True` if every target row occurs in `arr` (in particular if `targets` is empty), otherwise `False`.
    """
    return (_row_counts(targets, arr) > 0).all()

def get_same_rows(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    **Description:**
    Computes the rows of `A` that also occur as rows of `B`, in the order of `A`. Each such row is repeated once for every
    row of `B` equal to it, so the result contains duplicates if `B` does.
    **Arguments:**
    - `A (np.ndarray)`: First two-dimensional array.
    - `B (np.ndarray)`: Second two-dimensional array.
    **Returns:**
    - `np.ndarray`: The rows of `A` that occur in `B`, with the multiplicities described above.
    """
    A = np.asarray(A)
    return np.repeat(A, _row_counts(A, B), axis=0)

def same_rows(A: ArrayLike, B: ArrayLike) -> bool:
    """
    **Description:**
    Determines whether `A` and `B` contain the same rows with the same multiplicities, independent of row order. Integer
    arrays of equal dtype are compared by sorting every row as a single value, other numeric arrays of equal dtype by a
    lexicographic sort compared in blocks, and all remaining inputs with `np.unique`.
    **Arguments:**
    - `A (ArrayLike)`: First two-dimensional array.
    - `B (ArrayLike)`: Second two-dimensional array.
    **Returns:**
    - `bool`: `True` if `A` and `B` have the same rows with the same multiplicities, otherwise `False`.
    """
    A, B = np.asarray(A), np.asarray(B)
    if A.shape != B.shape:
        return False
    if A.dtype == B.dtype and A.dtype.kind in "biufc":
        if A.shape[1] == 0:
            return True
        if A.dtype.kind in "iu":
            row_dtype = np.dtype((np.void, A.shape[1] * A.dtype.itemsize))
            a = np.sort(np.ascontiguousarray(A).view(row_dtype).ravel())
            b = np.sort(np.ascontiguousarray(B).view(row_dtype).ravel())
            return np.array_equal(a, b)
        ia, ib = np.lexsort(A.T[::-1]), np.lexsort(B.T[::-1])
        step = max(1, _ROW_BUFFER // A.shape[1])
        for i in range(0, len(A), step):
            if not np.array_equal(A[ia[i:i + step]], B[ib[i:i + step]]):
                return False
        return True
    rows_a, counts_a = np.unique(A, axis=0, return_counts=True)
    rows_b, counts_b = np.unique(B, axis=0, return_counts=True)
    return np.array_equal(rows_a, rows_b) and np.array_equal(counts_a, counts_b)

def get_indices(arr: np.ndarray, targets: ArrayLike) -> np.ndarray:
    """
    **Description:**
    Finds the indices of rows of `arr` that also occur among the rows of `targets`.
    **Arguments:**
    - `arr (np.ndarray)`: The two-dimensional array to search.
    - `targets (ArrayLike)`: The target row (one-dimensional) or rows (two-dimensional).
    **Returns:**
    - `np.ndarray`: The indices of matching rows in `arr`, in increasing order.
    """
    return np.flatnonzero(_row_counts(arr, np.atleast_2d(targets)))

def get_index(arr: np.ndarray, target: ArrayLike) -> np.ndarray:
    """
    **Description:**
    Finds the indices of rows of `arr` equal to the one-dimensional array `target`, comparing in blocks of bounded size.
    **Arguments:**
    - `arr (np.ndarray)`: The two-dimensional array to search.
    - `target (ArrayLike)`: The target row.
    **Returns:**
    - `np.ndarray`: The indices of rows of `arr` equal to `target`, in increasing order.
    """
    arr, target = np.asarray(arr), np.asarray(target)
    step = max(1, _ROW_BUFFER // max(arr.shape[1], target.size, 1))
    matches = np.empty(len(arr), dtype=bool)
    for i in range(0, len(arr), step):
        matches[i:i + step] = (arr[i:i + step] == target).all(axis=1)
    return np.flatnonzero(matches)

def row_difference(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    **Description:**
    Computes all rows of `A` that do not occur as rows of `B`, keeping their order and multiplicities in `A`.
    **Arguments:**
    - `A (np.ndarray)`: First two-dimensional array.
    - `B (np.ndarray)`: Second two-dimensional array.
    **Returns:**
    - `np.ndarray`: The rows of `A` that are not rows of `B`.
    """
    A = np.asarray(A)
    return A[_row_counts(A, B) == 0]

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