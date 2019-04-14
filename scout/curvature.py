"""
Curvature Module
=================

This module implements a GPU-accelerated and CPU-only principal curvature calculations.

These routines are needed for curvature-based seeded watershed to segment densely-packed nuclei:
https://www.mdpi.com/2313-433X/2/4/31

"""

import logging
import itertools
import numpy as np
import torch


# Principal curvature calculations


def dx(x, xum=1.0):
    """Compute the gradient in the X direction

    Note that this function does not pad the image so the output is reduced
    in size.

    Parameters
    -----------
    x : ndarray
        Input array
    xum : float
        microns per voxel in the X direction

    Returns
    --------
    dx : ndarray
        The gradient in the X direction, reduced in size by one at each edge

    """
    return (x[1:-1, 1:-1, 2:] - x[1:-1, 1:-1, :-2]) / 2 / xum


def dy(x, yum=1.0):
    """Compute the gradient in the Y direction

    Note that this function does not pad the image so the output is reduced
    in size.

    Parameters
    -----------
    y : ndarray
        Input array
    yum : float
        The number of microns per voxel in the Y direction

    Returns
    --------
    dy : ndarray
        The gradient in the Y direction, reduced in size by one at each edge

    """
    return (x[1:-1, 2:, 1:-1] - x[1:-1, :-2, 1:-1]) / 2 / yum


def dz(x, zum=1.0):
    """Compute the gradient in the Z direction

    Note that this function does not pad the image so the output is reduced
    in size.

    Parameters
    -----------
    x : ndarray
        Input array
    zum : float
        microns per voxel in the z direction

    Returns
    --------
    dz : ndarray
        The gradient in the Z direction, reduced in size by one at each edge
    """
    return (x[2:, 1:-1, 1:-1] - x[:-2, 1:-1, 1:-1]) / 2 / zum


def gradient(x, zum=1.0, yum=1.0, xum=1.0):
    """Compute the gradient in all three directions

    Note that the images returned are reduced in size by 1 - there is no padding

    Parameters
    ------------
    x : ndarray
        Input array
    zum : float
        size of a voxel in the z direction - defaults to 1.0 micron
    yum : float
        size of a voxel in the y direction - defaults to 1.0 micron
    xum : float
        size of a voxel in the x direction - defaults to 1.0 micron

    Returns
    --------
    grad : tuple
        A tuple of the z, y, and x gradients.
    """
    return dz(x, zum), dy(x, yum), dx(x, xum)


def structure_tensor(dz, dy, dx):
    """Construct the structure tensor from the gradient

    The structure tensor is the cross product of the gradient with itself:
    dz * dz   dy * dz   dx * dz
    dz * dy   dy * dy   dx * dy
    dz * dx   dy * dx   dx * dx

    Note - the arrays of the structure tensor are reduced in size by 1 on
    each side to match the dimensions of the Hessian

    Parameters
    -----------
    dz : ndarray
        The gradient in the Z direction
    dy : ndarray
        The gradient in the Y direction
    dx : ndarray
        The gradient in the X direction

    Returns
    --------
    result : ndarray
        A 3 tuple of 3 tuples representing the structure tensor of the gradient

    """
    # Shorten dx, dy and dz by 1
    def shorten(x):
        return x[1:-1, 1:-1, 1:-1]
    dx = shorten(dx)
    dy = shorten(dy)
    dz = shorten(dz)
    result = [[None] * 3 for _ in range(3)]
    result[0][0] = dz * dz
    result[0][1] = result[1][0] = dy * dz
    result[0][2] = result[2][0] = dx * dz
    result[1][1] = dy * dy
    result[1][2] = result[2][1] = dx * dy
    result[2][2] = dx * dx
    return result


def _determinant2(x):
    """The determinant of a 2x2 matrix

    Parameters
    -----------
    x : ndarray
        A 2 x 2 matrix of arrays

    Returns
    --------
    det : ndarray
        An array that is the determinant of each 2 x 2 of x

    """
    return x[0][0] * x[1][1] - x[0][1] * x[1][0]


def _minor(x, i, j):
    """The minor matrix of x

    Parameters
    -----------
    i : int
        The column to eliminate
    j : int
        The row to eliminate

    Returns
    --------
    output : ndarray
        x without column i and row j

    """
    return [[x[n][m] for m in range(len(x[0])) if m != j]
            for n in range(len(x)) if n != i]


def _determinant3(x):
    """The determinant of a 3x3 matrix

    Parameters
    -----------
    x : ndarray
        A 3x3 matrix of arrays

    Return
    -------
    det : ndarray
        An array that is the determinant of each 3x3 of x

    """
    return x[0][0] * _determinant2(_minor(x, 0, 0)) -\
           x[0][1] * _determinant2(_minor(x, 0, 1)) +\
           x[0][2] * _determinant2(_minor(x, 0, 2))


def _transpose(x):
    """The transpose of a matrix

    Parameters
    -----------
    x : ndarray
        An NxM matrix, e.g. a list of lists

    Returns
    --------
    t : list
        A list of lists with the rows and columns reversed

    """
    return [[x[j][i] for j in range(len(x))] for i in range(len(x[0]))]


def _sign(i, j):
    """+1 if i+j is even, -1 if i+j is odd"""
    return 1 if (i + j) % 2 == 0 else -1


def _inverse(x):
    """The inverse of a 3x3 matrix

    X . _inverse(x) = I

    Parameters
    -----------
    x : ndarray
        A 3x3 matrix (e.g. a list of lists) of arrays

    Returns
    --------
    inv : ndarray
        a similar 3x3 matrix where each 3x3 is the inverse of the matrix

    """
    det_x = _determinant3(x)
    x_t = _transpose(x)
    return [[_sign(i, j) * _determinant2(_minor(x_t, i, j)) / det_x
             for j in range(len(x[0]))]
           for i in range(len(x))]


def weingarten(x, zum=1, yum=1, xum=1):
    """The Weingarten shape operator on a 3D image

    See http://mathworld.wolfram.com/ShapeOperator.html for instance.

    Parameters
    -----------
    x : ndarray
        The 3-D image to be processed
    zum : float
        size of a voxel in the z direction - defaults to 1.0 micron
    yum : float
        size of a voxel in the y direction - defaults to 1.0 micron
    xum : float
        size of a voxel in the x direction - defaults to 1.0 micron

    Returns
    --------
    weingarten : tuple
        A 3 tuple of 3 tuples representing the 3 x 3 matrix of the
        shape operator per voxel. The 3D elements of the matrix are reduced in
        size by 2 at each edge (a total of 4 voxels smaller in each dimension)
        because of the double differentiation of the Hessian.

    """
    dz_, dy_, dx_ = gradient(x, zum=zum, yum=yum, xum=xum)
    dzz, dzy, dzx = gradient(dz_, zum=zum, yum=yum, xum=xum)
    dyz = dzy
    dxz = dzx
    dyy = dy(dy_, yum=yum)
    dxy = dyx = dy(dx_, yum=yum)
    dxx = dx(dx_, xum=xum)
    H = [[dzz, dzy, dzx],
         [dyz, dyy, dyx],
         [dxz, dxy, dxx]]
    S = structure_tensor(dz_, dy_, dx_)
    l = torch.sqrt(1 + torch.sqrt(S[0][0] + S[1][1] + S[2][2]))
    B = [[S[0][0] + 1, S[0][1], S[0][2]],
         [S[1][0], S[1][1] + 1, S[1][2]],
         [S[2][0], S[1][2], S[2][2] + 1]]
    del S
    inv_B = _inverse(B)
    del B
    A = [[a * b / l for a, b in zip(aa, bb)] for aa, bb in zip(H, inv_B)]
    return A


def _sq(x):
    """The square of an array (like np.square)

    Parameters
    -----------
    x : ndarray
        Input array

    Returns
    --------
    sq : ndarray
        the elementwise square of the array

    """
    return x * x


def eigen3(A):
    """The eigenvalues of a 3x3 matrix of arrays

    Parameters
    ------------
    A : ndarray
        a 3x3 matrix of arrays, e.g. a list of lists

    Returns
    --------
    eigenvals : tuple
        a 3 tuple of arrays - the eigenvalues of each 3x3 of the matrix in ascending order.

    """
    #
    # From https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3%C3%973_matrices
    #
    p1 = _sq(A[0][1]) + _sq(A[0][2]) + _sq(A[1][2])
    q = (A[0][0] + A[1][1] + A[2][2])/3
    p2 = _sq(A[0][0] - q) + _sq(A[1][1] - q) + _sq(A[2][2] - q) + 2 * p1
    p = torch.sqrt(p2 / 6)
    del p2
    B = [[(1 / p) * ((A[i][j] - q) if i == j else A[i][j])
          for j in range(3)] for i in range(3)]
    r = _determinant3(B) / 2
    del B
    r = torch.clamp(r, -1, 1)
    phi = torch.acos(r) / 3
    eig1 = q + 2 * p * torch.cos(phi)
    eig3 = q + 2 * p * torch.cos(phi + (2*np.pi/3))
    del r
    del phi
    del p
    # the eigenvalues satisfy eig3 <= eig2 <= eig1
    eig2 = 3 * q - eig1 - eig3
    del q
    #
    # Sort in ascending order
    #
    e1gte2 = torch.lt(eig1, eig2)
    eig1a = torch.where(e1gte2, eig1, eig2)
    eig2a = torch.where(e1gte2, eig2, eig1)
    del e1gte2
    del eig1
    del eig2
    e1agte3 = torch.lt(eig1a, eig3)
    eig1 = torch.where(e1agte3, eig1a, eig3)
    eig3a = torch.where(e1agte3, eig3, eig1a)
    del e1agte3
    del eig1a
    del eig3
    e2agte3a = torch.lt(eig2a, eig3a)
    eig2 = torch.where(e2agte3a, eig2a, eig3a)
    eig3 = torch.where(e2agte3a, eig3a, eig2a)
    return eig1, eig2, eig3


def eigvals_of_weingarten(x, ew_block_size=64,
                          zum=1, yum=1, xum=1):
    """Find the eigenvalues of the weingarten operator

    Parameters
    ------------
    x : ndarray
        an NxMxP 3D array
    ew_block_size : int
        the block size for the blocks to be processed. The algorithm needs approximately 128 bytes per voxel processed.
    zum : float
        size of a voxel in the z direction - defaults to 1.0 micron
    yum : float
        size of a voxel in the y direction - defaults to 1.0 micron
    xum : float
        size of a voxel in the x direction - defaults to 1.0 micron

    Returns
    --------
    eigvals : ndarray
        an NxMxPx3 array of the 3 eigenvalues of the weingarten operator for the space.

    """
    x0 = np.arange(0, x.shape[2], ew_block_size)
    x1 = x0 + ew_block_size
    x1[-1] = x.shape[2]

    y0 = np.arange(0, x.shape[2], ew_block_size)
    y1 = y0 + ew_block_size
    y1[-1] = x.shape[1]

    z0 = np.arange(0, x.shape[0], ew_block_size)
    z1 = z0 + ew_block_size
    z1[-1] = x.shape[0]

    xpad = np.pad(x, 2, "reflect")
    result = np.zeros((x.shape[0], x.shape[1], x.shape[2], 3), x.dtype)
    a = [None, None]
    e = [None, None]
    try:
        #
        # Create a pipeline of operations where the GPU operation is interleaved
        # with the transfer of the last result to CPU memory and the transfer of
        # the next block of memory to the GPU.
        #
        def do_copy_to_gpu(x0a, x1a, y0a, y1a, z0a, z1a, idx):
            logging.debug("Starting copy to GPU: %d:%d, %d:%d, %d:%d idx=%d" %
                          (x0a, x1a, y0a, y1a, z0a, z1a, idx))
            a[idx] = torch.from_numpy(
                xpad[z0a:z1a + 4, y0a:y1a + 4, x0a:x1a + 4]).cuda()
            logging.debug("Finished copy to GPU: %d:%d, %d:%d, %d:%d idx=%d" %
                          (x0a, x1a, y0a, y1a, z0a, z1a, idx))

        def do_eigenvalues_of_weingarten(idx):
            logging.debug("Starting eigenvalues of weingarten, idx=%d" % idx)
            #zumc = torch.from_numpy(np.array([zum], np.float32)).expand_as(a[idx]).cuda()
            #yumc = torch.from_numpy(np.array([yum], np.float32)).expand_as(a[idx]).cuda()
            #xumc = torch.from_numpy(np.array([xum], np.float32)).expand_as(a[idx]).cuda()

            e[idx] = eigen3(weingarten(a[idx], zum=zum, yum=yum, xum=xum))
            logging.debug("Finished eigenvalues of weingarten, idx=%d" % idx)

        def do_copy_from_gpu(x0a, x1a, y0a, y1a, z0a, z1a, idx):
            logging.debug("Starting copy from GPU: %d:%d, %d:%d, %d:%d idx=%d" %
                          (x0a, x1a, y0a, y1a, z0a, z1a, idx))
            e1, e2, e3 = e[idx]
            result[z0a:z1a, y0a:y1a, x0a:x1a, 0] = e1.cpu().numpy()
            result[z0a:z1a, y0a:y1a, x0a:x1a, 1] = e2.cpu().numpy()
            result[z0a:z1a, y0a:y1a, x0a:x1a, 2] = e3.cpu().numpy()
            logging.debug("Finished copy from GPU: %d:%d, %d:%d, %d:%d idx=%d" %
                          (x0a, x1a, y0a, y1a, z0a, z1a, idx))

        to_gpu = []
        eow = []
        from_gpu = []
        idx = 0
        for (x0a, x1a), (y0a, y1a), (z0a, z1a) in itertools.product(
            zip(x0, x1), zip(y0, y1), zip(z0, z1)):
            to_gpu.append(
                lambda x0a=x0a, x1a=x1a, y0a=y0a, y1a=y1a, z0a=z0a, z1a=z1a,
                       idx=idx:
                       do_copy_to_gpu(x0a, x1a, y0a, y1a, z0a, z1a, idx))
            eow.append(lambda idx=idx: do_eigenvalues_of_weingarten(idx))
            from_gpu.append(
                lambda x0a=x0a, x1a=x1a, y0a=y0a, y1a=y1a, z0a=z0a, z1a=z1a,
                       idx=idx:
                       do_copy_from_gpu(x0a, x1a, y0a, y1a, z0a, z1a, idx))
            idx = 1 - idx

        operations = [to_gpu[0], eow[0]] + \
                      sum(map(list,
                              zip(to_gpu[1:], eow[1:], from_gpu[:-1])), []) +\
                     [from_gpu[-1]]
        [_() for _ in operations]
    finally:
        logging.debug("Deleting intermediate variables")
        del a
        del e
        del to_gpu
        del eow
        del from_gpu
        del operations

        logging.debug("Emptying CUDA cache")
        torch.cuda.empty_cache()
    logging.debug("Finishing eigvals_of_weingarten")
    return result


# CPU implementation


def gradient_numpy(data, zum=1.0, yum=1.0, xum=1.0):
    """Compute the gradient in units of intensity / micron

    Parameters
    -----------
    data : ndarray
        3d numpy array
    zum : float
        size of a voxel in the z direction - defaults to 1.0 micron
    yum : float
        size of a voxel in the y direction - defaults to 1.0 micron
    xum : float
        size of a voxel in the x direction - defaults to 1.0 micron

    Returns
    ---------
    grad : ndarray
        A 4-dimensional matrix with the last dimension being the z==0, y==1, x==2 selector of the gradient direction
    """
    fz, fy, fx = np.gradient(data, edge_order=2)
    grad = np.zeros((*data.shape, 3))
    grad[..., 0] = fz / zum
    grad[..., 1] = fy / yum
    grad[..., 2] = fx / xum
    return grad


def structure_tensor_numpy(grad):
    S = np.zeros((*grad.shape, 3))
    fz = grad[..., 0]
    fy = grad[..., 1]
    fx = grad[..., 2]
    S[..., 0, 0] = fz**2
    S[..., 1, 1] = fy**2
    S[..., 2, 2] = fx**2
    S[..., 0, 1] = fz*fy
    S[..., 0, 2] = fz*fx
    S[..., 1, 2] = fy*fx
    S[..., 1, 0] = fz*fy
    S[..., 2, 0] = fz*fx
    S[..., 2, 1] = fy*fx
    return S


def hessian_numpy(data, zum=1.0, yum=1.0, xum=1.0):
    microns = np.array([zum, yum, xum]).reshape(3, 1, 1, 1)
    grad = np.gradient(data) / microns
    n_dims = len(grad)
    H = np.zeros((*data.shape, n_dims, n_dims))
    for i, first_deriv in enumerate(grad):
        for j in range(i, n_dims):
            second_deriv = np.gradient(first_deriv, axis=j) / microns[j]
            H[..., i, j] = second_deriv
            if i != j:
                H[..., j, i] = second_deriv
    return H


def weingarten_numpy(g, zum=1.0, yum=1.0, xum=1.0):
    grad = gradient_numpy(g, zum=zum, yum=yum, xum=xum)
    l = np.sqrt(1+np.linalg.norm(grad, axis=-1))
    H = hessian_numpy(g, zum=zum, yum=yum, xum=xum)
    S = structure_tensor_numpy(grad)
    L = np.zeros(S.shape)
    for i in range(3):
        for j in range(3):
            L[..., i, j] = l
    eye = np.zeros(S.shape)
    eye[..., 0, 0] = 1
    eye[..., 1, 1] = 1
    eye[..., 2, 2] = 1
    B = S+eye
    A = H*np.linalg.inv(B)/L
    return A


def eigvals_of_weingarten_numpy(g, zum=1.0, yum=1.0, xum=1.0):
    return np.linalg.eigvalsh(weingarten_numpy(g, zum=zum, yum=yum, xum=xum))
