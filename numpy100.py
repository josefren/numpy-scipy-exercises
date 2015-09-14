"""
From
https://github.com/rougier/numpy-100
"""

import numpy as np

# Create a null vector of size 10"
def zeroes_np(n):
    return np.zeros(n)


def zeroes_py(n):
    return [0] * n

# ================================================================
# Create a null vector of size 10 but the fifth value which is 1
def zeroes_one_unit_np(size, pos):
    a = np.zeros(size)
    a[pos] = 1
    return a


def zeroes_one_unit_py(size, pos):
    a = [0] * size
    a[pos] = 1
    return a

# ================================================================
# Create a vector with values ranging from 10 to 49
def range_np(start, end):
    return np.arange(start, end)


def range_py(start, end):
    return list(range(start, end))

# Create a 3x3 matrix with values ranging from 0 to 8
def square_matrix_range_np(size):
    """
    >>> square_matrix_range_np(3)
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    """
    max_value = size ** 2
    return np.arange(max_value).reshape(size, size)


def square_matrix_range_py(size):
    """
    >>> square_matrix_range_py(3)
    [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """
    return [[i * size + j for j in range(size)] for i in range(size)]

# Find indices of non-zero elements from [1,2,0,0,4,0]
def non_zero_indicies_np(ar):
    """
    >>> non_zero_indicies_np([1, 2, 0, 0, 4, 0])
    array([0, 1, 4])
    """
    return np.nonzero(ar)[0]

def non_zero_indicies_py(ar):
    """
    >>> non_zero_indicies_py([1, 2, 0, 0, 4, 0])
    [0, 1, 4]
    """
    return [i for i in range(len(ar)) if ar[i]]

# Create a 3x3 identity matrix
def identity_matrix_np(size):
    return np.eye(size)

def identity_matrix_py(size):
    """
    >>> identity_matrix_py(3)
    [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    """
    return [[int(r == c) for c in range(size)] for r in range(size)]


# Z = np.eye(3)
# print(Z)
# Create a 5x5 matrix with values 1,2,3,4 just below the diagonal
#
# Z = np.diag(1+np.arange(4),k=-1)
# print(Z)
# Create a 3x3x3 array with random values
#
# Z = np.random.random((3,3,3))
# print(Z)
# Create a 8x8 matrix and fill it with a checkerboard pattern
#
# Z = np.zeros((8,8),dtype=int)
# Z[1::2,::2] = 1
# Z[::2,1::2] = 1
# print(Z)
# Create a 10x10 array with random values and find the minimum and maximum values
#
# Z = np.random.random((10,10))
# Zmin, Zmax = Z.min(), Z.max()
# print(Zmin, Zmax)
# Create a checkerboard 8x8 matrix using the tile function
#
# Z = np.tile( np.array([[0,1],[1,0]]), (4,4))
# print(Z)
# Normalize a 5x5 random matrix (between 0 and 1)
#
# Z = np.random.random((5,5))
# Zmax, Zmin = Z.max(), Z.min()
# Z = (Z - Zmin)/(Zmax - Zmin)
# print(Z)
# Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)
#
# Z = np.dot(np.ones((5,3)), np.ones((3,2)))
# print(Z)
# Create a 5x5 matrix with row values ranging from 0 to 4
#
# Z = np.zeros((5,5))
# Z += np.arange(5)
# print(Z)
# Create a vector of size 10 with values ranging from 0 to 1, both excluded
#
# Z = np.linspace(0,1,12,endpoint=True)[1:-1]
# print(Z)
# Create a random vector of size 10 and sort it
#
# Z = np.random.random(10)
# Z.sort()
# print(Z)
# Consider two random array A anb B, check if they are equal.
#
# A = np.random.randint(0,2,5)
# B = np.random.randint(0,2,5)
# equal = np.allclose(A,B)
# print(equal)
# Create a random vector of size 30 and find the mean value
#
# Z = np.random.random(30)
# m = Z.mean()
# print(m)
# Make an array immutable (read-only)
#
# Z = np.zeros(10)
# Z.flags.writeable = False
# Z[0] = 1
# Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates
#
# Z = np.random.random((10,2))
# X,Y = Z[:,0], Z[:,1]
# R = np.sqrt(X**2+Y**2)
# T = np.arctan2(Y,X)
# print(R)
# print(T)
# Create random vector of size 10 and replace the maximum value by 0
#
# Z = np.random.random(10)
# Z[Z.argmax()] = 0
# print(Z)
# Create a structured array with x and y coordinates covering the [0,1]x[0,1] area.
#
# Z = np.zeros((10,10), [('x',float),('y',float)])
# Z['x'], Z['y'] = np.meshgrid(np.linspace(0,1,10),
#                              np.linspace(0,1,10))
# print(Z)
# Print the minimum and maximum representable value for each numpy scalar type
#
# for dtype in [np.int8, np.int32, np.int64]:
#    print(np.iinfo(dtype).min)
#    print(np.iinfo(dtype).max)
# for dtype in [np.float32, np.float64]:
#    print(np.finfo(dtype).min)
#    print(np.finfo(dtype).max)
#    print(np.finfo(dtype).eps)
# Create a structured array representing a position (x,y) and a color (r,g,b)
#
#  Z = np.zeros(10, [ ('position', [ ('x', float, 1),
#                                    ('y', float, 1)]),
#                     ('color',    [ ('r', float, 1),
#                                    ('g', float, 1),
#                                    ('b', float, 1)])])
# print(Z)
# Consider a random vector with shape (100,2) representing coordinates, find point by point distances
#
# Z = np.random.random((10,2))
# X,Y = np.atleast_2d(Z[:,0]), np.atleast_2d(Z[:,1])
# D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
# print(D)
#
# # Much faster with scipy
# import scipy
# # Thanks Gavin Heverly-Coulson (#issue 1)
# import scipy.spatial
#
# Z = np.random.random((10,2))
# D = scipy.spatial.distance.cdist(Z,Z)
# print(D)
# Generate a generic 2D Gaussian-like array
#
# X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
# D = np.sqrt(X*X+Y*Y)
# sigma, mu = 1.0, 0.0
# G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
# print(G)
# Subtract the mean of each row of a matrix
#
# # Author: Warren Weckesser
#
# X = np.random.rand(5, 10)
#
# # Recent versions of numpy
# Y = X - X.mean(axis=1, keepdims=True)
#
# # Older versions of numpy
# Y = X - X.mean(axis=1).reshape(-1, 1)
# How to tell if a given 2D array has null columns ?
#
# # Author: Warren Weckesser
#
# Z = np.random.randint(0,3,(3,10))
# print((~Z.any(axis=0)).any())
# Find the nearest value from a given value in an array
#
# Z = np.random.uniform(0,1,10)
# z = 0.5
# m = Z.flat[np.abs(Z - z).argmin()]
# print(m)
# Consider the following file:
#
# 1,2,3,4,5
# 6,,,7,8
# ,,9,10,11
# How to read it ?
#
# Z = np.genfromtxt("missing.dat", delimiter=",")
# Consider a generator function that generates 10 integers and use it to build an array
#
# def generate():
#     for x in xrange(10):
#         yield x
# Z = np.fromiter(generate(),dtype=float,count=-1)
# print(Z)
# Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices) ?
#
# # Author: Brett Olsen
#
# Z = np.ones(10)
# I = np.random.randint(0,len(Z),20)
# Z += np.bincount(I, minlength=len(Z))
# print(Z)
# How to accumulate elements of a vector (X) to an array (F) based on an index list (I) ?
#
# # Author: Alan G Isaac
#
# X = [1,2,3,4,5,6]
# I = [1,3,9,3,4,1]
# F = np.bincount(I,X)
# print(F)
# Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors
#
# # Author: Nadav Horesh
#
# w,h = 16,16
# I = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
# F = I[...,0]*256*256 + I[...,1]*256 +I[...,2]
# n = len(np.unique(F))
# print(np.unique(I))
# Considering a four dimensions array, how to get sum over the last two axis at once ?
#
# A = np.random.randint(0,10,(3,4,3,4))
# sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
# print(sum)
# Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset indices ?
#
# # Author: Jaime Fernández del Río
#
# D = np.random.uniform(0,1,100)
# S = np.random.randint(0,10,100)
# D_sums = np.bincount(S, weights=D)
# D_counts = np.bincount(S)
# D_means = D_sums / D_counts
# print(D_means)
#
# I often need to compute the equivalent of
# How to get the diagonal of a dot product ?
#
# # Author: Mathieu Blondel
#
# # Slow version
# np.diag(np.dot(A, B))
#
# # Fast version
# np.sum(A * B.T, axis=1)
#
# # Faster version
# np.einsum("ij,ji->i", A, B).
# Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value ?
#
# # Author: Warren Weckesser
#
# Z = np.array([1,2,3,4,5])
# nz = 3
# Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz))
# Z0[::nz+1] = Z
# print(Z0)
# Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5) ?
#
# A = np.ones((5,5,3))
# B = 2*np.ones((5,5))
# print(A * B[:,:,None])
# How to swap two rows of an array ?
#
# # Author: Eelco Hoogendoorn
#
# A = np.arange(25).reshape(5,5)
# A[[0,1]] = A[[1,0]]
# print(A)
# Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1])
#
# # Author: Joe Kington / Erik Rigtorp
# from numpy.lib import stride_tricks
#
# def rolling(a, window):
#     shape = (a.size - window + 1, window)
#     strides = (a.itemsize, a.itemsize)
#     return stride_tricks.as_strided(a, shape=shape, strides=strides)
# Z = rolling(np.arange(10), 3)
# print(Z)
# Consider a set of 10 triplets describing 10 triangles (with shared vertices), find the set of unique line segments composing all the triangles.
#
# # Author: Nicolas P. Rougier
#
# faces = np.random.randint(0,100,(10,3))
# F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
# F = F.reshape(len(F)*3,2)
# F = np.sort(F,axis=1)
# G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
# G = np.unique(G)
# print(G)
# Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C ?
#
# # Author: Jaime Fernández del Río
#
# C = np.bincount([1,1,2,3,4,4,6])
# A = np.repeat(np.arange(len(C)), C)
# print(A)
# How to compute averages using a sliding window over an array ?
#
# # Author: Jaime Fernández del Río
#
# def moving_average(a, n=3) :
#     ret = np.cumsum(a, dtype=float)
#     ret[n:] = ret[n:] - ret[:-n]
#     return ret[n - 1:] / n
# Z = np.arange(20)
# print(moving_average(Z, n=3))
# How to get the documentation of the numpy add function from the command line ?
#
# python -c "import numpy; numpy.info(numpy.add)"
# How to negate a boolean, or to change the sign of a float inplace ?
#
# # Author: Nathaniel J. Smith
#
# Z = np.random.randint(0,2,100)
# np.logical_not(arr, out=arr)
#
# Z = np.random.uniform(-1.0,1.0,100)
# np.negative(arr, out=arr)
# Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3])
#
# # Author: Robert Kern
#
# Z = np.random.randint(0,5,(10,3))
# E = np.logical_and.reduce(Z[:,1:] == Z[:,:-1], axis=1)
# U = Z[~E]
# print(Z)
# print(U)
# Convert a vector of ints into a matrix binary representation.
#
# # Author: Warren Weckesser
#
# I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
# B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
# print(B[:,::-1])
#
# # Author: Daniel T. McDonald
#
# I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128], dtype=np.uint8)
# print(np.unpackbits(I[:, np.newaxis], axis=1))
# Consider 2 sets of points P0,P1 describing lines (2d) and a point p, how to compute distance from p to each line i (P0[i],P1[i]) ?
#
# def distance(P0, P1, p):
#     T = P1 - P0
#     L = (T**2).sum(axis=1)
#     U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
#     U = U.reshape(len(U),1)
#     D = P0 + U*T - p
#     return np.sqrt((D**2).sum(axis=1))
#
# P0 = np.random.uniform(-10,10,(10,2))
# P1 = np.random.uniform(-10,10,(10,2))
# p  = np.random.uniform(-10,10,( 1,2))
# print(distance(P0, P1, p))
# Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i]) ?
#
# # Author: Italmassov Kuanysh
# # based on distance function from previous question
# P0 = np.random.uniform(-10, 10, (10,2))
# P1 = np.random.uniform(-10,10,(10,2))
# p = np.random.uniform(-10, 10, (10,2))
# print np.array([distance(P0,P1,p_i) for p_i in p])
# Consider an arbitrary array, write a function that extract a subpart with a fixed shape and centered on a given element (pad with a fill value when necessary)
#
# # Author: Nicolas Rougier
#
# Z = np.random.randint(0,10,(10,10))
# shape = (5,5)
# fill  = 0
# position = (1,1)
#
# R = np.ones(shape, dtype=Z.dtype)*fill
# P  = np.array(list(position)).astype(int)
# Rs = np.array(list(R.shape)).astype(int)
# Zs = np.array(list(Z.shape)).astype(int)
#
# R_start = np.zeros((len(shape),)).astype(int)
# R_stop  = np.array(list(shape)).astype(int)
# Z_start = (P-Rs//2)
# Z_stop  = (P+Rs//2)+Rs%2
#
# R_start = (R_start - np.minimum(Z_start,0)).tolist()
# Z_start = (np.maximum(Z_start,0)).tolist()
# R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
# Z_stop = (np.minimum(Z_stop,Zs)).tolist()
#
# r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
# z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
# R[r] = Z[z]
# print(Z)
# print(R)
# Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]] ?
#
# # Author: Stéfan van der Walt
#
# Z = np.arange(1,15,dtype=uint32)
# R = stride_tricks.as_strided(Z,(11,4),(4,4))
# print(R)
# Consider two arrays A and B of shape (8,3) and (2,2). How to find rows of A that contain elements of each row of B regardless of the order of the elements in B ?
#
# # Author: Gabe Schwartz
#
# A = np.random.randint(0,5,(8,3))
# B = np.random.randint(0,5,(2,2))
#
# C = (A[..., np.newaxis, np.newaxis] == B)
# rows = (C.sum(axis=(1,2,3)) >= B.shape[1]).nonzero()[0]
# print(rows)
# Extract all the contiguous 3x3 blocks from a random 10x10 matrix.
#
# # Author: Chris Barker
#
# Z = np.random.randint(0,5,(10,10))
# n = 3
# i = 1 + (Z.shape[0]-3)
# j = 1 + (Z.shape[1]-3)
# C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
# print(C)
# Create a 2D array subclass such that Z[i,j] == Z[j,i]
#
# # Author: Eric O. Lebigot
# # Note: only works for 2d array and value setting using indices
#
# class Symetric(np.ndarray):
#     def __setitem__(self, (i,j), value):
#         super(Symetric, self).__setitem__((i,j), value)
#         super(Symetric, self).__setitem__((j,i), value)
#
# def symetric(Z):
#     return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric)
#
# S = symetric(np.random.randint(0,10,(5,5)))
# S[2,3] = 42
# print(S)
# Consider a set of p matrices wich shape (n,n) and a set of p vectors with shape (n,1). How to compute the sum of of the p matrix products at once ? (result has shape (n,1))
#
# # Author: Stéfan van der Walt
#
# p, n = 10, 20
# M = np.ones((p,n,n))
# V = np.ones((p,n,1))
# S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])
# print(S)
#
# # It works, because:
# # M is (p,n,n)
# # V is (p,n,1)
# # Thus, summing over the paired axes 0 and 0 (of M and V independently),
# # and 2 and 1, to remain with a (n,1) vector.
# Given a two dimensional array, how to extract unique rows ?
#
# Note
#
# See stackoverflow for explanations.
#
# # Author: Jaime Fernández del Río
#
# Z = np.random.randint(0,2,(6,3))
# T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
# _, idx = np.unique(T, return_index=True)
# uZ = Z[idx]
# print(uZ)
# How to implement the Game of Life using numpy arrays ?
#
# # Author: Nicolas Rougier
#
# def iterate(Z):
#     # Count neighbours
#     N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
#          Z[1:-1,0:-2]                + Z[1:-1,2:] +
#          Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])
#
#     # Apply rules
#     birth = (N==3) & (Z[1:-1,1:-1]==0)
#     survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
#     Z[...] = 0
#     Z[1:-1,1:-1][birth | survive] = 1
#     return Z
#
# Z = np.random.randint(0,2,(50,50))
# for i in range(100): Z = iterate(Z)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
