cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef c_convolve(double[:, :] kernel, double[:, :, :] image, int M, int N, int B, int m, int n, double[:, :, :] result):
    """
    Computes the convolution of image by kernel. Called by the 'MLIA.convolve'-function.
    :param kernel: kernel of size (m, n)
    :param image: padded image of size (M + m - 1, N + n - 1, B)
    :param M: original image height (unpadded)
    :param N: original image width (unpadded)
    :param B: image bandwidth 
    :param n: kernel width
    :param m: kernel height
    :param result: allocated memory for the output
    :return: 
    """

    cdef Py_ssize_t row, col, c, i, j

    for row in range(M):
        for col in range(N):
            for c in range(B):
                for i in range(m):
                    for j in range(n):
                        result[row, col, c] += image[row + i, col + j, c] * kernel[i, j]
    return result
