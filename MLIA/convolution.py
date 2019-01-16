import numpy as np


def smoothing_filter(n):
    return np.ones((n, n)) / n ** 2


def convolve_loop(kernel, image, mode='same'):
    """
    Returns the discrete convolution of a filter and an image.

    :param kernel: a numpy array of shape (m, n) describing the filter kernel
    :param image: a numpy array of shape (M, N) or (M, N, B) where M, N are the image dimensions,
        and B is the number of color channels.
    :param mode: 'full' / 'same' / 'valid'
        'full': generate a response at each point of overlap, yielding an output image of size (M + m - 1, N + n - 1)
        'same': generate a response for each point of overlap with the filter origin being inside the image,
        yielding an output image of size (max(M, m), max(N, n))
        'valid': generate a response for each point of complete overlap, yielding an output image of size (max(M,
        m) - min(M, m) + 1, max(N, n) - min(N, n) + 1)
    :return: Discrete convolution of filter and image of shape (., ., B)
    """
    if mode not in ["same"]:
        raise NotImplementedError(f'mode = "{mode}" is not implemented yet, use "same" for the time being')
    if image.ndim == 2:
        image = np.reshape(image, (image.shape[0], image.shape[1], 1))

    M, N, B = image.shape
    m, n = kernel.shape
    kernel_w, kernel_h = m // 2, n // 2

    image_padded = np.zeros(shape=(M + m - 1, N + n - 1, B))
    image_padded[kernel_w: -kernel_w, kernel_h: -kernel_h] = image

    result = np.zeros_like(image)
    for row in range(M):
        for col in range(N):
            for c in range(B):
                for i in range(m):
                    for j in range(n):
                        result[row, col, c] += image_padded[row + i, col + j] * kernel[i, j]
    return result


def convolve(kernel, image, mode='same'):
    """
    Returns the discrete convolution of a filter and an image.

    :param kernel: a numpy array of shape (m, n) describing the filter kernel
    :param image: a numpy array of shape (M, N) or (M, N, B) where M, N are the image dimensions,
        and B is the number of color channels.
    :param mode: 'full' / 'same' / 'valid'
        'full': generate a response at each point of overlap, yielding an output image of size (M + m - 1, N + n - 1)
        'same': generate a response for each point of overlap with the filter origin being inside the image,
        yielding an output image of size (max(M, m), max(N, n))
        'valid': generate a response for each point of complete overlap, yielding an output image of size (max(M,
        m) - min(M, m) + 1, max(N, n) - min(N, n) + 1)
    :return: Discrete convolution of filter and image of shape (., ., B)
    """
    if mode not in ["same"]:
        raise NotImplementedError(f'mode = "{mode}" is not implemented yet, use "same" for the time being')
    if image.ndim == 2:
        image = np.reshape(image, (image.shape[0], image.shape[1], 1))

    M, N, B = image.shape
    m, n = kernel.shape
    kernel_w, kernel_h = m // 2, n // 2

    image_padded = np.zeros(shape=(M + m - 1, N + n - 1, B))
    image_padded[kernel_w: -kernel_w, kernel_h: -kernel_h] = image

    result = np.zeros_like(image)
    kernel = kernel.ravel()
    for row in range(M):
        for col in range(N):
            for c in range(B):
                result[row, col, c] += kernel @ image_padded[row:row + m, col:col + n, c].ravel()
    return result
