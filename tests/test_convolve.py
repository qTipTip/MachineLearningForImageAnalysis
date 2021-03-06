import pytest
import numpy as np

from MLIA import convolve_loop, blur, convolve


@pytest.mark.parametrize('convolve_func', [convolve_loop, convolve])
def test_raise_errors(convolve_func):
    kernel = np.zeros((3, 3))
    image = np.zeros((5, 5))
    with pytest.raises(NotImplementedError) as error:
        convolve_func(kernel, image, mode='full')
        convolve_func(kernel, image, mode='valid')
    try:
        convolve_func(kernel, image, mode='same')
    except NotImplementedError:
        assert False


@pytest.mark.parametrize('convolve_func', [convolve_loop, convolve])
def test_mode_output_size(convolve_func):
    kernel = np.zeros((3, 3))
    image = np.zeros((5, 5))
    result = convolve_func(kernel, image, mode='same')

    assert result.shape == (5, 5, 1)


@pytest.mark.parametrize('convolve_func', [convolve_loop, convolve])
def test_convolve_valid_single_entry(convolve_func):
    image = np.array([
        [105, 102, 100],
        [103, 99, 103],
        [101, 98, 104]
    ])
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    result = convolve_func(kernel, image, mode='same')
    expected_result = np.array([
        [320, 206, 295],
        [210, 89, 212],
        [304, 186, 319]
    ]).reshape(3, 3, -1)

    np.testing.assert_allclose(result, expected_result)


@pytest.mark.parametrize('convolve_func', [convolve_loop, convolve])
def test_mode_smoothing_identity(convolve_func):
    image = np.eye(5)
    result = blur(image, 3)
    expected_result = np.array([
        [2, 2, 1, 0, 0],
        [2, 3, 2, 1, 0],
        [1, 2, 3, 2, 1],
        [0, 1, 2, 3, 2],
        [0, 0, 1, 2, 2]
    ]).reshape(5, 5, 1) / 9

    np.testing.assert_allclose(result, expected_result)


@pytest.mark.parametrize('convolve_func', [convolve_loop, convolve])
def test_mode_non_symmetric(convolve_func):
    kernel = np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    image = np.arange(25).reshape(5, 5)
    result = convolve_func(kernel, image, mode='same')
    expected_result = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 2, 3],
        [0, 5, 6, 7, 8],
        [0, 10, 11, 12, 13],
        [0, 15, 16, 17, 18]
    ]).reshape(5, 5, 1)

    np.testing.assert_allclose(result, expected_result)
