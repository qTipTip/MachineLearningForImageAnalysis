import numpy as np

import MLIA


def test_distances():
    X = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])
    x = np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ])

    KNN = MLIA.KNN()
    KNN.train(X, None)

    result = KNN.compute_distances(x)

    expected_result = np.array([
        [0, 9, 10.392304845],
        [9, 0, 9]
    ])

    np.testing.assert_allclose(result, expected_result)