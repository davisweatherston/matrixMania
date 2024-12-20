import unittest
import numpy as np
from numpy.linalg import LinAlgError
from matrix import Matrix  # Replace with the actual path or module name

class TestMatrixInit(unittest.TestCase):
    def test_matrix_initialization(self):
        # Define test dimensions
        test_rows = 5
        test_cols = 4
        # Create a new Matrix instance
        mat = Matrix(test_rows, test_cols)
        
        # Check that row and column attributes are set
        self.assertEqual(mat.row, test_rows, "Matrix row attribute not set correctly.")
        self.assertEqual(mat.column, test_cols, "Matrix column attribute not set correctly.")
        
        # Check that matrix attribute is a numpy array
        self.assertIsInstance(mat.matrix, np.ndarray, "Matrix attribute should be a NumPy array.")
        
        # Check the shape of the matrix
        self.assertEqual(mat.matrix.shape, (test_rows, test_cols), "Matrix shape does not match specified dimensions.")
        
        # Check the range of values in the matrix (0 <= value < 10)
        self.assertTrue(np.all((mat.matrix >= 0) & (mat.matrix < 10)), 
                        "Matrix values should be between 0 and 9 inclusive.")
        

class TestMatrixAdd(unittest.TestCase):
    def test_add_same_dimensions(self):
        # Create two matrices of the same size
        rows, cols = 3, 3
        m1 = Matrix(rows, cols)
        m2 = Matrix(rows, cols)
        
        # Perform the addition
        result = m1.add(m2)
        
        # Verify that the result is a numpy array
        self.assertIsInstance(result, np.ndarray, "Result of add should be a numpy array.")

        # Verify the shape of the result
        self.assertEqual(result.shape, (rows, cols), "Resulting matrix should have the same dimensions.")

        # Verify element-wise addition is correct
        # Since these are random, we check if result equals m1.matrix + m2.matrix
        expected = m1.matrix + m2.matrix
        np.testing.assert_array_equal(result, expected, "Result of add does not match element-wise addition.")

    def test_add_with_non_matrix(self):
        rows, cols = 2, 2
        m1 = Matrix(rows, cols)
        
        # Attempt to add a non-Matrix object (e.g., a list)
        with self.assertRaises(TypeError, msg="Adding non-Matrix object should raise TypeError."):
            m1.add([[1, 2], [3, 4]])  # Not a Matrix instance

    def test_add_dimension_mismatch(self):
        # Create two matrices of different dimensions
        m1 = Matrix(2, 3)
        m2 = Matrix(3, 2)
        
        # Attempt to add them and expect a ValueError
        with self.assertRaises(ValueError, msg="Adding matrices with mismatched dimensions should raise ValueError."):
            m1.add(m2)


class TestMatrixSubtract(unittest.TestCase):
    def test_subtract_same_dimensions(self):
        # Create two matrices of the same size
        rows, cols = 3, 3
        m1 = Matrix(rows, cols)
        m2 = Matrix(rows, cols)
        
        # Perform the subtraction
        result = m1.subtract(m2)
        
        # Verify that result is a numpy array
        self.assertIsInstance(result, np.ndarray, "Result of subtract should be a numpy array.")

        # Check the shape matches the input matrices
        self.assertEqual(result.shape, (rows, cols), "Resulting matrix should have the same dimensions.")
        
        # Verify element-wise subtraction is correct
        expected = m1.matrix - m2.matrix
        np.testing.assert_array_equal(result, expected, "Result of subtract does not match element-wise subtraction.")

    def test_subtract_with_non_matrix(self):
        rows, cols = 2, 2
        m1 = Matrix(rows, cols)

        # Attempt to subtract a non-Matrix object
        with self.assertRaises(TypeError, msg="Subtracting non-Matrix object should raise TypeError."):
            m1.subtract([[1, 2], [3, 4]])  # Not a Matrix instance

    def test_subtract_dimension_mismatch(self):
        # Create two matrices of different dimensions
        m1 = Matrix(2, 3)
        m2 = Matrix(3, 2)

        # Attempt to subtract them and expect a ValueError
        with self.assertRaises(ValueError, msg="Subtracting matrices with mismatched dimensions should raise ValueError."):
            m1.subtract(m2)


class TestMatrixMultiply(unittest.TestCase):
    def test_multiply_compatible_dimensions(self):
        # For matrix multiplication:
        # (2x3) * (3x2) = (2x2)
        rows_m1, cols_m1 = 2, 3
        rows_m2, cols_m2 = 3, 2
        
        m1 = Matrix(rows_m1, cols_m1)
        m2 = Matrix(rows_m2, cols_m2)
        
        # Perform the multiplication
        result = m1.multiply(m2)
        
        # Result should be a numpy array
        self.assertIsInstance(result, np.ndarray, "Result of multiply should be a numpy array.")
        
        # Check shape of the result
        expected_shape = (rows_m1, cols_m2)
        self.assertEqual(result.shape, expected_shape, "Resulting matrix shape is not correct.")

        # Check the values against np.dot()
        expected = np.dot(m1.matrix, m2.matrix)
        np.testing.assert_array_equal(result, expected, "Result of matrix multiplication does not match np.dot.")
        
    def test_multiply_with_non_matrix(self):
        rows, cols = 2, 2
        m = Matrix(rows, cols)
        
        # Attempt to multiply by a non-Matrix object
        with self.assertRaises(TypeError, msg="Multiplying by a non-Matrix object should raise TypeError."):
            m.multiply([[1, 2], [3, 4]])  # Not a Matrix instance

    def test_multiply_dimension_mismatch(self):
        # (2x3) * (2x2) is not valid because the
        # number of columns in the first (3) does not match the number of rows in the second (2).
        m1 = Matrix(2, 3)
        m2 = Matrix(2, 2)
        
        with self.assertRaises(ValueError, msg="Multiplying matrices with incompatible dimensions should raise ValueError."):
            m1.multiply(m2)

class TestMatrixDivide(unittest.TestCase):
    def test_divide_successful(self):
        # Create a known invertible 2x2 matrix for m2
        m1 = Matrix(2, 2)
        m1.matrix = np.array([[1, 2],
                              [3, 4]])

        m2 = Matrix(2, 2)
        m2.matrix = np.array([[2, 1],
                              [1, 2]])
        # m2 is invertible. Its inverse is (1/3)*[[2, -1], [-1, 2]]

        # Perform division: m1.divide(m2) = m1 * inv(m2)
        result = m1.divide(m2)
        
        # Check that result is a numpy array
        self.assertIsInstance(result, np.ndarray, "Result of divide should be a numpy array.")

        # Check the shape of the result
        self.assertEqual(result.shape, (2, 2), "Resulting matrix should have shape (2, 2).")

        # Compute the expected result using np.linalg.inv
        expected = np.dot(m1.matrix, np.linalg.inv(m2.matrix))
        np.testing.assert_array_almost_equal(result, expected, decimal=6, 
                                             err_msg="Result of division does not match expected (m1 * inverse(m2)).")

    def test_divide_with_non_matrix(self):
        m = Matrix(2, 2)
        with self.assertRaises(TypeError, msg="Dividing by a non-Matrix object should raise TypeError."):
            m.divide("not a matrix")

    def test_divide_non_invertible_matrix(self):
        m1 = Matrix(2, 2)
        m1.matrix = np.array([[1, 2],
                              [3, 4]])
        
        # Create a non-invertible matrix (determinant=0)
        m2 = Matrix(2, 2)
        m2.matrix = np.array([[1, 2],
                              [2, 4]])  # Rows are multiples, so this is singular

        with self.assertRaises(ValueError, msg="Dividing by a non-invertible matrix should raise ValueError."):
            m1.divide(m2)

    def test_divide_dimension_mismatch(self):
        # (2x2) divided by (3x3) won't work because dimensions won't match after inversion.
        m1 = Matrix(2, 2)
        m1.matrix = np.random.randint(1, 10, size=(2,2))  # Just some non-zero matrix
        
        m2 = Matrix(3, 3)
        m2.matrix = np.eye(3)  # Invertible 3x3 identity matrix
        
        # After inversion, m2 is still (3x3). 
        # For m1.divide(m2) = m1 * inv(m2), 
        #   m1 is (2x2), inv(m2) is (3x3).
        # The inner dimensions do not match (2 != 3).
        
        with self.assertRaises(ValueError, msg="Dividing matrices with dimension mismatch should raise ValueError."):
            m1.divide(m2)


class TestMatrixInverse(unittest.TestCase):
    def test_inverse_invertible_matrix(self):
        # Create a known invertible matrix
        m = Matrix(2, 2)
        m.matrix = np.array([[4, 7],
                             [2, 6]])

        # Compute the inverse
        inv_m = m.inverse()

        # Check that inv_m is a numpy array
        self.assertIsInstance(inv_m, np.ndarray, "The inverse should return a numpy array.")

        # Check that m * inv_m is approximately the identity matrix
        identity = np.dot(m.matrix, inv_m)
        np.testing.assert_array_almost_equal(identity, np.eye(2), decimal=6,
                                             err_msg="m * inverse(m) should be the identity matrix.")

    def test_inverse_non_invertible_matrix(self):
        # Create a non-invertible matrix (determinant = 0)
        m = Matrix(2, 2)
        m.matrix = np.array([[1, 2],
                             [2, 4]])  # Rows are linearly dependent

        # Ensure that attempting to invert a singular matrix raises LinAlgError
        with self.assertRaises(LinAlgError, msg="Inverting a non-invertible matrix should raise LinAlgError."):
            m.inverse()


class TestMatrixTranspose(unittest.TestCase):
    def test_transpose_square_matrix(self):
        m = Matrix(3, 3)
        m.matrix = np.array([[1, 2, 3],
                             [4, 5, 6],
                             [7, 8, 9]])

        # Transpose the matrix
        result = m.transpose()

        # Check that result is a numpy array
        self.assertIsInstance(result, np.ndarray, "The transpose should return a numpy array.")

        # Expected transpose of the given matrix
        expected = np.array([[1, 4, 7],
                             [2, 5, 8],
                             [3, 6, 9]])

        np.testing.assert_array_equal(result, expected, 
                                      "Transpose of a square matrix did not match expected result.")

    def test_transpose_non_square_matrix(self):
        m = Matrix(2, 3)
        m.matrix = np.array([[1, 2, 3],
                             [4, 5, 6]])

        # Transpose the matrix
        result = m.transpose()

        # Check the shape
        self.assertEqual(result.shape, (3, 2), 
                         "Transposed matrix shape should be (3, 2) for a (2, 3) input.")

        # Expected transpose
        expected = np.array([[1, 4],
                             [2, 5],
                             [3, 6]])

        np.testing.assert_array_equal(result, expected, 
                                      "Transpose of a non-square matrix did not match expected result.")

class TestMatrixGetDeterminant(unittest.TestCase):
    def test_getDeterminant_known_value(self):
        m = Matrix(2, 2)
        m.matrix = np.array([[4, 7],
                             [2, 6]])
        # The determinant of [[4, 7],
        #                    [2, 6]] is (4*6 - 7*2) = 24 - 14 = 10
        expected = 10
        result = m.getDeterminant()
        self.assertAlmostEqual(result, expected, places=7, 
                               msg="Determinant does not match the known expected value.")
        
    def test_getDeterminant_singular_matrix(self):
        m = Matrix(2, 2)
        m.matrix = np.array([[1, 2],
                             [2, 4]])  # This is singular (rows are linearly dependent)
        expected = 0.0
        result = m.getDeterminant()
        self.assertAlmostEqual(result, expected, places=7,
                               msg="Determinant of a singular matrix should be approximately 0.")

    def test_getDeterminant_random_matrix(self):
        # Just a random test to ensure the method calls np.linalg.det correctly
        # Compare with np.linalg.det directly.
        m = Matrix(3, 3)
        # Using a random integer matrix
        m.matrix = np.random.randint(1, 10, size=(3, 3))
        expected = np.linalg.det(m.matrix)
        result = m.getDeterminant()
        # Due to floating point issues, we use assertAlmostEqual
        self.assertAlmostEqual(result, expected, places=7, 
                               msg="Determinant of a random matrix does not match np.linalg.det.")


class TestMatrixGetEigenDecomp(unittest.TestCase):
    def test_getEigenDecomp_known_matrix(self):
        # For a simple diagonal matrix, eigenvalues are just the diagonal elements
        # and the eigenvectors are the identity matrix.
        m = Matrix(2, 2)
        m.matrix = np.array([[2, 0],
                             [0, 3]])

        eigen_values, eigen_vectors = m.getEigenDecomp()

        expected_eigenvalues = np.array([2, 3])
        expected_eigenvectors = np.array([[1, 0],
                                          [0, 1]])

        # Check that the returned eigenvalues match the expected values.
        # Since it's a diagonal matrix, we expect eigenvalues in the order [2, 3].
        self.assertCountEqual(eigen_values, expected_eigenvalues, 
                              "Eigenvalues do not match expected values for the diagonal matrix.")
        
        # For eigenvectors, corresponding to the eigenvalues, we may need to handle ordering.
        # Identify indices of the expected eigenvalues in the returned eigen_values
        idx = [np.where(np.isclose(eigen_values, ev))[0][0] for ev in expected_eigenvalues]

        # Reorder the eigenvectors to match the order of expected_eigenvalues
        reordered_vectors = eigen_vectors[:, idx]

        # Now compare the eigenvectors
        np.testing.assert_array_almost_equal(reordered_vectors, expected_eigenvectors, decimal=6,
                                             err_msg="Eigenvectors do not match the expected identity vectors.")

    def test_getEigenDecomp_random_matrix(self):
        # Just verify that getEigenDecomp returns something consistent with np.linalg.eig
        m = Matrix(3, 3)
        m.matrix = np.random.randint(1, 10, (3, 3))
        
        eigen_values, eigen_vectors = m.getEigenDecomp()
        expected_eigen_values, expected_eigen_vectors = np.linalg.eig(m.matrix)

        # Since eigen decomposition may differ by order, we can't just assert equality directly.
        # Instead, we can check that the set of eigenvalues is the same and the eigenvectors are consistent.
        
        # Check eigenvalues as a set (order-independent)
        self.assertCountEqual(list(np.round(eigen_values, decimals=6)), 
                              list(np.round(expected_eigen_values, decimals=6)),
                              "Eigenvalues from getEigenDecomp do not match np.linalg.eig.")
        
class TestMatrixGetIdentityMatrix(unittest.TestCase):
    def test_getIdentityMatrix_n2(self):
        n = 2
        identity = Matrix.getIdentityMatrix(n)
        
        # Check shape
        self.assertEqual(identity.shape, (n, n), "Identity matrix should be of shape (n, n).")
        
        # Check identity property
        expected = np.array([[1.0, 0.0],
                             [0.0, 1.0]])
        np.testing.assert_array_equal(identity, expected, 
                                      "2x2 identity matrix should have ones on the diagonal and zeros elsewhere.")
        
    def test_getIdentityMatrix_n3(self):
        n = 3
        identity = Matrix.getIdentityMatrix(n)
        
        # Check shape
        self.assertEqual(identity.shape, (n, n), "Identity matrix should be of shape (n, n).")
        
        # Check diagonal elements are 1 and off-diagonal are 0
        expected = np.array([[1.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, 1.0]])
        np.testing.assert_array_equal(identity, expected, 
                                      "3x3 identity matrix should have ones on the diagonal and zeros elsewhere.")

    def test_getIdentityMatrix_random_n(self):
        # Test a random size to ensure it's consistent
        n = 5
        identity = Matrix.getIdentityMatrix(n)

        # Check shape
        self.assertEqual(identity.shape, (n, n), "Identity matrix should be of shape (n, n).")

        # Check identity property
        self.assertTrue(np.allclose(identity, np.eye(n)), 
                        "Identity matrix should match np.eye(n) for any n.")

class TestGetDiagonal(unittest.TestCase):
    def test_getDiagonal_square_matrix(self):
        mat = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]])
        # The diagonal should be [1, 5, 9]
        expected = np.array([1, 5, 9])
        result = Matrix.getDiagonal(mat)
        np.testing.assert_array_equal(result, expected, 
                                      "Diagonal of a square matrix does not match expected values.")

    def test_getDiagonal_non_square_matrix(self):
        mat = np.array([[10, 20, 30],
                        [40, 50, 60]])
        # The diagonal should be [10, 50]
        expected = np.array([10, 50])
        result = Matrix.getDiagonal(mat)
        np.testing.assert_array_equal(result, expected, 
                                      "Diagonal of a non-square matrix does not match expected values.")

    def test_getDiagonal_single_row(self):
        mat = np.array([[5, 6, 7]])
        # Only one row, so diagonal is the first element
        expected = np.array([5])
        result = Matrix.getDiagonal(mat)
        np.testing.assert_array_equal(result, expected,
                                      "Diagonal of a single-row matrix should be the first element.")

    def test_getDiagonal_single_column(self):
        mat = np.array([[1],
                        [2],
                        [3]])
        # Only one column, so diagonal is still the first element from each row that exists
        expected = np.array([1])
        result = Matrix.getDiagonal(mat)
        np.testing.assert_array_equal(result, expected,
                                      "Diagonal of a single-column matrix should be the top element.")

class TestCreateZeroMatrix(unittest.TestCase):
    def test_createZeroMatrix_small(self):
        rows, cols = 2, 3
        mat = Matrix.createZeroMatrix(rows, cols)
        self.assertEqual(mat.shape, (rows, cols), "Zero matrix does not have the correct shape.")
        self.assertTrue(np.all(mat == 0), "Zero matrix should contain all zeros.")

    def test_createZeroMatrix_square(self):
        n = 4
        mat = Matrix.createZeroMatrix(n, n)
        self.assertEqual(mat.shape, (n, n), "Zero matrix should be square of dimension n x n.")
        self.assertTrue(np.all(mat == 0), "Zero matrix should contain all zeros.")

    def test_createZeroMatrix_large(self):
        rows, cols = 10, 5
        mat = Matrix.createZeroMatrix(rows, cols)
        self.assertEqual(mat.shape, (rows, cols), "Zero matrix does not have the correct dimensions for a larger matrix.")
        self.assertTrue(np.all(mat == 0), "All elements in the zero matrix should be zero for larger dimensions as well.")


if __name__ == '__main__':
    unittest.main()
