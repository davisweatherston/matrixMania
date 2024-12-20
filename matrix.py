import numpy as np

class Matrix:
    def __init__(self, row, column):
        self.row = row
        self.column = column
        self.matrix = np.random.randint(10, size=(self.row, self.column))

    def printMatrix(self):
        print(self.matrix)

    def add(self, other):

        if not isinstance(other, Matrix):
            raise TypeError("The object to add must be another Matrix.")
        
        if self.row != other.row or self.column != other.column:
            raise ValueError("Matrices must have the same dimensions to be added.")

        result = self.matrix + other.matrix
        return result
    
    def subtract(self, other):

        if not isinstance(other, Matrix):
            raise TypeError("The object to add must be another Matrix.")
        
        if self.row != other.row or self.column != other.column:
            raise ValueError("Matrices must have the same dimensions to be added.")

        result = self.matrix - other.matrix
        return result
    
    def multiply(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("The object to add must be another Matrix.")
        
        if self.row != other.column or self.column != other.row:
            raise ValueError("Matrix multiplication is not possible on these matrices")

        result = np.dot(self.matrix, other.matrix)
        return result
    
    def divide(self, other):
        if not isinstance(other, Matrix):
            raise TypeError("The object to add must be another Matrix.")
        
        try:
            inv_mat = other.inverse()

        except:
            raise ValueError("Matrix is not invertible")
        
        if self.row != other.column or self.column != other.row:
            raise ValueError("Matrix multiplication is not possible on these matrices")

        print("This technically is just multiplying the first matrix by the inverse of the second, but it acts like division just FYI")
        result = np.dot(self.matrix, inv_mat)
        return result
    
    def inverse(self):
        return np.linalg.inv(self.matrix)
    
    def transpose(self):
        return self.matrix.transpose()
    
    def getDeterminant(self):
        det = np.linalg.det(self.matrix)
        return det
    
    def getEigenDecomp(self):
        eigenValue, eigenVector = np.linalg.eig(self.matrix)
        return eigenValue, eigenVector

    # Special Matrices
    def getIdentityMatrix(n):
        return np.identity(n)
    
    def getDiagonal(mat):
        return np.diag(mat)
    
    def createZeroMatrix(row, column):
        zero_mat = np.zeros((row, column))
        return zero_mat


matrix1 = Matrix(3, 3)
matrix2 = Matrix(3, 3)
matrix3 = Matrix(3, 2)

matrix1.printMatrix()
matrix2.printMatrix()
matrix3.printMatrix()

print("\n\n")

print("inverse:")
print(matrix1.inverse())

print("\naddition:")
print(matrix1.add(matrix2))

print("\nsubtraction:")
print(matrix1.subtract(matrix2))

print("\ndot product")
print(matrix1.multiply(matrix2))
# print(matrix1.multiply(matrix3))

print("\ndivision:")
print(matrix1.divide(matrix2))
# print(matrix1.divide(matrix3))

print(matrix3.transpose())
print(matrix1.getDeterminant())
print("\nidentity matrices: ")
print(Matrix.getIdentityMatrix(3))

print("\ndiagonals:")
print(Matrix.getDiagonal(matrix1.matrix))

print("\nzero matrices:")
print(Matrix.createZeroMatrix(2, 3))



