"""Zero Matrix"""
"""
Write an algorithm such that if an element in as M x N matrix is 0,
its entire row and column are set to 0.
"""

def zeroMatrix(matrix):

    zero_positions = []
    column_len = len(matrix[0])

    for row in range(len(matrix)):
        for column in range(column_len):
            if matrix[row][column] == 0:
                zero_positions.append([row,column])

    for row in zero_positions:
        for i in range(column_len):
            matrix[row[0]][i] = 0
        for i in range(len(matrix)):
            matrix[i][row[1]] = 0

    return matrix
print(zeroMatrix([[0,9,3],[4,1,6],[7,8,9],[4,5,1]]))





