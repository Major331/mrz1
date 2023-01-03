import math
import numpy as np
import copy


class MatrixUtilities:
    @staticmethod
    def matrix_multiplication(matrix1, matrix2):
        m1_x, m1_y = len(matrix1[0]), len(matrix1)
        m2_x, m2_y = len(matrix2[0]), len(matrix2)
        if m1_x != m2_y:
            print("Матрицы нельзя перемножить")
            return
        final_matrix = []
        for current_line_in_m1 in range(m1_y):
            line_of_new_matrix = []
            for current_column_in_m2 in range(m2_x):
                result_slot = 0
                for current_column_in_m1 in range(m1_x):
                    result_slot += (matrix1[current_line_in_m1][current_column_in_m1] *
                                    matrix2[current_column_in_m1][current_column_in_m2])
                line_of_new_matrix.append(result_slot)
            final_matrix.append(line_of_new_matrix)
        return final_matrix

    # строка в матрице это результаты сумм произведений соответствующих элементов изначальных матриц

    @staticmethod
    def matrix_transpose(matrix0):
        transponed_matrix = []
        for original_column in range(len(matrix0[0])):
            transponed_matrix.append([])
            for original_row in range(len(matrix0)):
                transponed_matrix[original_column].append(matrix0[original_row][original_column])
        return transponed_matrix

    # транспонирование матрицы

    @staticmethod
    def vector_minus(vector1, vector2):
        if len(vector1) != len(vector2):
            print('Вектора разного размера!')
            return -1
        result_vector = []
        for v_index in range(len(vector1)):
            result_vector.append(vector1[v_index] - vector2[v_index])
        return result_vector

    # разница двух векторов (в1 - в2)

    @staticmethod
    def matrix_minus(matrix1, matrix2):
        if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
            print('Матрицы разного размера!')
            return -1
        minused_matrix = []
        for m_rows in range(len(matrix1)):
            minused_matrix.append([])
            for m_cols in range(len(matrix1[0])):
                minused_matrix[m_rows].append(matrix1[m_rows][m_cols] - matrix2[m_rows][m_cols])
        return minused_matrix

    # разница двух матриц (м1 - м2)

    @staticmethod
    def matrix_coeff_multiplication(matrix0, coeff):
        for rows in range(len(matrix0)):
            for cols in range(len(matrix0[0])):
                matrix0[rows][cols] *= coeff
        return matrix0

    # умножение матрицы на коэффициент

    @staticmethod
    def turn_into_image_matrix(list_of_pixel_rgb, rows, columns):
        for current_neuron_index in range(len(list_of_pixel_rgb)):
            list_of_pixel_rgb[current_neuron_index] = (list_of_pixel_rgb[current_neuron_index] + 1) / 2
        return np.reshape(list_of_pixel_rgb, (rows, columns, 3))

    # принимает вектор из значений от -1 до 1, возвращает готовую для показа матрицу

    @staticmethod
    def normalize_image_matrix(matrix0):
        new_matrix = copy.deepcopy(matrix0)
        for rows in range(len(matrix0)):
            for columns in range(len(matrix0[rows])):
                for colors in range(len(matrix0[rows][columns])):
                    new_matrix[rows][columns][colors] = matrix0[rows][columns][colors] * 2 - 1
        return new_matrix

    # принимает готовую для показа матрицу, возвращает нормализованную матрицу со значениями от -1 до 1

    @staticmethod
    def vector_mod(vector0):
        sum_of_squares = 0
        for el in vector0:
            sum_of_squares += (el ** 2)
        return math.sqrt(sum_of_squares)

    # модуль вектора

