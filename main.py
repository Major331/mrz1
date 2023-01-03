import matplotlib.image as image
import matplotlib.pyplot as plt
import math
import numpy as np
import random
import ast
from matrix_utilities import MatrixUtilities as mu

img_matrix = image.imread('hamster3.png')
compression_modifier = 16  # 4x4
neurons_on_compressed_layer = 4 * 3
e = 2500
learning_sample = 0

print("Выберите режим работы:\n1 - обучение матрицы весов\nОстальное - использование матрицы весов")
mode = input()
decision = '0'

if mode == '1':
    img_name = input("Введите название картинки: ")
    img_matrix = image.imread(img_name)
    compression_modifier = int(input("Введите кол-во пикселей в блоке: "))
    neurons_on_compressed_layer = int(input("Введите x, кол-во нейронов на первом скрытом слое (x = y^2 * 3): "))
    e = int(input("Введите пороговое значение ошибки: "))
    learning_sample = 0
else:
    weights_input_file_name = input("Введите название файла с матрицами весовых коэффициентов: ")
    decision = input("Введите 1, если хотите сжать картинку в архив;\nВведите что угодно, кроме 1, если "
                     "хотите разжать картинку из архива: ")
    img_input_to_transform = input("Введите название файла/картинки, которую вы хотите трансформировать: ")
    if weights_input_file_name == '' or img_input_to_transform == '':
        print("Не введены данные:")
        exit()
    weight_matrices_file = open(weights_input_file_name, "r")

    weight_matrix = ast.literal_eval(weight_matrices_file.readline())
    weight_matrix2 = ast.literal_eval(weight_matrices_file.readline())
    compression_modifier = len(weight_matrix2[0]) / 3

    if decision != '1':
        archive = open(img_input_to_transform, "r")
        compressed_image = ast.literal_eval(archive.readline())

        renewed_image_vector_of_rgbs = []
        for neuron_block in range(len(compressed_image)):
            renewed_matrix = mu.matrix_multiplication([compressed_image[neuron_block]], weight_matrix2)
            for x in renewed_matrix[0]:
                x += 1
                x /= 2
            renewed_image_vector_of_rgbs += renewed_matrix

            if int(neuron_block / len(compressed_image) * 100) % 10 == 0 and neuron_block % 100 == 0:
                print(int(neuron_block / len(compressed_image) * 100), "%")

        renewed_vector_of_rgbs_ready = []
        result_compression = (math.sqrt((compression_modifier * 3) / neurons_on_compressed_layer))
        block_size = int(math.sqrt(compression_modifier))

        compressed_block_size = int(math.sqrt(neurons_on_compressed_layer / 3))
        amount_of_blocks_in_row = img_matrix.shape[1] // block_size
        amount_of_blocks_in_col = img_matrix.shape[0] // block_size

        for block_row in range(amount_of_blocks_in_col):
            current_starting_block = block_row * amount_of_blocks_in_row
            for block_index_in_row in range(int(block_size)):
                for current_block in renewed_image_vector_of_rgbs[
                                     current_starting_block:current_starting_block + amount_of_blocks_in_row]:
                    renewed_vector_of_rgbs_ready.append(current_block[(3 * block_index_in_row * block_size):
                                                                      (3 * (block_index_in_row + 1) * block_size)])

        new_image = mu.turn_into_image_matrix(list(np.array(renewed_vector_of_rgbs_ready).flatten()),
                                              img_matrix.shape[0], img_matrix.shape[1])
        plt.imshow(new_image)
        # plt.show()
        plt.savefig(img_input_to_transform[:-4] + ".png")
        exit()

    img_matrix = image.imread(img_input_to_transform)

block_size = int(math.sqrt(compression_modifier))
block_matrix = []

img_matrix = img_matrix[:, :, :3]  # если у нас есть значение alpha, мы его не учитываем (сокращаем до 3 цветов)

leftover_lines = len(img_matrix) % math.sqrt(compression_modifier)
lines_to_complete_shape = math.sqrt(compression_modifier) - leftover_lines

leftover_columns = len(img_matrix[0]) % math.sqrt(compression_modifier)
columns_to_complete_shape = math.sqrt(compression_modifier) - leftover_columns

WHITE = [1., 1., 1.]
if leftover_lines != 0:  # добавляем строки белых пикселей, чтобы влезли все блоки
    blank_line = []
    for index_of_column in range(len(img_matrix[0])):
        blank_line.append(WHITE)
    blank_space: list = [blank_line]
    blank_space *= int(lines_to_complete_shape)
    img_matrix = np.append(img_matrix, blank_space, 0)

if leftover_columns != 0:  # добавляем столбцы белых пикселей
    blank_column = []
    for index_of_row in range(len(img_matrix)):
        blank_column.append([WHITE])
    for index_of_column in range(int(columns_to_complete_shape)):
        img_matrix = np.append(img_matrix, blank_column, 1)

shape = img_matrix.shape
number_of_neurons_layer1 = shape[0] * shape[1] * shape[2]  # вертикальный, горизонтальный, цвет
number_of_neurons_layer2 = number_of_neurons_layer1 / compression_modifier

neuron_matrix = mu.normalize_image_matrix(img_matrix)

for i in range(0, len(neuron_matrix), block_size):
    for j in range(0, len(neuron_matrix[i]), block_size):
        block_matrix.append([])
        for k in range(i, i + block_size):
            block_matrix[-1].extend(neuron_matrix[k][j:j + block_size])
# разбили поблочно

compressed_image_vector_of_rgbs = []
renewed_image_vector_of_rgbs = []

if learning_sample == 0:
    learning_sample = len(block_matrix)

# ---------------------------------------- тут начинается мясо -----------------------------------------------------

if mode == '1':
    print("Preparations complete!")

    iterations = 0

    # ------------ готовим матрицы весов
    total_quadratic_error = e + 1

    weight_matrix = []
    for wm_rows in range(compression_modifier * 3):
        weight_matrix.append([])
        for wm_column in range(neurons_on_compressed_layer):
            random_weight = ((random.random() * 2) - 1) / 1000
            weight_matrix[wm_rows].append(random_weight)
    weight_matrix2 = mu.matrix_transpose(weight_matrix)
    # рандомное заполнение матрицы весов

    while total_quadratic_error > e:
        total_quadratic_error = 0
        iterations += 1
        for learning_sample_index in range(learning_sample):
            # перебираем блоки для выборки, будем изменять веса до тех пор, пока во всех блоках не будет good ошибка
            layer0 = [list(np.array(block_matrix[learning_sample_index]).flatten())]
            layer0_number = len(layer0[0])

            result_matrix = list(np.matmul(np.array(layer0), np.array(weight_matrix)))
            renewed_matrix = list(np.matmul(np.array(result_matrix), np.array(weight_matrix2)))
            difference_vector = mu.vector_minus(renewed_matrix[0], layer0[0])

            layer0_transponed = np.array(layer0).T
            result_matrix_transponed = np.array(result_matrix).T

            # ------------------------ подсчёт коэффициентов обучения

            # alpha1 = 1 / (np.matmul(np.array(layer0), layer0_transponed)[0][0])
            # alpha2 = 1 / (np.matmul(np.array(result_matrix), result_matrix_transponed)[0][0])
            # АЛЬТЕРНАТИВА #
            alpha1 = alpha2 = 0.0005

            # --------- корректировка матриц весов ---------

            XiT_dXi = np.matmul(layer0_transponed, np.array([difference_vector]))
            XiT_dXi_W2T = np.matmul(XiT_dXi, np.array(weight_matrix2).T)
            weight_matrix = mu.matrix_minus(weight_matrix, list(alpha1 * XiT_dXi_W2T))

            YiT_dXi = np.matmul(result_matrix_transponed, np.array([difference_vector]))
            weight_matrix2 = mu.matrix_minus(weight_matrix2, list(alpha2 * YiT_dXi))

            # ----------- нормализация весовых матриц

            wm1_transponed = mu.matrix_transpose(weight_matrix)
            for wm1_rows in range(len(weight_matrix)):
                for wm1_cols in range(len(weight_matrix[wm1_rows])):
                    znamenatel1 = mu.vector_mod(wm1_transponed[wm1_cols])
                    weight_matrix[wm1_rows][wm1_cols] /= znamenatel1
            # нормализация весовой матрицы 1

            wm2_transponed = mu.matrix_transpose(weight_matrix2)
            for wm2_rows in range(len(weight_matrix2)):
                for wm2_cols in range(len(weight_matrix2[wm2_rows])):
                    znamenatel2 = mu.vector_mod(wm2_transponed[wm2_cols])
                    weight_matrix2[wm2_rows][wm2_cols] /= znamenatel2
            # нормализация весовой матрицы 2

            # ---------- подсчёт ошибки

            sum_quadratic_error = 0
            for i in range(layer0_number):
                sum_quadratic_error += (difference_vector[i] ** 2)
            total_quadratic_error += sum_quadratic_error

        print("Итерация " + str(iterations) + ": E = " + str(total_quadratic_error) + "; e = " + str(e))

for block_index in range(len(block_matrix)):
    layer1 = [list(np.array(block_matrix[block_index]).flatten())]
    layer1_number = len(layer1[0])

    result_matrix = mu.matrix_multiplication(layer1, weight_matrix)
    renewed_matrix = mu.matrix_multiplication(result_matrix, weight_matrix2)

    for x in renewed_matrix[0]:
        x += 1
        x /= 2
    # восстановление изображения

    compressed_image_vector_of_rgbs += result_matrix
    renewed_image_vector_of_rgbs += renewed_matrix

    if int(block_index / len(block_matrix) * 100) % 10 == 0 and block_index % 100 == 0:
        print(int(block_index / len(block_matrix) * 100), "%")

compressed_vector_of_rgbs_ready = []
renewed_vector_of_rgbs_ready = []
result_compression = (math.sqrt((compression_modifier * 3) / neurons_on_compressed_layer))

compressed_block_size = int(math.sqrt(neurons_on_compressed_layer / 3))
amount_of_blocks_in_row = img_matrix.shape[1] // block_size
amount_of_blocks_in_col = img_matrix.shape[0] // block_size

for block_row in range(amount_of_blocks_in_col):
    current_starting_block = block_row * amount_of_blocks_in_row
    for block_index_in_row in range(int(block_size)):
        for current_block in renewed_image_vector_of_rgbs[
                             current_starting_block:current_starting_block + amount_of_blocks_in_row]:
            renewed_vector_of_rgbs_ready.append(current_block[(3 * block_index_in_row * block_size):
                                                              (3 * (block_index_in_row + 1) * block_size)])

for x in compressed_image_vector_of_rgbs:
    for i in range(compressed_block_size):
        compressed_vector_of_rgbs_ready.append(x[(3 * i * compressed_block_size):
                                                 ((i + 1) * 3 * compressed_block_size)])

plt.imshow(mu.turn_into_image_matrix(list(np.array(compressed_vector_of_rgbs_ready).flatten()),
                                     int(img_matrix.shape[0] / result_compression),
                                     int(img_matrix.shape[1] / result_compression)))
plt.show()
# сжатая картинка

if decision == '1':
    archive_name = input("Введите название файла архива: ")
    new_archive = open(archive_name, "w")
    new_archive.write(str(compressed_image_vector_of_rgbs))

if decision != '1':
    plt.imshow(mu.turn_into_image_matrix(list(np.array(renewed_vector_of_rgbs_ready).flatten()),
                                         img_matrix.shape[0], img_matrix.shape[1]))
plt.show()
# разжатая картинка

if mode == '1':
    decision_to_save_matrices = input("Введите ENTER, если не хотите сохранять весовые матрицы; введите "
                                      "любое другое название для сохранения матриц в соответствующем файле: ")
    if decision_to_save_matrices != '':
        if not decision_to_save_matrices.endswith(".txt"):
            decision_to_save_matrices = decision_to_save_matrices + ".txt"
        weight_matrices_file = open(decision_to_save_matrices, 'w')
        weight_matrices_file.write(str(weight_matrix) + "\n" + str(weight_matrix2))
        weight_matrices_file.close()

print("End of program!")
