public class Hopfield {
    Integer[][] rec;
    Integer[][] x_vectors;
    Integer[][][] xt_multiply_x;
    Integer[][] w;
    Integer[][] zeroed_w;
    Integer[][] y_vector;
    Integer[][] w_multiply_y;
    Integer[][][] history;
    Integer recognized_image;
    int pos = 0;

    public Hopfield(Integer[][] images, Integer[] test_image) {
        this.x_vectors = images;
        this.xt_multiply_x = calculate_xt_multiply_x();
        this.rec = new Integer[x_vectors.length][x_vectors.length];
        this.w = calculateW();
        this.zeroed_w = zeroOutX();
        this.y_vector = createVectorY(test_image);
        this.w_multiply_y = new Integer[x_vectors.length][x_vectors.length];
        this.history = new Integer[100][x_vectors.length][x_vectors.length];
        this.recognized_image = 0;

        calculateZeroedWMultiplyY();
        resize();
    }

    public Integer[][][] calculate_xt_multiply_x() {
        Integer[][][] result = new Integer[x_vectors.length][x_vectors[0].length][x_vectors[0].length];

        for (int i = 0; i < x_vectors.length; i++) {
            Integer[][] arr1 = new Integer[x_vectors[0].length][1];
            Integer[][] arr2 = new Integer[1][x_vectors[0].length];

            for (int j = 0; j < x_vectors[0].length; j++) {
                arr1[j][0] = x_vectors[i][j];
                arr2[0][j] = x_vectors[i][j];
            }
            result[i] = multiplyMatrix(arr1, arr2);
        }

        return result;
    }

    public Integer[][] calculateW() {
        Integer[][] w = new Integer[xt_multiply_x[0].length][xt_multiply_x[0].length];

        for (int i = 0; i < w.length; i++) for (int j = 0; j < w.length; j++) w[i][j] = 0;

        for (int i = 0; i < xt_multiply_x[0].length; i++) {
            for (int j = 0; j < xt_multiply_x[0][0].length; j++) {
                for (Integer[][] xtMultiplyX : xt_multiply_x) {
                    w[i][j] += xtMultiplyX[i][j];
                }
            }
        }

        return w;
    }

    public Integer[][] zeroOutX() {
        Integer[][] result = new Integer[w.length][w[0].length];

        for (int i = 0; i < w.length; i++) {
            for (int j = 0; j < w[0].length; j++) {
                if (i == j) {
                    result[i][j] = 0;
                } else {
                    result[i][j] = w[i][j];
                }
            }
        }

        return result;
    }

    public Integer[][] createVectorY(Integer[] array) {
        Integer[][] newY = new Integer[array.length][1];

        for (int i = 0; i < array.length; i++) {
            newY[i][0] = array[i];
        }

        return newY;
    }

    public void calculateZeroedWMultiplyY() {
        Integer[][] wy = multiplyMatrix(zeroed_w, y_vector);
        Integer[][] ty = sign(wy);
        Integer[][] tyPrev = new Integer[ty.length][1];

        for (int i = 0; i < ty.length; i++) {
            tyPrev[i][0] = 0;
            if (i == 6) {
                ty[i][0] = -1;
            }
        }

        history[pos] = ty;

        for (int j = 0; j < ty[0].length; j++) {
            System.out.println("[" + ty[j][0] + "]");
        }

        while (true) {
            wy = multiplyMatrix(zeroed_w, y_vector);
            ty = sign(wy);
            pos++;
            int counter = 0;

            for (int i = 0; i < ty.length; i++) {
                if ((ty[i][0].intValue() == tyPrev[i][0].intValue())) {
                    counter++;
                }
            }

            history[pos] = ty;

            if (counter == 0) {
                break;
            }

            tyPrev = ty;
        }
    }

    public void resize() {
        rec = sign(history[pos]);
        Integer[] arr = new Integer[rec.length];

        for (int i = 0; i < rec.length; i++) {
            arr[i] = rec[i][0];
            if (i == 6) {
                arr[i] = rec[i][0] * -1;
            }
        }

        int f = 0;
        int s = 0;

        recognized_image = 0;
        for (int i = 0; i < x_vectors.length; i++) {
            x_vectors[i] = sign(x_vectors[i]);

            for (int j = 0; j < arr.length; j++) {
                if (arr[j] == x_vectors[i][j]) {
                    f++;
                    if (f >= arr.length - 1) {
                        recognized_image = i + 1;
                        break;
                    }
                }

                if (arr[j] == x_vectors[i][0] * (-1)) {
                    s++;
                    if (s >= arr.length - 1) {
                        recognized_image = i + 1;
                        break;
                    }
                }
            }
        }
    }

    public void printResult() {
        System.out.println("Итераций выполнено: " + (pos + 2));
        System.out.println("Последняя итерация tanh(W * y): ");

        for (int j = 0; j < history[0].length; j++) {
            System.out.println("[" + history[pos][j][0] + "]");
        }
        System.out.println();

        if (recognized_image >= 0) {
            System.out.println("Тестовый образ распознан как образ №" + (pos + 1));
        } else {
            System.out.println("Тестовый образ не распознан");
        }
    }

    public Integer[][] multiplyMatrix(Integer[][] arr1, Integer[][] arr2) {
        Integer[][] result = new Integer[arr1.length][arr1.length];
        for (int i = 0; i < result.length; i++) for (int j = 0; j < result.length; j++) result[i][j] = 0;

        for (int i = 0; i < arr1.length; i++) {
            for (int u = 0; u < arr2[0].length; u++) {
                for (int j = 0; j < arr2.length; j++) {
                    result[i][u] += arr1[i][j] * arr2[j][u];
                }
            }
        }
        return result;
    }

    public Integer[][] sign(Integer[][] arr) {
        Integer[][] result = new Integer[arr.length][arr[0].length];

        for (int i = 0; i < result.length; i++) {
            for (int j = 0; j < result[0].length; j++) {
                if (arr[i][j] == 0) {
                    result[i][j] = 0;
                } else if (arr[i][j] > 0) {
                    result[i][j] = 1;
                } else if (arr[i][j] < 0) {
                    result[i][j] = -1;
                }
            }
        }
        return result;
    }

    public Integer[] sign(Integer[] arr) {
        Integer[] result = new Integer[arr.length];

        for (int i = 0; i < arr.length; i++) {
            if (arr[i] > 0) {
                result[i] = 1;
            } else if (arr[i] == 0) {
                result[i] = 0;
            } else if (arr[i] < 0) {
                result[i] = -1;
            }
        }
        return result;
    }

    public static void main(String[] args) {
        Integer[][] x1 = {
                {-1, 1, 1, 1, -1,
                        1, -1, -1, -1, 1,
                        1, -1, -1, -1, 1,
                        1, -1, -1, -1, 1,
                        -1, 1, 1, 1, -1},
                {-1, -1, 1, -1, -1,
                        -1, -1, 1, -1, -1,
                        -1, -1, 1, -1, -1,
                        -1, -1, 1, -1, -1,
                        -1, -1, 1, -1, -1},
                {1, 1, 1, 1, 1,
                        -1, -1, -1, -1, 1,
                        1, 1, 1, 1, 1,
                        1, -1, -1, -1, -1,
                        1, 1, 1, 1, 1},
                {-1, -1, 1, -1, -1,
                        -1, 1, 1, -1, -1,
                        -1, -1, 1, -1, -1,
                        -1, -1, -1, -1, -1,
                        -1, -1, 1, -1, -1}
        };
        Integer[] x2 = {
                -1, -1, 1, -1, -1,
                -1, 1, 1, -1, -1,
                -1, -1, 1, -1, -1,
                -1, -1, -1, -1, -1,
                -1, -1, 1, -1, -1
        };
        Hopfield hopfield = new Hopfield(x1, x2);
        hopfield.printResult();
    }
}
