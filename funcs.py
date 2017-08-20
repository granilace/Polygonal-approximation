import numpy as np
import matplotlib.pyplot as plt
from Queue import Queue
from collections import deque
from rdp import rdp
from PIL import Image, ImageDraw

###CONSTANTS###
INF = 1e15
###############


# Plots curve by points
def plot_sample(list_of_points):
    plt.plot(*zip(*list_of_points), color='black')
    plt.show()
    
# Plots two curves(original image and approxed image)
def plot_curve(original_image, approx_image):
    plt.plot(*zip(*original_image), color='black')
    plt.plot(*zip(*approx_image), color='red')
    plt.scatter(*zip(*approx_image), color='green')
    plt.show()

# Checks pixel
def is_point(curr_x, curr_y, pixel, p):
    if curr_x < 200 and curr_y < 200 and curr_x >= 0 and curr_y >= 0 and pixel[curr_x, curr_y][0] <= 10 and p.count([curr_x, curr_y]) == 0:
        return True

# Gets list of pixels from curve
def vectorize_image(img):
    start_x = 0
    start_y = 0
    pix = img.load()
    for i in range(img.width):
        for j in range(img.height):
            if pix[i, j][0] == 0:
                start_x = i
                start_y = j
    points = [[start_x, start_y]]
    while True:
        if is_point(start_x + 1, start_y, pix, points):
            points.append([start_x + 1, start_y])
            start_x = start_x + 1
            continue
        if is_point(start_x - 1, start_y, pix, points):
            points.append([start_x - 1, start_y])
            start_x = start_x - 1
            continue
        if is_point(start_x, start_y + 1, pix, points):
            points.append([start_x, start_y + 1])
            start_y = start_y + 1
            continue
        if is_point(start_x, start_y - 1, pix, points):
            points.append([start_x, start_y - 1])
            start_y = start_y - 1
            continue
        if is_point(start_x + 1, start_y + 1, pix, points):
            points.append([start_x + 1, start_y + 1])
            start_x = start_x + 1
            start_y = start_y + 1
            continue
        if is_point(start_x - 1, start_y + 1, pix, points):
            points.append([start_x - 1, start_y + 1])
            start_x = start_x - 1
            start_y = start_y + 1
            continue
        if is_point(start_x + 1, start_y - 1, pix, points):
            points.append([start_x + 1, start_y - 1])
            start_x = start_x + 1
            start_y = start_y - 1
            continue
        if is_point(start_x - 1, start_y - 1, pix, points):
            points.append([start_x - 1, start_y - 1])
            start_x = start_x - 1
            start_y = start_y - 1
            continue
        break
    return points

# Smoothes image for desire count of nodes
# Arguments: inp_vector - list of nodes from curve, count_of_nodes - desire count of nodes in smoothed image
def smooth(inp_vector, count_of_nodes):
    for epsilon in [50, 10, 8, 5, 3.5, 3, 2.5, 2, 1.5, 1, 0.8, 0.7, 0.5, 0.4, 0.3, 0.2, 0.1]:
        if len(rdp(inp_vector, epsilon)) > count_of_nodes:
            return rdp(inp_vector, epsilon)

# Making appropriate vector for DP algorithm
def prepare_vector_for_DP(inp_vector):
    return [[0, 0]] + inp_vector + [inp_vector[0]]

# Returns first coordinate of pixel by serial number
def x(k, vector):
    return vector[k][0]

# Returns second coordinate of pixel by serial number
def y(k, vector):
    return vector[k][1]

# Calculates coefficient "a" from line equation 
def a(i, j, vector):
    if x(i, vector) == x(j, vector):
        return y(i, vector) - y(j, vector)
    else:
        return float(y(i, vector) - y(j, vector)) / (x(i, vector) - x(j, vector))

# Calculates coefficient "b" from line equation
def b(i, j, vector):
    if x(i, vector) == x(j, vector):
        return x(i, vector) * y(j, vector) - x(j, vector) * y(i, vector)
    else:
        return float(-x(j, vector) * y(i, vector) + x(i, vector) * y(j, vector)) / (x(i, vector) - x(j, vector))
    
# Calculates k-cosine for Rosenfeld-Johnston algorithm
# Arguments: i - start point, k - delta of points, inp_vector - array of points from curve
def k_cosine(i, k, inp_vector):
    N = len(inp_vector) - 1
    first = 1
    second = 1
    if i + k >= N:
        first = i + k - N + 1
    else:
        first = i + k
    if i - k <= 0:
        second = N + (i - k) - 1
    else:
        second = i - k
    return float(np.dot(inp_vector[first], inp_vector[second])) / ( np.linalg.norm(inp_vector[first]) * np.linalg.norm(inp_vector[second]))

# Calculates squared error
# Arguments: i - start point in curve, j - last point in curve, vector - array of points from curve
def error(i, j, vector):
    answ = 0.0
    if i > j:
        temp = i
        i = j
        j = temp
    for k in range(i + 1, j):
        answ += (y(k, vector) - a(i, j, vector) * x(k, vector) - b(i, j, vector)) ** 2
    answ /= (1 + a(i, j, vector) ** 2)
    return answ

# Calculates max Eucledian distance
# Arguments: i - start point in curve, j - last point in curve, vector - array of points from curve
def error_max(i, j, vector):
    max = 0
    p_c = i
    while p_c != j:
        p_c += 1
        d_x = x(j, vector) - x(i, vector)
        d_y = y(j, vector) - y(i, vector)
        d = (y(p_c, vector) - y(i, vector)) * d_x - (x(p_c, vector) - x(i, vector)) * d_y
        d = d * d / float(d_x * d_x + d_y * d_y)
        if d > max:
            max = d
    return max

# Returns the most distant point from line segment (i, j) by Eucledian distance
# Arguments: i - start point in curve, j - last point in curve, vector - array of points from curve
def far_point(i, j, vector):
    max = 0
    p_c = i
    far_point = p_c
    while p_c != j:
        p_c += 1
        d_x = x(j, vector) - x(i, vector)
        d_y = y(j, vector) - y(i, vector)
        d = (y(p_c, vector) - y(i, vector)) * d_x - (x(p_c, vector) - x(i, vector)) * d_y
        d = d * d / float(d_x * d_x + d_y * d_y)
        if d > max:
            max = d
            far_point = p_c
    return far_point

# Makes matrix of max Eucledian distances
def get_matrix_of_max_errors(vect_image):
    N = len(vect_image) - 1
    dist_graph = [[INF] * (N + 1) for i in range(N + 1)]
    for m in range (1, N + 1):
        for n in range (m + 1, N + 1):
            dist_graph[m][n] = error_max(m, n, vect_image)
    dist_graph[1][N] = INF
    return dist_graph

# Calculates DP
# Arguments: vect_image - list of nodes from curve, D - DP-matrix, A - supportive matrix for recovering answer, N - original count of nodes, M - desired count of nodes
def calc_DP(vect_image, D, A, N, M):
    for n in range(2, N + 1):
        for m in range(1, min(M + 1, n)):
            for j in range(max(1, m - 1), n):
                curr_result = D[j][m - 1] + error(j, n, vect_image)
                if curr_result < D[n][m]:
                    D[n][m] = curr_result
                    A[n][m] = j
    return A

# Calculates DP(optimized version)
# Arguments: vect_image - list of nodes from curve, D - DP-matrix, A - supportive matrix for recovering answer, N - original count of nodes, M - desired count of nodes
def calc_DP_optimized(vect_image, D, A, N, M):
    precalc_errors = [[INF] * (N + 1) for i in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            precalc_errors[i][j] = error(i, j, vect_image)
            
    for n in range(2, N + 1):
        for m in range(1, min(M + 1, n)):
            for j in range(max(1, m - 1), n):
                curr_result = D[j][m - 1] + precalc_errors[j][n]
                if curr_result < D[n][m]:
                    D[n][m] = curr_result
                    A[n][m] = j
    return A

# Recovers answer from array "A"
# Arguments: calculated_A - matrix from "calc_DP" function, vect_image - list of nodes from curve, N - original count of nodes, M - desired count of nodes
def recovered_answer(calculated_A, vect_image, N, M):
    answ_DP = [vect_image[1]]
    while M != 1:
        answ_DP.append(vect_image[calculated_A[N][M]])
        N = calculated_A[N][M]
        M -= 1
    answ_DP.append(vect_image[1])
    return answ_DP

# Returns polygonal approximation by naive DP algorithm
# Arguments: vect_image - list of nodes from curve, M - desired count of nodes
def DP_approx(vect_image, M):
    N = len(vect_image) - 1
    D = [[INF] * (M + 1) for i in range(N + 1)]
    A = [[0] * (M + 1) for i in range(N + 1)]
    D[1][0] = 0
    D[2][0] = (x(2, vect_image) - x(1, vect_image))**2 + (y(2, vect_image) - y(1, vect_image))**2
    calculated_A = calc_DP(vect_image, D, A, N, M)
    return recovered_answer(calculated_A, vect_image, N, M)

# Returns polygonal approximation by DP algorithm
# Arguments: vect_image - list of nodes from curve, M - desired count of nodes
def DP_approx_optimized(vect_image, M):
    N = len(vect_image) - 1
    D = [[INF] * (M + 1) for i in range(N + 1)]
    A = [[0] * (M + 1) for i in range(N + 1)]
    D[1][0] = 0
    D[2][0] = (x(2, vect_image) - x(1, vect_image))**2 + (y(2, vect_image) - y(1, vect_image))**2
    calculated_A = calc_DP_optimized(vect_image, D, A, N, M)
    return recovered_answer(calculated_A, vect_image, N, M)

# Returns polygonal approximation by Graph-Search algorithm
# Arguments: vect_image - list of nodes from curve, epsilon - threshold of epsilon
def graph_search_approx(vect_image, epsilon):
    N = len(vect_image) - 1
    dist_graph = get_matrix_of_max_errors(vect_image)
    queue = Queue()
    chosen_points = [vect_image[1]]
    queue.put(1)
    curve_finded = False
    while not curve_finded:
        p_c = queue.get()
        for p in range(N, p_c, -1):
            if dist_graph[p_c][p] < epsilon:
                if p == N:
                    curve_finded = True
                chosen_points.append(vect_image[p])
                queue.put(p)
                break
    chosen_points.append(vect_image[1])
    return chosen_points

# Makes piecewise linear approximation
# Arguments: vect_image - list of nodes from curve, M - desired count of nodes
def piecewise_linear_approx(vect_image, M):
    N = len(vect_image) - 1
    count_of_taken_points = 0
    vect_image = prepare_vector_for_DP(vect_image)
    taken_points = [False for i in range(N + 1)]
    taken_points[1] = True
    taken_points[N] = True
    answ = []
    queue = deque()
    queue.append([1, 1, N - 1])
    while count_of_taken_points + 1 < M:
        curr_pair = queue.popleft()
        far_p = far_point(curr_pair[1], curr_pair[2], vect_image)
        taken_points[far_p] = True
        count_of_taken_points += 1
        if curr_pair[0] * 2 <= count_of_taken_points:
            queue.appendleft([curr_pair[0] * 2, curr_pair[1], far_p])
            queue.appendleft([curr_pair[0] * 2, far_p, curr_pair[2]])
        else:
            queue.append([curr_pair[0] * 2, curr_pair[1], far_p])
            queue.append([curr_pair[0] * 2, far_p, curr_pair[2]])
    for i in range(N + 1):
        if taken_points[i]:
            answ.append(vect_image[i])
    answ.append(vect_image[1])
    return answ

# Makes polygonal approximation by RJ method
# Arguments: Arguments: vect_image - list of nodes from curve, m - parameter of smoothing
def RJ_approx(inp_vector, m):
    inp_vector = prepare_vector_for_DP(inp_vector)
    N = len(inp_vector) - 1
    m = int(m * N)
    best_values = [0 for i in range (N + 1)]

    for i in range (1, N):
        last_k_cosine = k_cosine(i, m, inp_vector)
        k = m - 1
        while k > 1 and k_cosine(i, k, inp_vector) > last_k_cosine:
            last_k_cosine = k_cosine(i, k, inp_vector)
            k -= 1
        best_values[i] = k
    answ = [inp_vector[1]]

    for i in range (2, N):
        curr_point_is_angle = True
        n = best_values[i]
        for j in range (1, n/2 + 1):
            left = 1
            right = 1
            if i + j >= N:
                right = i + j - N + 1
            else:
                right = i + j
            if i - j <= 0:
                left = N + (i - j) - 1
            else:
                left = i - j
            if k_cosine(i, n, inp_vector) > k_cosine(left, best_values[left], inp_vector) and k_cosine(i, n, inp_vector) > k_cosine(right, best_values[right], inp_vector):
                curr_point_is_angle = True
            else:
                curr_point_is_angle = False
                break
        if curr_point_is_angle:
            answ.append(inp_vector[i])
    answ.append(inp_vector[1])
    return answ
    