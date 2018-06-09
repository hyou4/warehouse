#! /usr/bin/env python
# -*- coding: utf-8 -*-

'''
Author: YouHao
'''

import sys
import os
import csv
import copy
import time
import Queue
import matplotlib.pyplot as plt
import numpy as np
import ttk
import pickle


import Tkinter as tk
from Tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


from item import Item
from point import Point
from shelf import Shelf
from treenode import TreeNode
from application import Application

warehouse_items_path = "warehouse-grid.csv"
orders_path = "warehouse-orders-v02-tabbed.txt"
item_weight_path = "item-dimensions-tabbed.txt"

database = {}       # (k,v) --> (itemId, shelf)
item_weight = {}
shelves = {}        # (k,v) --> (point, shelf)
lookup_table = {}   # (k,v) --> (point, index in source/destination)

orders = []
optimized_paths_nn = []
optimized_paths_bnb = []
optimized_paths_str_nn = []
optimized_paths_str_bnb = []
optimized_distance_nn = []
optimized_distance_bnb = []
default_distance = []

distance_matrix = None
branch_bound_matrix = None
source = None
destination = None

algo_choice = 1
INFINITY = sys.maxint

x_coord = []
y_coord = []
max_x = 0
max_y = 0

WEIGHT_LIMIT = 4
MAX_ITEMS_LIMIT = 15
startpoint = Point(0, 0)
endpoint = Point(0, 0)
ALGORITHM = 'nn'
FACTOR_WEIGHT = 'NO'

def read_input():
    global ALGORITHM
    # x,y = map(int, raw_input("Hello User, where is your worker? (X Y): ").split())
    # print x
    # print y
    ALGORITHM = raw_input("What algorithm you would choose for path calculation? (1 for nn, 2 for bnb): ")


def read_warehouse_data():
    global max_x,max_y

    with open(warehouse_items_path) as csvfile:
        reader = csv.reader(csvfile)
        for i, rows in enumerate(reader):
            if len(rows) == 3:
                item_id = rows[0]
                x = int(float(rows[1])) * 2 + 1
                y = int(float(rows[2])) * 2 + 1

                point = Point(x, y)
                item = Item(rows[0], x, y)
                new_shelf = Shelf(x, y)

                if shelves.get(point) is not None:
                    shelf = shelves.get(point)
                    shelf.put(item)
                    database[item_id] = shelf
                else:
                    new_shelf.put(item)
                    shelves[point] = new_shelf
                    database[item_id] = new_shelf

                x_coord.append(x)
                y_coord.append(y)

                if(x > max_x):
                    max_x = x
                if(y > max_y):
                    max_y = y


def read_orders_data():
    with open(orders_path) as orders_file:
        file = orders_file.readlines()
        for i, line in enumerate(file):
            order = line.split()
            orders.append(order)
            # print (order)


def read_orders_data_csv():
    with open(orders_path) as csvfile:
        reader = csv.reader(csvfile)
        for i, line in enumerate(reader):
            order = line[0].split()
            orders.append(order)
            # print (order)


def read_weights_data():
    with open(item_weight_path) as weight_file:
        file = weight_file.readlines()
        for i, line in enumerate(file):
            if(i == 0):
                continue
            item = line.split()[0]
            weight = float(line.split()[4])
            item_weight[item] = weight





def reorganize_order(order):
    total_weight = calculate_order_weight(order)
    for index,item in enumerate(order):
        weight = item_weight[item]
        total_weight = total_weight + weight

    if total_weight > WEIGHT_LIMIT:
        splitted_orders = split_order(order)
    else:
        order = combine_order(order)



def calculate_order_weight(order):
    total_weight = 0
    for index,item in enumerate(order):
        if item_weight.has_key(item):
            weight = item_weight[item]
            total_weight = total_weight + weight
    return total_weight


def split_order(order):
    first_half_order = []
    second_half_order = []
    item_weights = []

    new_order = []
    new_order_list = []
    total_weight = 0
    for item in order:
        if item_weight.has_key(item):
            weight = item_weight[item]
            total_weight = total_weight + weight

        new_order.append(item)

        if total_weight >= WEIGHT_LIMIT:
            new_order.pop()
            new_order_list.append(new_order)
            new_order = []
            new_order.append(item)
            total_weight = 0

    new_order_list.append(new_order)
    return new_order_list



    # for i in range(len(order) / 2):
    #     first_half_order.append(order[i])
    # for i in range(len(order) / 2, len(order)):
    #     second_half_order.append(order[i])
    #
    # return first_half_order, second_half_order


def combine_order(order):
    current_order_weight = calculate_order_weight(order)
    current_order_items = len(order)
    index = orders.index(order)

    for next in range(index + 1, len(orders) - 1):
        next_order = orders[next]
        next_order_weight = calculate_order_weight(next_order)
        if current_order_weight + next_order_weight > WEIGHT_LIMIT:
            continue

        next_order_items = len(next_order)
        if current_order_items + next_order_items > 15:
            continue

        list_new = []
        for item in order:
            list_new.append(item)
        for item in next_order:
            list_new.append(item)

        print ("Combine with order#: %d" %(next + 1))
        return list_new




def preprocess_data():
    global distance_matrix, source, destination

    values = shelves.values()
    source = np.array(values)
    destination = np.array(values)

    distance_matrix = np.zeros((2 * len(source), 2 * len(destination)))
    size = 2 * len(source)

    for i in range(size):
        if(i % 2 == 0):
            point = source[i / 2].get_left()
        else:
            point = source[i / 2].get_right()
        lookup_table[point] = i

    for i in range(size):
        for j in range(size):
            if(i % 2 == 0):
                start_shelf = source[i / 2].get_left()
                if(j % 2 == 0):
                    end_shelf = destination[j / 2].get_left()
                else:
                    end_shelf = destination[j / 2].get_right()
            else:
                start_shelf = source[i / 2].get_right()
                if (j % 2 == 0):
                    end_shelf = destination[j / 2].get_left()
                else:
                    end_shelf = destination[j / 2].get_right()

            distance_matrix[i][j] = manhattan_distance(start_shelf, end_shelf)


def process_all_orders():
    for index, order in enumerate(orders):
        if ALGORITHM == 'nn':
            path, cost, visited = nn_search(order)
            optimized_paths_nn.append(path)
            path = path_to_str(path)
            optimized_paths_str_nn.append(path)
            optimized_distance_nn.append(cost)
        elif ALGORITHM == 'bnb':
            path, cost, last_matrix = branch_bound_search(order)
            optimized_paths_bnb.append(path)
            path = path_to_str(path)
            optimized_paths_str_bnb.append(path)
            optimized_distance_bnb.append(cost)
        else:
            print ('Invalid input, will use NN search for order# ', order)
            path, cost, visited = nn_search(order)
        # path, distance = nn_search(order)

        default_length = default_order_length(order)
        default_distance.append(default_length)

        print ("order# ", index, " done! with method ", ALGORITHM)


def write2file():
    # with open("warehouse-orders-v01-optimized.csv", "w") as csvfile:
    #     writer = csv.writer(csvfile)
    #     writer.writerow(["Original Order", "Optimized Order", "Original Distance", "Optimized Distance"])
    #     for i in range(len(orders)):
    #         writer.writerow([orders[i], optimized_paths[i], default_distance[i], optimized_distance[i]])

    global optimized_paths
    if ALGORITHM == 'nn':
        if os.path.exists('./data_nn.pickle'):
            return
        pickle_out = open("data_nn.pickle", "w")
        pickle.dump(optimized_paths_nn, pickle_out)
    elif ALGORITHM == 'bnb':
        if os.path.exists('./data_bnb.pickle'):
            return
        pickle_out = open("data_bnb.pickle", "w")
        pickle.dump(optimized_paths_bnb, pickle_out)
    pickle_out.close()



def manhattan_distance(start, end):
    start_x = start.get_x()
    start_y = start.get_y()

    end_x = end.get_x()
    end_y = end.get_y()

    distance = np.abs(start_x - end_x) + np.abs(start_y - end_y)

    if(start_y == end_y and start_x <> end_x):
        return distance + 2         # move downward one more step in order not to go across items
    return distance


def find_path():
    if(algo_choice == 1):
        nn_search()
    else:
        branch_bound_search()


def default_order_length(order):
    distance = 0;
    lastpoint = startpoint
    for item in order:
        nextpoint = database[item].get_left()
        distance += manhattan_distance(lastpoint, nextpoint)
        lastpoint = nextpoint
    distance += manhattan_distance(lastpoint, endpoint)
    return distance


def calculate_path_distance(path):
    distance = 0
    lastpoint = startpoint
    for nextpoint in path:
        distance += manhattan_distance(lastpoint, nextpoint)
        lastpoint = nextpoint
    return distance


def find_common_point(points, point_list):
    visited = []
    for point in points:
        if point == startpoint or point == endpoint:
            continue

        visited.append(point)
        shared_point = get_neighbor(point, point_list)

        for other_point in points:
            if other_point in visited or other_point == startpoint or other_point == endpoint:
                continue
            neighbor = get_neighbor(other_point, point_list)
            if shared_point == neighbor:
                points.remove(point)
                points.remove(other_point)

    return points



def branch_bound_search(order):
    branch_bound_matrix, point_list = build_bb_matrix(order)
    points = sorted(set(point_list), key=point_list.index)
    points = find_common_point(points, point_list)

    start_point = startpoint
    reduced_matrix, cost = reduce_matrix(branch_bound_matrix)
    start_node = TreeNode(startpoint, reduced_matrix, cost)
    start_node.set_id(0)

    last_node = start_node
    queue = Queue.PriorityQueue()
    # queue.put(start_node)

    nn_path, _, _ = nn_search(order)
    upper_bound_cost = find_upper_bound(nn_path, point_list, start_node)
    # print upper_bound_cost

    id = start_node.get_id()
    for point in points:
        # if point == endpoint:
        #     continue

        child = build_tree_node(point, start_node, point_list)
        id = id + 1
        child.set_id(id)

        if child.get_cost() > upper_bound_cost:
            continue
        queue.put(child)

    candidate = nn_path
    paths = {}
    id = 0

    start = time.time()

    last_matrix = None
    visited = []
    while(queue.qsize() > 0):

        if(time.time() > start + 30):
            print ('Time Out!')
            return candidate, calculate_path_distance(candidate), last_matrix

        parent_node = queue.get()
        visited.append(parent_node.get_point())

        if(paths.has_key(parent_node)):
            path = paths[parent_node]
        else:
            path = []
            path.append(start_point)
            path.append(parent_node.get_point())

        parent_level = parent_node.get_level()

        for point in points:

            # neighbor = get_neighbor(point, point_list)

            if point == parent_node.get_point() or point in path or point == endpoint:
                continue

            # if(parent_node.get_point() <> neighbor):
            id = id + 1
            child = build_tree_node(point, parent_node, point_list)
            child.set_level(parent_level + 1)
            child.set_id(id)

            parent_node.add_child(child)

            # visited.append(point)
            path.append(child.get_point())

            # print len(path)
            # else:
            #     continue


            if (len(path) == len(nn_path) - 1):
                if (child.get_cost() > INFINITY - 2000):
                    path.remove(child.get_point())
                    continue

                path.append(endpoint)
                candidate = path

                return candidate, calculate_path_distance(candidate), last_matrix
            else:
                # print (len(path))
                paths[child] = copy.deepcopy(path)
                path.remove(child.get_point())
                last_matrix = child.get_matrix()

            if(child.get_cost() >= upper_bound_cost):
                continue
            queue.put(child)

    path.append(endpoint)
    return candidate, calculate_path_distance(candidate), last_matrix


def find_upper_bound(path, point_list, start_node):
    index = []
    for station in path:
        if station == endpoint or station == startpoint:
            continue
        index.append(point_list.index(station))
    index.append(len(point_list) - 1)

    matrix = copy.deepcopy(start_node.get_matrix())
    cost = start_node.get_cost()


    source = 0
    for target in index:
        cost_to_target = matrix[source][target]
        if source == 0 and target <> len(point_list) - 1:
            matrix[source] = INFINITY
            matrix[:, target] = INFINITY
            if target % 2 == 0:
                matrix[:, target - 1] = INFINITY
            else:
                matrix[:, target + 1] = INFINITY
        elif source <> 0 and target <> len(point_list) - 1:
            if source % 2 == 0:
                matrix[source] = INFINITY
                matrix[source - 1] = INFINITY
                if target % 2 == 0:
                    matrix[:, target - 1] = INFINITY
                else:
                    matrix[:, target + 1] = INFINITY
            else:
                matrix[source] = INFINITY
                matrix[source + 1] = INFINITY
                matrix[:, target] = INFINITY
                if target % 2 == 0:
                    matrix[:, target - 1] = INFINITY
                else:
                    matrix[:, target + 1] = INFINITY
        elif target == len(point_list) - 1:
            matrix[source] = INFINITY
            matrix[:, target] = INFINITY
            if source % 2 == 0:
                matrix[source - 1] = INFINITY
            else:
                matrix[source + 1] = INFINITY


        reduced_matrix, reduced_cost = reduce_matrix(matrix)
        matrix = copy.deepcopy(reduced_matrix)
        source = target
        cost = cost + reduced_cost + cost_to_target

    return cost


def calculate_effort(order, path):
    total_effort = 0
    remained_distance = calculate_path_distance(path)
    lastpoint = startpoint

    for point in path:
        remained_distance = remained_distance - manhattan_distance(lastpoint, point)
        for item in order:
            shelf = database[item]
            if shelf.get_left() == point or shelf.get_right() == point:
                if(item_weight.has_key(item)):
                    weight = item_weight[item]
                else:
                    print ('Item ', item, 'weight missing')
                    weight = 0

                total_effort = total_effort + weight * remained_distance
        lastpoint = point

    return total_effort

def get_neighbor(node, list):
    index = list.index(node)
    if(index % 2 == 1):
        neighbor = list[index + 1]
    else:
        neighbor = list[index - 1]

    return neighbor


def build_tree_node(target_point, parent_node, list):
    i = list.index(parent_node.get_point())
    j = list.index(target_point)
    matrix = copy.deepcopy(parent_node.get_matrix())


    cost_to_node = matrix[i][j]



    matrix[i] = INFINITY
    matrix[:, j] = INFINITY
    matrix[j][i] = INFINITY

    # if(i > 0):
    if(i > 0 and i < len(list) - 1):
        if (i % 2 == 1):
            matrix[i + 1] = INFINITY
            if(j + 1 < len(list)):
                matrix[j + 1][i] = INFINITY
        else:
            matrix[i - 1] = INFINITY
            matrix[j - 1][i] = INFINITY
        if(j % 2 == 1):
            if (j + 1 < len(list)):
                matrix[:, j + 1] = INFINITY
        else:
            matrix[:, j - 1] = INFINITY
    elif i == 0:
        if(j % 2 == 1):
            if (j + 1 < len(list)):
                matrix[:, j + 1] = INFINITY
        else:
            matrix[:, j - 1] = INFINITY

    reduced_matrix, cost = reduce_matrix(matrix)
    tree_node = TreeNode(target_point, reduced_matrix, cost + cost_to_node + parent_node.get_cost())

    return tree_node


def reduce_matrix(matrix):
    cost_row = []
    cost_col = []

    for i in range(0, matrix.shape[0]):
        cost = np.min(matrix[i])
        if(cost > INFINITY - 2000):
            continue
        matrix[i] = matrix[i] - np.min(matrix[i])
        cost_row.append(cost)

    for j in range(0, matrix.shape[1]):
        cost = np.min(matrix[:,j])
        if(cost > INFINITY - 2000):
            continue
        matrix[:,j] = matrix[:,j] - np.min(matrix[:,j])
        cost_col.append(cost)

    total_cost = np.sum(cost_row) + np.sum(cost_col)

    return matrix, total_cost


def build_bb_matrix(order):
    np.set_printoptions(suppress=True)

    shelves_list = []
    shelves_list.append(startpoint)
    for item in order:
        shelf = database[item]
        shelves_list.append(shelf.get_left())
        shelves_list.append(shelf.get_right())
    shelves_list.append(endpoint)

    size = len(shelves_list)
    matrix = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            if(i == 0 and j == 0):
                matrix[i][j] = INFINITY
                continue
            if(j == 0):
                matrix[i][j] = INFINITY
                continue
            if(i == size - 1):
                matrix[i][j] = INFINITY
                continue
            if(j == i):
                matrix[i][j] = INFINITY
            elif(i % 2 == 1 and i + 1 == j):
                matrix[i][j] = INFINITY
            elif(i % 2 == 0 and i - 1 == j ):
                matrix[i][j] = INFINITY
            else:
                start_shelf = shelves_list[i]
                end_shelf = shelves_list[j]
                matrix[i][j] = manhattan_distance(start_shelf, end_shelf)

    return matrix, shelves_list


def nn_search(single_order):
    min_distance = sys.maxint
    shelf_pool = []         # store candidate next shelf
    visited = []

    for item in single_order:       # build shelf_pool
        shelf = database[item]
        shelf_pool.append(shelf.get_left())         # two neighbor point associated with the shelf   -1 --shelf-- +1
        shelf_pool.append(shelf.get_right())
    shelf_pool_copy = shelf_pool[:]
    order_pool_copy = single_order[:]

    for next in single_order:
        path = []
        total_distance = 0
        path.append(startpoint)
        visited.append(startpoint)

        shelf_pool = shelf_pool_copy[:]
        order_pool = order_pool_copy[:]

        shelf = database[next]
        path.append(shelf.get_left())
        total_distance += manhattan_distance(startpoint, shelf.get_left())

        order_pool.remove(next)     # remove the current item
        shelf_pool.remove(shelf.get_right())        # remove the shelves point associated with current item
        shelf_pool.remove(shelf.get_left())
        last_shelf = shelf.get_left()

        i = 0
        while(i < len(order_pool)):         # remove shelf sharing the same pick up point with current starting shelf
            item = order_pool[i]
            other_shelf = database[item]
            # print item, other_shelf
            if(other_shelf.get_left() == last_shelf or other_shelf.get_right() == last_shelf):
                order_pool.remove(item)     # remove the item sharing the same pick up point with current item
                if(other_shelf.get_left() in shelf_pool):       # remove the associated shelves point in pool
                    shelf_pool.remove(other_shelf.get_left())
                if(other_shelf.get_right() in shelf_pool):
                    shelf_pool.remove(other_shelf.get_right())
            else:
                i += 1

        while(len(order_pool) > 0):
            next_shelf, distance = find_next_shelf(last_shelf, shelf_pool)
            path.append(next_shelf)
            total_distance += distance
            visited.append(next_shelf)

            delete_shelves, delete_orders = find_next_order(next_shelf, order_pool)
            for delete_shelf in delete_shelves:
                visited.append(delete_shelf)
                if delete_shelf.get_left() in shelf_pool:
                    shelf_pool.remove(delete_shelf.get_left())
                if delete_shelf.get_right() in shelf_pool:
                    shelf_pool.remove(delete_shelf.get_right())
            order_pool = [i for i in order_pool if i not in delete_orders]
            last_shelf = next_shelf

        path.append(endpoint)
        visited.append(endpoint)
        total_distance += manhattan_distance(last_shelf, endpoint)
        if(total_distance < min_distance):
            min_distance = total_distance
            best_path = path

    return best_path, min_distance, visited


def path_to_str(best_path):
    string = ''
    for point in best_path:
        x = str(point.get_x())
        y = str(point.get_y())
        string += "(" + x + "," + y + ")"
    return string


def find_next_order(next_shelf, order_pool):
    delete_shelves = []
    delete_orders = []
    for item in order_pool:
        shelf = database[item]
        if shelf.get_left() == next_shelf:
            delete_shelves.append(shelf)
            delete_orders.append(item)
        if shelf.get_right() == next_shelf:
            delete_shelves.append(shelf)
            delete_orders.append(item)

    return delete_shelves,delete_orders


def find_next_shelf(startpoint, shelf_pool):
    min_distance = sys.maxint
    next_nearest_shelf = None

    start_index = get_index(startpoint)
    for next_shelf in shelf_pool:
        next_index = get_index(next_shelf)
        distance = distance_matrix[start_index][next_index]
        if(distance < min_distance):
            min_distance = distance
            next_nearest_shelf = next_shelf

    return next_nearest_shelf, min_distance


def get_index(shelf):
    return lookup_table[shelf]


def plot_path(path):
    # root = tk.Tk()

    top = Toplevel()
    top.wm_geometry("300x300")

    figure = Figure(figsize=(5, 5), dpi=100)
    sp = figure.add_subplot(111)
    sp.set_axis_off()

    # draw shelves
    sp.scatter(x_coord, y_coord, s=50, marker='s', c='b')
    plt.xlim(0, max_x)
    plt.ylim(0, max_y)

    plt.plot()

    lastpoint = path[0]
    for nextpoint in path:

        x1 = lastpoint.get_x()
        y1 = lastpoint.get_y()
        x2 = nextpoint.get_x()
        y2 = nextpoint.get_y()

        startx = startpoint.get_x()
        starty = startpoint.get_y()
        sp.plot(startx, starty, 'ro')

        if nextpoint == endpoint:
            if x1 == x2:
                if not y1 == y2:
                    sp.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.6, head_length=0.1)
            elif y1 == y2:
                if not x1 == x2:
                    sp.arrow(x1, y1, 0, 1, head_width=0.6, head_length=0.3)
                    sp.arrow(x1, y1 + 1, x2 - x1, y2 - y1, head_width=0.6, head_length=0.3)
                    sp.arrow(x2, y2 + 1, 0, -1, head_width=0.6, head_length=0.3)
            else:
                dirx = (x2 - x1) / abs(x2 - x1)
                diry = (y2 - y1) / abs(y2 - y1)
                cory1 = diry * (abs(y2 - y1) - 1) + y1

                sp.arrow(x1, y1, 0, cory1 - y1 - 1, head_width=0.6, head_length=0.3)
                sp.arrow(x1, cory1 - 1, x2 - x1, 0, head_width=0.6, head_length=0.3)
                sp.arrow(x2, cory1 - 1, 0, y2 - cory1 - 1, head_width=0.6, head_length=0.3)
            continue

        if x1 == x2:
            if not y1 == y2:
                sp.arrow(x1, y1, x2 - x1, y2 - y1, head_width=0.6, head_length=0.1)
                sp.plot(x2, y2, 'ro')
        elif y1 == y2:
            if not x1 == x2:
                sp.arrow(x1, y1, 0, 1, head_width=0.6, head_length=0.3)
                sp.arrow(x1, y1 + 1, x2 - x1, y2 - y1, head_width=0.6, head_length=0.3)
                sp.arrow(x2, y2 + 1, 0, -1, head_width=0.6, head_length=0.3)
                sp.plot(x2, y2, 'ro')
        else:
            dirx = (x2 - x1) / abs(x2 - x1)
            diry = (y2 - y1) / abs(y2 - y1)
            cory1 = diry * (abs(y2 - y1) - 1) + y1

            sp.arrow(x1, y1, 0, cory1 - y1, head_width=0.6, head_length=0.3)
            sp.arrow(x1, cory1, x2 - x1, 0, head_width=0.6, head_length=0.3)
            sp.arrow(x2, cory1, 0, y2 - cory1, head_width=0.6, head_length=0.3)
            sp.plot(x2, y2, 'ro')

        lastpoint = nextpoint


    canvas = FigureCanvasTkAgg(figure, master=top)
    canvas.draw()
    canvas.get_tk_widget().pack(side="top", expand=1)



def draw_path(path):
    lastpoint = path[0]

    for nextpoint in path:
        draw_line(lastpoint, nextpoint)
        lastpoint = nextpoint


def draw_line(lastpoint, nextpoint):
    return



def draw_shelves():
    plt.scatter(x_coord, y_coord, s=50, marker='s', c='r')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, max_x + 1)
    plt.ylim(0, max_y + 1)


def algo_chosen_nn():
    global ALGORITHM
    ALGORITHM = 'nn'
    print (ALGORITHM)


def algo_chosen_bnb():
    global ALGORITHM
    ALGORITHM = 'bnb'
    print (ALGORITHM)


def factor_in_weight_yes():
    global FACTOR_WEIGHT
    FACTOR_WEIGHT = 'YES'


def factor_in_weight_no():
    global FACTOR_WEIGHT
    FACTOR_WEIGHT = 'NO'


def read_pickle():
    global optimized_paths_nn
    global optimized_paths_bnb
    global num_of_path

    if ALGORITHM == 'nn':
        if os.path.exists('./data_nn.pickle'):
            pickle_nn = open("data_nn.pickle", "r")
            optimized_paths_nn = pickle.load(pickle_nn)
            num_of_path = len(optimized_paths_nn)
            print ("data exist, previous nn data loaded.")
        else:
            process_all_orders()
    elif ALGORITHM == 'bnb':
        if os.path.exists('./data_bnb.pickle'):
            pickle_bnb = open("data_bnb.pickle", "r")
            optimized_paths_bnb = pickle.load(pickle_bnb)
            num_of_path = len(optimized_paths_bnb)
            print ("data exist, previous bnb data loaded.")
        else:
            process_all_orders()



def get_gui_input():
    global startpoint
    global endpoint
    global max_weight
    global ALGORITHM
    global order

    startpoint.set_x(int(start_x.get()))
    startpoint.set_y(int(start_y.get()))

    endpoint.set_x(int(end_x.get()))
    endpoint.set_y(int(end_y.get()))

    if FACTOR_WEIGHT == "YES":
        WEIGHT_LIMIT = int(weight_limit.get())

    print ("Start point is: %s" %startpoint)
    print ("End point is: %s" %endpoint)
    if FACTOR_WEIGHT == "YES":
        print ("Weight limit: %d" %WEIGHT_LIMIT)
    print ("Algorithm chosen: %s" %ALGORITHM)



    read_warehouse_data()
    if len(orders) == 0:
        read_orders_data()
    preprocess_data()
    read_weights_data()
    read_pickle()
    write2file()

    if int(order_number.get()) is not None:
        order_index = int(order_number.get()) - 1
        order = orders[order_index]
        print ('The original order is: %s' %order)


        if FACTOR_WEIGHT == 'YES':
            weight = calculate_order_weight(order)
            print ('The order weight is: %d' %weight)

            if weight > WEIGHT_LIMIT:
                order_list = split_order(order)
                print ("The order has been spliited into %d orders" %len(order_list))

                for index,order in enumerate(order_list):
                    print ("Order: %s" %order)

                    if ALGORITHM == 'nn':
                        path1, cost1, _ = nn_search(order)
                        weight = calculate_order_weight(order)

                        print ("The optimal path for this order: %s" % path_to_str(path1))
                        print ("The total distance for this order: %d" % calculate_path_distance(path1))
                        print ("The total effort for this order: %d" % calculate_effort(orders[order_index], path1))

                        plot_path(path1)

                    if ALGORITHM == 'bnb':
                        path1, cost1, _ = branch_bound_search(order)

                        print ("The optimal path for this order: %s" % path_to_str(path1))
                        print ("The total distance for this order: %d" % calculate_path_distance(path1))
                        print ("The total effort for this order: %d" % calculate_effort(orders[order_index], path1))

                        plot_path(path1)

            else:
                order = combine_order(order)
                print ("Combine order is: %s" %order)

                if ALGORITHM == 'nn':
                    path1, cost1, _ = nn_search(order)
                    print ("The optimal path for combined order is: %s" % path_to_str(path1))
                    print ("The total distance for combined order is: %d" % calculate_path_distance(path1))
                    print ("The total effort for this order: %d" % calculate_effort(orders[order_index], path1))

                    plot_path(path1)

                if ALGORITHM == 'bnb':
                    path2, cost2, _ = branch_bound_search(order)
                    print ("The optimal path for combined order is: %s" % path_to_str(path2))
                    print ("The total distance for combined order is: %d" % calculate_path_distance(path2))
                    print ("The total effort for this order: %d" % calculate_effort(orders[order_index], path2))

                    plot_path(path2)

        else:
            if ALGORITHM == 'nn':
                path = optimized_paths_nn[order_index]
            if ALGORITHM == 'bnb':
                path = optimized_paths_bnb[order_index]

            print ("The optimal path is: %s" %path_to_str(path))
            print ("The total distance is: %d" %calculate_path_distance(path))
            print ("The total effort is: %d" %calculate_effort(orders[order_index], path))
            print ("-------------------------------------------------")

            plot_path(path)





if __name__ == "__main__":

    window = Tk()
    window.geometry("500x500")
    window.title("EECS221 App")

    var1 = tk.StringVar()
    var2 = tk.StringVar()

    # algorithm chosen section
    lbl1 = Label(window, text="Algorithm chosen", font=("Arial", 16))
    lbl1.grid(column=0, row=0)
    rad1 = Radiobutton(window, text='Nearest Neighbor', value=1, variable=var1, command=algo_chosen_nn)
    rad1.grid(column=1, row=0)
    rad2 = Radiobutton(window, text='Branch & Bound', value=2, variable=var1, command=algo_chosen_bnb)
    rad2.grid(column=2, row=0)

    # startpoint and endpoint section
    lbl2 = Label(window, text="Start Point (x, y)", font=("Arial", 16))
    lbl2.grid(column=0, row=1)
    start_x = Entry(window, width=5)
    start_x.grid(column=1, row=1)
    start_y = Entry(window, width=5)
    start_y.grid(column=2, row=1)

    lbl3 = Label(window, text="End Point (x, y)", font=("Arial", 16))
    lbl3.grid(column=0, row=2)
    end_x = Entry(window, width=5)
    end_x.grid(column=1, row=2)
    end_y = Entry(window, width=5)
    end_y.grid(column=2, row=2)

    # weight factor
    lbl4 = Label(window, text="Factor in weight?", font=("Arial", 16))
    lbl4.grid(column=0, row=3)
    rad3 = Radiobutton(window, text='Yes', value=3, variable=var2, command=factor_in_weight_yes)
    rad3.grid(column=1, row=3)
    rad4 = Radiobutton(window, text='No', value=4, variable=var2, command=factor_in_weight_no)
    rad4.grid(column=2, row=3)

    # max weight
    lbl5 = Label(window, text="Weight limit for one worker?", font=("Arial", 16))
    lbl5.grid(column=0, row=4)
    weight_limit = Entry(window, width=5)
    weight_limit.grid(column=1, row=4)

    # reading batch file input
    lbl6 = Label(window, text="Available path info: ", font=("Arial", 16))
    lbl6.grid(column=0, row=5)
    num_of_path = tk.StringVar()
    path_info = tk.Label(window, textvariable=num_of_path, bg='white', font=('Arial', 16), width=15,
                   height=1)
    path_info.grid(column=1, row=5)

    # show result
    lbl7 = Label(window, text="Show path for order#: ", font=("Arial", 16))
    lbl7.grid(column=0, row=6)
    order_number = Entry(window, width=5)
    order_number.grid(column=1, row=6)


    btn = Button(window, text="Go", command=get_gui_input)
    btn.grid(column=0, row=10)

    window.mainloop()




    # read_warehouse_data()
    # read_orders_data_csv()
    # preprocess_data()
    # read_weights_data()
    # # read_pickle()
    # # write2file()
    #
    #
    #
    # order = orders[166]
    # order_list = split_order(order)
    # print (order_list)

    #
    # path, cost, _ = branch_bound_search(combined_order)
    # print (path)

    # ALGORITHM = 'nn'
    # paths_nn = read_pickle()
    # path = optimized_paths[7]
    #
    # ALGORITHM = 'bnb'
    # paths_nn = read_pickle()
    # path1 = optimized_paths[7]
    #
    # print ('nn cost: ', calculate_path_distance(path))
    # print ('bnb cost: ', calculate_path_distance(path1))
    #
    #
    # print ('nn path: ', path)
    # print ('bnb path: ', path1)
    # plot_path(path1)
    #
    # if ALGORITHM == '1':
    #     path, cost, visited = nn_search(order)
    # elif ALGORITHM == '2':
    #     path, cost, last_matrix, visited = branch_bound_search(order)
    # else:
    #     print ('Invalid input, will use NN search as default algorithm')
    #     path, cost, visited = nn_search(order)
    #
    # effort = calculate_effort(order, path)
    # print ('Total effort: ' ,effort)

    # for point in visited:
    #     print point,
    # plot_path(path)
    # for i in range(distance_matrix.shape[0]):
    #     for j in range(distance_matrix.shape[1]):
    #         if(distance_matrix[i][j] > INFINITY - 2000):
    #             print ('INFINITY'),
    #         else:
    #             print distance_matrix[i][j],
    #     print ('\n')
    # print (distance_matrix.shape)
    # matrix,_ = build_bb_matrix(order)
    # for i in range(matrix.shape[0]):
    #     for j in range(matrix.shape[1]):
    #         if(matrix[i][j] > INFINITY - 2000):
    #             print ('INFINITY'),
    #         else:
    #             print matrix[i][j],
    #     print ('\n')
    #
    # matrix, cost = reduce_matrix(matrix)
    # for i in range(matrix.shape[0]):
    #     for j in range(matrix.shape[1]):
    #         print matrix[i][j],
    #     print ('\n')
    #
    # print (cost)
    #


    # path, cost, last_matrix, visited = branch_bound_search(order)
    # print ('nn cost: ', cost)
    # print ('bnb cost: ', cost1)

    #
    # for node in visited:
    #     print node,

    #
    #
    # for i in range(last_matrix.shape[0]):
    #     for j in range(last_matrix.shape[1]):
    #         if(last_matrix[i][j] > INFINITY - 2000):
    #             print ('INFINITY'),
    #         else:
    #             print last_matrix[i][j],
    #     print ('\n')

    # path2, cost2 = nn_search(order)

    # print ("cost1: ", cost1)
    # print ("cost2: ", cost2)
    # plot_path(path1)
    # plot_path(path2)

    # print (.)
    # end = start + 5
    #
    # print end - start


    # mat,_ = build_bb_matrix(order)
    # for i in range(len(mat)):
    #     for j in range(len(mat[0])):
    #         print mat[i][j],
    #     print ('\n')
    # print ("-------------------")
    # print (mat[:,1])
    # print (np.min(mat[:,1]))
    # print (mat[:,1] - np.min(mat[:,1]))
    # print ("-------------------")
    # plot_path(optimized_paths[11])
    print ('This is main of "smartpath.py"')
