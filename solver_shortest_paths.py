import os
import sys
sys.path.append('..')
sys.path.append('../..')
import argparse
import utils

from operator import itemgetter
from collections import deque

from student_utils import *
"""
======================================================================
  Complete the following function.
======================================================================
"""

def solve(list_of_locations, list_of_homes, starting_car_location, adjacency_matrix, params=[]):
    """
    Write your algorithm here.
    Input:
        list_of_locations: A list of locations such that node i of the graph corresponds to name at index i of the list
        list_of_homes: A list of homes
        starting_car_location: The name of the starting location for the car
        adjacency_matrix: The adjacency matrix from the input file
    Output:
        A list of locations representing the car path
        A dictionary mapping drop-off location to a list of homes of TAs that got off at that particular location
        NOTE: both outputs should be in terms of indices not the names of the locations themselves
    """
    size = len(list_of_locations)

    """ MST """

    edges = []

    #create list of [edge index from, edge index to, edge weight]
    for i in range(0, len(adjacency_matrix[0])):
        for j in range(i, len(adjacency_matrix[0])):
            if(adjacency_matrix[i][j]!='x'):
                edges.append([i, j, adjacency_matrix[i][j]])
    """
    sorted_edges = sorted(edges, key=itemgetter(2))

    mst_edges = []

    #set disjoint set
    disjoint_set = [None]*len(adjacency_matrix)
    for i in range(0, len(disjoint_set)):
        disjoint_set[i]=i

    #create mst
    while (len(mst_edges)<len(adjacency_matrix)-1):
        edge = sorted_edges.pop(0)
        if (not is_connected(disjoint_set, edge[0],edge[1])) :
            union(disjoint_set, edge[0],edge[1])
            mst_edges.append(edge)
    #print(mst_edges)
    """


    #convert source and home to indices
    source_index = list_of_locations.index(starting_car_location)
    home_indices = []
    for h in list_of_homes:
        home_indices.append(list_of_locations.index(h))
    paths = []

    """
    #trims mst to create a list of paths from homes to source
    for h in home_indices:
        path = []
        path.append(h)
        end = path[-1]
        next = source_index
        bad_vertices = []
        while(end!=source_index):
            reverse = True
            for e in mst_edges:
                if e[0]==end and e[1] not in path and e[1] not in bad_vertices:
                    next = e[1]
                    reverse = False
                    break
                if e[1]==end and e[0] not in path and e[0] not in bad_vertices:
                    next = e[0]
                    reverse = False
                    break
            if (reverse):
                bad_vertex = path.pop(-1)
                bad_vertices.append(bad_vertex)
                end = path[-1]
            else:
                path.append(next)
                end = next
        paths.append(path)
    #print(paths) #[[home,...,source],...]
    """

    """
    #create adjacency list
    adjacency_list = [float("inf")]*size
    for row in range(size):
        for col in range(size):
            if adjacency_matrix[row][col] != 'x':
                print(row)
                if adjacency_list[row]== float("inf"):
                    adjacency_list[row] = [[col, adjacency_matrix[row][col]]]
                else:
                    adjacency_list[row].append([col,adjacency_matrix[row][col]])
    print(adjacency_list)
    """

    #run dijkstra's to create paths = list or list = paths from source to homes
    paths = []
    deque_paths = []
    for h in home_indices:
        path = list(dijkstra(source_index, h, list(range(size)), edges))
        paths.append(path[::-1])
    #print(paths)

    #create list of vertices that overlap in paths
    if(len(paths)==1):
        repeated_vertices = [source_index]
    else:
        repeated_vertices = []
        count_iter = [0] * size
        for p in paths:
            for loc in p:
                count_iter[loc]+=1
        for i in range(0,len(count_iter)):
            if count_iter[i] > 1:
                repeated_vertices.append(i)
    #print(repeated_vertices)

    #create dict mapping locations to TAs dropped off there
    repeatedToDropoff = {}
    for p in paths:
        if p == []:
            p = [source_index]
        home = p[0]
        for loc in p:
            if (loc in repeated_vertices):
                if loc in repeatedToDropoff:
                    repeatedToDropoff[loc].append(home)
                else:
                    repeatedToDropoff[loc] = [home]
                break
    #repeated_vertices.remove(source_index)

    """
    final_path = []
    start = source_index
    final_path.append(start)
    while(repeated_vertices):
        for edge in mst_edges:
            if edge[0] in repeated_vertices:
                final_path.append(edge[0])
                start = edge[0]
                repeated_vertices.remove(edge[0])
                break
            if edge[1] in repeated_vertices:
                final_path.append(edge[1])
                start = edge[1]
                repeated_vertices.remove(edge[1])
                break
    """
    #run dfs on mst to get route
    visited = []
    backtrack = [-1] * (len(list_of_locations))
    car_path = []
    dfs(visited, edges, source_index, repeated_vertices, backtrack, -1, car_path)

    print(car_path)
    print(repeatedToDropoff)
    ret = [car_path, repeatedToDropoff]
    return ret

def dfs(visited, graph, node, repeated_vertices, backtrack, prev, car_path):
    if node not in visited:
        car_path.append(node)
        backtrack[node] = prev
        visited.append(node)
        for e in graph:
            if e[0]==node and e[1] in repeated_vertices:
                dfs(visited, graph, e[1], repeated_vertices, backtrack, e[0], car_path)
            if e[1]==node and e[0] in repeated_vertices:
                dfs(visited, graph, e[0], repeated_vertices, backtrack, e[1], car_path)
        if (backtrack[node] != -1) :
            car_path.append(backtrack[node])

def dijkstra(start, end, vertices, edges):
    inf = float('inf')
    distances = {vertex: inf for vertex in vertices}
    previous_vertices = {vertex: None for vertex in vertices}
    distances[start] = 0

    while vertices:
        current_vertex = min(vertices, key=lambda vertex: distances[vertex])
        vertices.remove(current_vertex)
        if distances[current_vertex] == inf:
            break
        for edge in edges:
            if (edge[0] == current_vertex):
                neighbor = edge[1]
                alternative_route = distances[current_vertex] + edge[2]
                if alternative_route  < distances[neighbor]:
                    distances[neighbor] = alternative_route
                    previous_vertices[neighbor] = current_vertex
            if (edge[1] ==  current_vertex):
                neighbor = edge[0]
                alternative_route = distances[current_vertex] + edge[2]
                if alternative_route  < distances[neighbor]:
                    distances[neighbor] = alternative_route
                    previous_vertices[neighbor] = current_vertex

    path = deque()
    current_vertex = end

    while previous_vertices[current_vertex] is not None:
        path.appendleft(current_vertex)
        current_vertex = previous_vertices[current_vertex]
    if path:
        path.appendleft(current_vertex)
    return path


""" disjoint set helper functions """

def find(parent, i):
	if parent[i] == i:
		return i
	return find(parent, parent[i])

def union(parent, x, y):
        xroot = find(parent, x)
        yroot = find(parent, y)

        parent[xroot] = yroot

def is_connected(parent, x, y):
    return find(parent, x)==find(parent, y)

"""
======================================================================
   No need to change any code below this line
======================================================================
"""

"""
Convert solution with path and dropoff_mapping in terms of indices
and write solution output in terms of names to path_to_file + file_number + '.out'
"""
def convertToFile(path, dropoff_mapping, path_to_file, list_locs):
    string = ''
    for node in path:
        string += list_locs[node] + ' '
    string = string.strip()
    string += '\n'

    dropoffNumber = len(dropoff_mapping.keys())
    string += str(dropoffNumber) + '\n'
    for dropoff in dropoff_mapping.keys():
        strDrop = list_locs[dropoff] + ' '
        for node in dropoff_mapping[dropoff]:
            strDrop += list_locs[node] + ' '
        strDrop = strDrop.strip()
        strDrop += '\n'
        string += strDrop
    utils.write_to_file(path_to_file, string)

def solve_from_file(input_file, output_directory, params=[]):
    print('Processing', input_file)

    input_data = utils.read_file(input_file)
    num_of_locations, num_houses, list_locations, list_houses, starting_car_location, adjacency_matrix = data_parser(input_data)
    car_path, drop_offs = solve(list_locations, list_houses, starting_car_location, adjacency_matrix, params=params)

    basename, filename = os.path.split(input_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    output_file = utils.input_to_output(input_file, output_directory)

    convertToFile(car_path, drop_offs, output_file, list_locations)


def solve_all(input_directory, output_directory, params=[]):
    input_files = utils.get_files_with_extension(input_directory, 'in')

    for input_file in input_files:
        solve_from_file(input_file, output_directory, params=params)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Parsing arguments')
    parser.add_argument('--all', action='store_true', help='If specified, the solver is run on all files in the input directory. Else, it is run on just the given input file')
    parser.add_argument('input', type=str, help='The path to the input file or directory')
    parser.add_argument('output_directory', type=str, nargs='?', default='.', help='The path to the directory where the output should be written')
    parser.add_argument('params', nargs=argparse.REMAINDER, help='Extra arguments passed in')
    args = parser.parse_args()
    output_directory = args.output_directory
    if args.all:
        input_directory = args.input
        solve_all(input_directory, output_directory, params=args.params)
    else:
        input_file = args.input
        solve_from_file(input_file, output_directory, params=args.params)
