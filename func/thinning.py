import pickle
from sys import setrecursionlimit
import numpy as np
import pandas as pd
import skimage.io as io
import skimage.measure as measure
import SimpleITK as sitk
import os

# find Articulation Points in 3d binary numpy array, assuming the points are connected in 26 directions (3-connectivity)
# articulation point: a point in the binary image, if removed, the number of connected components will increase
# param: binary_img: 3d numpy array
# return: list of coordinates of articulation points

setrecursionlimit(50000)


def find_articulation_points(binary_img: np.ndarray):

    assert binary_img.ndim == 3
    if binary_img.dtype != bool:
        binary_img = binary_img.astype(bool)

    # find the coordinates of all the points (the points with value 1) in the binary image

    # return points
    ret = []
    visited = np.zeros((np.argwhere(binary_img).shape[0]), dtype=bool)
    points = np.argwhere(binary_img)
    point_indexes = {}
    for i in range(points.shape[0]):
        point_indexes[tuple(points[i])] = i

    print(len(points))

    # this subroutine perfomres Articulation Points detection in 3d binary image assuming that the image has only one connected component
    def subroutine(root_point_idx: np.ndarray):
        # use tarjan's algorithm to find articulation points
        # record dfs order and low value for each point
        # low value: the lowest order of the points that can be reached from the subtree rooted at the point
        # dfs order: the order of the point in the depth first search+

        nonlocal binary_img, ret, visited, points, point_indexes

        dx = [-1, 0, 1]
        dy = [-1, 0, 1]
        dz = [-1, 0, 1]

        dfs_order = np.ones(
            (np.argwhere(binary_img).shape[0]), dtype=int) * 10000000
        # indexes of the points in the points array

        # visited: whether the point has been visited

        # low: the low value of the point
        low = np.ones((np.argwhere(binary_img).shape[0]), dtype=int) * 10000000
        # is_cut: whether the point is an articulation point
        is_cut = np.zeros((np.argwhere(binary_img).shape[0]), dtype=bool)
        # order: the order of the point in the depth first search
        order = 0

        def dfs(point_idx, parent):
            # point_idx: the index of the point in the points array

            # low: the low value of the point
            # parent: the parent of the point
            # dfs_order: the dfs order of the point
            # visited: whether the point has been visited
            nonlocal binary_img, low, dfs_order, visited
            nonlocal order
            nonlocal ret
            nonlocal dx, dy, dz
            nonlocal points
            nonlocal point_indexes
            nonlocal is_cut

            visited[point_idx] = True
            dfs_order[point_idx] = order
            low[point_idx] = order
            point = points[point_idx]
            children = 0
            order += 1

            for i, j, k in [(i, j, k) for i in dx for j in dy for k in dz]:
                if i == 0 and j == 0 and k == 0:
                    continue

                x = point[0] + i
                y = point[1] + j
                z = point[2] + k
                # check if the point is out of the boundary
                if x < 0 or x >= binary_img.shape[0] or y < 0 or y >= binary_img.shape[1] or z < 0 or z >= binary_img.shape[2]:
                    continue
                # check if the point is not 1
                if not binary_img[x, y, z]:
                    continue

                child_idx = point_indexes[tuple((x, y, z))]
                if not visited[child_idx]:
                    children += 1
                    dfs(child_idx, point_idx)
                    low[point_idx] = min(low[point_idx], low[child_idx])
                    if parent != -1 and low[child_idx] >= dfs_order[point_idx]:
                        is_cut[point_idx] = True
                    elif parent == -1 and children > 1:
                        is_cut[point_idx] = True
                elif child_idx != parent:
                    low[point_idx] = min(low[point_idx], dfs_order[child_idx])

        dfs(root_point_idx, -1)
        # poot articulation points into ret
        return np.argwhere(is_cut)

    # check if each point is visited and run subroutine if the point is not visited
    for i in range(points.shape[0]):
        if not visited[i]:
            ret.extend(subroutine(i))

    return ret

# find neighbor points count for each point in 3d binary numpy array (in 26 directions)
# param: binary_img: 3d numpy array
# return: 3d numpy array, the number of 26-neighbors of each point in the binary image


def find_neighbor_points_count(binary_img: np.ndarray):
    # binary_img: 3d numpy array
    # return: 3d numpy array, the number of 6-neighbors of each point in the binary image
    assert binary_img.ndim == 3
    if binary_img.dtype != np.bool:
        binary_img = binary_img.astype(np.bool)

    dx = [-1, 0, 1]
    dy = [-1, 0, 1]
    dz = [-1, 0, 1]

    directions = [(i, j, k) for i in dx for j in dy for k in dz if not (
        i == 0 and j == 0 and k == 0)]

    # the number of 26-neighbors of each point in the binary image
    neighbor_points_count = np.zeros(binary_img.shape, dtype=int)

    # for each point in the binary image
    for i in range(binary_img.shape[0]):
        for j in range(binary_img.shape[1]):
            for k in range(binary_img.shape[2]):
                # if the point is 0, skip
                if not binary_img[i, j, k]:
                    continue
                # count the number of 26-neighbors of the point
                count = 0
                for direction in directions:
                    x = i + direction[0]
                    y = j + direction[1]
                    z = k + direction[2]
                    # if the neighbor point is out of the boundary, skip
                    if x < 0 or x >= binary_img.shape[0] or y < 0 or y >= binary_img.shape[1] or z < 0 or z >= binary_img.shape[2]:
                        continue
                    # if the neighbor point is 1, count
                    if binary_img[x, y, z]:
                        count += 1
                neighbor_points_count[i, j, k] = count

    return neighbor_points_count


'''
perform 3d thinning on a binary image using the following algorithm

repeat
    for each direction U, N, E, S, W, and D do
        mark all border points according to the actual direction that are simple points (not articulation points) and not line-end points
        for each marked point p do
            if p is simple (articulative) in the actual image then
                if p is not a line-end point then
                    delete p
                else if #(deleted 6-neighbors of p)≥ t then
                    delete p
        endfor
    endfor
until changes occur

Additional explanation: A “1” voxel is called line-end point if it has exactly one “1” 26-neighbor.
'''


def thinning_3d(binary_img: np.ndarray, t=1):
    # binary_img: 3d numpy array
    # t: the threshold for the number of 6-neighbors of a point, if the number of 6-neighbors of a point is greater than or equal to t, the point will be deleted
    # the paper said t=1 or 2 is good, use 1 for default
    # return: the thinned binary image
    assert binary_img.ndim == 3
    if binary_img.dtype != np.bool:
        binary_img = binary_img.astype(np.bool)

    # find the coordinates of all the points (the points with value 1) in the binary image
    points = np.argwhere(binary_img)

    # find the coordinates of all the line-end points in the binary image (e. g, the points with value 1 and the number of 6-neighbors is 1)
    neighbor_points_count = find_neighbor_points_count(binary_img)
    line_end_points = np.argwhere(np.logical_and(
        binary_img, neighbor_points_count == 1))

    # find the coordinates of all the articulation points in the binary image
    articulation_points = find_articulation_points(binary_img)
    articulation_points_in_thinned_img = articulation_points.copy()

    # the directions
    directions = [(-1, 0, 0), (0, 1, 0), (1, 0, 0),
                  (0, -1, 0), (0, 0, 1), (0, 0, -1)]

    # the thinned binary image
    thinned_img = binary_img.copy()

    # repeat until no changes occur
    while True:
        changes = 0
        for direction in directions:
            changes_in_cur_direction = 0
            # mark all border points according to the actual direction that are simple points (not articulation points) and not line-end points
            marked_points = []

            # 1. find the border points
            for point in points:
                if point[0] == 0 or point[0] == binary_img.shape[0]-1 or point[1] == 0 or point[1] == binary_img.shape[1]-1 or point[2] == 0 or point[2] == binary_img.shape[2]-1:
                    marked_points.append(point)
            # 2. find the simple points
            marked_points = [
                point for point in marked_points if point not in articulation_points_in_thinned_img]
            # 3. find the points that are not line-end points
            marked_points = [
                point for point in marked_points if point not in line_end_points]

            for point in marked_points:
                # if p is simple (articulative) in the actual image then
                if not point in articulation_points:
                    # if p is not a line-end point then
                    if not point in line_end_points:
                        # delete p
                        thinned_img[point[0], point[1], point[2]] = False
                        changes += 1
                        changes_in_cur_direction += 1
                    # else if #(deleted 6-neighbors of p)≥ t then
                    elif np.sum(thinned_img[point[0]-1:point[0]+2, point[1]-1:point[1]+2, point[2]-1:point[2]+2]) >= t:
                        # delete p
                        thinned_img[point[0], point[1], point[2]] = False
                        changes += 1
                        changes_in_cur_direction += 1

            # if change occured, update points list, line_end_points list, and articulation_points_in_thinned_img list
            if changes_in_cur_direction > 0:
                points = np.argwhere(thinned_img)
                neighbor_points_count = find_neighbor_points_count(thinned_img)
                line_end_points = np.argwhere(
                    np.logical_and(thinned_img, neighbor_points_count == 1))
                articulation_points_in_thinned_img = find_articulation_points(
                    thinned_img)

        if changes == 0:
            break

    return thinned_img
