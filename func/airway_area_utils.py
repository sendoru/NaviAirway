from detect_tree import *
from collections import deque

def bfs(node, matrix, component_no: int):
    # do BFS on 2d matrix
    # unvisited node == 0, unreachable node == -1
    dr = [1, -1, 0, 0]
    dc = [0, 0, -1, 1]
    q = deque()
    q.append(node)

    if matrix[node[0]][node[1]] != 0:
        return 0
    
    matrix[node[0]][node[1]] == component_no
    area_size = 1

    while len(q) > 0:
        r_cur, c_cur = q[-1]
        q.popleft()
        for i in range(4):
            r_next = r_cur + dr[i]
            c_next = c_cur + dc[i]
            if r_next < 0 or r_next >= matrix.shape[0] or c_next < 0 or c_next >= matrix.shape[1] or \
                matrix[r_next][c_next] != 1:
                continue
            matrix[r_next][c_next] = component_no
            area_size += 1
            q.append([r_next, c_next])
            

def get_voxel_by_generation(seg_result: np.array, connection_dict: dict):
    ret = seg_result - 1
    for key, val in connection_dict.items():
        bfs([val['loc'][1], val['loc'][2]], seg_result[val['loc'][0]], val['generation'])
    return ret

def get_voxel_count_by_generation(seg_result: np.array, connection_dict: dict):
    voxel_by_generation = get_voxel_by_generation(seg_result)
    # generation higher than 16 is likely to be noise, ignore them
    ret = np.zeros(16, int)
    for i in range(16):
        ret[i] = (voxel_by_generation == i).astype(np.int32).sum()
    return ret

