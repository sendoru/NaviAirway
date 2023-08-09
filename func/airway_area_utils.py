from .detect_tree import *
from collections import deque
from skimage.measure import label

def bfs(nodes, matrix, unvisited_node_no=-1):
    # do BFS on 3D matrix
    # unvisited node == -1, unreachable node == very big value
    ret = matrix.copy()
    directions = [
        (0, 0, -1),
        (0, 0, 1),
        (0, -1, 0),
        (0, 1, 0),
        (-1, 0, 0),
        (1, 0, 0)
    ]

    q = deque()
    for node in nodes:
        q.append(node)
        if ret[node[0]][node[1]][node[2]] == unvisited_node_no:
            ret[node[0]][node[1]][node[2]] = node[3]

    while len(q) > 0:
        x_cur, y_cur, z_cur, gen_no = q[0]
        q.popleft()
        for i in range(len(directions)):
            x_next = x_cur + directions[i][0]
            y_next = y_cur + directions[i][1]
            z_next = z_cur + directions[i][2]
            if x_next < 0 or x_next >= ret.shape[0] or \
                y_next < 0 or y_next >= ret.shape[1] or \
                z_next < 0 or z_next >= ret.shape[2] or \
                ret[x_next][y_next][z_next] != unvisited_node_no:
                    continue
            ret[x_next][y_next][z_next] = gen_no
            q.append((x_next, y_next, z_next, gen_no))

    return ret

def is_upside_down(onehot:np.ndarray):
    onehot_copied = onehot.copy()
    comp_counts = []
    for i in range(len(onehot_copied)):
        _, comp_count = label(onehot_copied[i], return_num=True, connectivity=1)
        if comp_count > 0:
            comp_counts.append(comp_count * comp_count / onehot_copied[i].sum())

    comp_counts = np.array(comp_counts)
    forward = np.dot(comp_counts, np.linspace(1, len(comp_counts), len(comp_counts)))
    backward = np.dot(comp_counts, np.linspace(len(comp_counts), 1, len(comp_counts)))
    return backward < forward

def get_only_largest_component(slice: np.ndarray):
    slice_label, comp_count = label(slice, return_num=True, connectivity=1)
    comp_size = np.zeros(comp_count + 1)
    for i in range(1, comp_count + 1):
        comp_size[i] = (slice_label == i).astype(int).sum()
    dtype = slice.dtype
    slice_label = (slice_label == (comp_size.argmax())).astype(dtype)
    return slice_label

def get_voxel_by_generation(seg_result: np.ndarray, connection_dict: dict, max_valid_gen=15):
    ret = seg_result.astype(np.int32) - 2
    nodes = []
    for key, val in connection_dict.items():
        if val['generation'] <= max_valid_gen:
            gen = val['generation']
        else:
            gen = max_valid_gen
        nodes.append((*val['loc'], gen))
    nodes.sort(key=lambda x:x[3])
    ret = bfs(nodes, ret)
    return ret

def get_voxel_by_generation_without_bfs(seg_result: np.ndarray, connection_dict: dict, max_valid_gen=15):
    ret = np.zeros_like(seg_result) - 1
    voxel_coords = np.argwhere(seg_result == 1)
    min_dist = np.ones((len(voxel_coords), max_valid_gen + 1)) * 1e6
    for key, val in connection_dict.items():
        if val['generation'] <= max_valid_gen:
            gen = val['generation']
        else:
            gen = max_valid_gen
        min_dist[:,gen] = np.minimum(min_dist[:,gen], np.linalg.norm(np.array(val['loc'], dtype=float) - voxel_coords, axis=1))
    gen_no = np.argmin(min_dist, axis=1)
    for idx, voxel_coord in enumerate(voxel_coords):
        ret[voxel_coord[0]][voxel_coord[1]][voxel_coord[2]] = gen_no[idx]
    
    return ret

def get_voxel_count_by_generation(voxel_by_generation: np.array, connection_dict: dict, max_valid_gen=15):
    # generation higher than 16 is likely to be noise, ignore them
    ret = np.zeros(max_valid_gen + 1, int)
    for i in range(max_valid_gen + 1):
        ret[i] = (voxel_by_generation == i).astype(np.int32).sum()
    return ret

