# tree detection V3
import gc
from typing import Optional
import numpy as np
from skimage.morphology import skeletonize_3d
from skimage.measure import label as label_regions
import heapq

def tree_detection(seg_onehot, search_range = 2, branch_penalty=16., pixdim_info=None, need_skeletonize_3d=True):
    center_map, center_dict, nearby_dict = get_the_skeleton_and_center_nearby_dict(seg_onehot, search_range=search_range, need_skeletonize_3d=need_skeletonize_3d)
    connection_dict = get_connection_dict(center_dict, nearby_dict, pixdim_info, branch_penalty)
    number_of_branch = get_number_of_branch(connection_dict)
    tree_length = get_tree_length(connection_dict, is_3d_len=True)
    
    return center_map, connection_dict, number_of_branch, tree_length

def get_crop(input_3d_img, search_center, search_range=1):
    shape_of_input_3d_img = input_3d_img.shape
    
    x = search_center[0]
    y = search_center[1]
    z = search_center[2]
    
    x_s = np.clip(x-search_range, 0, shape_of_input_3d_img[0])
    x_e = np.clip(x+search_range+1, 0, shape_of_input_3d_img[0])
    y_s = np.clip(y-search_range, 0, shape_of_input_3d_img[1])
    y_e = np.clip(y+search_range+1, 0, shape_of_input_3d_img[1])
    z_s = np.clip(z-search_range, 0, shape_of_input_3d_img[2])
    z_e = np.clip(z+search_range+1, 0, shape_of_input_3d_img[2])
    
    #print("crop: x from "+str(x_s)+" to "+str(x_e)+"; y from "+str(y_s)+" to "+str(y_e)+"; z from "+str(z_s)+" to "+str(z_e), end="\r")
    
    crop_3d_img = input_3d_img[x_s:x_e,y_s:y_e,z_s:z_e]
    
    return crop_3d_img

# step 1 get the skeleton
def get_the_skeleton_and_center_nearby_dict(seg_input, search_range = 10, need_skeletonize_3d=True):
    if need_skeletonize_3d:
        center_map = np.array(skeletonize_3d(seg_input)>0, dtype=np.int)
    else:
        center_map = seg_input
    
    center_dict = {}
    nearby_dict = {}
    
    center_locs = np.where(center_map>0)
    base_count = 1
    for i in range(len(center_locs[0])):
        center_dict[i+base_count]=[center_locs[0][i],center_locs[1][i],center_locs[2][i]]
        center_map[center_locs[0][i],center_locs[1][i],center_locs[2][i]] = i+base_count
    
    for i in center_dict.keys():
        center_map_crop = get_crop(center_map, center_dict[i], search_range)
        crop_img_vals = np.unique(center_map_crop).astype(np.int)
        crop_img_vals = crop_img_vals[crop_img_vals!=0]
        crop_img_vals = crop_img_vals[crop_img_vals!=i]
        nearby_dict[i] = crop_img_vals
    
    return center_map, center_dict, nearby_dict

# step 2 get_connection_dict
def get_connection_dict(center_dict, nearby_dict, pixdim_info=None, branch_penalty=16.):
    slice_idxs = list(center_dict.keys())
    slice_idxs.reverse()

    if pixdim_info is None:
        pixdim = np.array([1., 1., 1.])
    else:
        pixdim = np.array([pixdim_info['pixdim_z'], pixdim_info['pixdim_x'], pixdim_info['pixdim_y']])
    
    # init connection dict
    connection_dict = {}
    for slice_idx in slice_idxs:
        connection_dict[slice_idx] = {}
        connection_dict[slice_idx]["loc"] = center_dict[slice_idx]
        connection_dict[slice_idx]["before"] = []
        connection_dict[slice_idx]["next"] = []
        connection_dict[slice_idx]["is_bifurcation"] = False
        connection_dict[slice_idx]["number_of_next"] = 0
        connection_dict[slice_idx]["generation"] = 0
        connection_dict[slice_idx]["segment_no"] = 0
        connection_dict[slice_idx]["is_processed"] = False
    
    def find_connection(current_label, before_label):
        nonlocal connection_dict
        
        nearby_labels = nearby_dict[current_label]
        valid_next_labels = []
        dist_to_valid_labels = []
        processed_count = 0
        for nearby_label in nearby_labels:
            if connection_dict[nearby_label]["is_processed"]==False:
                valid_next_labels.append(nearby_label)
                dist_to_valid_labels.append(np.sum(((np.array(connection_dict[nearby_label]["loc"])-np.array(connection_dict[current_label]["loc"]))**2)))
            else:
                processed_count+=1
        
        if len(valid_next_labels)>0:
            valid_next_labels = np.array(valid_next_labels)
            dist_to_valid_labels = np.array(dist_to_valid_labels)
            sort_locs = np.argsort(dist_to_valid_labels)
            valid_next_labels = valid_next_labels[sort_locs]
        
        connection_dict[current_label]["before"].append(before_label)
        connection_dict[current_label]["is_processed"] = True

        print("current_label is "+str(current_label), end="\r")
        print(connection_dict[current_label], end="\r")

        if len(valid_next_labels)==0 or len(nearby_labels)==processed_count:
            return connection_dict
        else:
            for valid_next_label in valid_next_labels:
                if connection_dict[valid_next_label]["is_processed"]==False:
                    connection_dict[current_label]["next"].append(valid_next_label)
                    find_connection(valid_next_label, current_label)
    
    def find_connection_using_regularized_prim(start_label, null_label=0):
        nonlocal connection_dict
        pq = []

        # (cost of edge, outdegree of before_label, before_label, current_label)
        pq.append((-10000, 0, null_label, start_label))
        while len(pq) > 0:
            _, outdegree, before_label, current_label= pq[0]
            heapq.heappop(pq)

            # if the outdegree info retrieved from pq is different from true outdegree from conneciton_dict,
            # it should be discarded
            if before_label != null_label:
                true_outdegree = len(connection_dict[before_label]["next"])
                if true_outdegree > 2:
                    true_outdegree = 2
                if true_outdegree != outdegree:
                    continue

            if connection_dict[current_label]["is_processed"]:
                continue
            if before_label != null_label:
                connection_dict[before_label]["next"].append(current_label)
            connection_dict[current_label]["before"].append(before_label)
            connection_dict[current_label]["is_processed"] = True

            # if outdegree of before_label changes like (0 -> 1) or (1 -> 2),
            # edges start from before_label should be pushed to pq again
            if before_label != null_label and len(connection_dict[before_label]["next"]) <= 2:
                nearby_labels = nearby_dict[before_label]
                for nearby_label in nearby_labels:
                    if connection_dict[nearby_label]["is_processed"] == False:
                        valid_next_labels.append(nearby_label)
                        dir_vec = (np.array(connection_dict[nearby_label]["loc"]) - np.array(connection_dict[before_label]["loc"])) * pixdim
                        cost = np.sum((dir_vec ** 2))
                        # + 1 because we appended connection_dict[before_label]["next"]
                        # if true_outdegree + 1 >= 2:
                        #     cost -= branch_penalty
                        if true_outdegree + 1 == 1:
                            cost += branch_penalty

                        heapq.heappush(pq, (cost, true_outdegree + 1, before_label, nearby_label))

            nearby_labels = nearby_dict[current_label]
            valid_next_labels = []
            dist_to_valid_labels = []
            processed_count = 0
            for nearby_label in nearby_labels:
                if connection_dict[nearby_label]["is_processed"] == False:
                    valid_next_labels.append(nearby_label)
                    cost = np.sum(((np.array(connection_dict[nearby_label]["loc"])-np.array(connection_dict[current_label]["loc"]))**2))

                    heapq.heappush(pq, (cost, 0, current_label, nearby_label))
                else:
                    processed_count+=1
    
    find_connection_using_regularized_prim(start_label=slice_idxs[0], null_label=0)
    
    for item in connection_dict.keys():
        assert len(connection_dict[item]["before"])<=1, (item, connection_dict[item])
        connection_dict[item]["number_of_next"] = len(connection_dict[item]["next"])
        connection_dict[item]["is_bifurcation"] = (connection_dict[item]["number_of_next"]>1)
        #if connection_dict[item]["is_bifurcation"]:
        #    print(connection_dict[item])

    segment_no = 1
    def find_generation_and_node_no(current_label, generation):
        nonlocal connection_dict
        nonlocal segment_no
        connection_dict[current_label]["generation"] = generation
        connection_dict[current_label]["segment_no"] = segment_no
        if connection_dict[current_label]["number_of_next"]>0:
            for next_label in connection_dict[current_label]["next"]:
                if connection_dict[current_label]["is_bifurcation"]:
                    segment_no += 1
                    find_generation_and_node_no(next_label, generation+1)
                else:
                    find_generation_and_node_no(next_label, generation)
        else:
            return connection_dict
    
    find_generation_and_node_no(current_label=slice_idxs[0], generation=1)

    gc.collect()
    
    return connection_dict

def get_number_of_branch(connection_dict):
    number_of_branch = 1
    for label in connection_dict.keys():
        if connection_dict[label]["is_bifurcation"]:
            number_of_branch+=connection_dict[label]["number_of_next"]
    return number_of_branch

def get_tree_length(connection_dict, is_3d_len=True):
    global tree_length
    tree_length = 0
    for label in connection_dict.keys():
        if connection_dict[label]["before"][0]==0:
            start_label = label
            break
    def get_tree_length_func(connection_dict, current_label):
        global tree_length
        if connection_dict[current_label]["number_of_next"]==0:
            return
        else:
            current_branch_length = 0
            for next_label in connection_dict[current_label]["next"]:
                if is_3d_len:
                    current_branch_length += np.sqrt(np.sum((np.array(connection_dict[current_label]["loc"])-np.array(connection_dict[next_label]["loc"]))**2))
                else:
                    current_branch_length += 1
            print("len of "+str(current_label)+" branch is "+str(current_branch_length),end="\r")
            tree_length += current_branch_length
            for next_label in connection_dict[current_label]["next"]:
                get_tree_length_func(connection_dict, next_label)
    get_tree_length_func(connection_dict, start_label)
    return tree_length

# 1. get segment connection info (adj. list + segment length)
def get_segment_dict(connection_dict: dict, pixdim_info=None) -> dict:
    segment_dict = {}
    for key, val in connection_dict.items():
        if not val["segment_no"] in segment_dict.keys():
            segment_dict[val["segment_no"]] = {
                    "points": [],
                    "before": 0,
                    "next": [],
                    "length": 0.,
                    "pruned": False,
                    "generation": 0,
                    "endpoint": 0,
                    "endpoint_loc": [0, 0, 0]
                }
        segment_dict[val["segment_no"]]["points"].append(key)
        if val["number_of_next"] != 1:
            segment_dict[val["segment_no"]]["endpoint_loc"] = val["loc"].copy()
            segment_dict[val["segment_no"]]["endpoint"] = key
    if pixdim_info is None:
        pixdim = np.array([1., 1., 1.])
    else:
        pixdim = np.array([pixdim_info['pixdim_z'], pixdim_info['pixdim_x'], pixdim_info['pixdim_y']])

    def dfs(current_label: int) -> None:
        nonlocal connection_dict, segment_dict
        current_segment_no = connection_dict[current_label]["segment_no"]
        segment_dict[current_segment_no]["generation"] = connection_dict[current_label]["generation"]
        if connection_dict[current_label]["number_of_next"] > 0:
            for next_label in connection_dict[current_label]["next"]:
                next_segment_no = connection_dict[next_label]["segment_no"]
                dir_vec = np.array(connection_dict[current_label]["loc"]) - np.array(connection_dict[next_label]["loc"]) * pixdim
                segment_dict[next_segment_no]["length"] += np.sqrt(np.sum((dir_vec)**2))
                if connection_dict[current_label]["is_bifurcation"]:
                    segment_dict[current_segment_no]["next"].append(next_segment_no)
                    segment_dict[next_segment_no]["before"] = current_segment_no
                    dfs(next_label)
                else:
                    dfs(next_label)

    dfs(list(connection_dict.keys())[0])
    return segment_dict

# 2. get sum of weights (lenght of segments) of subtree for each node using tree DP
# This function inplaces given variable
def get_subtree_length(segment_dict: dict) -> dict:
    for key in segment_dict.keys():
        segment_dict[key]["subtree_length"] = 0.

    def tree_dp(current_segment: int) -> float:
        nonlocal segment_dict
        for next_segment in segment_dict[current_segment]["next"]:
            segment_dict[current_segment]["subtree_length"] += tree_dp(next_segment)
        segment_dict[current_segment]["subtree_length"] += segment_dict[current_segment]["length"]
        return segment_dict[current_segment]["subtree_length"]
    
    tree_dp(list(segment_dict.keys())[0])

    return segment_dict

# 3. if a subtree of a segment has too low weight (compared to the sum of weights of its siblings)
# remove the segment and its subtree
def prune_segment(segment_dict: dict, th_ratio: float = 0.05, max_gen: Optional[int] = None) -> dict:
    def prune_subtree(current_segment):
        nonlocal segment_dict
        if segment_dict[current_segment]["pruned"]:
            return
        segment_dict[current_segment]["pruned"] = True
        for next_segment in segment_dict[current_segment]["next"]:
            prune_subtree(next_segment)

    for segment, val in segment_dict.items():
        # root segment
        if val["before"] == 0:
            continue
        parent = val["before"]
        siblings_subtree_length_sum = segment_dict[parent]["subtree_length"] - segment_dict[parent]["length"]
        if segment_dict[segment]["subtree_length"] < th_ratio * siblings_subtree_length_sum:
            prune_subtree(segment)

    if max_gen is not None:
        for segment, val in segment_dict.items():
            if segment_dict[segment]["generation"] > max_gen:
                segment_dict[segment]["pruned"] = True

    return segment_dict

# 4. reconstruct tree using pruned segment info
def prune_conneciton_dict(connection_dict: dict, th_ratio: float = 0.05, max_gen: Optional[int] = None):
    segment_dict = get_segment_dict(connection_dict)
    segment_dict = get_subtree_length(segment_dict)
    segment_dict = prune_segment(segment_dict, th_ratio=th_ratio, max_gen=max_gen)

    for current_label, val in connection_dict.items():
        next_labels_list = []
        for next_label in val["next"]:
            next_segment = connection_dict[next_label]["segment_no"]
            if not segment_dict[next_segment]["pruned"]:
                next_labels_list.append(next_label)

        connection_dict[current_label]["next"] = next_labels_list
        connection_dict[current_label]["number_of_next"] = len(connection_dict[current_label]["next"])
        connection_dict[current_label]["is_bifurcation"] = connection_dict[current_label]["number_of_next"] > 1

    new_connection_dict = {}
    keys = connection_dict.keys()
    for current_label in keys:
        current_segment = connection_dict[current_label]["segment_no"]
        if not segment_dict[current_segment]["pruned"]:
            new_connection_dict[current_label] = connection_dict[current_label]

    connection_dict = new_connection_dict

    segment_no = 1
    call_cnt = 0
    def find_generation_and_node_no(current_label, generation):
        nonlocal segment_no, call_cnt
        call_cnt += 1
        if (call_cnt % 10000) == 0:
            print(call_cnt)
        nonlocal connection_dict
        nonlocal segment_no
        connection_dict[current_label]["generation"] = generation
        connection_dict[current_label]["segment_no"] = segment_no
        if connection_dict[current_label]["number_of_next"]>0:
            for next_label in connection_dict[current_label]["next"]:
                if connection_dict[current_label]["is_bifurcation"]:
                    segment_no += 1
                    find_generation_and_node_no(next_label, generation+1)
                else:
                    find_generation_and_node_no(next_label, generation)
        else:
            return connection_dict
    
    find_generation_and_node_no(current_label=list(connection_dict.keys())[0], generation=1)
    gc.collect()
    return connection_dict

def get_trace_voxel_count_by_gen_from_root(segment_dict: dict, voxel_count_by_segment_no: np.ndarray,max_vaild_gen=10) -> dict:
    
    vol = np.zeros(max_vaild_gen + 1, dtype=float)
    def dfs(cur_segment: int, prev_vol: np.ndarray, cur_depth=1) -> None:
        nonlocal segment_dict
        cur_vol = prev_vol.copy()
        cur_vol[cur_depth] += voxel_count_by_segment_no[cur_segment]
        segment_dict[cur_segment]["trace_voxel_count_by_gen"] = cur_vol
        if cur_depth >= max_vaild_gen:
            return
        for next_segment in segment_dict[cur_segment]["next"]:
            dfs(next_segment, cur_vol, cur_depth + 1)

    for key in segment_dict.keys():
        segment_dict[key]["trace_voxel_count_by_gen"] = vol.copy()
        
    dfs(list(segment_dict.keys())[0], vol)
    return segment_dict



# # tree detection V2
# import edt
# #from sklearn.cluster import DBSCAN
# import numpy as np
# from skimage.measure import label as label_regions 
# from skimage.feature import peak_local_max

# def tree_detection(seg_onehot, axis=0):
#     seg_slice_label, center_dict, touching_dict = label_each_slice_and_get_center_of_each_slice(seg_onehot, axis=axis)
#     connection_dict = get_connection_dict(seg_slice_label, center_dict, touching_dict)
#     number_of_branch = get_number_of_branch(connection_dict)
#     tree_length = get_tree_length(connection_dict, is_3d_len=True)
    
#     return seg_slice_label, connection_dict, number_of_branch, tree_length

# def get_distance_transform(img_oneshot):
#     return edt.edt(np.array(img_oneshot>0, dtype=np.uint32, order='F'), black_border=True, order='F', parallel=1)

# """
# def get_outlayer_of_a_3d_shape(a_3d_shape_onehot):
#     shape=a_3d_shape_onehot.shape
    
#     a_3d_crop_diff_x1 = a_3d_shape_onehot[0:shape[0]-1,:,:]-a_3d_shape_onehot[1:shape[0],:,:]
#     a_3d_crop_diff_x2 = -a_3d_shape_onehot[0:shape[0]-1,:,:]+a_3d_shape_onehot[1:shape[0],:,:]
#     a_3d_crop_diff_y1 = a_3d_shape_onehot[:,0:shape[1]-1,:]-a_3d_shape_onehot[:,1:shape[1],:]
#     a_3d_crop_diff_y2 = -a_3d_shape_onehot[:,0:shape[1]-1,:]+a_3d_shape_onehot[:,1:shape[1],:]
#     a_3d_crop_diff_z1 = a_3d_shape_onehot[:,:,0:shape[2]-1]-a_3d_shape_onehot[:,:,1:shape[2]]
#     a_3d_crop_diff_z2 = -a_3d_shape_onehot[:,:,0:shape[2]-1]+a_3d_shape_onehot[:,:,1:shape[2]]

#     outlayer = np.zeros(shape)
#     outlayer[1:shape[0],:,:] += np.array(a_3d_crop_diff_x1==1, dtype=np.int8)
#     outlayer[0:shape[0]-1,:,:] += np.array(a_3d_crop_diff_x2==1, dtype=np.int8)
#     outlayer[:,1:shape[1],:] += np.array(a_3d_crop_diff_y1==1, dtype=np.int8)
#     outlayer[:,0:shape[1]-1,:] += np.array(a_3d_crop_diff_y2==1, dtype=np.int8)
#     outlayer[:,:,1:shape[2]] += np.array(a_3d_crop_diff_z1==1, dtype=np.int8)
#     outlayer[:,:,0:shape[2]-1] += np.array(a_3d_crop_diff_z2==1, dtype=np.int8)
    
#     outlayer = np.array(outlayer>0, dtype=np.int8)
    
#     return outlayer
#     """

# def get_crop_by_pixel_val(input_3d_img, val, boundary_extend=1, axis=0):
#     locs = np.where(input_3d_img==val)
    
#     shape_of_input_3d_img = input_3d_img.shape
    
#     min_x = np.min(locs[0])
#     max_x =np.max(locs[0])
#     min_y = np.min(locs[1])
#     max_y =np.max(locs[1])
#     min_z = np.min(locs[2])
#     max_z =np.max(locs[2])
    
#     if axis==0:
#         x_s = np.clip(min_x-boundary_extend, 0, shape_of_input_3d_img[0])
#         x_e = np.clip(max_x+boundary_extend+1, 0, shape_of_input_3d_img[0])
#         y_s = np.clip(min_y, 0, shape_of_input_3d_img[1])
#         y_e = np.clip(max_y+1, 0, shape_of_input_3d_img[1])
#         z_s = np.clip(min_z, 0, shape_of_input_3d_img[2])
#         z_e = np.clip(max_z+1, 0, shape_of_input_3d_img[2])
    
#     #print("crop: x from "+str(x_s)+" to "+str(x_e)+"; y from "+str(y_s)+" to "+str(y_e)+"; z from "+str(z_s)+" to "+str(z_e), end="\r")
    
#     crop_3d_img = input_3d_img[x_s:x_e,y_s:y_e,z_s:z_e]
    
#     return crop_3d_img

# """
# def find_cluster_by_dbscan(img_slice, base_count, threshold=0, dbscan_min_samples=1, dbscan_eps=1):
#     locs=np.where(img_slice>0)
#     locs_x=locs[0]
#     locs_y=locs[1]
#     locs_len=locs[0].shape[0]
#     locs_reshape=np.concatenate((locs[0].reshape(locs_len,1),
#                                  locs[1].reshape(locs_len,1)),axis=1)
    
#     clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric='euclidean').fit(locs_reshape)
#     clustering_labels=clustering.labels_
#     clustering_labels_unique,clustering_labels_counts=np.unique(clustering_labels, return_counts=True)
    
#     #clustering_labels_counts=clustering_labels_counts[clustering_labels_unique>-1]
#     #clustering_labels_unique=clustering_labels_unique[clustering_labels_unique>-1] # delete noise
#     clustering_labels_unique=clustering_labels_unique+1
    
#     clustering_labels_unique=clustering_labels_unique[np.where(clustering_labels_counts>threshold)]
#     clustering_labels_counts=clustering_labels_counts[np.where(clustering_labels_counts>threshold)]
    
#     seg_img_slice=np.zeros(img_slice.shape)
    
#     center_dict = {}
    
#     for i in range(0, len(clustering_labels_unique)):
#         temp_label=clustering_labels_unique[i]
#         temp_label_locs=np.where(clustering_labels==temp_label-1)
#         seg_img_slice[locs_x[temp_label_locs],locs_y[temp_label_locs]]= \
#         clustering_labels_unique[i]+base_count
        
#         seg_img_slice_temp = np.zeros(seg_img_slice.shape)
#         seg_img_slice_temp[seg_img_slice==clustering_labels_unique[i]+base_count]=1        
#         seg_img_slice_edt_temp = get_distance_transform(seg_img_slice_temp)
#         center_loc = np.where(seg_img_slice_edt_temp==np.max(seg_img_slice_edt_temp))
#         center_loc = [center_loc[0][0], center_loc[1][0]]
#         center_dict[clustering_labels_unique[i]+base_count] = center_loc
    
#     return seg_img_slice, center_dict, clustering_labels_unique, clustering_labels_counts
#     """
# def find_cluster(img_slice, base_count):
#     clustering_labels = label_regions(img_slice, connectivity=1)
#     clustering_labels[clustering_labels>0]+=base_count
    
#     img_slice_edt = get_distance_transform(img_slice)
    
#     clustering_labels_unique, clustering_labels_counts = np.unique(clustering_labels, return_counts=True)
#     clustering_labels_counts = clustering_labels_counts[clustering_labels_unique>0]
#     clustering_labels_unique = clustering_labels_unique[clustering_labels_unique>0]
    
#     center_dict = {}
    
#     for i in range(0, len(clustering_labels_unique)):
#         seg_img_slice_edt_temp = np.zeros(img_slice_edt.shape)
#         seg_img_slice_edt_temp[clustering_labels==clustering_labels_unique[i]]=img_slice_edt[clustering_labels==clustering_labels_unique[i]]
#         center_loc = np.where(seg_img_slice_edt_temp==np.max(seg_img_slice_edt_temp))
#         center_loc = [center_loc[0][0], center_loc[1][0]]
#         center_dict[clustering_labels_unique[i]] = center_loc
    
#     return clustering_labels, center_dict, clustering_labels_unique, clustering_labels_counts


# # step 1 label each slice and get centers of each slice
# def label_each_slice_and_get_center_of_each_slice(seg_onehot, axis=0):
#     base_count = 1
#     center_dict = {}
#     seg_slice_label = np.zeros(seg_onehot.shape)
    
#     touching_dict = {}
    
#     if axis==0:
#         for i in np.arange(seg_onehot.shape[0]):
#             print("processing slice: "+str(i), end="\r")
#             current_slice = seg_onehot[i,:,:]
#             if np.sum(current_slice)>0:
#                 current_slice_labeled, center_dict_slice, clustering_labels_unique, clustering_labels_counts = \
#                 find_cluster(current_slice, base_count=base_count)
#                 for label in center_dict_slice.keys():
#                     center_dict_slice[label] = [i, center_dict_slice[label][0], center_dict_slice[label][1]]
#                 center_dict.update(center_dict_slice)
#                 seg_slice_label[i,:,:] = current_slice_labeled
#                 base_count += len(clustering_labels_unique)
                
#                 if i>0:
#                     for label in center_dict_slice.keys():
#                         crop_img = get_crop_by_pixel_val(seg_slice_label[i-1:i+1,:,:], label, boundary_extend=1)
#                         crop_img_vals = np.unique(crop_img).astype(np.int)
#                         crop_img_vals = crop_img_vals[crop_img_vals!=0]
#                         crop_img_vals = crop_img_vals[crop_img_vals!=label]
                        
#                         if label not in touching_dict.keys():
#                             touching_dict[label]=[]
#                         for crop_img_val in crop_img_vals:
#                             touching_dict[label].append(crop_img_val)
#                         for crop_img_val in crop_img_vals:
#                             if crop_img_val not in touching_dict.keys():
#                                 touching_dict[crop_img_val]=[]
#                             touching_dict[crop_img_val].append(label)
                            
#             else:
#                 seg_slice_label[i,:,:] = 0
        
#         for label in center_dict.keys():
#             if label not in touching_dict.keys():
#                 crop_img = get_crop_by_pixel_val(seg_slice_label, label, boundary_extend=1)
#                 crop_img_vals = np.unique(crop_img).astype(np.int)
#                 crop_img_vals = crop_img_vals[crop_img_vals!=0]
#                 crop_img_vals = crop_img_vals[crop_img_vals!=label]
#                 if len(crop_img_vals)==0:
#                     touching_dict[label] = []
#                 else:
#                     touching_dict[label] = crop_img_vals
                
#             #print("clustering_labels_unique, clustering_labels_counts: "+str((clustering_labels_unique, clustering_labels_counts)))
#             #print("center_dict_slice: "+str(center_dict_slice))
#             #print("base count: "+str(base_count))
#             #print("----------")
    
#         for label in touching_dict.keys():
#             touching_dict[label] = np.array(touching_dict[label])
#             touching_dict[label] = np.unique(touching_dict[label])
#     """
#     for idx,label in enumerate(center_dict.keys()):
#         print("find touching relationship of each slice: "+str(idx/len(center_dict.keys())), end="\r")
#         crop_img = get_crop_by_pixel_val(seg_slice_label, label, boundary_extend=1)
#         crop_img_vals = np.unique(crop_img).astype(np.int)
#         crop_img_vals = crop_img_vals[crop_img_vals!=0]
#         crop_img_vals = crop_img_vals[crop_img_vals!=label]
#         touching_dict[label] = crop_img_vals
#     """
#     return seg_slice_label, center_dict, touching_dict

# # step 2 get_connection_dict
# def get_connection_dict(seg_slice_label, center_dict, touching_dict=None):
#     slice_idxs = list(center_dict.keys())
#     slice_idxs.reverse()
    
#     # init connection dict
#     global connection_dict
#     connection_dict = {}
#     for slice_idx in slice_idxs:
#         connection_dict[slice_idx] = {}
#         connection_dict[slice_idx]["loc"] = center_dict[slice_idx]
#         connection_dict[slice_idx]["before"] = []
#         connection_dict[slice_idx]["next"] = []
#         connection_dict[slice_idx]["is_bifurcation"] = False
#         connection_dict[slice_idx]["number_of_next"] = 0
#         connection_dict[slice_idx]["generation"] = 0
#         connection_dict[slice_idx]["is_processed"] = False
    
#     def get_touching_labels(input_img, val):
#         if touching_dict is not None:
#             return touching_dict[val]
#         else:
#             crop_img = get_crop_by_pixel_val(input_img, val, boundary_extend=1)
#             crop_img_vals = np.unique(crop_img).astype(np.int)
#             crop_img_vals = crop_img_vals[crop_img_vals!=0]
#             crop_img_vals = crop_img_vals[crop_img_vals!=val]

#             return crop_img_vals
    
#     def find_connection(slice_label_img, current_label, before_label):
#         global connection_dict
        
#         touching_labels = get_touching_labels(slice_label_img, current_label)
#         valid_next_labels = []
#         dist_to_valid_labels = []
#         processed_count = 0
#         for touching_label in touching_labels:
#             if connection_dict[touching_label]["is_processed"]==False \
#             and connection_dict[touching_label]["loc"][0]!=connection_dict[current_label]["loc"][0]:
#                 valid_next_labels.append(touching_label)
#                 dist_to_valid_labels.append(np.sum(((np.array(connection_dict[touching_label]["loc"])-np.array(connection_dict[current_label]["loc"]))**2)))
#             if connection_dict[touching_label]["is_processed"]==True:
#                 processed_count+=1
        
#         if len(valid_next_labels)>0:
#             valid_next_labels = np.array(valid_next_labels)
#             dist_to_valid_labels = np.array(dist_to_valid_labels)
#             sort_locs = np.argsort(dist_to_valid_labels)
#             valid_next_labels = valid_next_labels[sort_locs]
        
#         connection_dict[current_label]["before"].append(before_label)
#         connection_dict[current_label]["is_processed"] = True
#         #connection_dict[current_label]["generation"] = generation

#         print("current_label is "+str(current_label), end="\r")
#         print(connection_dict[current_label], end="\r")

#         if len(valid_next_labels)==0 or len(touching_labels)==processed_count:
#             return connection_dict
#         else:
#             for valid_next_label in valid_next_labels:
#                 if connection_dict[valid_next_label]["is_processed"]==False:
#                     connection_dict[current_label]["next"].append(valid_next_label)
#                     find_connection(slice_label_img, valid_next_label, current_label)
    
#     find_connection(seg_slice_label, current_label=slice_idxs[0], before_label=0)
    
#     for item in connection_dict.keys():
#         assert len(connection_dict[item]["before"])<=1, (item, connection_dict[item])
#         connection_dict[item]["number_of_next"] = len(connection_dict[item]["next"])
#         connection_dict[item]["is_bifurcation"] = (connection_dict[item]["number_of_next"]>1)
#         #if connection_dict[item]["is_bifurcation"]:
#         #    print(connection_dict[item])
            
#     def find_generation(current_label, generation):
#         global connection_dict
#         connection_dict[current_label]["generation"]=generation
#         if connection_dict[current_label]["number_of_next"]>0:
#             for next_label in connection_dict[current_label]["next"]:
#                 if connection_dict[current_label]["is_bifurcation"]:
#                     find_generation(next_label, generation+1)
#                 else:
#                     find_generation(next_label, generation)
#         else:
#             return connection_dict
    
#     find_generation(current_label=slice_idxs[0], generation=0)
    
#     return connection_dict

# def get_number_of_branch(connection_dict):
#     number_of_branch = 1
#     for label in connection_dict.keys():
#         if connection_dict[label]["is_bifurcation"]:
#             number_of_branch+=connection_dict[label]["number_of_next"]
#     return number_of_branch

# def get_tree_length(connection_dict, is_3d_len=True):
#     global tree_length
#     tree_length = 0
#     for label in connection_dict.keys():
#         if connection_dict[label]["before"][0]==0:
#             start_label = label
#             break
#     def get_tree_length_func(connection_dict, current_label):
#         global tree_length
#         if connection_dict[current_label]["number_of_next"]==0:
#             return
#         else:
#             current_branch_length = 0
#             for next_label in connection_dict[current_label]["next"]:
#                 if is_3d_len:
#                     current_branch_length += np.sqrt(np.sum((np.array(connection_dict[current_label]["loc"])-np.array(connection_dict[next_label]["loc"]))**2))
#                 else:
#                     current_branch_length += 1
#             print("len of "+str(current_label)+" branch is "+str(current_branch_length),end="\r")
#             tree_length += current_branch_length
#             for next_label in connection_dict[current_label]["next"]:
#                 get_tree_length_func(connection_dict, next_label)
#     get_tree_length_func(connection_dict, start_label)
#     return tree_length

# # tree detection V1

# import edt
# import numpy as np
# from skimage.measure import label, regionprops
# from skimage.feature import peak_local_max
# #from sklearn.cluster import DBSCAN

# def tree_detection(seg_onehot, axis=0):
#     seg_slice_label, center_dict, touching_dict = label_each_slice_and_get_center_of_each_slice(seg_onehot, axis=axis)
#     connection_dict = get_connection_dict(seg_slice_label, center_dict, touching_dict)
#     number_of_branch = get_number_of_branch(connection_dict)
#     tree_length = get_tree_length(connection_dict, is_3d_len=True)
    
#     return seg_slice_label, connection_dict, number_of_branch, tree_length

# def get_distance_transform(img_oneshot):
#     return edt.edt(np.array(img_oneshot>0, dtype=np.uint32, order='F'), black_border=True, order='F', parallel=1)

# """
# def get_outlayer_of_a_3d_shape(a_3d_shape_onehot):
#     shape=a_3d_shape_onehot.shape
    
#     a_3d_crop_diff_x1 = a_3d_shape_onehot[0:shape[0]-1,:,:]-a_3d_shape_onehot[1:shape[0],:,:]
#     a_3d_crop_diff_x2 = -a_3d_shape_onehot[0:shape[0]-1,:,:]+a_3d_shape_onehot[1:shape[0],:,:]
#     a_3d_crop_diff_y1 = a_3d_shape_onehot[:,0:shape[1]-1,:]-a_3d_shape_onehot[:,1:shape[1],:]
#     a_3d_crop_diff_y2 = -a_3d_shape_onehot[:,0:shape[1]-1,:]+a_3d_shape_onehot[:,1:shape[1],:]
#     a_3d_crop_diff_z1 = a_3d_shape_onehot[:,:,0:shape[2]-1]-a_3d_shape_onehot[:,:,1:shape[2]]
#     a_3d_crop_diff_z2 = -a_3d_shape_onehot[:,:,0:shape[2]-1]+a_3d_shape_onehot[:,:,1:shape[2]]

#     outlayer = np.zeros(shape)
#     outlayer[1:shape[0],:,:] += np.array(a_3d_crop_diff_x1==1, dtype=np.int8)
#     outlayer[0:shape[0]-1,:,:] += np.array(a_3d_crop_diff_x2==1, dtype=np.int8)
#     outlayer[:,1:shape[1],:] += np.array(a_3d_crop_diff_y1==1, dtype=np.int8)
#     outlayer[:,0:shape[1]-1,:] += np.array(a_3d_crop_diff_y2==1, dtype=np.int8)
#     outlayer[:,:,1:shape[2]] += np.array(a_3d_crop_diff_z1==1, dtype=np.int8)
#     outlayer[:,:,0:shape[2]-1] += np.array(a_3d_crop_diff_z2==1, dtype=np.int8)
    
#     outlayer = np.array(outlayer>0, dtype=np.int8)
    
#     return outlayer
#     """

# def get_crop_by_pixel_val(input_3d_img, val, boundary_extend=1, axis=0):
#     locs = np.where(input_3d_img==val)
    
#     shape_of_input_3d_img = input_3d_img.shape
    
#     min_x = np.min(locs[0])
#     max_x =np.max(locs[0])
#     min_y = np.min(locs[1])
#     max_y =np.max(locs[1])
#     min_z = np.min(locs[2])
#     max_z =np.max(locs[2])
    
#     if axis==0:
#         x_s = np.clip(min_x-boundary_extend, 0, shape_of_input_3d_img[0])
#         x_e = np.clip(max_x+boundary_extend+1, 0, shape_of_input_3d_img[0])
#         y_s = np.clip(min_y, 0, shape_of_input_3d_img[1])
#         y_e = np.clip(max_y+1, 0, shape_of_input_3d_img[1])
#         z_s = np.clip(min_z, 0, shape_of_input_3d_img[2])
#         z_e = np.clip(max_z+1, 0, shape_of_input_3d_img[2])
    
#     #print("crop: x from "+str(x_s)+" to "+str(x_e)+"; y from "+str(y_s)+" to "+str(y_e)+"; z from "+str(z_s)+" to "+str(z_e), end="\r")
    
#     crop_3d_img = input_3d_img[x_s:x_e,y_s:y_e,z_s:z_e]
    
#     return crop_3d_img

# """
# def find_cluster_by_dbscan(img_slice, base_count, threshold=0, dbscan_min_samples=1, dbscan_eps=1):
#     locs=np.where(img_slice>0)
#     locs_x=locs[0]
#     locs_y=locs[1]
#     locs_len=locs[0].shape[0]
#     locs_reshape=np.concatenate((locs[0].reshape(locs_len,1),
#                                  locs[1].reshape(locs_len,1)),axis=1)
    
#     clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric='euclidean').fit(locs_reshape)
#     clustering_labels=clustering.labels_
#     clustering_labels_unique,clustering_labels_counts=np.unique(clustering_labels, return_counts=True)
    
#     #clustering_labels_counts=clustering_labels_counts[clustering_labels_unique>-1]
#     #clustering_labels_unique=clustering_labels_unique[clustering_labels_unique>-1] # delete noise
#     clustering_labels_unique=clustering_labels_unique+1
    
#     clustering_labels_unique=clustering_labels_unique[np.where(clustering_labels_counts>threshold)]
#     clustering_labels_counts=clustering_labels_counts[np.where(clustering_labels_counts>threshold)]
    
#     seg_img_slice=np.zeros(img_slice.shape)
    
#     center_dict = {}
    
#     for i in range(0, len(clustering_labels_unique)):
#         temp_label=clustering_labels_unique[i]
#         temp_label_locs=np.where(clustering_labels==temp_label-1)
#         seg_img_slice[locs_x[temp_label_locs],locs_y[temp_label_locs]]= \
#         clustering_labels_unique[i]+base_count
        
#         seg_img_slice_temp = np.zeros(seg_img_slice.shape)
#         seg_img_slice_temp[seg_img_slice==clustering_labels_unique[i]+base_count]=1        
#         seg_img_slice_edt_temp = get_distance_transform(seg_img_slice_temp)
#         center_loc = np.where(seg_img_slice_edt_temp==np.max(seg_img_slice_edt_temp))
#         center_loc = [center_loc[0][0], center_loc[1][0]]
#         center_dict[clustering_labels_unique[i]+base_count] = center_loc
    
#     return seg_img_slice, center_dict, clustering_labels_unique, clustering_labels_counts
#     """
# def find_cluster(img_slice, base_count):
#     clustering_labels = label(img_slice, connectivity=1)
#     clustering_labels[clustering_labels>0]+=base_count
    
#     img_slice_edt = get_distance_transform(img_slice)
    
#     clustering_labels_unique, clustering_labels_counts = np.unique(clustering_labels, return_counts=True)
#     clustering_labels_counts = clustering_labels_counts[clustering_labels_unique>0]
#     clustering_labels_unique = clustering_labels_unique[clustering_labels_unique>0]
    
#     center_dict = {}
    
#     for i in range(0, len(clustering_labels_unique)):
#         seg_img_slice_edt_temp = np.zeros(img_slice_edt.shape)
#         seg_img_slice_edt_temp[clustering_labels==clustering_labels_unique[i]]=img_slice_edt[clustering_labels==clustering_labels_unique[i]]
#         center_loc = np.where(seg_img_slice_edt_temp==np.max(seg_img_slice_edt_temp))
#         center_loc = [center_loc[0][0], center_loc[1][0]]
#         center_dict[clustering_labels_unique[i]] = center_loc
    
#     return clustering_labels, center_dict, clustering_labels_unique, clustering_labels_counts


# # step 1 label each slice and get centers of each slice
# def label_each_slice_and_get_center_of_each_slice(seg_onehot, axis=0):
#     base_count = 1
#     center_dict = {}
#     seg_slice_label = np.zeros(seg_onehot.shape)
    
#     touching_dict = {}
    
#     if axis==0:
#         for i in np.arange(seg_onehot.shape[0]):
#             print("processing slice: "+str(i), end="\r")
#             current_slice = seg_onehot[i,:,:]
#             if np.sum(current_slice)>0:
#                 current_slice_labeled, center_dict_slice, clustering_labels_unique, clustering_labels_counts = \
#                 find_cluster(current_slice, base_count=base_count)
#                 for label in center_dict_slice.keys():
#                     center_dict_slice[label] = [i, center_dict_slice[label][0], center_dict_slice[label][1]]
#                 center_dict.update(center_dict_slice)
#                 seg_slice_label[i,:,:] = current_slice_labeled
#                 base_count += len(clustering_labels_unique)
                
#                 if i>0:
#                     for label in center_dict_slice.keys():
#                         crop_img = get_crop_by_pixel_val(seg_slice_label[i-1:i+1,:,:], label, boundary_extend=1)
#                         crop_img_vals = np.unique(crop_img).astype(np.int)
#                         crop_img_vals = crop_img_vals[crop_img_vals!=0]
#                         crop_img_vals = crop_img_vals[crop_img_vals!=label]
                        
#                         if label not in touching_dict.keys():
#                             touching_dict[label]=[]
#                         for crop_img_val in crop_img_vals:
#                             touching_dict[label].append(crop_img_val)
#                         for crop_img_val in crop_img_vals:
#                             if crop_img_val not in touching_dict.keys():
#                                 touching_dict[crop_img_val]=[]
#                             touching_dict[crop_img_val].append(label)
                            
#             else:
#                 seg_slice_label[i,:,:] = 0
        
#         for label in center_dict.keys():
#             if label not in touching_dict.keys():
#                 crop_img = get_crop_by_pixel_val(seg_slice_label, label, boundary_extend=1)
#                 crop_img_vals = np.unique(crop_img).astype(np.int)
#                 crop_img_vals = crop_img_vals[crop_img_vals!=0]
#                 crop_img_vals = crop_img_vals[crop_img_vals!=label]
#                 if len(crop_img_vals)==0:
#                     touching_dict[label] = []
#                 else:
#                     touching_dict[label] = crop_img_vals
                
#             #print("clustering_labels_unique, clustering_labels_counts: "+str((clustering_labels_unique, clustering_labels_counts)))
#             #print("center_dict_slice: "+str(center_dict_slice))
#             #print("base count: "+str(base_count))
#             #print("----------")
    
#         for label in touching_dict.keys():
#             touching_dict[label] = np.array(touching_dict[label])
#             touching_dict[label] = np.unique(touching_dict[label])
#     """
#     for idx,label in enumerate(center_dict.keys()):
#         print("find touching relationship of each slice: "+str(idx/len(center_dict.keys())), end="\r")
#         crop_img = get_crop_by_pixel_val(seg_slice_label, label, boundary_extend=1)
#         crop_img_vals = np.unique(crop_img).astype(np.int)
#         crop_img_vals = crop_img_vals[crop_img_vals!=0]
#         crop_img_vals = crop_img_vals[crop_img_vals!=label]
#         touching_dict[label] = crop_img_vals
#     """
#     return seg_slice_label, center_dict, touching_dict

# # step 2 get_connection_dict
# def get_connection_dict(seg_slice_label, center_dict, touching_dict=None):
#     slice_idxs = list(center_dict.keys())
#     slice_idxs.reverse()
    
#     # init connection dict
#     global connection_dict
#     connection_dict = {}
#     for slice_idx in slice_idxs:
#         connection_dict[slice_idx] = {}
#         connection_dict[slice_idx]["loc"] = center_dict[slice_idx]
#         connection_dict[slice_idx]["before"] = 0
#         connection_dict[slice_idx]["next"] = 0
#         connection_dict[slice_idx]["is_bifurcation"] = False
#         connection_dict[slice_idx]["number_of_next"] = 0
#         connection_dict[slice_idx]["generation"] = 0
#         connection_dict[slice_idx]["is_processed"] = False
    
#     def get_touching_labels(input_img, val):
#         if touching_dict is not None:
#             return touching_dict[val]
#         else:
#             crop_img = get_crop_by_pixel_val(input_img, val, boundary_extend=1)
#             crop_img_vals = np.unique(crop_img).astype(np.int)
#             crop_img_vals = crop_img_vals[crop_img_vals!=0]
#             crop_img_vals = crop_img_vals[crop_img_vals!=val]

#             return crop_img_vals
    
#     def find_connection(slice_label_img, current_label, before_label, generation):
#         global connection_dict
        
#         touching_labels = get_touching_labels(slice_label_img, current_label)
#         valid_next_labels = []
#         processed_count = 0
#         for touching_label in touching_labels:
#             if connection_dict[touching_label]["is_processed"]==False \
#             and connection_dict[touching_label]["loc"][0]!=connection_dict[current_label]["loc"][0]:
#                 valid_next_labels.append(touching_label)
#             if connection_dict[touching_label]["is_processed"]==True:
#                 processed_count+=1
        
#         connection_dict[current_label]["before"] = before_label
#         connection_dict[current_label]["generation"] = generation
#         connection_dict[current_label]["is_processed"] = True
#         connection_dict[current_label]["next"] = valid_next_labels
#         connection_dict[current_label]["is_bifurcation"] = (len(valid_next_labels)>=2)
#         connection_dict[current_label]["number_of_next"] = len(valid_next_labels)

#         print("current_label is "+str(current_label), end="\r")
#         print(connection_dict[current_label], end="\r")

#         if connection_dict[current_label]["number_of_next"]==0 or len(touching_labels)==processed_count:
#             return connection_dict
#         else:
#             for valid_next_label in valid_next_labels:
#                 if connection_dict[current_label]["is_bifurcation"]:
#                     find_connection(slice_label_img, valid_next_label, current_label, generation+1)
#                 else:
#                     find_connection(slice_label_img, valid_next_label, current_label, generation)
    
#     find_connection(seg_slice_label, current_label=slice_idxs[0], before_label=0, generation=0)
    
#     return connection_dict

# def get_number_of_branch(connection_dict):
#     number_of_branch = 1
#     for label in connection_dict.keys():
#         if connection_dict[label]["is_bifurcation"]:
#             number_of_branch+=connection_dict[label]["number_of_next"]
#     return number_of_branch

# def get_tree_length(connection_dict, is_3d_len=True):
#     global tree_length
#     tree_length = 0
#     for label in connection_dict.keys():
#         if connection_dict[label]["before"]==0:
#             start_label = label
#             break
#     def get_tree_length_func(connection_dict, current_label):
#         global tree_length
#         if connection_dict[current_label]["number_of_next"]==0:
#             return
#         else:
#             current_branch_length = 0
#             for next_label in connection_dict[current_label]["next"]:
#                 if is_3d_len:
#                     current_branch_length += np.sqrt(np.sum((np.array(connection_dict[current_label]["loc"])-np.array(connection_dict[next_label]["loc"]))**2))
#                 else:
#                     current_branch_length += 1
#             print("len of "+str(current_label)+" branch is "+str(current_branch_length),end="\r")
#             tree_length += current_branch_length
#             for next_label in connection_dict[current_label]["next"]:
#                 get_tree_length_func(connection_dict, next_label)
#     get_tree_length_func(connection_dict, start_label)
#     return tree_length