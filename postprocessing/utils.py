import os, shutil, json, cv2, time, copy
import numpy as np


def create_archive(path_to_result_folder, targets_dict):
    result_dict = {"Objects": []}
    for i, target in enumerate(targets_dict.values()):

        target.pop('description', None)
        target.pop('box', None)
        target.pop('best_img', None)
        target.pop('closure', None)
        target.pop('conf', None)

        # target['name'] ='object_'+str(i) 
        result_dict["Objects"].append(target)

    object_file = os.path.join(path_to_result_folder, "Objects.json")
    with open(object_file, 'w') as object_folder:
        json.dump(result_dict, object_folder)

    archive_path = os.path.join(
        os.path.dirname(path_to_result_folder), 
        'archive'
    )

    s = time.time()
    os.system(f"zip -j -0 {archive_path}.zip {path_to_result_folder}/*")
    # shutil.make_archive(archive_path, 'zip', path_to_result_folder)
    e = time.time()
    print(f"Time for ziping {e-s}")
    print("########### GENERATING SHA256SUM ##################")
    txt = os.path.join(os.path.dirname(path_to_result_folder), 'checksum.txt')
    os.system(f'sha256sum {archive_path}.zip > {txt}')


def haversine(p1, pointlist):

    R = 6378137

    delta_lat = (p1[1] - pointlist[:, 1])*np.pi/360 
    delta_lon = (p1[0] - pointlist[:, 0])*np.pi/360

    a = np.sin(delta_lat)**2 + np.cos(p1[1])*np.cos(pointlist[:, 1])*np.sin(delta_lon)**2
    c = 2*np.arcsin(a**0.5)
    return c*R



def fill_dict_nearest(targets_coords, targets_dict, filename): # Problem: first coordinate has higher prioruty in matchig

    new_objects = {}
    matched = set()
    is_new= []

    existing_coords = np.array([list(k) for k in targets_dict.keys()])

    for coord, pred_dict in targets_coords.items():
        
        coord_k = copy.deepcopy(coord)
        coord = np.array(list(coord))
        if len(existing_coords)>0:
            distances = haversine(coord, existing_coords)
            near_objs = existing_coords[distances<32] # empirical max error
            near_dists = distances[distances<32]
        else:
            near_objs = []

        if len(near_objs)==0:
            new_objects[tuple(coord)] = {
                "name": f"object_{len(targets_dict)+len(new_objects)+1}",
                "dd.dddddd_lat": str( # maybe agregate more all observations not first
                    np.round(coord[1], 6)
                ),
                "dd.dddddd_lon": str(
                    np.round(coord[0], 6)
                ),
                "images": [filename],
                "best_img": filename,
                "closure": pred_dict['closure'],
                "box": pred_dict['box'],
                'conf': pred_dict['conf']

            }

            # is_new.append(True)
        else:
            for near_coord in near_objs[near_dists.argsort()]: 
                if tuple(near_coord) in matched:
                    continue

                k = tuple(near_coord)
                candidate_dict=targets_dict[k]

                if candidate_dict['closure']>pred_dict['closure'] and candidate_dict['conf']<pred_dict['conf']:
                    obj_name = targets_dict.pop(k)['name']
                    # new_objects[coord_k] = {
                    targets_dict[coord_k] = {
                        "name": obj_name,
                        "dd.dddddd_lat": str( # maybe agregate more all observations not first
                            np.round(coord[1], 6)
                        ),
                        "dd.dddddd_lon": str(
                            np.round(coord[0], 6)
                        ),
                        "images": [filename],
                        "best_img": filename,
                        "closure": pred_dict['closure'],
                        "box": pred_dict['box'],
                        "conf": pred_dict['conf']
                    }

                    # matched.add(coord_k)
                    existing_coords = np.array([list(k) for k in targets_dict.keys()])
                # else:
                #     matched.add(k)
                break

            print("same coord")
            # is_new.append(False)

    targets_dict.update(new_objects)    
    # return is_new




def fill_dict_nearest_SIFT(targets_coords, targets_dict, filename): # Problem: first coordinate has higher prioruty in matchig

    matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

    new_objects = {}
    matched = set()

    existing_coords = np.array([list(k) for k in targets_dict.keys()])

    for coord, description in targets_coords.items():
        
        coord = np.array(list(coord))
        if len(existing_coords)>0:
            distances = haversine(coord, existing_coords)
            near_objs = existing_coords[distances<32] # empirical max error
            near_dists = distances[distances<32]
        else:
            near_objs = []

        if len(near_objs)==0:
            new_objects[tuple(coord)] = {
                "name": f"object_{len(targets_dict)+1}",
                "dd.dddddd_lat": str( # maybe agregate more all observations not first
                    np.round(coord[1], 6)
                ),
                "dd.dddddd_lon": str(
                    np.round(coord[0], 6)
                ),
                "images": [filename],
                "description": description
            }
        else:
            if len(near_objs)>1:
                best_score = 0
                best_coord = None
                # for near_coords in near_objs[distances.argsort()]:
                for near_coord, dist in zip(near_objs, near_dists):
                    k = tuple(near_coord)
                    
                    if near_coord in matched:
                        continue

                    knn_matches = matcher.knnMatch(
                        description, 
                        targets_dict[k]["description"], 
                        2
                    )

                    ratio_thresh = 0.7
                    # good_matches = []
                    good_matches_num = 0
                    for m,n in knn_matches:
                        if m.distance < ratio_thresh * n.distance:
                            # good_matches.append(m)
                            good_matches_num +=1
                    
                    score = (dist/20)*0.6 + (len(knn_matches)-good_matches_num/len(knn_matches))*0.4

                    if score < best_score:
                        best_coord = k
                        best_score = score

                matched.add(best_coord)
                targets_dict[best_coord]["images"].append(filename)
            
            else: 
                k = tuple(near_objs[0])
                matched.add(k)
                targets_dict[k]["images"].append(filename)
            


    targets_dict.update(new_objects) 