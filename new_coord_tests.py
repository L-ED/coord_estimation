from zipfile import ZipFile
import xml
from bs4 import BeautifulSoup
import numpy as np

from osgeo import osr, gdal


import shutil, os, cv2, copy, gpxpy, torch
# from
import time
from tqdm import tqdm
from postprocessing import load_meta
from postprocessing import  calc_coords_rectified, calc_coords, fill_dict_nearest, fill_dict_nearest_SIFT, create_archive, Height_Estimator, calc_coords_rectified_interpolated, calc_coords_interpolated

from ultralytics import YOLO

from ultralytics.utils.plotting import save_one_box


def take_latest_folder(path):
    missions_dirs = [
            os.path.join(path, dir_) 
            for dir_ in os.listdir(path)
        ]
    return max(missions_dirs, key=os.path.getmtime)
    

def create_result_folder(parent):
    if os.path.exists(parent):
        folder_idx = len(
            [
                fold 
                for fold in os.listdir(parent) 
                if fold.startswith("result")
            ]
        ) +1
    else:
        folder_idx = 1
    return os.path.join(parent, f'result_{folder_idx}')


def create_gpx():
    gpx = gpxpy.gpx.GPX()
    # Create first track in our GPX:
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)
    # Create first segment in our GPX track:
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)
    return gpx


def add_gpx_point(gpx, meta):
    gpx.tracks[0].segments[0].points.append(
        gpxpy.gpx.GPXTrackPoint(
                latitude=meta['GPSLatitude'], 
                longitude=meta['GPSLongitude'], 
                elevation=meta['RelativeAltitude'], 
                time=meta['datetime']
            )
    )


def create_descriptor(box, img, feature_detector):
    crop = save_one_box(
                box.xyxy,
                img,
                BGR=False, # BGR=True orig img in bgr, leave it for grayscale 
                gain=1.5, 
                pad=10, 
                save=False,
    )

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    keyp, descr = feature_detector.detectAndCompute(gray,None)
    return descr


def prepare_folders(src_path, res_path):

    if src_path is None:
        # src_path = take_latest_folder()
        src_path = "/media/user/SD_Card/DCIM"
        print(f"Source path {src_path}")

    if res_path is None:
        res_path = create_result_folder(
            "/home/user/Emergency_Search/results"
        )
        print(f"Path to result folder {res_path}")

    det_path = os.path.join(res_path, 'detections')
    os.makedirs(det_path, exist_ok=True)

    return src_path, res_path, det_path


def take_start_point(mission_dir):
    kml_filename = None
    for file in os.listdir(mission_dir):
        if file.endswith('kmz'):
            kml_filename = file
    kmz_path = os.path.join(mission_dir, kml_filename)
    with ZipFile(kmz_path, 'r') as kmz:
        kml = kmz.open(kmz.filelist[0].filename, 'r').read()
    
    kml_parse = BeautifulSoup(kml, 'xml')
    lat,long, _ = [
        float(c) for c in 
        kml_parse.find(
            "wpml:takeOffRefPoint").text.split(',')
    ]
    planned_flight_height = float(kml_parse.find(
            "wpml:surfaceRelativeHeight").text)
    return (lat, long), planned_flight_height
    


def wait_for_dir(folder, line):
    while not os.path.exists(folder):
        print(line, end="\r")
        line += "." 
        time.sleep(1)


def take_mission_plan(parent_dir= "/run/user/1000/gvfs/mtp:host=DJI_DJI_RC_Pro_Enterprise_5YSZKB10020SML/Внутренний общий накопитель/DJI/Mission/KML", take_latest=False):

    print('mission_plan dir: ', parent_dir)
    wait_for_dir(parent_dir, "Waiting for Controller")

    if take_latest:
        latest_mission_dir = take_latest_folder(parent_dir)
    else:
        latest_mission_dir = parent_dir
    dsm_maps_dir = os.path.join(latest_mission_dir, "wpmz/res/dsm")
    dsm_maps_paths = [
        os.path.join(
            dsm_maps_dir,
            dsm_name
        ) for dsm_name in os.listdir(dsm_maps_dir)
    ]
    start_point, planned_height = take_start_point(latest_mission_dir) 

    return dsm_maps_paths, start_point, planned_height


def plot_coords(img, targets, coords, results_dir, filename):

    color=(255, 51, 153)#
    box_color = (0, 0, 255) 
    txt_color=(255, 255, 255) 

    for bbox, coord in zip(targets, coords):

        box = bbox.xyxy.squeeze()
        # label = f"lon: {np.round(coord[0], 6)},lat: {np.round(coord[1], 6)}"
        label_lon = f"longitude: {np.round(coord[0], 6)}"
        label_lat = f'latitude: {np.round(coord[1], 6)}'
        conf = bbox.conf[0].cpu().numpy()
        conf_label = f'conf: {np.round(conf, 3)}'
        
        """Add one xyxy box to image with label."""
        if isinstance(box, torch.Tensor):
            box = box.tolist()

        space = 15
        lw= 10
        imgh, imgw = img.shape[:2]

        min_point = (min(int(box[0]), int(box[2])), min(int(box[1]), int(box[3])))
        max_point = (max(int(box[0]), int(box[2])), max(int(box[1]), int(box[3])))

        cv2.rectangle(img, min_point, max_point, box_color, thickness=lw, lineType=cv2.LINE_AA)

        tf = max(lw - 5, 1)  # font thickness
        fs = lw /5
        w, h = cv2.getTextSize(label_lon, 0, fontScale=fs, thickness=tf)[0]  # text width, height
        
        x_offset = max(0, min_point[0]+w - imgw)
        y_offset = 0
        
        if min_point[1]-h*3 - space*5//2 - tf<0:
            y_offset = h*3+max_point[1]-min_point[1]+space*5//2+tf*2

        x_text_start = min_point[0] - x_offset
        y_text_start = min_point[1] + y_offset

        cv2.rectangle(
            img, 
            (x_text_start, y_text_start-tf),#p1, 
            (x_text_start+w, y_text_start-h*3-space*5//2-tf),#p2, 
            color, 
            -1, 
            cv2.LINE_AA
        )  # filled
        cv2.putText(img,
                    label_lat, 
                    (x_text_start, y_text_start-tf-space//2),#(p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    fs,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)

        cv2.putText(img,
                    label_lon, 
                    (x_text_start, y_text_start-h-space*2-tf),#(p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    fs,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        
        cv2.putText(img,
                    conf_label, 
                    (x_text_start, y_text_start-h*2-space*5//2-tf),#(p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    fs,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)
            
        
        img_path= os.path.join(results_dir, filename)
        cv2.imwrite(img_path,img)


def compare_closest(origs, comp_dict, preds, preds_inter, preds_rect, preds_rect_inter):
    for pred, pred_i, pred_r, pred_ri in zip(preds, preds_inter, preds_rect, preds_rect_inter):
        diff = origs-pred
        print("Distances: ", diff)
        print("Distances: ", )

        dists = np.sum(np.absolute(diff), axis=1)
        min_arg = np.argmin(dists, axis=0)
        # comp_dict
        closest = origs[min_arg]
        print('closest: ', closest)
        if abs(dists[min_arg]) < 0.001:
            comp_dict[tuple(closest.tolist())].append({
                'orig': dists[min_arg],
                'inter': np.sum(np.absolute(closest-pred_i)),
                'rect': np.sum(np.absolute(closest-pred_r)),
                'inter_rect': np.sum(np.absolute(closest-pred_ri)),
            })



def create_result(
        orig_coords,
        comparison_dict,
        src_path: str = None,
        res_path: str = None, #"/home/user/Emergency_Search/result"
        path_to_model_weights: str = "/home/user/Emergency_Search/weights/new/best_s.engine",
        iou_threshold = 0.2,
        conf_threshold = 0.2,
        use_descriptors = False,
        static_height = None,
        model=None
    ):

    s = time.time()

    src_path, res_path, det_path = prepare_folders(
        src_path, res_path
    )

    dsm_maps_paths, start_point, planned_heigth = take_mission_plan(
        os.path.join(src_path, 'mission_plan'))

    Height_estimator = Height_Estimator(dsm_maps_paths)
        
    gpx = create_gpx()

    if model is None:
        model = YOLO(path_to_model_weights, task='detect')
    device = "cuda"

    sift = cv2.SIFT_create()

    wait_for_dir(src_path, "Waiting for Copter")

    # src_path = take_latest_folder(src_path)

    # src_path, res_path, det_path = prepare_folders(
    #     src_path, res_path
    # )

    folder_content = sorted(os.listdir(src_path))
    targets_dict = {}

    ### Processing

    for filename in folder_content:
        if filename.endswith('JPG'):
            source_path = os.path.join(src_path, filename)
            meta = load_meta(
                source_path, Height_estimator, planned_heigth)

            add_gpx_point(gpx, meta)
            
            targets = model(
                source_path, 
                iou=iou_threshold, 
                conf=conf_threshold, 
                device=device, 
                save=False,#True, 
                project=det_path,#res_path,
                name='./',
                exist_ok=True,
                imgsz = (3968,5280),

            )

            if len(targets[0]) == 0:
                # os.remove(os.path.join(det_path, filename))
                continue


            descriptors = []
            for d in copy.deepcopy(targets[0].boxes.cpu()):

                if use_descriptors:
                    descr = create_descriptor(
                        d, targets[0].orig_img.copy(), sift
                    )
                else:
                    descr = []

                descriptors.append(descr)

            img = targets[0].orig_img
            # targets = targets[0].boxes.xywhn

            targets_coords = calc_coords(meta, targets[0].boxes.xywhn, Height_estimator)
            targets_coords_interpolated = calc_coords_interpolated(meta, targets[0].boxes.xywhn, Height_estimator)
            targets_coords_rectified = calc_coords_rectified(meta, targets[0].boxes.xywhn, Height_estimator)
            targets_coords_rectified_interpolated = calc_coords_rectified_interpolated(meta, targets[0].boxes.xywhn, Height_estimator)

            print('targets are', targets_coords)
            compare_closest(
                orig_coords, 
                comparison_dict,
                targets_coords,
                targets_coords_interpolated,
                targets_coords_rectified,
                targets_coords_rectified_interpolated
            )


            final_coords = {tuple(coord.tolist()): desc for coord, desc in zip(targets_coords, descriptors)}

            new_objs = fill_dict_nearest(final_coords, targets_dict, filename)

            new_targs = targets[0].boxes[new_objs]
            new_coords = targets_coords[new_objs]

            if len(new_targs)==0:
                continue

            plot_coords(img.copy(), new_targs, new_coords, det_path, filename)

    
    create_archive(det_path, targets_dict)

    gpx_path = os.path.join(res_path, 'flight_track.gpx')
    with open(gpx_path, 'w')as grpxf:
        grpxf.write(gpx.to_xml())
    
    e = time.time()
    print(f"Time elapsed: {e-s}")
    


if __name__=='__main__':
    # root='/home/user/Emergency_Search/tests/train2'

    # source_dirs = [
    #     os.path.join(root, elem)
    #     for elem in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, elem)) and elem.startswith('DJI')
    # ]

    source_dirs=[
        '/home/user/Emergency_Search/tests/coords_rect_check'
    ]

    print('Source: ', source_dirs)

    # source_dirs= [
    #     '/home/user/Emergency_Search/tests/train1/DJI_202309121727_022_Картография14_60metr',
    #     '/home/user/Emergency_Search/tests/train1/DJI_202309121727_024_Картография14_60metr',
    #     '/home/user/Emergency_Search/tests/train1/DJI_202309121727_025_Картография14_65metr',
    #     '/home/user/Emergency_Search/tests/train1/DJI_202309121808_026_Картография14_70metr',
    #     '/home/user/Emergency_Search/tests/train1/DJI_202309121808_027_Картография14_75metr',
    #     '/home/user/Emergency_Search/tests/train1/DJI_202309121836_029_Картография14time60',
    #     '/home/user/Emergency_Search/tests/train1/DJI_202309121808_028_Картография14time70metr',
    # ]

    res_dir='/home/user/Emergency_Search/tests/coords_rect_check/results'
    # res_dir='/home/user/Emergency_Search/tests/train2/results'

    subdirs=[
        os.path.basename(source)
        for source in source_dirs
    ]

    orig_coords = np.array(
        [
            [38.9888, 56.6563],
            [38.9879, 56.6551],
            [38.9859, 56.6542],
            [38.9894, 56.6528],
            [38.9914, 56.6528],
            [38.9958, 56.6526],
            [38.9881, 56.6623],
            [38.9865, 56.6608],
            [38.9905, 56.6611],
            [38.9901, 56.6597],
            [38.9904, 56.6588],

        ]
    )

    model_weights = [
        # "/home/user/Emergency_Search/weights/new/best_s.engine",
        # "/home/user/Emergency_Search/weights/new/best_m.engine",
        "/home/user/Emergency_Search/weights/old/final.engine",
        # "/home/user/Emergency_Search/weights/new/m65.engine"
    ]

    comparisons = {}
    for model_weight in model_weights:
        result_dir = os.path.join(
            res_dir,
            os.path.basename(model_weight)
        )


        model = YOLO(model_weight, task='detect')
        for src, res in zip(source_dirs, subdirs):

            comp_dict = {tuple(coord.tolist()):[] for coord in orig_coords}
            time.sleep(10)
            res_tree = os.path.join(
                result_dir, res
            )
            os.makedirs(res_tree, exist_ok=True)
            create_result(
                    orig_coords=orig_coords,
                    comparison_dict=comp_dict,
                    src_path=src,
                    res_path=res_tree,
                    # path_to_model_weights=model_weight,
                    model=model
                )
            comparisons[res]=comp_dict

    print(comparisons)
    import json
    with open('/home/user/Emergency_Search/tests/coords_rect_check/difs.json', 'w') as h:
        json.dump(comparisons, h)
        # time.sleep(60)
