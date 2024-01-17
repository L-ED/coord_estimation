
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
from postprocessing import  calc_coords, fill_dict_nearest, fill_dict_nearest_SIFT, create_archive, Height_Estimator

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
    planned_flight_height = float(kml_parse.find(
            "wpml:surfaceRelativeHeight").text)
    return planned_flight_height
    

def wait_for_dir(folder, line):
    while not os.path.exists(folder):
        print(line, end="\r")
        line += "." 
        time.sleep(1)


def take_mission_plan(parent_dir= "/run/user/1000/gvfs/mtp:host=DJI_DJI_RC_Pro_Enterprise_5YSZKB10020SML/Внутренний общий накопитель/DJI/Mission/KML"):

    wait_for_dir(parent_dir, "Waiting for Controller")

    latest_mission_dir = take_latest_folder(parent_dir)
    print('Latest mission is: ', latest_mission_dir)
    dsm_maps_dir = os.path.join(latest_mission_dir, "wpmz/res/dsm")
    dsm_maps_paths = [
        os.path.join(
            dsm_maps_dir,
            dsm_name
        ) for dsm_name in os.listdir(dsm_maps_dir)
    ]
    # planned_height, start_point = take_start_point(latest_mission_dir) 
    planned_height = take_start_point(latest_mission_dir) 

    # return dsm_maps_paths, start_point, planned_height
    return dsm_maps_paths, planned_height


def plot_coords(imgs_dict, source_dir, results_dir):

    color=(255, 51, 153)#
    box_color = (0, 0, 255) 
    txt_color=(255, 255, 255) 

    for filename, objs_dicts in imgs_dict.items():
        img = cv2.imread(
            os.path.join(source_dir, filename)
        )

        for obj_dict in objs_dicts:

            box = obj_dict['box'].xyxy.squeeze()
            label_name = obj_dict['obj_name'] + " conf:{:0.3f}".format(obj_dict['conf'])
            label_lon = f"longitude: {obj_dict['longitude']}"
            label_lat = f"latitude: {obj_dict['latitude']}"
            
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
            
            if min_point[1]-h*3 - space*4 - tf<0:
                y_offset = h*3+max_point[1]-min_point[1]+space*4+tf*2

            x_text_start = min_point[0] - x_offset
            y_text_start = min_point[1] + y_offset

            cv2.rectangle(
                img, 
                (x_text_start, y_text_start-tf),#p1, 
                (x_text_start+w, y_text_start-h*3-space*4-tf),#p2, 
                color, -1, cv2.LINE_AA
            )  # filled
            cv2.putText(img,
                        label_name, 
                        (x_text_start, y_text_start-h*2-tf-space*7//2),#(p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,fs,txt_color,thickness=tf,lineType=cv2.LINE_AA)
            
            cv2.putText(img,
                        label_lat, 
                        (x_text_start, y_text_start-h-space*2-tf),#(p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,fs,txt_color,thickness=tf,lineType=cv2.LINE_AA)

            cv2.putText(img,
                        label_lon, 
                        (x_text_start, y_text_start-tf-space//2),#(p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,fs,txt_color,thickness=tf,lineType=cv2.LINE_AA)
            
            img_path= os.path.join(results_dir, filename)
            cv2.imwrite(img_path,img)




def create_imgs_dict(targs_dict):
    imgs_dict = {}

    for coord, obj_dict in targs_dict.items():
        img_filename = obj_dict['best_img']
        if img_filename in imgs_dict:
            imgs_dict[img_filename].append(
                {
                    'obj_name':obj_dict['name'],
                    'box':obj_dict['box'],
                    'longitude':obj_dict['dd.dddddd_lon'],
                    'latitude':obj_dict['dd.dddddd_lat'],
                    'conf':np.round(obj_dict['conf'], 3)
                }
            )
        else:
            imgs_dict[img_filename]=[
                {
                    'obj_name':obj_dict['name'],
                    'box':obj_dict['box'],
                    'longitude':obj_dict['dd.dddddd_lon'],
                    'latitude':obj_dict['dd.dddddd_lat'],
                    'conf':np.round(obj_dict['conf'], 3)
                }
            ]
        
    return imgs_dict


def filter_small(targets, meta):
    height=meta['Aprox_H']

    min_area = 2.5*(-height*72 + 7200)/4
    min_side = 2.5*(-height*0.6 + 78.9)/3

    correct_results=[]
    for (_, _, w, h) in targets.boxes.xywh:
        area = w*h
        side = min(w, h)
        if area<=min_area or side<=min_side:
            correct_results.append(False)
        else:
            correct_results.append(True)
    
    targets = targets[correct_results]
    return targets


def create_result(
        src_path: str = None,
        res_path: str = None, #"/home/user/Emergency_Search/result"
        path_to_model_weights: str = "/home/user/Emergency_Search/weights/old/final.engine",
        iou_threshold = 0.2,
        conf_threshold = 0.45,
        use_descriptors = False,
        static_height = None
    ):

    s = time.time()

    src_path, res_path, det_path = prepare_folders(
        src_path, res_path
    )

    dsm_maps_paths, planned_heigth = take_mission_plan()
    # dsm_maps_paths, start_point, planned_heigth = take_mission_plan()

    Height_estimator = Height_Estimator(dsm_maps_paths)
        
    gpx = create_gpx()

    model = YOLO(path_to_model_weights, task='detect')
    device = "cuda"

    sift = cv2.SIFT_create()

    wait_for_dir(src_path, "Waiting for Copter")

    src_path = take_latest_folder(src_path)

    print('Latest mission shots: ', src_path)
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
                max_det=4

            )[0]

            if len(targets) == 0:
                # os.remove(os.path.join(det_path, filename))
                continue

            targets = filter_small(targets, meta)
            
            descriptors = []
            for d in copy.deepcopy(targets.boxes.cpu()):

                if use_descriptors:
                    descr = create_descriptor(
                        d, targets.orig_img.copy(), sift
                    )
                else:
                    descr = []

                descriptors.append(descr)

            img = targets.orig_img
            # targets = targets.boxes.xywhn

            # targets_coords = calc_coords(meta, targets.boxes.xywhn, Height_estimator)
            targets_coords, closures = calc_coords(
                meta, targets.boxes.xywhn, Height_estimator)

            final_coords = {
                tuple(coord): {'descriptor':desc, 'closure': closure, 'box':box, 'conf':box.conf[0].cpu().numpy()} 
                for coord, desc, closure, box in zip(targets_coords, descriptors, closures, targets.boxes)
            }

            fill_dict_nearest(final_coords, targets_dict, filename)

    imgs_dict = create_imgs_dict(targets_dict)
    plot_coords(imgs_dict, src_path, det_path)

    
    create_archive(det_path, targets_dict)

    gpx_path = os.path.join(res_path, 'flight_track.gpx')
    with open(gpx_path, 'w')as grpxf:
        grpxf.write(gpx.to_xml())
    
    e = time.time()
    print(f"Time elapsed: {e-s}")
    
create_result(
    # src_path="/media/user/SD_Card/DCIM/DJI_202308171809_016_Картография12"
    # '/home/user/Desktop/calibration/DJI_202308051312_008'
    )