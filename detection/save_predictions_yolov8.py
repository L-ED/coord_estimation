from ultralytics import YOLO
from glob import glob
from tqdm import tqdm
from ultralytics.utils import ops
import typing as tp
import torch


model_path = "/storage_labs/emergency-search/modeling/private/LyginE/v8m_1920crop_pt_vis_noheridal_mf1bf_sahi/weights/best.pt"
images_folder = "/storage_labs/3030/LyginE/emergency/private/images_test" 
results_folder = "/storage_labs/3030/LyginE/emergency/gitlab/res"
images = glob(f"{images_folder}/*")
device = "cuda"
iou_thres = 0.7
conf_thres = 0.6

model = YOLO(model_path)

def get_yolov8_predictions(model: YOLO, 
                           image_path: str, 
                           *args, **kwargs) -> torch.Tensor:
    
    """
    Returns model prediction
    Arguments: 
    model: ultralytics.YOLO, object used for predictions
    image_path: str, path to image to be forwarded
    Returns:
    results: torch.Tensor, (N, 4): normalized xywh values for each prediction
    
    """
    
    results = model(image_path, *args, **kwargs)[0]
    return results.boxes.xywhn 

for image_path in images:
    prediction = get_yolov8_predictions(model=model, 
                                        image_path=image_path, 
                                        iou=iou_thres, 
                                        conf=conf_thres, 
                                        device=device, 
                                        save=True, 
                                        project=results_folder,
                                        name='./',
                                        exist_ok=True,
                                        )
    

