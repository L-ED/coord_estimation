# competition_postprocessing



## Description

Model prediction postprocessing for Emergency Search competition 

## Usage

create_result in main.py accept: 
    path_to_source_folder: str, 
    path_to_model_weights: str, 
    path_to_result_folder: str

Result of program work should be .zip archive in same directory as path_to_result_folder
Content of .zip:
    Images with prediction rectangulars
    Json with objects global coords and images filenames with this object
