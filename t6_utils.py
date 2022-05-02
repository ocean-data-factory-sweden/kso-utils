# base imports
import argparse, os, ffmpeg
import kso_utils.db_utils as db_utils
import pandas as pd
import numpy as np
import math
import subprocess
import shutil
import logging
import pims
import cv2
import difflib
import wandb
import pprint
from ast import literal_eval

# widget imports
from IPython.display import HTML, display, update_display, clear_output, Image
from ipywidgets import interact, Layout, Video
from PIL import Image as PILImage, ImageDraw
import ipywidgets as widgets

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
out_df = pd.DataFrame()


def get_data_viewer(data_path):
    imgs = list(filter(lambda fn:fn.lower().endswith('.jpg'), os.listdir(data_path)))
    def loadimg(k):
        display(draw_box(os.path.join(data_path,imgs[k])))
    return widgets.interact(loadimg ,k=(0,len(imgs)-1))


def draw_box(path):
    im = PILImage.open(path)
    d = { line.split()[0] : line.split()[1:] for line in open(
        path.replace("images", "labels").replace(".jpg", ".txt")) }
    dw, dh = im._size
    img1 = ImageDraw.Draw(im)
    for i, vals in d.items():
        vals = tuple(float(val) for val in vals)
        vals_adjusted = tuple([int((vals[0]-vals[2]/2)*dw), int((vals[1]-vals[3]/2)*dh),
                               int((vals[0]+vals[2]/2)*dw), int((vals[1]+vals[3]/2)*dh)])
        img1.rectangle(vals_adjusted, outline="red", width=2)
    return im


def get_dataset(project_name, model):
    api = wandb.Api()
    run_id = model.split("_")[1]
    run = api.run(f'koster/{project_name.lower()}/runs/{run_id}')
    datasets = [artifact for artifact in run.used_artifacts() if artifact.type == 'dataset']
    if len(datasets) == 0:
        logging.error("No datasets are linked to these runs. Please try another run.")
        return None, None
    dirs = []
    for i in range(len(["train", "val"])):
        artifact = datasets[i]
        print(f"Downloading {artifact.name} checkpoint...")
        artifact_dir = artifact.download()
        print(f"{artifact.name} - Dataset downloaded.")
        dirs.append(artifact_dir)
    return dirs


def get_model(model_name, project_name):
    api = wandb.Api()
    collections = [
        coll for coll in api.artifact_type(type_name='model', project=f'koster/{project_name.lower()}').collections()
    ]
    model = [i for i in collections if i.name == model_name]
    if len(model) == 1:
        model = model[0]
    else:
        print("No model found")
    artifact = api.artifact(f"koster/{project_name.lower()}/" + model.name + ":latest")
    print("Downloading model checkpoint...")
    artifact_dir = artifact.download()
    print("Checkpoint downloaded.")
    return artifact_dir


# Function to compare original to modified frames
def choose_model(project_name):
    model_dict = {}
    model_info = {}
    api = wandb.Api()
    # weird error fix
    api.runs(path=f'koster/{project_name.lower()}')
    for edge, obj in zip(api.runs(path=f'koster/{project_name.lower()}').last_response["project"]["runs"]["edges"],
                    api.runs(path=f'koster/{project_name.lower()}').objects):
        model_dict[edge["node"]["displayName"]] = "run_"+obj.id+"_model"
        model_info["run_"+obj.id+"_model"] = literal_eval(edge["node"]["summaryMetrics"])
    # Add "no movie" option to prevent conflicts
    #models = np.append(list(model_dict.keys()),"No model")
    
    model_widget = widgets.Dropdown(
                    options=[(name, model)  for name, model in model_dict.items()],
                    description="Select model:",
                    ensure_option=False,
                    disabled=False,
                    layout=Layout(width='50%'),
                    style = {'description_width': 'initial'}
                )
    
    main_out = widgets.Output()
    display(model_widget, main_out)
    
    # Display the original and modified clips
    def on_change(change):
        with main_out:
            clear_output()
            if change["new"]=="No file":
                print("Choose another file")
            else:
                print({k:v for k,v in model_info[change["new"]].items() if "metrics" in k})
                #display(a)
                   
    model_widget.observe(on_change, names='value')
    
    return model_widget


# Function to compare original to modified frames
def choose_files(path):
    
    # Add "no movie" option to prevent conflicts
    files = np.append([path+i for i in os.listdir(path)],"No file")
    
    clip_path_widget = widgets.Dropdown(
                    options=tuple(np.sort(files)),
                    description="Select file:",
                    ensure_option=True,
                    disabled=False,
                    layout=Layout(width='50%'),
                    style = {'description_width': 'initial'}
                )
    
    main_out = widgets.Output()
    display(clip_path_widget, main_out)
    
    # Display the original and modified clips
    def on_change(change):
        with main_out:
            clear_output()
            if change["new"]=="No file":
                print("Choose another file")
            else:
                a = view_file(change["new"])
                display(a)
                   
    clip_path_widget.observe(on_change, names='value')

# Display the frames using html
def view_file(path):
    # Get path of the modified clip selected
    extension = os.path.splitext(path)[1]
    file = open(path, "rb").read()
    if extension.lower() in [".jpeg", ".png", ".jpg"]:
        widget = widgets.Image(value=file, format=extension)
    elif extension.lower() in [".mp4", ".mov", ".avi"]:
        if os.path.exists('linked_content'):
            shutil.rmtree('linked_content')
        os.mkdir('linked_content')
        os.symlink(path, 'linked_content/'+os.path.basename(path))
        widget = HTML(f"""
                    <video width=800 height=400 alt="test" controls>
                        <source src="linked_content/{os.path.basename(path)}" type="video/{extension.lower().replace(".", "")}">
                    </video>
                """)
    else:
        print("File format not supported. Supported formats: jpeg, png, jpg, mp4, mov, avi.")

    return widget
