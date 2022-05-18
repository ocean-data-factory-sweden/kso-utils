# base imports
import argparse, os, ffmpeg, re
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
import torch
from ast import literal_eval
from pathlib import Path
from natsort import index_natsorted

# widget imports
from IPython.display import HTML, display, update_display, clear_output, Image
from ipywidgets import interact, Layout, Video
from PIL import Image as PILImage, ImageDraw
import ipywidgets as widgets

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
out_df = pd.DataFrame()


def generate_csv_report(evaluation_path):
    labels = os.listdir(Path(evaluation_path, "labels"))
    data_dict = {}
    for f in labels:
        frame_no = int(f.split("_")[-1].replace(".txt", ""))
        data_dict[f] = []
        with open(Path(evaluation_path, "labels", f), "r") as infile:
            lines = infile.readlines()
            for line in lines:
                class_id, x, y, w, h, conf = line.split(" ")
                data_dict[f].append([class_id, frame_no, x, y, w, h, float(conf)])
    dlist = [
            [key, i[0], i[1], i[2], i[3], i[4], i[5], i[6]] for key, value in data_dict.items() for i in value
            ]
    detect_df = pd.DataFrame.from_records(
                dlist, columns=["filename", "class_id", "frame_no", "x", "y", "w", "h", "conf"]
                )
    csv_out = Path(evaluation_path,"annotations.csv")
    detect_df.sort_values(
                    by="frame_no",
                    key=lambda x: np.argsort(index_natsorted(detect_df["filename"]))
                    ).to_csv(csv_out, index=False)
    print("Report created at {}".format(csv_out))
    return detect_df

def generate_tracking_report(tracker_dir, eval_dir):
    data_dict = {}
    for track_file in os.listdir(tracker_dir):
        if track_file.endswith(".txt"):
            data_dict[track_file] = []
            with open(Path(tracker_dir, track_file), "r") as infile:
                lines = infile.readlines()
                for line in lines:
                    vals = line.split(" ")
                    class_id, frame_no, tracker_id = vals[0], vals[1], vals[2]
                    data_dict[track_file].append([class_id, frame_no, tracker_id])
    dlist = [
            [os.path.splitext(key)[0]+f"_{i[1]}.txt", i[0], i[1], i[2]] for key, value in data_dict.items() for i in value
            ]
    detect_df = pd.DataFrame.from_records(
                dlist, columns=["filename", "class_id", "frame_no", "tracker_id"]
                )
    csv_out = Path(eval_dir,"tracking.csv")
    detect_df.sort_values(
                    by="frame_no",
                    key=lambda x: np.argsort(index_natsorted(detect_df["filename"]))
                    ).to_csv(csv_out, index=False)
    print("Report created at {}".format(csv_out))
    return detect_df

def generate_counts(eval_dir, tracker_dir, model_dir):
    model = torch.load(model_dir+"/best.pt")
    names = {i: model["model"].names[i] for i in range(len(model["model"].names))}
    class_df = generate_csv_report(eval_dir)
    tracker_df = generate_tracking_report(tracker_dir, eval_dir)
    tracker_df["frame_no"] = tracker_df["frame_no"].astype(int)
    combined_df = pd.merge(class_df, tracker_df, on=["filename", "frame_no", "class_id"])
    combined_df["species_name"] = combined_df["class_id"].apply(lambda x: names[int(x)])
    print("--- DETECTION REPORT ---")
    print("--------------------------------")
    print(combined_df.groupby(["species_name"])["tracker_id"].nunique())

def track_objects(source_dir, artifact_dir, conf_thres=0.5, img_size=720):
    # Enter the correct folder
    try:
        os.chdir("Yolov5_DeepSort_OSNet")
    except:
        pass
    best_model = artifact_dir+"/best.pt"
    try:
        subprocess.check_output([f'python {os.getcwd()}/track.py --conf-thres {str(conf_thres)}  \
                                 --save-txt --save-vid --yolo_model {best_model} --source "{source_dir}" \
                                 --imgsz {str(img_size)}'],
                     shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    # Go up one directory
    if "Yolov5_DeepSort_OSNet" in os.getcwd():
        os.chdir("..")
    tracker_root = "Yolov5_DeepSort_OSNet/runs/track/"
    latest_tracker = tracker_root + sorted(os.listdir(tracker_root))[-1] + "/tracks"
    print("Tracking completed succesfully")
    return latest_tracker

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
    return os.path.realpath(artifact_dir)


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
