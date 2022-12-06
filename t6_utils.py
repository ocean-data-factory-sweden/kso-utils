# base imports
import os
import pandas as pd
import numpy as np
import subprocess
import tempfile
import shutil
import logging
import wandb
import torch
import base64
from ast import literal_eval
from pathlib import Path
from natsort import index_natsorted

# widget imports
from IPython.display import HTML, display, clear_output
from ipywidgets import Layout
from PIL import Image as PILImage, ImageDraw
import ipywidgets as widgets
from jupyter_bbox_widget import BBoxWidget
import imagesize

import yolov5_tracker.track as track
from yolov5.utils import torch_utils

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def set_config(conf_thres: float, model: str, eval_dir: str):
    config = wandb.config
    config.confidence_threshold = conf_thres
    config.model_name = model
    config.evaluation_directory = eval_dir
    return config


def add_data_wandb(path: str, name: str, run):
    my_data = wandb.Artifact(name, type="raw_data")
    my_data.add_dir(path)
    run.log_artifact(my_data)


def choose_conf():
    w = widgets.FloatSlider(
        value=0.5,
        min=0,
        max=1.0,
        step=0.1,
        description="Confidence threshold:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )
    display(w)
    return w


def generate_csv_report(evaluation_path: str, wandb_log: bool = False):
    """
    > We read the labels from the `labels` folder, and create a dictionary with the filename as the key,
    and the list of labels as the value. We then convert this dictionary to a dataframe, and write it to
    a csv file

    :param evaluation_path: The path to the evaluation folder
    :type evaluation_path: str
    :return: A dataframe with the following columns:
        filename, class_id, frame_no, x, y, w, h, conf
    """
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
    dlist = [[key, *i] for key, value in data_dict.items() for i in value]
    detect_df = pd.DataFrame.from_records(
        dlist, columns=["filename", "class_id", "frame_no", "x", "y", "w", "h", "conf"]
    )
    csv_out = Path(evaluation_path, "annotations.csv")
    detect_df.sort_values(
        by="frame_no", key=lambda x: np.argsort(index_natsorted(detect_df["filename"]))
    ).to_csv(csv_out, index=False)
    logging.info("Report created at {}".format(csv_out))
    if wandb_log:
        wandb.log({"predictions": wandb.Table(dataframe=detect_df)})
    return detect_df


def generate_tracking_report(tracker_dir: str, eval_dir: str):
    """
    > It takes the tracking output from the tracker and creates a csv file that can be used for
    evaluation

    :param tracker_dir: The directory where the tracking results are stored
    :type tracker_dir: str
    :param eval_dir: The directory where the evaluation results will be stored
    :type eval_dir: str
    :return: A dataframe with the following columns: filename, class_id, frame_no, tracker_id
    """
    data_dict = {}
    if os.path.exists(tracker_dir):
        track_files = os.listdir(tracker_dir)
    else:
        track_files = []
    if len(track_files) == 0:
        logging.error("No tracks found.")
    else:
        for track_file in track_files:
            if track_file.endswith(".txt"):
                data_dict[track_file] = []
                with open(Path(tracker_dir, track_file), "r") as infile:
                    lines = infile.readlines()
                    for line in lines:
                        vals = line.split(" ")
                        class_id, frame_no, tracker_id = vals[0], vals[1], vals[2]
                        data_dict[track_file].append([class_id, frame_no, tracker_id])
        dlist = [
            [os.path.splitext(key)[0] + f"_{i[1]}.txt", i[0], i[1], i[2]]
            for key, value in data_dict.items()
            for i in value
        ]
        detect_df = pd.DataFrame.from_records(
            dlist, columns=["filename", "class_id", "frame_no", "tracker_id"]
        )
        csv_out = Path(eval_dir, "tracking.csv")
        detect_df.sort_values(
            by="frame_no",
            key=lambda x: np.argsort(index_natsorted(detect_df["filename"])),
        ).to_csv(csv_out, index=False)
        logging.info("Report created at {}".format(csv_out))
        return detect_df


def generate_counts(
    eval_dir: str, tracker_dir: str, artifact_dir: str, wandb_log: bool = False
):
    model = torch.load(
        Path(
            [
                f
                for f in Path(artifact_dir).iterdir()
                if f.is_file() and ".pt" in str(f)
            ][-1]
        )
    )
    names = {i: model["model"].names[i] for i in range(len(model["model"].names))}
    tracker_df = generate_tracking_report(tracker_dir, eval_dir)
    if tracker_df is None:
        logging.error("No tracks to count.")
    else:
        tracker_df["frame_no"] = tracker_df["frame_no"].astype(int)
        tracker_df["species_name"] = tracker_df["class_id"].apply(
            lambda x: names[int(x)]
        )
        print("------- DETECTION REPORT -------")
        print("--------------------------------")
        print(tracker_df.groupby(["species_name"])["tracker_id"].nunique())
        final_df = (
            tracker_df.groupby(["species_name"])["tracker_id"]
            .nunique()
            .to_frame()
            .reset_index()
        )
        if wandb_log:
            wandb.log({"tracking_counts": wandb.Table(dataframe=final_df)})
        return final_df


def track_objects(
    source_dir: str,
    artifact_dir: str,
    tracker_folder: str,
    conf_thres: float = 0.5,
    img_size: tuple = (720, 540),
    gpu: bool = False,
):
    """
    This function takes in the source directory of the video, the artifact directory, the tracker
    folder, the confidence threshold, and the image size. It then copies the best model from the
    artifact directory to the tracker folder, and runs the tracking script. It then returns the latest
    tracker folder

    :param source_dir: The directory where the images are stored
    :param artifact_dir: The directory where the model is saved
    :param tracker_folder: The folder where tracker runs will be stored
    :param conf_thres: The confidence threshold for the YOLOv5 model
    :param img_size: The size of the image to be used for tracking. The default is 720, defaults to 720
    (optional)
    :return: The latest tracker folder
    """
    # Check that tracker folder specified exists
    if not os.path.exists(tracker_folder):
        logging.error("The tracker folder does not exist. Please try again")
        return None

    model_path = [
        f
        for f in Path(artifact_dir).iterdir()
        if f.is_file() and ".pt" in str(f) and "osnet" not in str(f)
    ][0]

    best_model = Path(model_path)
    main_tracker_folder = Path("../yolov5_tracker").resolve()

    if not gpu:
        track.run(
            source=source_dir,
            conf_thres=conf_thres,
            yolo_weights=best_model,
            strong_sort_weights=Path(tracker_folder, "osnet_x0_25_msmt17.pt"),
            config_strongsort=Path(
                main_tracker_folder, "strong_sort/configs/strong_sort.yaml"
            ),
            imgsz=img_size,
            project=Path(f"{tracker_folder}/runs/track/"),
            save_vid=True,
            save_conf=True,
            save_txt=True,
        )
    else:
        track.run(
            source=source_dir,
            conf_thres=conf_thres,
            yolo_weights=best_model,
            strong_sort_weights=Path(tracker_folder, "osnet_x0_25_msmt17.pt"),
            config_strongsort=Path(
                main_tracker_folder, "strong_sort/configs/strong_sort.yaml"
            ),
            imgsz=img_size,
            project=Path(f"{tracker_folder}/runs/track/"),
            device=torch_utils.select_device(""),
            save_vid=True,
            save_conf=True,
            save_txt=True,
            half=True,
        )

    tracker_root = os.path.join(tracker_folder, "runs", "track")
    latest_tracker = os.path.join(
        tracker_root, sorted(os.listdir(tracker_root))[-1], "tracks"
    )
    logging.info(f"Tracking saved succesfully to {latest_tracker}")
    return latest_tracker


def encode_image(filepath):
    """
    It takes a filepath to an image, opens the image, reads the bytes, encodes the bytes as base64, and
    returns the encoded string

    :param filepath: The path to the image file
    :return: the base64 encoding of the image.
    """
    with open(filepath, "rb") as f:
        image_bytes = f.read()
    encoded = str(base64.b64encode(image_bytes), "utf-8")
    return "data:image/jpg;base64," + encoded


def get_annotations_viewer(data_path: str, species_list: list):
    """
    It takes a path to a folder containing images and annotations, and a list of species names, and
    returns a widget that allows you to view the images and their annotations, and to edit the
    annotations

    :param data_path: the path to the data folder
    :type data_path: str
    :param species_list: a list of species names
    :type species_list: list
    :return: A VBox widget containing a progress bar and a BBoxWidget.
    """
    image_path = os.path.join(data_path, "images")
    annot_path = os.path.join(data_path, "labels")

    images = sorted(
        [
            f
            for f in os.listdir(image_path)
            if os.path.isfile(os.path.join(image_path, f))
        ]
    )
    annotations = sorted(
        [
            f
            for f in os.listdir(annot_path)
            if os.path.isfile(os.path.join(annot_path, f))
        ]
    )

    # a progress bar to show how far we got
    w_progress = widgets.IntProgress(value=0, max=len(images), description="Progress")
    # the bbox widget
    image = os.path.join(image_path, images[0])
    width, height = imagesize.get(image)
    label_file = annotations[w_progress.value]
    bboxes = []
    labels = []
    with open(os.path.join(annot_path, label_file), "r") as f:
        for line in f:
            s = line.split(" ")
            labels.append(s[0])

            left = (float(s[1]) - (float(s[3]) / 2)) * width
            top = (float(s[2]) - (float(s[4]) / 2)) * height

            bboxes.append(
                {
                    "x": left,
                    "y": top,
                    "width": float(s[3]) * width,
                    "height": float(s[4]) * height,
                    "label": species_list[int(s[0])],
                }
            )
    w_bbox = BBoxWidget(image=encode_image(image), classes=species_list)

    # here we assign an empty list to bboxes but
    # we could also run a detection model on the file
    # and use its output for creating inital bboxes
    w_bbox.bboxes = bboxes

    # combine widgets into a container
    w_container = widgets.VBox(
        [
            w_progress,
            w_bbox,
        ]
    )

    def on_button_clicked(b):
        w_progress.value = 0
        image = os.path.join(image_path, images[0])
        width, height = imagesize.get(image)
        label_file = annotations[w_progress.value]
        bboxes = []
        labels = []
        with open(os.path.join(annot_path, label_file), "r") as f:
            for line in f:
                s = line.split(" ")
                labels.append(s[0])

                left = (float(s[1]) - (float(s[3]) / 2)) * width
                top = (float(s[2]) - (float(s[4]) / 2)) * height

                bboxes.append(
                    {
                        "x": left,
                        "y": top,
                        "width": float(s[3]) * width,
                        "height": float(s[4]) * height,
                        "label": species_list[int(s[0])],
                    }
                )
        w_bbox.image = encode_image(image)

        # here we assign an empty list to bboxes but
        # we could also run a detection model on the file
        # and use its output for creating inital bboxes
        w_bbox.bboxes = bboxes
        w_container.children = tuple(list(w_container.children[1:]))
        b.close()

    # when Skip button is pressed we move on to the next file
    def on_skip():
        w_progress.value += 1
        if w_progress.value == len(annotations):
            button = widgets.Button(
                description="Click to restart.",
                disabled=False,
                display="flex",
                flex_flow="column",
                align_items="stretch",
            )
            if isinstance(w_container.children[0], widgets.Button):
                w_container.children = tuple(list(w_container.children[1:]))
            w_container.children = tuple([button] + list(w_container.children))
            button.on_click(on_button_clicked)

        # open new image in the widget
        else:
            image_file = images[w_progress.value]
            image_p = os.path.join(image_path, image_file)
            width, height = imagesize.get(image_p)
            w_bbox.image = encode_image(image_p)
            label_file = annotations[w_progress.value]
            bboxes = []
            with open(os.path.join(annot_path, label_file), "r") as f:
                for line in f:
                    s = line.split(" ")
                    left = (float(s[1]) - (float(s[3]) / 2)) * width
                    top = (float(s[2]) - (float(s[4]) / 2)) * height
                    bboxes.append(
                        {
                            "x": left,
                            "y": top,
                            "width": float(s[3]) * width,
                            "height": float(s[4]) * height,
                            "label": species_list[int(s[0])],
                        }
                    )

            # here we assign an empty list to bboxes but
            # we could also run a detection model on the file
            # and use its output for creating initial bboxes
            w_bbox.bboxes = bboxes

    w_bbox.on_skip(on_skip)

    # when Submit button is pressed we save current annotations
    # and then move on to the next file
    def on_submit():
        image_file = images[w_progress.value]
        width, height = imagesize.get(os.path.join(image_path, image_file))
        # save annotations for current image
        open(os.path.join(annot_path, label_file), "w").write(
            "\n".join(
                [
                    "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                        species_list.index(
                            i["label"]
                        ),  # single class vs multiple classes
                        min((i["x"] + i["width"] / 2) / width, 1.0),
                        min((i["y"] + i["height"] / 2) / height, 1.0),
                        min(i["width"] / width, 1.0),
                        min(i["height"] / height, 1.0),
                    )
                    for i in w_bbox.bboxes
                ]
            )
        )
        # move on to the next file
        on_skip()

    w_bbox.on_submit(on_submit)

    return w_container


def get_data_viewer(data_path: str):
    """
    It takes a path to a directory of images, and returns a function that displays the images in that
    directory

    :param data_path: the path to the data folder
    :type data_path: str
    :return: A function that takes in a parameter k and returns a widget that displays the image at
    index k in the list of images.
    """
    if "empty_string" in data_path:
        logging.info("No files.")
        return None
    imgs = list(filter(lambda fn: fn.lower().endswith(".jpg"), os.listdir(data_path)))

    def loadimg(k):
        display(draw_box(os.path.join(data_path, imgs[k])))

    return widgets.interact(loadimg, k=(0, len(imgs) - 1))


def draw_box(path: str):
    """
    It takes a path to an image, opens it, opens the corresponding label file, and draws a box around
    each object in the image

    :param path: the path to the image
    :type path: str
    :return: The image with the bounding boxes drawn on it.
    """
    im = PILImage.open(path)
    d = {
        line.split()[0]: line.split()[1:]
        for line in open(path.replace("images", "labels").replace(".jpg", ".txt"))
    }
    dw, dh = im._size
    img1 = ImageDraw.Draw(im)
    for i, vals in d.items():
        vals = tuple(float(val) for val in vals)
        vals_adjusted = tuple(
            [
                int((vals[0] - vals[2] / 2) * dw),
                int((vals[1] - vals[3] / 2) * dh),
                int((vals[0] + vals[2] / 2) * dw),
                int((vals[1] + vals[3] / 2) * dh),
            ]
        )
        img1.rectangle(vals_adjusted, outline="red", width=2)
    return im


def get_dataset(project_name: str, model: str, team_name: str = "koster"):
    """
    It takes in a project name and a model name, and returns the paths to the train and val datasets

    :param project_name: The name of the project you want to download the dataset from
    :type project_name: str
    :param model: The model you want to use
    :type model: str
    :return: The return value is a list of two directories, one for the training data and one for the
    validation data.
    """
    api = wandb.Api()
    if "_" in model:
        run_id = model.split("_")[1]
        run = api.run(f"{team_name}/{project_name.lower()}/runs/{run_id}")
        datasets = [
            artifact for artifact in run.used_artifacts() if artifact.type == "dataset"
        ]
        if len(datasets) == 0:
            logging.error(
                "No datasets are linked to these runs. Please try another run."
            )
            return "empty_string", "empty_string"
        dirs = []
        for i in range(len(["train", "val"])):
            artifact = datasets[i]
            logging.info(f"Downloading {artifact.name} checkpoint...")
            artifact_dir = artifact.download()
            logging.info(f"{artifact.name} - Dataset downloaded.")
            dirs.append(artifact_dir)
        return dirs
    else:
        logging.error("Externally trained model. No data available.")
        return "empty_string", "empty_string"


def get_model(
    model_name: str, project_name: str, download_path: str, team_name: str = "koster"
):
    """
    It downloads the latest model checkpoint from the specified project and model name

    :param model_name: The name of the model you want to download
    :type model_name: str
    :param project_name: The name of the project you want to download the model from
    :type project_name: str
    :param download_path: The path to download the model to
    :type download_path: str
    :return: The path to the downloaded model checkpoint.
    """
    api = wandb.Api()
    try:
        api.artifact_type(
            type_name="model", project=f"{team_name}/{project_name.lower()}"
        ).collections()
    except Exception as e:
        logging.error(f"No model collections found. No artifacts have been logged. {e}")
        return None
    collections = [
        coll
        for coll in api.artifact_type(
            type_name="model", project=f"{team_name}/{project_name.lower()}"
        ).collections()
    ]
    model = [i for i in collections if i.name == model_name]
    if len(model) > 0:
        model = model[0]
    else:
        logging.error("No model found")
    artifact = api.artifact(
        f"{team_name}/{project_name.lower()}/" + model.name + ":latest"
    )
    logging.info("Downloading model checkpoint...")
    artifact_dir = artifact.download(root=download_path)
    logging.info("Checkpoint downloaded.")
    return os.path.realpath(artifact_dir)


# Function to compare original to modified frames
def choose_model(project_name: str, team_name: str = "koster"):
    """
    It takes a project name and returns a dropdown widget that displays the metrics of the model
    selected

    :param project_name: The name of the project you want to load the model from
    :return: The model_widget is being returned.
    """
    model_dict = {}
    model_info = {}
    api = wandb.Api()
    # weird error fix (initialize api another time)
    api.runs(path=f"{team_name}/{project_name.lower()}")
    if (
        len(
            api.runs(path=f"{team_name}/{project_name.lower()}").last_response[
                "project"
            ]["runs"]["edges"]
        )
        == 0
    ):
        logging.error("No models currently available for the project.")
        model_dict = {}
    else:
        for edge, obj in zip(
            api.runs(path=f"{team_name}/{project_name.lower()}").last_response[
                "project"
            ]["runs"]["edges"],
            api.runs(path=f"{team_name}/{project_name.lower()}").objects,
        ):
            if project_name == "model-registry":
                model_dict[edge["node"]["displayName"]] = edge["node"]["displayName"]
            else:
                model_dict[edge["node"]["displayName"]] = "run_" + obj.id + "_model"
            model_info["run_" + obj.id + "_model"] = literal_eval(
                edge["node"]["summaryMetrics"]
            )
    # Add "no movie" option to prevent conflicts
    # models = np.append(list(model_dict.keys()),"No model")

    model_widget = widgets.Dropdown(
        options=[(name, model) for name, model in model_dict.items()],
        description="Select model:",
        ensure_option=False,
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )

    main_out = widgets.Output()
    display(model_widget, main_out)

    # Display the original and modified clips
    def on_change(change):
        with main_out:
            clear_output()
            if change["new"] == "No file":
                print("Choose another file")
            else:
                print(
                    {
                        k: v
                        for k, v in model_info[change["new"]].items()
                        if "metrics" in k
                    }
                )

    model_widget.observe(on_change, names="value")

    return model_widget


# Function to compare original to modified frames
def choose_files(path: str):
    """
    It creates a dropdown menu of all the files in the specified directory, and displays the selected
    file

    :param path: the path to the folder containing the clips
    :type path: str
    """

    # Add "no movie" option to prevent conflicts
    files = np.append([path + i for i in os.listdir(path)], "No file")

    clip_path_widget = widgets.Dropdown(
        options=tuple(np.sort(files)),
        description="Select file:",
        ensure_option=True,
        disabled=False,
        layout=Layout(width="50%"),
        style={"description_width": "initial"},
    )

    main_out = widgets.Output()
    display(clip_path_widget, main_out)

    # Display the original and modified clips
    def on_change(change):
        with main_out:
            clear_output()
            if change["new"] == "No file":
                print("Choose another file")
            else:
                a = view_file(change["new"])
                display(a)

    clip_path_widget.observe(on_change, names="value")


# Display the frames using html
def view_file(path: str):
    """
    It takes a path to a file, opens it, and returns a widget that can be displayed in the notebook

    :param path: The path to the file you want to view
    :return: A widget that displays the image or video.
    """
    # Get path of the modified clip selected
    extension = os.path.splitext(path)[1]
    file = open(path, "rb").read()
    if extension.lower() in [".jpeg", ".png", ".jpg"]:
        widget = widgets.Image(value=file, format=extension)
    elif extension.lower() in [".mp4", ".mov", ".avi"]:
        if os.path.exists("linked_content"):
            shutil.rmtree("linked_content")
        else:
            try:
                os.mkdir("linked_content")
                os.symlink(path, "linked_content/" + os.path.basename(path))
                widget = HTML(
                    f"""
                            <video width=800 height=400 alt="test" controls>
                                <source src="linked_content/{os.path.basename(path)}" type="video/{extension.lower().replace(".", "")}">
                            </video>
                        """
                )
            except:
                logging.error(
                    "Cannot write to local files, viewing not currently possible."
                )
                widget = widgets.Image()

    else:
        logging.error(
            "File format not supported. Supported formats: jpeg, png, jpg, mp4, mov, avi."
        )
        widget.Image()

    return widget
