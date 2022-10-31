# base imports
import pandas as pd
import os
import yaml
import paramiko
import logging
import wandb
from paramiko import SSHClient
from scp import SCPClient
from pathlib import Path

# widget imports
from IPython.display import display
import ipywidgets as widgets

# util imports
import kso_utils.db_utils as db_utils

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def setup_paths(output_folder: str):
    """
    It takes the output folder and returns the path to the data file and the path to the hyperparameters
    file

    :param output_folder: The folder where the output of the experiment is stored
    :type output_folder: str
    :return: The data_path and hyps_path
    """
    try:
        data_path = [
            str(Path(output_folder, _))
            for _ in os.listdir(output_folder)
            if _.endswith(".yaml") and "hyp" not in _
        ][-1]
        hyps_path = str(Path(output_folder, "hyp.yaml"))
        
        # Rewrite main path to images and labels
        with open(data_path, "r") as yamlfile:
            cur_yaml = yaml.safe_load(yamlfile) 
            cur_yaml["path"] = output_folder

        if cur_yaml:
            with open(data_path, "w") as yamlfile:
                yaml.safe_dump(cur_yaml, yamlfile)

        logging.info("Success! Paths to data.yaml and hyps.yaml found.")
    except Exception as e:
        logging.error(
            f"{e}, Either data.yaml or hyps.yaml was not found in your folder. Ensure they are located in the selected directory."
        )
    return data_path, hyps_path


def choose_experiment_name():
    """
    It creates a text box that allows you to enter a name for your experiment
    :return: The text box widget.
    """
    exp_name = widgets.Text(
        value="exp_name",
        placeholder="Choose an experiment name",
        description="Experiment name:",
        disabled=False,
        display="flex",
        flex_flow="column",
        align_items="stretch",
        style={"description_width": "initial"},
    )
    display(exp_name)
    return exp_name


def choose_baseline_model():
    """
    It downloads the latest version of the baseline model from WANDB
    :return: The path to the baseline model.
    """
    api = wandb.Api()
    # weird error fix (initialize api another time)
    api.runs(path=f"koster/model-registry")
    api = wandb.Api()
    collections = [
        coll
        for coll in api.artifact_type(
            type_name="model", project="koster/model-registry"
        ).collections()
    ]

    for artifact in collections[-1].versions():
        try:
            artifact_dir = artifact.download()
            artifact_file = [
                str(Path(artifact_dir, "yolov5m.pt"))
                for i in os.listdir(artifact_dir)
                if i.endswith(".pt")
            ][-1]
            logging.info("Baseline YOLO model successfully downloaded from WANDB")
        except Exception as e:
            logging.error(
                "Failed to download the baseline model. Please ensure you are logged in to WANDB."
            )

    return artifact_file


def transfer_model(
    model_name: str, artifact_dir: str, project_name: str, user: str, password: str
):
    """
    It takes the model name, the artifact directory, the project name, the user and the password as
    arguments and then downloads the latest model from the project and uploads it to the server

    :param model_name: the name of the model you want to transfer
    :type model_name: str
    :param artifact_dir: the directory where the model is stored
    :type artifact_dir: str
    :param project_name: The name of the project you want to transfer the model from
    :type project_name: str
    :param user: the username of the remote server
    :type user: str
    :param password: the password for the user you're using to connect to the server
    :type password: str
    """
    ssh = SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.load_system_host_keys()
    ssh.connect(hostname="80.252.221.46", port=2230, username=user, password=password)

    # SCPCLient takes a paramiko transport as its only argument
    scp = SCPClient(ssh.get_transport())
    scp.put(
        f"{artifact_dir}/weights/best.pt",
        f"/home/koster/model_config/weights/ \
            {os.path.basename(project_name)}_{os.path.basename(os.path.dirname(artifact_dir))}_{model_name}",
    )
    scp.close()


def choose_classes(db_path: str = "koster_lab.db"):
    """
    It creates a dropdown menu of all the species in the database, and returns the species that you
    select

    :param db_path: The path to the database, defaults to koster_lab.db
    :type db_path: str (optional)
    :return: A widget object
    """
    conn = db_utils.create_connection(db_path)
    species_list = pd.read_sql_query("SELECT label from species", conn)[
        "label"
    ].tolist()
    w = widgets.SelectMultiple(
        options=species_list,
        value=[species_list[0]],
        description="Species",
        disabled=False,
    )

    display(w)
    return w


def choose_train_params():
    """
    It creates three sliders, one for batch size, one for epochs, and one for confidence threshold
    :return: the values of the sliders.
    """
    v = widgets.FloatLogSlider(
        value=1,
        base=2,
        min=0,  # max exponent of base
        max=10,  # min exponent of base
        step=1,  # exponent step
        description="Batch size:",
        readout=True,
        readout_format="d",
    )

    z = widgets.IntSlider(
        value=1,
        min=0,
        max=1000,
        step=10,
        description="Epochs:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format="d",
    )

    z1 = widgets.FloatSlider(
        value=0.5,
        min=0.0,
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

    box = widgets.HBox([v, z, z1])
    display(box)
    return v, z, z1


def choose_test_prop():
    """
    > The function `choose_test_prop()` creates a slider widget that allows the user to choose the
    proportion of the data to be used for testing
    :return: A widget object
    """

    w = widgets.FloatSlider(
        value=0.2,
        min=0.0,
        max=1.0,
        step=0.1,
        description="Test proportion:",
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
