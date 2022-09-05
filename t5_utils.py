# base imports
import pandas as pd
import os
import paramiko
import logging
from paramiko import SSHClient
from scp import SCPClient

# widget imports
from IPython.display import display
import ipywidgets as widgets

# util imports
import kso_utils.db_utils as db_utils

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

def transfer_model(model_name: str, artifact_dir: str, project_name: str, user: str, password: str):
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
    #api = wandb.Api()
    #collection = [
    #    coll for coll in api.artifact_type(type_name='model', project=project_name).collections()
    #][-1]
    #artifact = api.artifact(f"{project_name}/" + collection.name + ":latest")
    # Download the artifact's contents
    #artifact_dir = artifact.download()
    ssh = SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy()) 
    ssh.load_system_host_keys()
    ssh.connect(hostname="80.252.221.46", 
                port = 2230,
                username=user,
                password=password)

    # SCPCLient takes a paramiko transport as its only argument
    scp = SCPClient(ssh.get_transport())
    scp.put(f"{artifact_dir}/weights/best.pt", 
            f"/home/koster/model_config/weights/ \
            {os.path.basename(project_name)}_{os.path.basename(os.path.dirname(artifact_dir))}_{model_name}")
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
    species_list = pd.read_sql_query("SELECT label from species", conn)["label"].tolist()
    w = widgets.SelectMultiple(
        options=species_list,
        value=[species_list[0]],
        description='Species',
        disabled=False
    )

    display(w)
    return w

def choose_train_params():
    """
    It creates three sliders, one for batch size, one for epochs, and one for confidence threshold
    :return: the values of the sliders.
    """
    v = widgets.FloatLogSlider(
        value=3,
        base=2,
        min=0, # max exponent of base
        max=10, # min exponent of base
        step=1, # exponent step
        description='Batch size:'
    )
    
    z = widgets.IntSlider(
        value=10,
        min=0,
        max=1000,
        step=10,
        description='Epochs:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='d'
    )
    
    z1 = widgets.FloatSlider(
        value=0.5,
        min=0.0,
        max=1.0,
        step=0.1,
        description='Confidence threshold:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        display='flex',
        flex_flow='column',
        align_items='stretch',
        style= {'description_width': 'initial'}
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
        description='Test proportion:',
        disabled=False,
        continuous_update=False,
        orientation='horizontal',
        readout=True,
        readout_format='.1f',
        display='flex',
        flex_flow='column',
        align_items='stretch',
        style= {'description_width': 'initial'}
    )
   
    display(w)
    return w


