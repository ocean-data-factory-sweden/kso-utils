# base imports
import os
import numpy as np
import shutil
import wandb
from ast import literal_eval

# widget imports
from IPython.display import HTML, display, update_display, clear_output, Image
from ipywidgets import interact, Layout, Video
import ipywidgets as widgets

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
                print({k.replace("metrics/", ""):v for k,v in model_info[change["new"]].items() if "metrics" in k})
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
        widget = widgets.Image(value=file, format=extension, width=800, height=400)
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
