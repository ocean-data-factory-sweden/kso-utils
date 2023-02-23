# base imports
import os
import logging
import asyncio
import pandas as pd
from dataclasses import dataclass
import fiftyone as fo
import ipywidgets as widgets
from itertools import chain
from pathlib import Path
import imagesize

# util imports
import kso_utils.tutorials_utils as t_utils
import kso_utils.db_utils as db_utils
import kso_utils.movie_utils as movie_utils
import kso_utils.server_utils as server_utils
import kso_utils.yolo_utils as yolo_utils
from IPython.display import display, HTML
import kso_utils.t1_utils as t1_utils
import kso_utils.t3_utils as t3_utils
import kso_utils.t4_utils as t4_utils
import kso_utils.t8_utils as t8_utils
import kso_utils.t2_utils as t2_utils
import kso_utils.t6_utils as t6_utils


# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


@dataclass
class Project:
    Project_name: str
    Zooniverse_number: int = 0
    db_path: str = None
    server: str = None
    bucket: str = None
    key: str = None
    csv_folder: str = None
    movie_folder: str = None
    photo_folder: str = None
    ml_folder: str = None


class ProjectProcessor:
    def __init__(self, project: Project):
        self.project = project
        self.db_connection = None
        self.server_info = {}
        self.db_info = {}
        self.zoo_info = {}
        self.local_movies_csv = pd.DataFrame()
        self.local_species_csv = pd.DataFrame()
        self.local_sites_csv = pd.DataFrame()
        self.server_movies_csv = pd.DataFrame()
        self.annotation_engine = None
        self.annotations = pd.DataFrame()
        self.classifications = pd.DataFrame()
        self.generated_clips = pd.DataFrame()

        self.setup_db()
        self.get_server_info()
        self.get_movie_info()
        self.load_meta()

    def update_meta(self, new_table, meta_name):
        return t1_utils.update_csv(
            self.db_info,
            self.project,
            new_table,
            getattr(self, "local_" + meta_name + "_csv"),
            "local_" + meta_name + "_csv",
            "server_" + meta_name + "_csv",
        )

    def preview_media(self):
        movie_selected = t_utils.select_movie(self.server_movies_csv)

        async def f(project, db_info, server_movies_csv):
            x = await t_utils.single_wait_for_change(movie_selected, "value")
            display(t_utils.preview_movie(project, db_info, server_movies_csv, x)[0])

        asyncio.create_task(f(self.project, self.db_info, self.server_movies_csv))

    def map_sites(self):
        return t1_utils.map_site(self.db_info, self.project)

    def load_meta(self, base_keys=["sites", "species", "movies"]):
        for key, val in self.db_info.items():
            if any(ext in key for ext in base_keys):
                setattr(self, key, pd.read_csv(val))

    def check_subject_size(self, subject_type: str, subjects_df: pd.DataFrame):
        if subject_type == "clip":
            t3_utils.check_clip_size(subjects_df)
        elif subject_type == "frame":
            t4_utils.check_frame_size(subjects_df)

    def check_movies_meta(self):
        return t1_utils.check_movies_csv(
            self.db_info, self.server_movies_csv, self.project, "Basic", False
        )

    def check_sites_meta(self):
        # code for processing movie metadata (t1_utils.check_sites_csv)
        pass

    def check_species_meta(self):
        return t1_utils.check_species_csv(self.db_info, self.project)

    def setup_db(self):
        # code for setting up the SQLite database goes here
        db_utils.init_db(self.project.db_path)
        self.db_info = t_utils.initiate_db(self.project)
        # connect to the database and add to project
        self.db_connection = db_utils.create_connection(self.project.db_path)

    def get_server_info(self):
        self.server_info = server_utils.connect_to_server(self.project)

    def get_zoo_info(self):
        self.zoo_project = t_utils.connect_zoo_project(self.project)
        self.zoo_info = t_utils.retrieve__populate_zoo_info(
            self.project,
            self.db_info,
            self.zoo_project,
            zoo_info=["subjects", "workflows", "classifications"],
        )

    def get_movie_info(self):
        self.server_movies_csv = server_utils.retrieve_movie_info_from_server(
            self.project, self.db_info
        )

    def load_movie(self, filepath):
        return movie_utils.get_movie_path(filepath, self.db_info, self.project)

    def view_annotations(self, folder_path: str, annotation_classes: list):
        return t8_utils.get_annotations_viewer(
            folder_path, species_list=annotation_classes
        )

    def upload_zu_subjects(self, upload_data: pd.DataFrame, subject_type: str):
        if subject_type == "clip":
            upload_df, sitename, created_on = t3_utils.set_zoo_metadata(
                self.db_info, upload_data, self.project
            )
            t3_utils.upload_clips_to_zooniverse(
                upload_df, sitename, created_on, self.project.Zooniverse_number
            )
        elif subject_type == "frame":
            species_list = []
            upload_df = t4_utils.set_zoo_metadata(
                upload_data, species_list, self.project, self.db_info
            )
            t4_utils.upload_frames_to_zooniverse(
                upload_df, species_list, self.db_info, self.project
            )

    def generate_zu_clips(self, movie_name, movie_path):
        # t3_utils.create_clips

        clip_selection = t3_utils.select_clip_n_len(
            movie_i=movie_name, db_info_dict=self.db_info
        )

        clip_modification = t3_utils.clip_modification_widget()

        button = widgets.Button(
            description="Click to extract clips.",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
        )

        def on_button_clicked(b):
            self.generated_clips = t3_utils.create_clips(
                self.server_movies_csv,
                movie_name,
                movie_path,
                self.db_info,
                clip_selection,
                self.project,
                clip_modification.checks,
                False,
                1,
            )

        button.on_click(on_button_clicked)
        display(clip_modification)
        display(button)

    def generate_zu_frames(self):
        # t4_utils.create_frames
        pass

    def get_ml_data(self):
        # get template ml data
        pass

    def upload_movies(self, movie_list: list):
        return t2_utils.upload_new_movies(self.project, self.db_info, movie_list)

    def add_movies(self):
        movie_list = t2_utils.choose_new_videos_to_upload()
        button = widgets.Button(
            description="Click to upload movies",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"width": "initial"},
        )

        def on_button_clicked(b):
            new_sheet = self.upload_movies(movie_list)
            button2 = widgets.Button(
                description="Save changes",
                disabled=False,
                display="flex",
                flex_flow="column",
                align_items="stretch",
                style={"width": "initial"},
            )

            def on_button_clicked2(b):
                self.local_movies_csv = t2_utils.add_new_rows_to_csv(
                    self.db_info, new_sheet
                )
                logging.info("Changed saved locally")

            button2.on_click(on_button_clicked2)
            display(button2)

        button.on_click(on_button_clicked)
        # t2_utils.upload_new_movies_to_snic
        # t2_utils.update_csv
        # t2_utils.sync_server_csv
        display(button)

    def add_sites(self):
        pass

    def add_species(self):
        pass

    def process_image(self):
        # code for processing image goes here
        pass

    def prepare_metadata(self):
        # code for preparing metadata goes here
        pass

    def prepare_movies(self):
        # code for preparing movie files (standardising formats)
        pass

    def process_classifications(
        self,
        classifications_data,
        subject_type: str,
        agg_params: list,
        summary: bool = False,
    ):
        # code for prepare dataset for machine learning

        t = False
        if subject_type == "clip":
            t = len(agg_params) == 2
        elif subject_type == "frame":
            t = len(agg_params) == 5

        if not t:
            logging.error("Incorrect agg_params length for subject type")
            return

        def get_classifications(classes_df, subject_type):
            conn = self.db_connection
            if subject_type == "frame":
                # Query id and subject type from the subjects table
                subjects_df = pd.read_sql_query(
                    "SELECT id, subject_type, \
                                                https_location, filename, frame_number, movie_id FROM subjects \
                                                WHERE subject_type=='frame'",
                    conn,
                )

            else:
                # Query id and subject type from the subjects table
                subjects_df = pd.read_sql_query(
                    "SELECT id, subject_type, \
                                                https_location, filename, clip_start_time, movie_id FROM subjects \
                                                WHERE subject_type=='clip'",
                    conn,
                )

            # Ensure id format matches classification's subject_id
            classes_df["subject_ids"] = classes_df["subject_ids"].astype("Int64")
            subjects_df["id"] = subjects_df["id"].astype("Int64")

            # Add subject information based on subject_ids
            classes_df = pd.merge(
                classes_df,
                subjects_df,
                how="left",
                left_on="subject_ids",
                right_on="id",
            )

            if classes_df[["subject_type", "https_location"]].isna().any().any():
                # Exclude classifications from missing subjects
                filtered_class_df = classes_df.dropna(
                    subset=["subject_type", "https_location"], how="any"
                ).reset_index(drop=True)

                # Report on the issue
                logging.info(
                    f"There are {(classes_df.shape[0]-filtered_class_df.shape[0])}"
                    f" classifications out of {classes_df.shape[0]}"
                    f" missing subject info. Maybe the subjects have been removed from Zooniverse?"
                )

                classes_df = filtered_class_df

            logging.info(
                f"{classes_df.shape[0]} Zooniverse classifications have been retrieved"
            )

            return classes_df

        agg_class_df, raw_class_df = t8_utils.aggregrate_classifications(
            get_classifications(classifications_data, subject_type),
            subject_type,
            self.project,
            agg_params,
        )
        if summary:
            agg_class_df = (
                agg_class_df.groupby("label")["subject_ids"].agg("count").to_frame()
            )
        return agg_class_df

    def process_annotations(self):
        # code for prepare dataset for machine learning
        pass

    def get_db_view(self, table_name):
        cursor = self.db_connection.cursor()
        # Get column names
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        # Get column names
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns = [col[1] for col in cursor.fetchall()]

        # Create a DataFrame from the data
        df = pd.DataFrame(rows, columns=columns)

        html = (
            f"<div style='height:300px;overflow:auto'>{df.to_html(index=False)}</div>"
        )

        # Display the HTML
        display(HTML(html))

    def get_db_table(self, table_name):
        cursor = self.db_connection.cursor()
        # Get column names
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        # Get column names
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns = [col[1] for col in cursor.fetchall()]

        # Create a DataFrame from the data
        df = pd.DataFrame(rows, columns=columns)

        return df


class MLProject:
    def __init__(
        self,
        project: ProjectProcessor,
        team_name: str,
        config_path,
        weights_path,
        classes,
        model_type,
    ):
        self.project_process = project
        self.project_name = self.project_process.project.Project_name.lower().replace(
            " ", "_"
        )
        self.team_name = team_name
        self.config_path = config_path
        self.weights_path = weights_path
        self.classes = classes
        self.model_type = model_type
        self.run_history = None
        self.best_model_path = None

    def train_yolov5(self, train_data, val_data, epochs=50, batch_size=16, lr=0.001):
        # Train a new YOLOv5 model on the given training and validation data
        import torch
        import wandb
        import yaml
        from yolov5.models.yolo import Model
        from yolov5.utils.dataloaders import create_dataloader
        from yolov5.utils.general import (
            check_dataset,
            check_file,
            check_img_size,
            non_max_suppression,
            set_logging,
        )
        from yolov5.utils.downloads import attempt_download
        from torch.utils.data import DataLoader

        wandb.login()
        set_logging()

        # Check if the specified data directories and configuration files exist
        check_dataset(train_data)
        check_dataset(val_data)
        check_file(self.config_path)
        check_file(self.weights_path)

        # Load the configuration file and create the model
        with open(self.config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        model = Model(
            config["nc"], config["anchors"], config["ch"], self.weights_path
        ).to("cuda")

        # Create the data loaders for training and validation data
        train_loader = create_dataloader(
            train_data,
            batch_size,
            config["img_size"],
            "cuda",
            augment=True,
            hyp=None,
            rect=True,
            cache=False,
            image_weights=False,
            quad=False,
            prefix="",
        )
        val_loader = create_dataloader(
            val_data,
            batch_size,
            config["img_size"],
            "cuda",
            augment=False,
            hyp=None,
            rect=True,
            cache=False,
            image_weights=False,
            quad=False,
            prefix="",
        )

        # Set up the optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=1, T_mult=2
        )

        # Set up Weights and Biases logging
        wandb.init(
            project="yolov5",
            name="run",
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": lr,
                "train_data": train_data,
                "val_data": val_data,
                "model_type": self.model_type,
                "classes": self.classes,
            },
        )

        # Train the model
        for epoch in range(epochs):
            model.train()
            loss = 0.0
            for i, (images, targets, paths, _) in enumerate(train_loader):
                images = images.to("cuda").float() / 255.0
                targets = targets.to("cuda")
                loss_item = model(images, targets)
                loss += loss_item.item()
                optimizer.zero_grad()
                loss_item.backward()
                optimizer.step()
                if i % 100 == 0:
                    wandb.log({"train_loss": loss / (i + 1)})
            wandb.log({"epoch": epoch + 1, "train_loss": loss / len(train_loader)})
            scheduler.step()

            # Validate the model
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, targets, paths, _ in val_loader:
                    images = images.to("cuda").float() / 255.0
                    targets = targets.to("cuda")
                    loss_item = model(images, targets)
                    val_loss += loss_item.item()
                wandb.log({"val_loss": val_loss / len(val_loader)})

            # Save the model checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"yolov5_{self.model_type}_epoch{epoch+1}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                wandb.save(checkpoint_path)

            # Export the best model
            best_model_path = f"yolov5_{self.model_type}_best.pt"
            best_val_loss = float("inf")
            for file in os.listdir(wandb.run.dir):
                if file.endswith(".pt"):
                    val_loss = float(file.split("_")[3][:-3])
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_path = os.path.join(wandb.run.dir, file)
            optimized_model_path = f"yolov5_{self.model_type}_optimized.pt"
            model.load_state_dict(torch.load(best_model_path))
            model.to("cpu")
            model = model.fuse().eval()
            model.export(optimized_model_path)
            wandb.save(optimized_model_path)
            wandb.finish()

    def download_project_runs(self):
        # Download all the runs from the given project ID using Weights and Biases API,
        # sort them by the specified metric, and assign them to the run_history attribute
        import wandb

        wandb.login()
        runs = wandb.Api().runs(f"{self.team_name}/{self.project_name}")
        self.run_history = []
        for run in runs:
            run_info = {}
            run_info["run"] = run
            metrics = run.history()
            run_info["metrics"] = metrics
            self.run_history.append(run_info)
        # self.run_history = sorted(
        #    self.run_history, key=lambda x: x["metrics"]["metrics/"+sort_metric]
        # )

    def prepare_dataset(
        self,
        out_path,
        perc_test,
        class_list,
        img_size,
        remove_nulls,
        track_frames,
        n_tracked_frames,
        agg_df,
    ):
        # code for prepare dataset for machine learning
        yolo_utils.frame_aggregation(
            self.project_process.project,
            self.project_process.db_info,
            out_path,
            perc_test,
            class_list,
            img_size,
            remove_nulls,
            track_frames,
            n_tracked_frames,
            agg_df,
        )

    def get_best_model(self, metric="mAP_0.5", download_path: str = ""):
        # Get the best model from the run history according to the specified metric
        if self.run_history is not None:
            best_run = self.run_history[0]
        else:
            self.download_project_runs()
            best_run = self.run_history[0]
        try:
            best_metric = best_run["metrics"][metric]
            for run in self.run_history:
                if run["metrics"][metric] < best_metric:
                    best_run = run
                    best_metric = run["metrics"][metric]
        except KeyError:
            logging.error(
                "No run with the given metric has been recorded. Using first run as best run."
            )
        best_model = [
            artifact
            for artifact in chain(
                best_run["run"].logged_artifacts(), best_run["run"].used_artifacts()
            )
            if artifact.type == "model"
        ][0]
        import wandb

        api = wandb.Api()
        artifact = api.artifact(
            f"{self.team_name}/{self.project_name}"
            + "/"
            + best_model.name.split(":")[0]
            + ":latest"
        )
        logging.info("Downloading model checkpoint...")
        artifact_dir = artifact.download(root=download_path)
        logging.info("Checkpoint downloaded.")
        return os.path.realpath(artifact_dir)

    def export_best_model(self, output_path):
        # Export the best model to PyTorch format
        import torch
        import tensorflow as tf

        model = tf.keras.models.load_model(self.best_model_path)
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open("temp.tflite", "wb") as f:
            f.write(tflite_model)
        converter = torch.onnx.TFLiteParser.parse("temp.tflite")
        with open(output_path, "wb") as f:
            f.write(converter)


class Annotator:
    def __init__(self, dataset_name, images_path, potential_labels=None):
        self.dataset_name = dataset_name
        self.images_path = images_path
        self.potential_labels = potential_labels
        self.bboxes = []

    def fiftyone_annotate(self):

        # Create a new dataset
        try:
            dataset = fo.load_dataset(self.dataset_name)
            dataset.delete()
        except ValueError:
            pass
        dataset = fo.Dataset(self.dataset_name)

        # Add all the images in the directory to the dataset
        for filename in os.listdir(self.images_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(self.images_path, filename)
                sample = fo.Sample(filepath=image_path)
                dataset.add_sample(sample)

        # Add the potential labels to the dataset
        # Set default classes
        if self.potential_labels is not None:
            label_field = "my_label"
            dataset.add_sample_field(
                label_field, fo.core.fields.StringField, classes=self.potential_labels
            )

        # Create a view with the desired labels

        dataset.annotate(
            self.dataset_name,
            label_type="scalar",
            label_field=label_field,
            launch_editor=True,
            backend="labelbox",
        )
        # Open the dataset in the FiftyOne App
        # Connect to FiftyOne session
        # session = fo.launch_app(dataset, view=view)

        # Start annotating
        # session.wait()

        # Save the annotations
        dataset.save()

    def annotate(self, autolabel_model: str = None):
        return t6_utils.get_annotator(
            self.images_path, self.potential_labels, autolabel_model
        )

    def load_annotations(self):
        images = sorted(
            [
                f
                for f in os.listdir(self.images_path)
                if os.path.isfile(os.path.join(self.images_path, f))
                and f.endswith(".jpg")
            ]
        )
        annot_path = os.path.join(Path(self.images_path).parent, "labels")
        if len(os.listdir(annot_path)) > 0:
            for label_file in os.listdir(annot_path):
                image = os.path.join(self.images_path, images[0])
                width, height = imagesize.get(image)
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
                                "label": self.potential_labels[int(s[0])],
                            }
                        )
            self.bboxes = bboxes
        else:
            self.bboxes = []
