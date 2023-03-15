# base imports
import os
import logging
import asyncio
import numpy as np
import pandas as pd
from dataclasses import dataclass
import fiftyone as fo
import ipywidgets as widgets
from itertools import chain
from pathlib import Path
import imagesize
import traitlets

# util imports
import kso_utils.tutorials_utils as t_utils
import kso_utils.project_utils as p_utils
import kso_utils.db_utils as db_utils
import kso_utils.movie_utils as movie_utils
import kso_utils.server_utils as server_utils
import kso_utils.yolo_utils as yolo_utils
from IPython.display import display, HTML, clear_output


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


def import_modules(module_names, utils: bool = True):
    importlib = __import__("importlib")
    modules = {}
    for module_name in module_names:
        if utils:
            module_full = "kso_utils." + module_name
        else:
            module_full = module_name
        try:
            modules[module_name] = importlib.import_module(module_full)
        except ModuleNotFoundError:
            logging.error(f"Module {module_name} could not be imported.")
    return modules


class ProjectProcessor:
    def __init__(self, project: Project):
        self.project = project
        self.db_connection = None
        self.server_info = {}
        self.db_info = {}
        self.zoo_info = {}
        self.annotation_engine = None
        self.annotations = pd.DataFrame()
        self.classifications = pd.DataFrame()
        self.generated_clips = pd.DataFrame()

        # Create empty meta tables
        self.init_meta()
        # Get server details
        self.get_server_info()
        # Setup initial db
        self.setup_db()
        # Check movies on server
        self.get_movie_info()
        # Reads csv files
        self.load_meta()
        # Import modules
        self.modules = import_modules(
            ["t1_utils", "t2_utils", "t3_utils", "t4_utils", "t8_utils"]
        )

    def __repr__(self):
        return repr(self.__dict__)

    def keys(self):
        """Print keys of ProjectProcessor object"""
        logging.info("Stored variable names.")
        return list(self.__dict__.keys())

    # general
    def mount_snic(self, snic_path: str = "/mimer/NOBACKUP/groups/snic2021-6-9/"):
        """
        It mounts the remote directory to the local machine

        :param snic_path: The path to the SNIC directory on the remote server, defaults to
        /mimer/NOBACKUP/groups/snic2021-6-9/
        :type snic_path: str (optional)
        :return: The return value is the exit status of the command.
        """
        cmd = "sshfs {}:{} {}".format(
            self.server_info["client"].get_transport().get_username(),
            snic_path,
            snic_path,
        )
        stdin, stdout, stderr = self.server_info["client"].exec_command(cmd)
        # Print output and errors (if any)
        print("Output:", stdout.read().decode("utf-8"))
        print("Errors:", stderr.read().decode("utf-8"))
        # Verify that the remote directory is mounted
        if os.path.ismount(snic_path):
            logging.info("Remote directory mounted successfully!")
            return 1
        else:
            logging.error("Failed to mount remote directory!")
            return 0

    def setup_db(self):
        """
        The function checks if the project is running on the SNIC server, if not it attempts to mount
        the server. If the server is available, the function creates a database and adds the database to
        the project
        :return: The database connection object.
        """
        if self.project.server == "SNIC":
            if not os.path.exists(self.project.csv_folder):
                logging.error("Not running on SNIC server, attempting to mount...")
                status = self.mount_snic()
                if status == 0:
                    return
        db_utils.init_db(self.project.db_path)
        self.db_info = t_utils.initiate_db(self.project)
        # connect to the database and add to project
        self.db_connection = db_utils.create_connection(self.project.db_path)

    def get_db_table(self, table_name, interactive: bool = False):
        """
        It takes a table name as an argument, connects to the database, gets the column names, gets the
        data, and returns a DataFrame or an interactive view of the table using HTML.

        :param table_name: The name of the table you want to get from the database
        :param interactive: A boolean which displays the table as HTML
        :return: A dataframe
        """
        cursor = self.db_connection.cursor()
        # Get column names
        cursor.execute(f"SELECT * FROM {table_name}")
        rows = cursor.fetchall()

        # Get column names
        cursor.execute(f"PRAGMA table_info('{table_name}')")
        columns = [col[1] for col in cursor.fetchall()]

        # Create a DataFrame from the data
        df = pd.DataFrame(rows, columns=columns)

        if interactive:
            html = f"<div style='height:300px;overflow:auto'>{df.to_html(index=False)}</div>"

            # Display the HTML
            display(HTML(html))
        else:
            return df

    def get_server_info(self):
        """
        It connects to the server and returns the server info
        :return: The server_info is added to the ProjectProcessor class.
        """
        try:
            self.server_info = server_utils.connect_to_server(self.project)
        except BaseException as e:
            logging.error(f"Server connection could not be established. Details {e}")
            return

    def get_zoo_info(self):
        """
        It connects to the Zooniverse project, and then retrieves and populates the Zooniverse info for
        the project
        :return: The zoo_info is being returned.
        """
        if self.project.Zooniverse_number is not None:
            self.zoo_project = t_utils.connect_zoo_project(self.project)
            self.zoo_info = t_utils.retrieve__populate_zoo_info(
                self.project,
                self.db_info,
                self.zoo_project,
                zoo_info=["subjects", "workflows", "classifications"],
            )
        else:
            logging.error("This project is not registered with ZU.")
            return

    def get_movie_info(self):
        """
        It retrieves a csv file from the server, and then updates the local variable server_movies_csv
        with the contents of that csv file
        """
        self.server_movies_csv = server_utils.retrieve_movie_info_from_server(
            self.project, self.db_info
        )
        logging.info("server_movies_csv updated")

    def load_movie(self, filepath):
        """
        It takes a filepath, and returns a movie path

        :param filepath: The path to the movie file
        :return: The movie path.
        """
        return movie_utils.get_movie_path(filepath, self.db_info, self.project)

    # t1
    def init_meta(self, init_keys=["movies", "species", "sites"]):
        """
        This function creates a new attribute for the class, which is a pandas dataframe.

        The attribute name is a concatenation of the string "local_" and the value of the variable
        meta_name, and the string "_csv".

        The value of the attribute is a pandas dataframe.

        The function is called with the argument init_keys, which is a list of strings.

        The function loops through the list of strings, and for each string, it creates a new attribute
        for the class.

        :param init_keys: a list of strings that are the names of the metadata files you want to
        initialize
        """
        for meta_name in init_keys:
            setattr(self, "local_" + meta_name + "_csv", pd.DataFrame())
            setattr(self, "server_" + meta_name + "_csv", pd.DataFrame())

    def load_meta(self, base_keys=["movies", "species", "sites"]):
        """
        It loads the metadata from the local csv files into the `db_info` dictionary

        :param base_keys: the base keys to load
        """
        for key, val in self.db_info.items():
            if any("local_" + ext in key for ext in base_keys):
                setattr(self, key, pd.read_csv(val))

    def select_meta_range(self, meta_key: str):
        """
        > This function takes a meta key as input and returns a dataframe, range of rows, and range of
        columns

        :param meta_key: str
        :type meta_key: str
        :return: meta_df, range_rows, range_columns
        """
        meta_df, range_rows, range_columns = self.modules[
            "t1_utils"
        ].select_sheet_range(
            db_info_dict=self.db_info, orig_csv=f"local_{meta_key}_csv"
        )
        return meta_df, range_rows, range_columns

    def edit_meta(self, meta_df: pd.DataFrame, range_rows, range_columns):
        """
        > This function opens a Google Sheet with the dataframe passed as an argument

        :param meta_df: the dataframe that contains the metadata
        :type meta_df: pd.DataFrame
        :param range_rows: a list of row numbers to include in the sheet
        :param range_columns: a list of columns to display in the sheet
        :return: df_filtered, sheet
        """
        df_filtered, sheet = self.modules["t1_utils"].open_csv(
            df=meta_df, df_range_rows=range_rows, df_range_columns=range_columns
        )
        display(sheet)
        return df_filtered, sheet

    def view_meta_changes(self, df_filtered, sheet):
        """
        > This function takes a dataframe and a sheet name as input, and returns a dataframe with the
        changes highlighted

        :param df_filtered: a dataframe that has been filtered by the user
        :param sheet: the name of the sheet you want to view
        :return: A dataframe with the changes highlighted.
        """
        highlight_changes, sheet_df = self.modules["t1_utils"].display_changes(
            self.db_info, isheet=sheet, df_filtered=df_filtered
        )
        display(highlight_changes)
        return sheet_df

    def update_meta(self, new_table, meta_name):
        """
        `update_meta` takes a new table, a meta name, and updates the local and server meta files

        :param new_table: the name of the table that you want to update
        :param meta_name: the name of the metadata file (e.g. "movies")
        :return: The return value is a boolean.
        """
        return self.modules["t1_utils"].update_csv(
            self.db_info,
            self.project,
            new_table,
            getattr(self, "local_" + meta_name + "_csv"),
            "local_" + meta_name + "_csv",
            "server_" + meta_name + "_csv",
        )

    def map_sites(self):
        """
        It takes the database information and the project name and returns a dictionary of the sites in
        the project. This information is displayed using a map widget.
        """
        return self.modules["t1_utils"].map_site(self.db_info, self.project)

    def preview_media(self):
        """
        > The function `preview_media` is a function that takes in a `self` argument and returns a
        function `f` that takes in three arguments: `project`, `db_info`, and `server_movies_csv`. The
        function `f` is an asynchronous function that takes in the value of the `movie_selected` widget
        and displays the movie preview
        """
        movie_selected = t_utils.select_movie(self.server_movies_csv)

        async def f(project, db_info, server_movies_csv):
            x = await t_utils.single_wait_for_change(movie_selected, "value")
            display(t_utils.preview_movie(project, db_info, server_movies_csv, x)[0])

        asyncio.create_task(f(self.project, self.db_info, self.server_movies_csv))

    def check_meta_sync(self, meta_key: str):
        """
        It checks if the local and server versions of a metadata file are the same

        :param meta_key: str
        :type meta_key: str
        :return: The return value is a list of the names of the files in the directory.
        """
        try:
            local_csv, server_csv = getattr(
                self, "local_" + meta_key + "_csv"
            ), getattr(self, "server_" + meta_key + "_csv")
            common_keys = np.intersect1d(local_csv.columns, server_csv.columns)
            assert local_csv[common_keys].equals(server_csv[common_keys])
            logging.info(f"Local and server versions of {meta_key} are synced.")
        except AssertionError:
            logging.error(f"Local and server versions of {meta_key} are not synced.")
            return

    def check_movies_meta(self, review_method: str = "Basic", gpu: bool = False):
        """
        `check_movies_meta` checks the metadata of the movies in the database

        :param review_method: This is the method used to review the movies. The options are: Basic, Advanced, defaults to
        Basic
        :type review_method: str (optional)
        :param gpu: bool = False, defaults to False
        :type gpu: bool (optional)
        """
        return self.modules["t1_utils"].check_movies_csv(
            self.db_info, self.server_movies_csv, self.project, review_method, gpu
        )

    def check_species_meta(self):
        """
        This function checks the species metadata file for the project and returns a boolean value
        :return: the result of the check_species_csv function.
        """
        return self.modules["t1_utils"].check_species_csv(self.db_info, self.project)

    def check_sites_meta(self):
        # TODO: code for processing sites metadata (t1_utils.check_sites_csv)
        pass

    # t2
    def upload_movies(self, movie_list: list):
        """
        > This function takes a list of movie objects and uploads them to the server

        :param movie_list: a list of dictionaries, each dictionary containing the following keys:
        :type movie_list: list
        :return: A list of movie objects
        """
        return self.modules["t2_utils"].upload_new_movies(
            self.project, self.db_info, movie_list
        )

    def add_movies(self):
        """
        > It creates a button that, when clicked, creates a new button that, when clicked, saves the
        changes to the local csv file of the new movies that should be added. It creates a metadata row
        for each new movie, which should be filled in by the user before uploading can continue.
        """
        movie_list = self.modules["t2_utils"].choose_new_videos_to_upload()
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
                self.local_movies_csv = self.modules["t2_utils"].add_new_rows_to_csv(
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

    def view_annotations(self, folder_path: str, annotation_classes: list):
        """
        > This function takes in a folder path and a list of annotation classes and returns a widget that
        allows you to view the annotations in the folder

        :param folder_path: The path to the folder containing the images you want to annotate
        :type folder_path: str
        :param annotation_classes: list of strings
        :type annotation_classes: list
        :return: A list of dictionaries, each dictionary containing the following keys:
            - 'image_path': the path to the image
            - 'annotations': a list of dictionaries, each dictionary containing the following keys:
                - 'class': the class of the annotation
                - 'bbox': the bounding box of the annotation
        """
        return self.modules["t8_utils"].get_annotations_viewer(
            folder_path, species_list=annotation_classes
        )

    # t3 / t4
    def generate_zu_clips(
        self,
        movie_name,
        movie_path,
        use_gpu: bool = False,
        pool_size: int = 4,
        is_example: bool = False,
    ):
        """
        > This function takes a movie name and path, and returns a list of clips from that movie

        :param movie_name: The name of the movie you want to extract clips from
        :param movie_path: The path to the movie you want to extract clips from
        :param use_gpu: If you have a GPU, set this to True, defaults to False
        :type use_gpu: bool (optional)
        :param pool_size: number of threads to use for clip extraction, defaults to 4
        :type pool_size: int (optional)
        :param is_example: If True, the clips will be selected randomly. If False, the clips will be
        selected based on the number of clips and the length of each clip, defaults to False
        :type is_example: bool (optional)
        """
        # t3_utils.create_clips

        if is_example:
            clip_selection = self.modules["t3_utils"].select_random_clips(
                movie_i=movie_name, db_info_dict=self.db_info
            )
        else:
            clip_selection = self.modules["t3_utils"].select_clip_n_len(
                movie_i=movie_name, db_info_dict=self.db_info
            )

        clip_modification = self.modules["t3_utils"].clip_modification_widget()

        button = widgets.Button(
            description="Click to extract clips.",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
        )

        def on_button_clicked(b):
            self.generated_clips = self.modules["t3_utils"].create_clips(
                self.server_movies_csv,
                movie_name,
                movie_path,
                self.db_info,
                clip_selection,
                self.project,
                clip_modification.checks,
                use_gpu,
                pool_size,
            )

        button.on_click(on_button_clicked)
        display(clip_modification)
        display(button)

    def check_movies_uploaded(self, movie_name: str):
        """
        This function checks if a movie has been uploaded to Zooniverse

        :param movie_name: The name of the movie you want to check if it's uploaded
        :type movie_name: str
        """
        self.modules["t3_utils"].check_movie_uploaded(
            movie_i=movie_name, db_info_dict=self.db_info
        )

    def upload_zu_subjects(self, upload_data: pd.DataFrame, subject_type: str):
        """
        This function uploads clips or frames to Zooniverse, depending on the subject_type argument

        :param upload_data: a pandas dataframe with the following columns:
        :type upload_data: pd.DataFrame
        :param subject_type: str = "clip" or "frame"
        :type subject_type: str
        """
        if subject_type == "clip":
            upload_df, sitename, created_on = self.modules["t3_utils"].set_zoo_metadata(
                self.db_info, upload_data, self.project
            )
            self.modules["t3_utils"].upload_clips_to_zooniverse(
                upload_df, sitename, created_on, self.project.Zooniverse_number
            )
            # Clean up subjects after upload
            self.modules["t3_utils"].remove_temp_clips(upload_df)
        elif subject_type == "frame":
            species_list = []
            upload_df = self.modules["t4_utils"].set_zoo_metadata(
                upload_data, species_list, self.project, self.db_info
            )
            self.modules["t4_utils"].upload_frames_to_zooniverse(
                upload_df, species_list, self.db_info, self.project
            )

    def generate_zu_frames(self):
        """
        This function takes a dataframe of frames to upload, a species of interest, a project, and a
        dictionary of modifications to make to the frames, and returns a dataframe of modified frames.
        """

        # t4_utils.create_frames

        frame_modification = self.modules["t3_utils"].clip_modification_widget()

        button = widgets.Button(
            description="Click to modify frames",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
        )

        def on_button_clicked(b):
            self.generated_frames = self.modules["t4_utils"].modify_frames(
                frames_to_upload_df=self.frames_to_upload_df.df.reset_index(drop=True),
                species_i=self.species_of_interest,
                modification_details=frame_modification.checks,
                project=self.project,
            )

        button.on_click(on_button_clicked)
        display(frame_modification)
        display(button)

    def get_frames(self, n_frames_subject: int = 3, subsample_up_to: int = 3):
        """
        > This function allows you to choose a species of interest, and then it will fetch a random
        sample of frames from the database for that species

        :param n_frames_subject: number of frames to fetch per subject, defaults to 3
        :type n_frames_subject: int (optional)
        :param subsample_up_to: If you have a lot of frames for a given species, you can subsample them.
        This parameter controls how many frames you want to subsample to, defaults to 3
        :type subsample_up_to: int (optional)
        """

        species_list = self.modules["t4_utils"].choose_species(self.db_info)

        button = widgets.Button(
            description="Click to fetch frames",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
        )

        def on_button_clicked(b):
            self.species_of_interest = species_list.value
            self.frames_to_upload_df = self.modules["t4_utils"].get_frames(
                species_names=species_list.value,
                db_path=self.db_info["db_path"],
                zoo_info_dict=self.zoo_info,
                server_dict=self.db_info,
                project=self.project,
                n_frames_subject=n_frames_subject,
                subsample_up_to=subsample_up_to,
            )

        button.on_click(on_button_clicked)
        display(button)

    def check_frames_uploaded(self):
        """
        This function checks if the frames in the frames_to_upload_df dataframe have been uploaded to
        the database
        """
        self.modules["t4_utils"].check_frames_uploaded(
            self.frames_to_upload_df,
            self.project,
            self.species_of_interest,
            self.db_connection,
        )

    # t5, t6, t7
    def get_ml_data(self):
        # get template ml data
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

    # t8
    def process_classifications(
        self,
        classifications_data,
        subject_type: str,
        agg_params: list,
        summary: bool = False,
    ):

        """
        It takes in a dataframe of classifications, a subject type (clip or frame), a list of
        aggregation parameters, and a boolean for whether or not to return a summary of the
        classifications.

        It then returns a dataframe of aggregated classifications.

        Let's break it down.

        First, we check that the length of the aggregation parameters is correct for the subject type.

        Then, we define a function called `get_classifications` that takes in a dataframe of
        classifications and a subject type.

        This function queries the subjects table for the subject type and then merges the
        classifications dataframe with the subjects dataframe.

        It then returns the merged dataframe.

        Finally, we call the `aggregrate_classifications` function from the `t8_utils` module, passing
        in the dataframe returned by `get_classifications`, the subject

        :param classifications_data: the dataframe of classifications from the Zooniverse API
        :param subject_type: This is the type of subject you want to retrieve classifications for. This
        can be either "clip" or "frame"
        :type subject_type: str
        :param agg_params: list
        :type agg_params: list
        :param summary: If True, the output will be a summary of the classifications, with the number of
        classifications per label, defaults to False
        :type summary: bool (optional)
        """

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

        agg_class_df, raw_class_df = self.modules[
            "t8_utils"
        ].aggregrate_classifications(
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


class MLProjectProcessor(ProjectProcessor):
    def __init__(
        self,
        project_process: ProjectProcessor,
        team_name: str,
        config_path,
        weights_path,
        classes,
        model_type,
    ):
        self.__dict__ = project_process.__dict__.copy()
        super().__init__(self.project)
        self.project_name = self.project.Project_name.lower().replace(" ", "_")
        self.team_name = team_name
        self.config_path = config_path
        self.weights_path = weights_path
        self.classes = classes
        self.model_type = model_type
        self.run_history = None
        self.best_model_path = None
        self.modules = import_modules(["t5_utils", "t6_utils", "t7_utils"])
        self.modules.update(
            import_modules(["torch", "wandb", "yaml", "yolov5"], utils=False)
        )

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
            self.project,
            self.db_info,
            out_path,
            perc_test,
            class_list,
            img_size,
            remove_nulls,
            track_frames,
            n_tracked_frames,
            agg_df,
        )

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
        self.modules = import_modules(["t5_utils", "t6_utils", "t7_utils"])

    def __repr__(self):
        return repr(self.__dict__)

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
        return self.modules["t6_utils"].get_annotator(
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
