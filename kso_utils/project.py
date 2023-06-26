# base impkrts
import os
import sys
import glob
import logging
import asyncio
import wandb
import numpy as np
import pandas as pd
import ipywidgets as widgets
import ffmpeg
import shutil
import paramiko
from paramiko import SSHClient
from scp import SCPClient
from itertools import chain
from pathlib import Path
from tqdm import tqdm
from ast import literal_eval
import imagesize
import ipysheet
from IPython.display import display, clear_output
from IPython.core.display import HTML

# util imports
import kso_utils.tutorials_utils as t_utils
import kso_utils.project_utils as project_utils
import kso_utils.db_utils as db_utils
import kso_utils.movie_utils as movie_utils
import kso_utils.server_utils as server_utils
import kso_utils.yolo_utils as yolo_utils
import kso_utils.zooniverse_utils as zu_utils
import kso_utils.general as g_utils
import kso_utils.widgets as kso_widgets

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


class ProjectProcessor:
    # The ProjectProcessor class initializes various attributes and methods for processing a project,
    # including importing modules, setting up a database, and loading metadata.
    def __init__(self, project: project_utils.Project):
        self.project = project
        self.db_connection = None
        self.init_keys = ["movies", "species", "photos", "surveys", "sites"]
        self.server_connection = {}
        self.csv_paths = {}
        self.zoo_info = {}
        self.annotation_engine = None
        self.annotations = pd.DataFrame()
        self.classifications = pd.DataFrame()
        self.generated_clips = pd.DataFrame()

        # Import modules
        self.modules = g_utils.import_modules([])

        # Get server details and connect to server
        self.connect_to_server()

        # Map initial csv files
        self.map_init_csv()

        # Create empty db and populate with local csv files data
        self.setup_db()

        ############ TO REVIEW #############
        # Check if template project
        if self.project.server == "SNIC":
            if not os.path.exists(self.project.csv_folder):
                logging.error("Not running on SNIC server, attempting to mount...")
                status = self.mount_snic()
                if status == 0:
                    return

    ############# Finish Review ###################

    def __repr__(self):
        return repr(self.__dict__)

    def keys(self):
        """Print keys of ProjectProcessor object"""
        logging.info("Stored variable names.")
        return list(self.__dict__.keys())

    # Functions to initiate the project
    def connect_to_server(self):
        """
        It connects to the server and returns the server info
        :return: The server_connection is added to the ProjectProcessor class.
        """
        try:
            self.server_connection = server_utils.connect_to_server(self.project)
        except BaseException as e:
            logging.error(f"Server connection could not be established. Details {e}")
            return

    def map_init_csv(self):
        """
        This function maps the csv files, download them from the server (if needed) and
        stores the server/local paths of the csv files
        """

        # Create the folder to store the csv files if not exist
        if not os.path.exists(self.project.csv_folder):
            Path(self.project.csv_folder).mkdir(parents=True, exist_ok=True)
            # Recursively add permissions to folders created
            [
                os.chmod(root, 0o777)
                for root, dirs, files in os.walk(self.project.csv_folder)
            ]

        # Download csv files from the server if needed and store their server path
        self.csv_paths = server_utils.download_init_csv(
            self.project, self.init_keys, self.server_connection
        )

        # Store the paths of the local csv files
        self.load_meta()

    def load_meta(self):
        """
        It loads the metadata from the relevant local csv files into the `csv_paths` dictionary
        """
        # Retrieve a list with all the csv files in the folder with initival csvs
        local_files = os.listdir(self.project.csv_folder)
        local_csv_files = [
            filename for filename in local_files if filename.endswith("csv")
        ]

        # Select only csv files that are relevant to start the db
        local_csvs_db = [
            file
            for file in local_csv_files
            if any(local_csv_files in file for local_csv_files in self.init_keys)
        ]

        # Store the paths of the local csv files of interest into the "csv_paths" dictionary
        for local_csv in local_csvs_db:
            # Specify the key of the csv
            init_key = [key for key in self.init_keys if key in local_csv][0]

            # Specify the key in the dictionary of the csv file
            csv_key = str("local_" + init_key + "_csv")

            # Store the path of the csv file
            self.csv_paths[csv_key] = Path(self.project.csv_folder, local_csv)

            # Read the local csv files into a pd df
            setattr(self, csv_key, pd.read_csv(self.csv_paths[csv_key]))

    def setup_db(self):
        """
        The function creates a database and populates it with the data from the local csv files.
        It also return the db connection
        :return: The database connection object.
        """
        # Create a new database for the project
        db_utils.create_db(self.project.db_path)

        # Connect to the database and add the db connection to project
        self.db_connection = db_utils.create_connection(self.project.db_path)

        # Select only attributes of the propjectprocessor that are df of local csvs
        # (sorted in reverse alphabetically to load sites before movies)
        local_dfs = sorted(
            [str(file) for file in list(self.keys()) if "local" in str(file)],
            reverse=True,
        )

        # Populate the db with initial info from the local_csvs
        [
            db_utils.populate_db(
                project=self.project,
                conn=self.db_connection,
                local_df=getattr(self, i),
                init_key=i.split("_", 2)[1],
            )
            for i in local_dfs
        ]

    def choose_workflows(self, generate_export: bool = False):
        self.set_zoo_info(generate_export=generate_export)
        self.workflow_widget = zu_utils.WidgetMaker(self.zoo_info["workflows"])
        display(self.workflow_widget)

    def set_zoo_info(self, generate_export: bool = False):
        if self.project.Zooniverse_number is not None:
            self.zoo_project = zu_utils.connect_zoo_project(self.project)
        else:
            logging.error("This project is not registered with Zooniverse.")
            return
        if self.zoo_info is None or self.zoo_info == {}:
            self.zoo_info = zu_utils.retrieve_zoo_info(
                self.project,
                self.zoo_project,
                zoo_info=["subjects", "workflows", "classifications"],
                generate_export=generate_export,
            )

    def get_zoo_info(self, generate_export: bool = False):
        """
        It connects to the Zooniverse project, and then retrieves and populates the Zooniverse info for
        the project
        :return: The zoo_info is being returned.
        """
        if hasattr(self.project, "db_path"):
            if hasattr(self, "workflow_widget"):
                # If the workflow widget is used, retrieve a subset of the subjects to build the db
                names, workflow_versions = [], []
                for i in range(0, len(self.workflow_widget.checks), 3):
                    names.append(list(self.workflow_widget.checks.values())[i])
                    workflow_versions.append(
                        list(self.workflow_widget.checks.values())[i + 2]
                    )

                self.project.zu_workflows = zu_utils.get_workflow_ids(
                    self.zoo_info["workflows"], names
                )

                if not isinstance(self.project.zu_workflows, list):
                    self.project.zu_workflows = literal_eval(self.project.zu_workflows)

                self.zoo_info["subjects"]["workflow_id"] = self.zoo_info["subjects"][
                    "workflow_id"
                ].astype("Int64")
                subjects_series = self.zoo_info["subjects"][
                    self.zoo_info["subjects"].workflow_id.isin(
                        self.project.zu_workflows
                    )
                ].copy()

            else:
                self.set_zoo_info(generate_export=generate_export)
                subjects_series = self.zoo_info["subjects"].copy()

            # Safely remove subjects table
            db_utils.drop_table(conn=self.db_connection, table_name="subjects")

            if len(subjects_series) > 0:
                # Fill or re-fill subjects table
                zu_utils.populate_subjects(
                    subjects_series, project=self.project, conn=self.db_connection
                )
            else:
                logging.error(
                    "No subjects to populate database from the workflows selected."
                )
        else:
            logging.info("No database path found. Subjects have not been added to db")

    def get_movie_info(self):
        """
        This function checks what movies from the movies csv are available
        """
        self.server_movies_csv = movie_utils.retrieve_movie_info_from_server(
            project=self.project,
            db_connection=self.db_connection,
            server_connection=self.server_connection,
        )

        logging.info("Information of available movies has been retrieved")

    def load_movie(self, filepath):
        """
        It takes a filepath, and returns a movie path

        :param filepath: The path to the movie file
        :return: The movie path.
        """
        return movie_utils.get_movie_path(filepath, self)

    # t1

    def select_meta_range(self, meta_key: str):
        """
        > This function takes a meta key as input and returns a dataframe, range of rows, and range of
        columns

        :param meta_key: str
        :type meta_key: str
        :return: meta_df, range_rows, range_columns
        """
        meta_df, range_rows, range_columns = kso_widgets.select_sheet_range(
            project=self.project,
            orig_csv=f"local_{meta_key}_csv",
            csv_paths=self.csv_paths,
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
        df_filtered, sheet = kso_widgets.open_csv(
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
        highlight_changes, sheet_df = kso_widgets.display_changes(
            isheet=sheet, df_filtered=df_filtered
        )
        display(highlight_changes)
        return sheet_df

    def update_meta(
        self,
        sheet_df: pd.DataFrame,
        meta_name: str,
    ):
        return kso_widgets.update_meta(
            project=self.project,
            conn=self.db_connection,
            server_connection=self.server_connection,
            sheet_df=sheet_df,
            df=getattr(self, "local_" + meta_name + "_csv"),
            meta_name=meta_name,
            csv_paths=self.csv_paths,
        )

    def map_sites(self):
        return kso_widgets.map_sites(project=self.project, csv_paths=self.csv_paths)

    def preview_media(self):
        """
        > The function `preview_media` is a function that takes in a `self` argument and returns a
        function `f` that takes in three arguments: `project`, `csv_paths`, and `server_movies_csv`. The
        function `f` is an asynchronous function that takes in the value of the `movie_selected` widget
        and displays the movie preview
        """
        movie_selected = kso_widgets.select_movie(self.server_movies_csv)

        async def f(project, server_connection, server_movies_csv):
            x = await kso_widgets.single_wait_for_change(movie_selected, "value")
            html, movie_path = movie_utils.preview_movie(
                project=project,
                available_movies_df=server_movies_csv,
                movie_i=x,
                server_connection=server_connection,
            )
            display(html)
            self.movie_selected = x
            self.movie_path = movie_path

        asyncio.create_task(
            f(self.project, self.server_connection, self.server_movies_csv)
        )

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

    def check_movies_meta(
        self,
        review_method: str,
        gpu_available: bool = False,
    ):
        """
        > The function `check_movies_csv` loads the csv with movies information and checks if it is empty

        :param review_method: The method used to review the movies
        :param gpu_available: Boolean, whether or not a GPU is available
        """

        movie_utils.check_movies_meta(
            project=self.project,
            csv_paths=self.csv_paths,
            server_movies_csv=self.server_movies_csv,
            conn=self.db_connection,
            server_connection=self.server_connection,
            review_method=review_method,
            gpu_available=gpu_available,
        )

    def check_species_meta(self):
        return db_utils.check_species_meta(
            csv_paths=self.csv_paths, db_connection=self.db_connection
        )

    def check_sites_meta(self):
        # TODO: code for processing sites metadata (t1_utils.check_sites_csv)
        pass

    # t2
    def upload_movies(self, movie_list: list):
        """
        It uploads the new movies to the SNIC server and creates new rows to be updated
        with movie metadata and saved into movies.csv

        :param movie_list: list of new movies that are to be added to movies.csv
        """
        # Get number of new movies to be added
        movie_folder = self.project.movie_folder
        number_of_movies = len(movie_list)
        # Get current movies
        movies_df = pd.read_csv(self.csv_paths["local_movies_csv"])
        # Set up a new row for each new movie
        new_movie_rows_sheet = ipysheet.sheet(
            rows=number_of_movies,
            columns=movies_df.shape[1],
            column_headers=movies_df.columns.tolist(),
        )
        if len(movie_list) == 0:
            logging.error("No valid movie found to upload.")
            return
        for index, movie in enumerate(movie_list):
            if self.project.server == "SNIC":
                # Specify volume allocated by SNIC
                snic_path = "/mimer/NOBACKUP/groups/snic2021-6-9"
                remote_fpath = Path(f"{snic_path}/tmp_dir/", movie[1])
            else:
                remote_fpath = Path(f"{movie_folder}", movie[1])
            if os.path.exists(remote_fpath):
                logging.info(
                    "Filename "
                    + str(movie[1])
                    + " already exists on SNIC, try again with a new file"
                )
                return
            else:
                # process video
                stem = "processed"
                p = Path(movie[0])
                processed_video_path = p.with_name(f"{p.stem}_{stem}{p.suffix}").name
                logging.info("Movie to be uploaded: " + processed_video_path)
                ffmpeg.input(p).output(
                    processed_video_path,
                    crf=22,
                    pix_fmt="yuv420p",
                    vcodec="libx264",
                    threads=4,
                ).run(capture_stdout=True, capture_stderr=True, overwrite_output=True)

                if self.project.server == "SNIC":
                    server_utils.upload_object_to_snic(
                        self.server_connection["sftp_client"],
                        str(processed_video_path),
                        str(remote_fpath),
                    )
                elif self.project.server in ["LOCAL", "TEMPLATE"]:
                    shutil.copy2(str(processed_video_path), str(remote_fpath))
                logging.info("movie uploaded\n")
            # Fetch movie metadata that can be calculated from movie file
            fps, duration = movie_utils.get_fps_duration(movie[0])
            movie_id = str(max(movies_df["movie_id"]) + 1)
            ipysheet.cell(index, 0, movie_id)
            ipysheet.cell(index, 1, movie[1])
            ipysheet.cell(index, 2, "-")
            ipysheet.cell(index, 3, "-")
            ipysheet.cell(index, 4, "-")
            ipysheet.cell(index, 5, fps)
            ipysheet.cell(index, 6, duration)
            ipysheet.cell(index, 7, "-")
            ipysheet.cell(index, 8, "-")
        logging.info("All movies uploaded:\n")
        logging.info(
            "Complete this sheet by filling the missing info on the movie you just uploaded"
        )
        display(new_movie_rows_sheet)
        return new_movie_rows_sheet

    def add_movies(self):
        """
        > It creates a button that, when clicked, creates a new button that, when clicked, saves the
        changes to the local csv file of the new movies that should be added. It creates a metadata row
        for each new movie, which should be filled in by the user before uploading can continue.
        """
        movie_list = kso_widgets.choose_new_videos_to_upload()
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
                movies_df = pd.read_csv(self.csv_paths["local_movies_csv"])
                new_movie_rows_df = ipysheet.to_dataframe(new_sheet)
                self.local_movies_csv = pd.concat(
                    [movies_df, new_movie_rows_df], ignore_index=True
                )
                logging.info("Changed saved locally")

            button2.on_click(on_button_clicked2)
            display(button2)

        button.on_click(on_button_clicked)

        # TO BE COMPLETED with Chloudina
        # upload new movies and update csvs
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
        :return: A list of dictionaries, each dictionary containing the following keys
                 - 'image_path': the path to the image
                 - 'annotations': a list of dictionaries, each dictionary containing the following keys:
                 - 'class': the class of the annotation
                 - 'bbox': the bounding box of the annotation
        """
        return t_utils.get_annotations_viewer(
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
            clip_selection = kso_widgets.select_random_clips(
                project=self.project, movie_i=movie_name
            )
        else:
            clip_selection = kso_widgets.select_clip_n_len(
                project=self.project, movie_i=movie_name
            )

        clip_modification = kso_widgets.clip_modification_widget()

        button = widgets.Button(
            description="Click to extract clips.",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
        )

        def on_button_clicked(b):
            self.generated_clips = t_utils.create_clips(
                available_movies_df=self.server_movies_csv,
                movie_i=movie_name,
                movie_path=movie_path,
                clip_selection=clip_selection,
                project=self.project,
                modification_details={},
                gpu_available=use_gpu,
                pool_size=pool_size,
            )
            mod_clips = t_utils.create_modified_clips(
                self.project,
                self.generated_clips.clip_path,
                movie_name,
                clip_modification.checks,
                use_gpu,
                pool_size,
            )
            # Temporary workaround to get both clip paths
            self.generated_clips["modif_clip_path"] = mod_clips

        button.on_click(on_button_clicked)
        display(clip_modification)
        display(button)

    def check_movies_uploaded(self, movie_name: str):
        """
        This function checks if a movie has been uploaded to Zooniverse

        :param movie_name: The name of the movie you want to check if it's uploaded
        :type movie_name: str
        """
        movie_utils.check_movie_uploaded(
            project=self.project, db_connection=self.db_connection, movie_i=movie_name
        )

    def generate_zu_frames(self):
        """
        This function takes a dataframe of frames to upload, a species of interest, a project, and a
        dictionary of modifications to make to the frames, and returns a dataframe of modified frames.
        """

        frame_modification = kso_widgets.clip_modification_widget()

        button = widgets.Button(
            description="Click to modify frames",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
        )

        def on_button_clicked(b):
            self.generated_frames = zu_utils.modify_frames(
                project=self.project,
                frames_to_upload_df=self.frames_to_upload_df.df.reset_index(drop=True),
                species_i=self.species_of_interest,
                modification_details=frame_modification.checks,
            )

        button.on_click(on_button_clicked)
        display(frame_modification)
        display(button)

    def generate_custom_frames(
        self,
        skip_start: int,
        skip_end: int,
        input_path: str,
        output_path: str,
        num_frames: int = None,
        frames_skip: int = None,
    ):
        """
        This function generates custom frames from input movie files and saves them in an output directory.

        :param input_path: The directory path where the input movie files are located
        :type input_path: str
        :param output_path: The directory where the extracted frames will be saved
        :type output_path: str
        :param num_frames: The number of frames to extract from each video file. If not specified, all
        frames will be extracted
        :type num_frames: int
        :param frames_skip: The `frames_skip` parameter is an optional integer that specifies the number of
        frames to skip between each extracted frame. For example, if `frames_skip` is set to 2, every other
        frame will be extracted. If `frames_skip` is not specified, all frames will be extracted
        :type frames_skip: int
        :return: the results of calling the `parallel_map` function with the `extract_custom_frames` function from
        the `t4_utils` module, passing in the `movie_files` list as the input and the `args` tuple
        containing `output_dir`, `num_frames`, and `frames_skip`. The `parallel_map` function is a custom
        function that applies the given function to each element of a list of movie_files.
        """
        frame_modification = kso_widgets.clip_modification_widget()
        species_list = kso_widgets.choose_species(self.project)

        button = widgets.Button(
            description="Click to modify frames",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
        )

        def on_button_clicked(b):
            movie_files = sorted(
                [
                    f
                    for f in glob.glob(f"{input_path}/*")
                    if os.path.isfile(f)
                    and os.path.splitext(f)[1].lower()
                    in [".mov", ".mp4", ".avi", ".mkv"]
                ]
            )

            results = g_utils.parallel_map(
                kso_widgets.extract_custom_frames,
                movie_files,
                args=(
                    [output_path] * len(movie_files),
                    [skip_start] * len(movie_files),
                    [skip_end] * len(movie_files),
                    [num_frames] * len(movie_files),
                    [frames_skip] * len(movie_files),
                ),
            )
            if len(results) > 0:
                self.frames_to_upload_df = pd.concat(results)
                self.frames_to_upload_df["species_id"] = pd.Series(
                    [t_utils.get_species_ids(self.project, species_list.value)]
                    * len(self.frames_to_upload_df)
                )
                self.frames_to_upload_df = self.frames_to_upload_df.merge(
                    db_utils.get_df_from_db_table(self.db_connection, "movies").rename(
                        columns={"id": "movie_id"}
                    ),
                    how="left",
                    left_on="movie_filename",
                    right_on="filename",
                )
                # Ensure necessary metadata fields are available
                self.frames_to_upload_df = self.frames_to_upload_df[
                    [
                        "frame_path",
                        "site_id",
                        "movie_id",
                        "created_on",
                        "frame_number",
                        "species_id",
                    ]
                ]

            else:
                logging.error("No results.")
                self.frames_to_upload_df = pd.DataFrame()
            self.project.output_path = output_path
            self.generated_frames = zu_utils.modify_frames(
                frames_to_upload_df=self.frames_to_upload_df.reset_index(drop=True),
                species_i=species_list.value,
                modification_details=frame_modification.checks,
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

        species_list = kso_widgets.choose_species(self.project)

        button = widgets.Button(
            description="Click to fetch frames",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
        )

        def on_button_clicked(b):
            self.species_of_interest = species_list.value
            self.frames_to_upload_df = zu_utils.get_frames(
                project=self.project,
                zoo_info_dict=self.zoo_info,
                species_names=species_list.value,
                n_frames_subject=n_frames_subject,
                subsample_up_to=subsample_up_to,
            )

        button.on_click(on_button_clicked)
        display(button)

    def upload_zu_subjects(self, subject_type: str):
        """
        This function uploads clips or frames to Zooniverse, depending on the subject_type argument

        :param
        :param subject_type: str = "clip" or "frame"
        :type subject_type: str
        """
        if subject_type == "clip":
            upload_df, sitename, created_on = zu_utils.set_zoo_clip_metadata(
                project=self.project,
                generated_clipsdf=self.generated_clips,
                sitesdf=self.local_sites_csv,
                moviesdf=self.local_movies_csv,
            )
            zu_utils.upload_clips_to_zooniverse(
                project=self.project,
                upload_to_zoo=upload_df,
                sitename=sitename,
                created_on=created_on,
            )
            # Clean up subjects after upload
            zu_utils.remove_temp_clips(upload_df)
        elif subject_type == "frame":
            species_list = []
            upload_df = zu_utils.set_zoo_frame_metadata(
                project=self.project,
                df=upload_data,
                species_list=species_list,
                csv_paths=self.csv_paths,
            )
            zu_utils.upload_frames_to_zooniverse(
                project=self.project,
                upload_to_zoo=upload_df,
                species_list=species_list,
            )

    # t5, t6, t7
    def get_team_name(self):
        """
        > If the project name is "Spyfish_Aotearoa", return "wildlife-ai", otherwise return "koster"

        :param project_name: The name of the project you want to get the data from
        :type project_name: str
        :return: The team name is being returned.
        """

        if self.project.Project_name == "Spyfish_Aotearoa":
            return "wildlife-ai"
        else:
            return "koster"

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

    def check_frames_uploaded(self):
        """
        This function checks if the frames in the frames_to_upload_df dataframe have been uploaded to
        the database
        """
        t_utils.check_frames_uploaded(
            self.project,
            self.frames_to_upload_df,
            self.species_of_interest,
        )

    # t8
    def get_classifications(
        self,
        workflow_dict: dict,
        workflows_df: pd.DataFrame,
        subj_type: str,
        class_df: pd.DataFrame,
    ):
        return zu_utils.get_classifications(
            project=self.project,
            workflow_dict=workflow_dict,
            workflows_df=workflows_df,
            subj_type=subj_type,
            class_df=class_df,
        )

    def process_classifications(
        self, classifications_data, subject_type, agg_params, summary
    ):
        return zu_utils.process_classifications(
            project=self.project,
            conn=self.db_connection,
            classifications_data=classifications_data,
            subject_type=subject_type,
            agg_params=agg_params,
            summary=summary,
        )

    def process_annotations(self):
        # code for prepare dataset for machine learning
        pass

    def format_to_gbif(self, agg_df: pd.DataFrame, subject_type: str):
        return zu_utils.format_to_gbif_occurence(
            project=self.project,
            csv_paths=self.csv_paths,
            zoo_info_dict=self.zoo_info,
            df=agg_df,
            classified_by="citizen_scientists",
            subject_type=subject_type,
        )


class MLProjectProcessor(ProjectProcessor):
    def __init__(
        self,
        project_process: ProjectProcessor,
        config_path: str = None,
        weights_path: str = None,
        output_path: str = None,
        classes: list = [],
    ):
        self.__dict__ = project_process.__dict__.copy()
        self.project_name = self.project.Project_name.lower().replace(" ", "_")
        self.data_path = config_path
        self.weights_path = weights_path
        self.output_path = output_path
        self.classes = classes
        self.run_history = None
        self.best_model_path = None
        self.model_type = None
        self.train, self.run, self.test = (None,) * 3

        # Before t6_utils gets loaded in, the val.py file in yolov5_tracker repository needs to be removed
        # to prevent the batch_size error, see issue kso-object-detection #187
        path_to_val = os.path.join(sys.path[0], "yolov5_tracker/val.py")
        try:
            os.remove(path_to_val)
        except OSError:
            pass

        self.modules = g_utils.import_modules([])
        self.modules.update(
            g_utils.import_modules(["torch", "wandb", "yaml", "yolov5"], utils=False)
        )

        self.team_name = "koster"

        model_selected = t_utils.choose_model_type()

        async def f():
            x = await kso_widgets.single_wait_for_change(model_selected, "value")
            self.model_type = x
            self.modules.update(self.load_yolov5_modules())
            if all(["train", "detect", "val"]) in self.modules:
                self.train, self.run, self.test = (
                    self.modules["train"],
                    self.modules["detect"],
                    self.modules["val"],
                )

        asyncio.create_task(f())

    def load_yolov5_modules(self):
        # Model-specific imports
        if self.model_type == 1:
            module_names = ["yolov5.train", "yolov5.detect", "yolov5.val"]
            logging.info("Object detection model loaded")
            return g_utils.import_modules(module_names, utils=False, models=True)
        elif self.model_type == 2:
            logging.info("Image classification model loaded")
            module_names = [
                "yolov5.classify.train",
                "yolov5.classify.predict",
                "yolov5.classify.val",
            ]
            return g_utils.import_modules(module_names, utils=False, models=True)
        elif self.model_type == 3:
            logging.info("Image segmentation model loaded")
            module_names = [
                "yolov5.segment.train",
                "yolov5.segment.predict",
                "yolov5.segment.val",
            ]
            return g_utils.import_modules(module_names, utils=False, models=True)
        else:
            logging.info("Invalid model specification")

    def prepare_dataset(
        self,
        agg_df: pd.DataFrame,
        out_path: str,
        perc_test: float = 0.2,
        img_size: tuple = (224, 224),
        remove_nulls: bool = False,
        track_frames: bool = False,
        n_tracked_frames: int = 0,
    ):
        species_list = kso_widgets.choose_species(self.project)

        button = widgets.Button(
            description="Aggregate frames",
            disabled=False,
            display="flex",
            flex_flow="column",
            align_items="stretch",
            style={"description_width": "initial"},
        )

        def on_button_clicked(b):
            self.species_of_interest = species_list.value
            # code for prepare dataset for machine learning
            yolo_utils.frame_aggregation(
                project=self.project,
                server_connection=self.server_connection,
                db_connection=self.db_connection,
                out_path=out_path,
                perc_test=perc_test,
                class_list=self.species_of_interest,
                img_size=img_size,
                remove_nulls=remove_nulls,
                track_frames=track_frames,
                n_tracked_frames=n_tracked_frames,
                agg_df=agg_df,
            )

        button.on_click(on_button_clicked)
        display(button)

    def choose_entity(self, alt_name: bool = False):
        if self.team_name is None:
            return t_utils.choose_entity()
        else:
            if not alt_name:
                logging.info(
                    f"Found team name: {self.team_name}. If you want"
                    " to use a different team name for this experiment"
                    " set the argument alt_name to True"
                )
            else:
                return t_utils.choose_entity()

    # Function to choose a model to evaluate
    def choose_model(self):
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

        project_name = self.project.Project_name.replace(" ", "_")
        if self.team_name == "wildlife-ai":
            logging.info("Please note: Using models from adi-ohad-heb-uni account.")
            full_path = "adi-ohad-heb-uni/project-wildlife-ai"
            api.runs(path=full_path).objects
        else:
            full_path = f"{self.team_name}/{project_name.lower()}"

        runs = api.runs(full_path)

        for run in runs:
            model_artifacts = [
                artifact
                for artifact in chain(run.logged_artifacts(), run.used_artifacts())
                if artifact.type == "model"
            ]
            if len(model_artifacts) > 0:
                model_dict[run.name] = model_artifacts[0].name.split(":")[0]
                model_info[model_artifacts[0].name.split(":")[0]] = run.summary

        # Add "no movie" option to prevent conflicts
        # models = np.append(list(model_dict.keys()),"No model")

        model_widget = widgets.Dropdown(
            options=[(name, model) for name, model in model_dict.items()],
            description="Select model:",
            ensure_option=False,
            disabled=False,
            layout=widgets.Layout(width="50%"),
            style={"description_width": "initial"},
        )

        main_out = widgets.Output()
        display(model_widget, main_out)

        # Display model metrics
        def on_change(change):
            with main_out:
                clear_output()
                if change["new"] == "No file":
                    logging.info("Choose another file")
                else:
                    if project_name == "model-registry":
                        logging.info("No metrics available")
                    else:
                        logging.info(
                            {
                                k: v
                                for k, v in model_info[change["new"]].items()
                                if "metrics" in k
                            }
                        )

        model_widget.observe(on_change, names="value")
        return model_widget

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
        ssh.connect(
            hostname="80.252.221.46", port=2230, username=user, password=password
        )

        # SCPCLient takes a paramiko transport as its only argument
        scp = SCPClient(ssh.get_transport())
        scp.put(
            f"{artifact_dir}/weights/best.pt",
            f"/home/koster/model_config/weights/ \
                {os.path.basename(project_name)}_{os.path.basename(os.path.dirname(artifact_dir))}_{model_name}",
        )
        scp.close()

    def setup_paths(self):
        if not isinstance(self.output_path, str) and self.output_path is not None:
            self.output_path = self.output_path.selected
        self.data_path, self.hyp_path = yolo_utils.setup_paths(
            self.output_path, self.model_type
        )

    def choose_train_params(self):
        return t_utils.choose_train_params(self.model_type)

    def train_yolov5(
        self, exp_name, weights, epochs=50, batch_size=16, img_size=[720, 540]
    ):
        if self.model_type == 1:
            self.modules["train"].run(
                entity=self.team_name,
                data=self.data_path,
                hyp=self.hyp_path,
                weights=weights,
                project=self.project_name,
                name=exp_name,
                img_size=img_size,
                batch_size=int(batch_size),
                epochs=epochs,
                workers=1,
                single_cls=False,
                cache_images=True,
            )
        elif self.model_type == 2:
            self.modules["train"].run(
                entity=self.team_name,
                data=self.data_path,
                model=weights,
                project=self.project_name,
                name=exp_name,
                img_size=img_size[0],
                batch_size=int(batch_size),
                epochs=epochs,
                workers=1,
            )
        else:
            logging.error("Segmentation model training not yet supported.")

    def eval_yolov5(self, exp_name: str, model_folder: str, conf_thres: float):
        # Find trained model weights
        project_path = str(Path(self.output_path, self.project.Project_name.lower()))
        self.tuned_weights = f"{Path(project_path, model_folder, 'weights', 'best.pt')}"
        try:
            self.modules["val"].run(
                data=self.data_path,
                weights=self.tuned_weights,
                conf_thres=conf_thres,
                imgsz=640 if self.model_type == 1 else 224,
                half=False,
                project=self.project_name,
                name=str(exp_name) + "_val",
            )
        except Exception as e:
            logging.error(f"Encountered {e}, terminating run...")
            self.modules["wandb"].finish()
        logging.info("Run succeeded, finishing run...")
        self.modules["wandb"].finish()

    def detect_yolov5(
        self, source: str, save_dir: str, conf_thres: float, artifact_dir: str
    ):
        self.run = self.modules["wandb"].init(
            entity=self.team_name,
            project="model-evaluations",
            settings=self.modules["wandb"].Settings(start_method="thread"),
        )
        self.modules["detect"].run(
            weights=[
                f
                for f in Path(artifact_dir).iterdir()
                if f.is_file()
                and str(f).endswith((".pt", ".model"))
                and "osnet" not in str(f)
            ][0],
            source=source,
            conf_thres=conf_thres,
            save_txt=True,
            save_conf=True,
            project=save_dir,
            name="detect",
        )

    def save_detections_wandb(self, conf_thres: float, model: str, eval_dir: str):
        yolo_utils.set_config(conf_thres, model, eval_dir)
        yolo_utils.add_data_wandb(eval_dir, "detection_output", self.run)
        self.csv_report = yolo_utils.generate_csv_report(eval_dir, wandb_log=True)
        wandb.finish()

    def track_individuals(
        self,
        source: str,
        artifact_dir: str,
        eval_dir: str,
        conf_thres: float,
        img_size: tuple = (540, 540),
    ):
        latest_tracker = yolo_utils.track_objects(
            source_dir=source,
            artifact_dir=artifact_dir,
            tracker_folder=eval_dir,
            conf_thres=conf_thres,
            img_size=img_size,
            gpu=True if self.modules["torch"].cuda.is_available() else False,
        )
        yolo_utils.add_data_wandb(
            Path(latest_tracker).parent.absolute(), "tracker_output", self.run
        )
        self.csv_report = yolo_utils.generate_csv_report(eval_dir, wandb_log=True)
        self.tracking_report = yolo_utils.generate_counts(
            eval_dir, latest_tracker, artifact_dir, wandb_log=True
        )
        self.modules["wandb"].finish()

    def enhance_yolov5(self, conf_thres: float, img_size=[640, 640]):
        if self.model_type == 1:
            logging.info("Enhancement running...")
            self.modules["detect"].run(
                weights=self.tuned_weights,
                source=str(Path(self.output_path, "images")),
                imgsz=img_size,
                conf_thres=conf_thres,
                save_txt=True,
            )
            self.modules["wandb"].finish()
        elif self.model_type == 2:
            logging.info(
                "Enhancements not supported for image classification models at this time."
            )
        else:
            logging.info(
                "Enhancements not supported for segmentation models at this time."
            )

    def enhance_replace(self, run_folder: str):
        if self.model_type == 1:
            os.rename(f"{self.output_path}/labels", f"{self.output_path}/labels_org")
            os.rename(f"{run_folder}/labels", f"{self.output_path}/labels")
        else:
            logging.error("This option is not supported for other model types.")

    def download_project_runs(self):
        # Download all the runs from the given project ID using Weights and Biases API,
        # sort them by the specified metric, and assign them to the run_history attribute

        self.modules["wandb"].login()
        runs = self.modules["wandb"].Api().runs(f"{self.team_name}/{self.project_name}")
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

    def get_model(self, model_name: str, download_path: str):
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
        if self.team_name == "wildlife-ai":
            logging.info("Please note: Using models from adi-ohad-heb-uni account.")
            full_path = "adi-ohad-heb-uni/project-wildlife-ai"
        else:
            full_path = f"{self.team_name}/{self.project.Project_name.lower()}"
        api = wandb.Api()
        try:
            api.artifact_type(type_name="model", project=full_path).collections()
        except Exception as e:
            logging.error(
                f"No model collections found. No artifacts have been logged. {e}"
            )
            return None
        collections = [
            coll
            for coll in api.artifact_type(
                type_name="model", project=full_path
            ).collections()
        ]
        model = [i for i in collections if i.name == model_name]
        if len(model) > 0:
            model = model[0]
        else:
            logging.error("No model found")
        artifact = api.artifact(full_path + "/" + model.name + ":latest")
        logging.info("Downloading model checkpoint...")
        artifact_dir = artifact.download(root=download_path)
        logging.info("Checkpoint downloaded.")
        return os.path.realpath(artifact_dir)

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

        api = self.modules["wandb"].Api()
        artifact = api.artifact(
            f"{self.team_name}/{self.project_name}"
            + "/"
            + best_model.name.split(":")[0]
            + ":latest"
        )
        logging.info("Downloading model checkpoint...")
        artifact_dir = artifact.download(root=download_path)
        logging.info("Checkpoint downloaded.")
        self.best_model_path = os.path.realpath(artifact_dir)

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

    def get_dataset(self, model: str, team_name: str = "koster"):
        """
        It takes in a project name and a model name, and returns the paths to the train and val datasets

        :param project_name: The name of the project you want to download the dataset from
        :type project_name: str
        :param model: The model you want to use
        :type model: str
        :return: The return value is a list of two directories, one for the training data and one for the validation data.
        """
        api = wandb.Api()
        if "_" in model:
            run_id = model.split("_")[1]
            try:
                run = api.run(
                    f"{team_name}/{self.project.Project_name.lower()}/runs/{run_id}"
                )
            except wandb.CommError:
                logging.error("Run data not found")
                return "empty_string", "empty_string"
            datasets = [
                artifact
                for artifact in run.used_artifacts()
                if artifact.type == "dataset"
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


class Annotator:
    def __init__(self, dataset_name, images_path, potential_labels=None):
        self.dataset_name = dataset_name
        self.images_path = images_path
        self.potential_labels = potential_labels
        self.bboxes = {}
        self.modules = g_utils.import_modules([])
        self.modules.update(g_utils.import_modules(["fiftyone"], utils=False))

    def __repr__(self):
        return repr(self.__dict__)

    def fiftyone_annotate(self):
        # Create a new dataset
        try:
            dataset = self.modules["fiftyone"].load_dataset(self.dataset_name)
            dataset.delete()
        except ValueError:
            pass
        dataset = self.modules["fiftyone"].Dataset(self.dataset_name)

        # Add all the images in the directory to the dataset
        for filename in os.listdir(self.images_path):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(self.images_path, filename)
                sample = self.modules["fiftyone"].Sample(filepath=image_path)
                dataset.add_sample(sample)

        # Add the potential labels to the dataset
        # Set default classes
        if self.potential_labels is not None:
            label_field = "my_label"
            dataset.add_sample_field(
                label_field,
                self.modules["fiftyone"].core.fields.StringField,
                classes=self.potential_labels,
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
        # session = self.modules["fiftyone"].launch_app(dataset, view=view)

        # Start annotating
        # session.wait()

        # Save the annotations
        dataset.save()

    def annotate(self, autolabel_model: str = None):
        return t_utils.get_annotator(
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
        bbox_dict = {}
        annot_path = os.path.join(Path(self.images_path).parent, "labels")
        if len(os.listdir(annot_path)) > 0:
            for label_file in os.listdir(annot_path):
                image = os.path.join(self.images_path, images[0])
                width, height = imagesize.get(image)
                bboxes = []
                bbox_dict[image] = []
                with open(os.path.join(annot_path, label_file), "r") as f:
                    for line in f:
                        s = line.split(" ")
                        left = (float(s[1]) - (float(s[3]) / 2)) * width
                        top = (float(s[2]) - (float(s[4]) / 2)) * height
                        bbox_dict[image].append(
                            {
                                "x": left,
                                "y": top,
                                "width": float(s[3]) * width,
                                "height": float(s[4]) * height,
                                "label": self.potential_labels[int(s[0])],
                            }
                        )
                        bboxes.append(
                            {
                                "x": left,
                                "y": top,
                                "width": float(s[3]) * width,
                                "height": float(s[4]) * height,
                                "label": self.potential_labels[int(s[0])],
                            }
                        )
            self.bboxes = bbox_dict
        else:
            self.bboxes = {}
