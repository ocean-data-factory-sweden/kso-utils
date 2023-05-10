# base imports
import os
import sqlite3
import logging
import pandas as pd
from pathlib import Path

# util imports
import kso_utils.db_starter.schema as schema
import kso_utils.project_utils as project_utils

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


# Utility functions for common database operations
def init_db(db_path: str):
    """Initiate a new database for the project

    :param db_path: path of the database file
    :return:
    """

    # Delete previous database versions if exists
    if os.path.exists(db_path):
        os.remove(db_path)

    # Get sql command for db setup
    sql_setup = schema.sql
    # create a database connection
    conn = create_connection(r"{:s}".format(db_path))

    # create tables
    if conn is not None:
        # execute sql
        execute_sql(conn, sql_setup)
        return "Database creation success"
    else:
        return "Database creation failure"


def create_connection(db_file: str):
    """create a database connection to the SQLite database
        specified by db_file

    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        conn.execute("PRAGMA foreign_keys = 1")
        return conn
    except sqlite3.Error as e:
        logging.error(e)

    return conn


def insert_many(conn: sqlite3.Connection, data: list, table: str, count: int):
    """
    Insert multiple rows into table

    :param conn: the Connection object
    :param data: data to be inserted into table
    :param table: table of interest
    :param count: number of fields
    :return:
    """

    values = (1,) * count
    values = str(values).replace("1", "?")

    cur = conn.cursor()
    cur.executemany(f"INSERT INTO {table} VALUES {values}", data)


def retrieve_query(conn: sqlite3.Connection, query: str):
    """
    Execute SQL query and returns output

    :param conn: the Connection object
    :param query: a SQL query
    :return:
    """
    try:
        cur = conn.cursor()
        cur.execute(query)
    except sqlite3.Error as e:
        logging.error(e)

    rows = cur.fetchall()

    return rows


def execute_sql(conn: sqlite3.Connection, sql: str):
    """Execute multiple SQL statements without return

    :param conn: Connection object
    :param sql: a string of SQL statements
    :return:
    """
    try:
        c = conn.cursor()
        c.executescript(sql)
    except sqlite3.Error as e:
        logging.error(e)


def add_to_table(db_path: str, table_name: str, values: list, num_fields: int):
    """
    This function adds multiple rows of data to a specified table in a SQLite database.
    
    :param db_path: The path to the SQLite database file
    :type db_path: str
    :param table_name: The name of the table in the database where the values will be added
    :type table_name: str
    :param values: The `values` parameter is a list of tuples, where each tuple represents a row of data
    to be inserted into the specified table. The number of values in each tuple should match the
    `num_fields` parameter, which specifies the number of columns in the table
    :type values: list
    :param num_fields: The parameter `num_fields` is an integer that represents the number of fields or
    columns in the table where the values will be inserted. This parameter is used to ensure that the
    correct number of values are being inserted into the table
    :type num_fields: int
    """
    conn = create_connection(db_path)

    try:
        insert_many(
            conn,
            values,
            table_name,
            num_fields,
        )
    except sqlite3.Error as e:
        logging.error(e)

    conn.commit()

    logging.info(f"Updated {table_name}")


def test_table(df: pd.DataFrame, table_name: str, keys: list = ["id"]):
    """
    The function checks if a given DataFrame has any NULL values in the specified key columns and logs
    an error message if it does.
    
    :param df: A pandas DataFrame that represents a table in a database
    :type df: pd.DataFrame
    :param table_name: The name of the table being tested, which is a string
    :type table_name: str
    :param keys: The `keys` parameter is a list of column names that are used as keys to uniquely
    identify each row in the DataFrame `df`. The function `test_table` checks that there are no NULL
    values in these key columns, which would indicate that some rows were not properly matched
    :type keys: list
    """
    try:
        # check that there are no id columns with a NULL value, which means that they were not matched
        assert len(df[df[keys].isnull().any(axis=1)]) == 0
    except AssertionError:
        logging.error(
            f"The table {table_name} has invalid entries, please ensure that all columns are non-zero"
        )
        logging.error(f"The invalid entries are {df[df[keys].isnull().any(axis=1)]}")


def get_id(
    row: int,
    field_name: str,
    table_name: str,
    conn: sqlite3.Connection,
    conditions: dict = {"a": "=b"},
):
    """
    This function retrieves an ID value from a specified table in a SQLite database based on specified
    conditions.
    
    :param row: The row number of the data in the table
    :type row: int
    :param field_name: The name of the field/column from which we want to retrieve data
    :type field_name: str
    :param table_name: The name of the table in the database where the data is stored
    :type table_name: str
    :param conn: The `conn` parameter is a connection object to a SQLite database. It is used to execute
    SQL queries and retrieve data from the database
    :type conn: sqlite3.Connection
    :param conditions: The `conditions` parameter is an optional dictionary that specifies the
    conditions that need to be met in order to retrieve the `id_value` from the specified table. The
    keys of the dictionary represent the column names in the table, and the values represent the
    conditions that need to be met for that column
    :type conditions: dict
    :return: the value of the specified field (field_name) from the specified table (table_name) where
    the specified conditions (conditions) are met. If no value is found, it returns None.
    """
    # Get id from a table where a condition is met

    if isinstance(conditions, dict):
        condition_string = " AND ".join(
            [k + v[0] + f"{v[1:]}" for k, v in conditions.items()]
        )
    else:
        raise ValueError("Conditions should be specified as a dict, e.g. {'a', '=b'}")

    try:
        id_value = retrieve_query(
            conn, f"SELECT {field_name} FROM {table_name} WHERE {condition_string}"
        )[0][0]
    except IndexError:
        id_value = None
    return id_value


def get_column_names_db(db_info_dict: pd.DataFrame, table_i: str):
    """
    > This function returns the "column" names of the sql table of interest

    :param db_info_dict: The dictionary containing the database information
    :param table_i: a string of the name of the table of interest
    :return: A list of column names of the table of interest
    """
    # Connect to the db
    conn = create_connection(db_info_dict["db_path"])

    # Get the data of the table of interest
    data = conn.execute(f"SELECT * FROM {table_i}")

    # Get the names of the columns inside the table of interest
    field_names = [i[0] for i in data.description]

    return field_names


def find_duplicated_clips(conn: sqlite3.Connection):
    """
    This function finds duplicated clips in a database and returns a count of how many times each clip
    has been uploaded.
    
    :param conn: The parameter `conn` is a connection object to a SQLite database
    :type conn: sqlite3.Connection
    :return: a Pandas Series object that contains the count of how many times each duplicated clip has
    been uploaded. The count is grouped by the number of times a clip has been uploaded.
    """
    # Retrieve the information of all the clips uploaded
    subjects_df = pd.read_sql_query(
        "SELECT id, movie_id, clip_start_time, clip_end_time FROM subjects WHERE subject_type='clip'",
        conn,
    )

    # Find clips uploaded more than once
    duplicated_subjects_df = subjects_df[
        subjects_df.duplicated(
            ["movie_id", "clip_start_time", "clip_end_time"], keep=False
        )
    ]

    # Count how many time each clip has been uploaded
    times_uploaded_df = (
        duplicated_subjects_df.groupby(["movie_id", "clip_start_time"], as_index=False)
        .size()
        .to_frame("times")
    )

    return times_uploaded_df["times"].value_counts()


def get_movies_id(df: pd.DataFrame, db_path: str):
    """
    This function retrieves movie IDs based on movie filenames from a database and merges them with a
    given DataFrame.
    
    :param df: A pandas DataFrame containing information about movie filenames and clip subjects
    :type df: pd.DataFrame
    :param db_path: The path to the database file
    :type db_path: str
    :return: a pandas DataFrame with the movie_ids added to the input DataFrame based on matching movie
    filenames with the movies table in a SQLite database. The function drops the movie_filename column
    before returning the DataFrame.
    """
    # Create connection to db
    conn = create_connection(db_path)

    # Query id and filenames from the movies table
    movies_df = pd.read_sql_query("SELECT id, filename FROM movies", conn)
    movies_df = movies_df.rename(
        columns={"id": "movie_id", "filename": "movie_filename"}
    )

    # Check all the movies have a unique ID
    df_unique = df.movie_filename.unique()
    movies_df_unique = movies_df.movie_filename.unique()
    diff_filenames = set(df_unique).difference(movies_df_unique)

    if diff_filenames:
        raise ValueError(
            f"There are clip subjects that don't have movie_id. The movie filenames are {diff_filenames}"
        )

    # Reference the manually uploaded subjects with the movies table
    df = pd.merge(df, movies_df, how="left", on="movie_filename")

    # Drop the movie_filename column
    df = df.drop(columns=["movie_filename"])

    return df


# Project db functions


def get_project_details(project: project_utils.Project):
    """
    > This function connects to the server (or folder) hosting the csv files, and gets the initial info
    from the database

    :param project: the project object
    """

    from kso_utils.server_utils import connect_to_server

    # Connect to the server (or folder) hosting the csv files
    server_i_dict = connect_to_server(project)

    # Get the initial info
    db_initial_info = get_db_init_info(project, server_i_dict)

    return server_i_dict, db_initial_info


def initiate_db(project: project_utils.Project):
    """
    This function takes a project name as input and returns a dictionary with all the information needed
    to connect to the project's database

    :param project: The name of the project. This is used to get the project-specific info from the config file
    :return: A dictionary with the following keys (db_path, project_name, server_i_dict, db_initial_info)

    """

    # Check if template project
    if project.Project_name == "model-registry":
        return {}

    # Get project specific info
    server_i_dict, db_initial_info = get_project_details(project)

    # Check if server and db info
    if len(server_i_dict) == 0 and len(db_initial_info) == 0:
        return {}

    # Initiate the sql db
    init_db(db_initial_info["db_path"])

    # List the csv files of interest
    list_of_init_csv = [
        "local_sites_csv",
        "local_movies_csv",
        "local_photos_csv",
        "local_species_csv",
    ]

    # Populate the sites, movies, photos, info
    for local_i_csv in list_of_init_csv:
        if local_i_csv in db_initial_info.keys():
            populate_db(
                project=project, db_initial_info=db_initial_info, local_csv=local_i_csv
            )

    # Combine server/project info in a dictionary
    db_info_dict = {**db_initial_info, **server_i_dict}

    return db_info_dict


def populate_db(project: project_utils.Project, db_initial_info: dict, local_csv: str):
    """
    > This function populates a sql table of interest based on the info from the respective csv

    :param db_initial_info: The dictionary containing the initial database information
    :param local_csv: a string of the names of the local csv to populate from
    """

    # Process the csv of interest and tests for compatibility with sql table
    csv_i, df = process_test_csv(
        project=project, db_info_dict=db_initial_info, local_csv=local_csv
    )

    # Add values of the processed csv to the sql table of interest
    add_to_table(
        db_initial_info["db_path"],
        csv_i,
        [tuple(i) for i in df.values],
        len(df.columns),
    )


def process_test_csv(
    project: project_utils.Project, db_info_dict: dict, local_csv: str
):
    """
    > This function process a csv of interest and tests for compatibility with the respective sql table of interest

    :param db_info_dict: The dictionary containing the database information
    :param project: The project object
    :param local_csv: a string of the names of the local csv to populate from
    :return: a string of the category of interest and the processed dataframe

    """
    # Load the csv with the information of interest
    df = pd.read_csv(db_info_dict[local_csv])

    # Save the category of interest and process the df
    if "sites" in local_csv:
        field_names, csv_i, df = process_sites_df(project, df)

    if "movies" in local_csv:
        field_names, csv_i, df = process_movies_df(project, db_info_dict, df)

    if "species" in local_csv:
        field_names, csv_i, df = process_species_df(df)

    if "photos" in local_csv:
        field_names, csv_i, df = process_photos_df(project, db_info_dict, df)

    # Add the names of the basic columns in the sql db
    field_names = field_names + get_column_names_db(db_info_dict, csv_i)
    field_names.remove("id")

    # Select relevant fields
    df.rename(columns={"Author": "author"}, inplace=True)
    df = df[[c for c in field_names if c in df.columns]]

    # Roadblock to prevent empty rows in id_columns
    test_table(df, csv_i, [df.columns[0]])

    return csv_i, df


def process_sites_df(
    project: project_utils.Project,
    df: pd.DataFrame,
):
    """
    > This function processes the sites dataframe and returns a string with the category of interest

    :param df: a pandas dataframe of the information of interest
    :return: a string of the category of interest and the processed dataframe
    """

    # Check if the project is the Spyfish Aotearoa
    if project.Project_name == "Spyfish_Aotearoa":
        # Rename columns to match schema fields
        from kso_utils.spyfish_utils import process_spyfish_sites

        df = process_spyfish_sites(df)

    # Specify the category of interest
    csv_i = "sites"

    # Specify the id of the df of interest
    field_names = ["site_id"]

    return field_names, csv_i, df


def process_movies_df(
    project: project_utils.Project, db_info_dict: dict, df: pd.DataFrame
):
    """
    > This function processes the movies dataframe and returns a string with the category of interest

    :param db_info_dict: The dictionary containing the database information
    :param df: a pandas dataframe of the information of interest
    :return: a string of the category of interest and the processed dataframe
    """

    # Check if the project is the Spyfish Aotearoa
    if project.Project_name == "Spyfish_Aotearoa":
        from kso_utils.spyfish_utils import process_spyfish_movies

        df = process_spyfish_movies(df)

    # Check if the project is the KSO
    if project.Project_name == "Koster_Seafloor_Obs":
        from kso_utils.koster_utils import process_koster_movies_csv

        df = process_koster_movies_csv(df)

    # Connect to database
    conn = create_connection(db_info_dict["db_path"])

    # Reference movies with their respective sites
    sites_df = pd.read_sql_query("SELECT id, siteName FROM sites", conn)
    sites_df = sites_df.rename(columns={"id": "site_id"})

    # Merge movies and sites dfs
    df = pd.merge(df, sites_df, how="left", on="siteName")

    # Select only those fields of interest
    if "fpath" not in df.columns:
        df["fpath"] = df["filename"]

    # Specify the category of interest
    csv_i = "movies"

    # Specify the id of the df of interest
    field_names = ["movie_id"]

    return field_names, csv_i, df


def process_photos_df(
    project: project_utils.Project, db_info_dict: dict, df: pd.DataFrame
):
    """
    > This function processes the photos dataframe and returns a string with the category of interest

    :param db_info_dict: The dictionary containing the database information
    :param df: a pandas dataframe of the information of interest
    :param project: The project object
    :return: a string of the category of interest and the processed dataframe
    """
    # Check if the project is the SGU
    if project.Project_name == "SGU":
        from kso_utils.sgu_utils import process_sgu_photos_csv

        df = process_sgu_photos_csv(db_info_dict)

    # Specify the category of interest
    csv_i = "photos"

    # Specify the id of the df of interest
    field_names = ["ID"]

    return field_names, csv_i, df


def process_species_df(df: pd.DataFrame):
    """
    > This function processes the species dataframe and returns a string with the category of interest

    :param df: a pandas dataframe of the information of interest
    :param project: The project object
    :return: a string of the category of interest and the processed dataframe
    """

    # Rename columns to match sql fields
    df = df.rename(columns={"commonName": "label"})

    # Specify the category of interest
    csv_i = "species"

    # Specify the id of the df of interest
    field_names = ["species_id"]

    return field_names, csv_i, df


def get_db_init_info(project: project_utils.Project, server_dict: dict) -> dict:
    """
    This function downloads the csv files from the server and returns a dictionary with the paths to the
    csv files

    :param project: the project object
    :param server_dict: a dictionary containing the server information
    :type server_dict: dict
    :return: A dictionary with the paths to the csv files with the initial info to build the db.
    """

    from kso_utils.server_utils import download_csv_aws, download_gdrive

    # Define the path to the csv files with initial info to build the db
    db_csv_info = project.csv_folder

    # Get project-specific server info
    server = project.server
    project_name = project.Project_name

    # Create the folder to store the csv files if not exist
    if not os.path.exists(db_csv_info):
        Path(db_csv_info).mkdir(parents=True, exist_ok=True)
        # Recursively add permissions to folders created
        [os.chmod(root, 0o777) for root, dirs, files in os.walk(db_csv_info)]

    if server == "AWS":
        # Download csv files from AWS
        db_initial_info = download_csv_aws(project_name, server_dict, db_csv_info)

        if project_name == "Spyfish_Aotearoa":
            from kso_utils.spyfish_utils import get_spyfish_choices

            db_initial_info = get_spyfish_choices(
                server_dict, db_initial_info, db_csv_info
            )

    elif server in ["LOCAL", "SNIC"]:
        csv_folder = db_csv_info

        # Define the path to the csv files with initial info to build the db
        if not os.path.exists(csv_folder):
            logging.error(
                "Invalid csv folder specified, please provide the path to the species, sites and movies (optional)"
            )

        for file in Path(csv_folder).rglob("*.csv"):
            if "sites" in file.name:
                sites_csv = file
            if "movies" in file.name:
                movies_csv = file
            if "photos" in file.name:
                photos_csv = file
            if "survey" in file.name:
                surveys_csv = file
            if "species" in file.name:
                species_csv = file

        if (
            "movies_csv" not in vars()
            and "photos_csv" not in vars()
            and os.path.exists(csv_folder)
        ):
            logging.info(
                "No movies or photos found, an empty movies file will be created."
            )
            with open(str(Path(f"{csv_folder}", "movies.csv")), "w") as fp:
                fp.close()

        db_initial_info = {}

        if "sites_csv" in vars():
            db_initial_info["local_sites_csv"] = sites_csv

        if "species_csv" in vars():
            db_initial_info["local_species_csv"] = species_csv

        if "movies_csv" in vars():
            db_initial_info["local_movies_csv"] = movies_csv

        if "photos_csv" in vars():
            db_initial_info["local_photos_csv"] = photos_csv

        if "surveys_csv" in vars():
            db_initial_info["local_surveys_csv"] = surveys_csv

        if len(db_initial_info) == 0:
            logging.error(
                "Insufficient information to build the database. Please fix the path to csv files."
            )

    elif server == "TEMPLATE":
        # Specify the id of the folder with csv files of the template project
        gdrive_id = "1PZGRoSY_UpyLfMhRphMUMwDXw4yx1_Fn"

        # Download template csv files from Gdrive
        db_initial_info = download_gdrive(gdrive_id, db_csv_info)

        for file in Path(db_csv_info).rglob("*.csv"):
            if "sites" in file.name:
                sites_csv = file
            if "movies" in file.name:
                movies_csv = file
            if "photos" in file.name:
                photos_csv = file
            if "survey" in file.name:
                surveys_csv = file
            if "species" in file.name:
                species_csv = file

        db_initial_info = {}

        if "sites_csv" in vars():
            db_initial_info["local_sites_csv"] = sites_csv

        if "species_csv" in vars():
            db_initial_info["local_species_csv"] = species_csv

        if "movies_csv" in vars():
            db_initial_info["local_movies_csv"] = movies_csv

        if "photos_csv" in vars():
            db_initial_info["local_photos_csv"] = photos_csv

        if "surveys_csv" in vars():
            db_initial_info["local_surveys_csv"] = surveys_csv

        if len(db_initial_info) == 0:
            logging.error(
                "Insufficient information to build the database. Please fix the path to csv files."
            )

    else:
        raise ValueError(
            "The server type you have chosen is not currently supported. Supported values are AWS, SNIC and LOCAL."
        )

    # Add project-specific db_path
    db_initial_info["db_path"] = project.db_path
    if "client" in server_dict:
        db_initial_info["client"] = server_dict["client"]

    return db_initial_info


def get_col_names(project: project_utils.Project, local_csv: str):
    """Return a dictionary with the project-specific column names of a csv of interest
    This function helps matching the schema format without modifying the column names of the original csv.

    :param project: The project object
    :param local_csv: a string of the name of the local csv of interest
    :return: a dictionary with the names of the columns
    """

    # Get project-specific server info
    project_name = project.Project_name

    if "sites" in local_csv:
        # Get spyfish specific column names
        if project_name == "Spyfish_Aotearoa":
            from kso_utils.spyfish_utils import get_spyfish_col_names

            col_names_sites = get_spyfish_col_names("sites")

        else:
            # Save the column names of interest in a dict
            col_names_sites = {
                "siteName": "siteName",
                "decimalLatitude": "decimalLatitude",
                "decimalLongitude": "decimalLongitude",
                "geodeticDatum": "geodeticDatum",
                "countryCode": "countryCode",
            }

        return col_names_sites

    if "movies" in local_csv:
        # Get spyfish specific column names
        if project_name == "Spyfish_Aotearoa":
            col_names_movies = get_spyfish_col_names("movies")

        elif project_name == "Koster_Seafloor_Obs":
            # Save the column names of interest in a dict
            col_names_movies = {
                "filename": "filename",
                "created_on": "created_on",
                "fps": "fps",
                "duration": "duration",
                "sampling_start": "SamplingStart",
                "sampling_end": "SamplingEnd",
                "author": "Author",
                "site_id": "site_id",
                "fpath": "fpath",
            }

        else:
            # Save the column names of interest in a dict
            col_names_movies = {
                "filename": "filename",
                "created_on": "created_on",
                "fps": "fps",
                "duration": "duration",
                "sampling_start": "sampling_start",
                "sampling_end": "sampling_end",
                "author": "author",
                "site_id": "site_id",
                "fpath": "fpath",
            }

        return col_names_movies

    if "species" in local_csv:
        # Save the column names of interest in a dict
        col_names_species = {
            "label": "label",
            "scientificName": "scientificName",
            "taxonRank": "taxonRank",
            "kingdom": "kingdom",
        }
        return col_names_species

    else:
        raise ValueError("The local csv doesn't have a table match in the schema")


def check_species_meta(project: project_utils.Project):
    """
    > The function `check_species_meta` loads the csv with species information and checks if it is empty

    :param db_info_dict: a dictionary with the following keys:
    :param project: The project name
    """
    # Load the csv with movies information
    species_df = pd.read_csv(project.db_info["local_species_csv"])

    # Retrieve the names of the basic columns in the sql db
    conn = create_connection(project.db_info["db_path"])
    data = conn.execute("SELECT * FROM species")
    field_names = [i[0] for i in data.description]

    # Select the basic fields for the db check
    df_to_db = species_df[[c for c in species_df.columns if c in field_names]]

    # Roadblock to prevent empty lat/long/datum/countrycode
    test_table(df_to_db, "species", df_to_db.columns)

    logging.info("The species dataframe is complete")
