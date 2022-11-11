# base imports
import os
import sqlite3
import logging
import pandas as pd

# util imports
import kso_utils.db_starter.schema as schema
import kso_utils.koster_utils as koster_utils
import kso_utils.spyfish_utils as spyfish_utils
import kso_utils.sgu_utils as sgu_utils
import kso_utils.project_utils as project_utils

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# Utility functions for common database operations


def init_db(db_path: str):
    """Initiate a new database for the project
    :param db_path: path of the database file
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


def process_test_csv(
    db_info_dict: dict, project: project_utils.Project, local_csv: str
):
    """
    > This function process a csv of interest and tests for compatibility with the respective sql table of interest

    :param db_info_dict: The dictionary containing the database information
    :param project: The project object
    :param local_csv: a string of the names of the local csv to populate from
    :return a string of the category of interest and the processed dataframe
    """
    # Load the csv with the information of interest
    df = pd.read_csv(db_info_dict[local_csv])

    # Save the category of interest and process the df
    if "sites" in local_csv:
        field_names, csv_i, df = process_sites_df(db_info_dict, df, project)

    if "movies" in local_csv:
        field_names, csv_i, df = process_movies_df(db_info_dict, df, project)

    if "species" in local_csv:
        field_names, csv_i, df = process_species_df(db_info_dict, df, project)

    if "photos" in local_csv:
        field_names, csv_i, df = process_photos_df(db_info_dict, df, project)

    # Add the names of the basic columns in the sql db
    field_names = field_names + get_column_names_db(db_info_dict, csv_i)
    field_names.remove("id")

    # Select relevant fields
    df.rename(columns={"Author": "author"}, inplace=True)
    df = df[[c for c in field_names if c in df.columns]]

    # Roadblock to prevent empty rows
    test_table(df, csv_i, df.columns)

    return csv_i, df


def populate_db(db_initial_info: dict, project: project_utils.Project, local_csv: str):
    """
    > This function populates a sql table of interest based on the info from the respective csv

    :param db_initial_info: The dictionary containing the initial database information
    :param project: The project object
    :param local_csv: a string of the names of the local csv to populate from
    """

    # Process the csv of interest and tests for compatibility with sql table
    csv_i, df = process_test_csv(
        db_info_dict=db_initial_info, project=project, local_csv=local_csv
    )

    # Add values of the processed csv to the sql table of interest
    add_to_table(
        db_initial_info["db_path"],
        csv_i,
        [tuple(i) for i in df.values],
        len(df.columns),
    )


def process_sites_df(
    db_info_dict: dict, df: pd.DataFrame, project: project_utils.Project
):
    """
    > This function processes the sites dataframe and returns a string with the category of interest

    :param db_info_dict: The dictionary containing the database information
    :param df: a pandas dataframe of the information of interest
    :param project: The project object
    :return: a string of the category of interest and the processed dataframe
    """

    # Check if the project is the Spyfish Aotearoa
    if project.Project_name == "Spyfish_Aotearoa":
        # Rename columns to match schema fields
        df = spyfish_utils.process_spyfish_sites(df)

    # Specify the category of interest
    csv_i = "sites"

    # Specify the id of the df of interest
    field_names = ["site_id"]

    return field_names, csv_i, df


def process_movies_df(
    db_info_dict: dict, df: pd.DataFrame, project: project_utils.Project
):
    """
    > This function processes the movies dataframe and returns a string with the category of interest

    :param db_info_dict: The dictionary containing the database information
    :param df: a pandas dataframe of the information of interest
    :param project: The project object
    :return: a string of the category of interest and the processed dataframe
    """

    # Check if the project is the Spyfish Aotearoa
    if project.Project_name == "Spyfish_Aotearoa":
        df = spyfish_utils.process_spyfish_movies(df)

    # Check if the project is the KSO
    if project.Project_name == "Koster_Seafloor_Obs":
        df = koster_utils.process_koster_movies_csv(df)

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
    db_info_dict: dict, df: pd.DataFrame, project: project_utils.Project
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
        df = sgu_utils.process_sgu_photos_csv(db_info_dict)

    # Specify the category of interest
    csv_i = "photos"

    # Specify the id of the df of interest
    field_names = ["ID"]

    return field_names, csv_i, df


def process_species_df(
    db_info_dict: dict, df: pd.DataFrame, project: project_utils.Project
):
    """
    > This function processes the species dataframe and returns a string with the category of interest

    :param db_info_dict: The dictionary containing the database information
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


def find_duplicated_clips(conn: sqlite3.Connection):

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
