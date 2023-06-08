# base imports
import os
import sqlite3
import logging
import pandas as pd
from pathlib import Path

# util imports
import kso_utils.db_starter.schema as schema
from kso_utils.project_utils import Project


# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


# SQL specific functions
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


def drop_table(conn: sqlite3.Connection, table_name: str):
    """
    Safely remove a table from a Sql db

    :param conn: the Connection object
    :param table_name: table of interest
    """
    # Creating a cursor object using the cursor() method
    cursor = conn.cursor()

    try:
        cursor.execute(f"DELETE FROM {table_name}")
    except Exception as e:
        logging.info(f"Table doesn't exist, {e}")
        return
    logging.info(f"Table contents {table_name} cleared. ")

    # Commit your changes in the database
    conn.commit()


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


def add_to_table(
    conn: sqlite3.Connection, table_name: str, values: list, num_fields: int
):
    """
    This function adds multiple rows of data to a specified table in a SQLite database.

    :param conn: SQL connection object
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


def get_df_from_db_table(conn: sqlite3.Connection, table_name: str):
    """
    This function connects to a specific table from the sql database
    and returns it as a pd DataFrame.

    :param conn: SQL connection object
    :param table_name: The name of the table you want to get from the database
    :return: A dataframe
    """

    if conn is not None:
        cursor = conn.cursor()
    else:
        return
    # Get column names
    cursor.execute(f"SELECT * FROM {table_name}")
    rows = cursor.fetchall()

    # Get column names
    cursor.execute(f"PRAGMA table_info('{table_name}')")
    columns = [col[1] for col in cursor.fetchall()]

    # Create a DataFrame from the data
    df = pd.DataFrame(rows, columns=columns)

    return df


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


def get_column_names_db(conn: sqlite3.Connection, table_i: str):
    """
    > This function returns the "column" names of the sql table of interest

    :param conn: SQL connection object
    :param table_i: a string of the name of the table of interest
    :return: A list of column names of the table of interest
    """

    # Get the data of the table of interest
    data = conn.execute(f"SELECT * FROM {table_i}")

    # Get the names of the columns inside the table of interest
    field_names = [i[0] for i in data.description]

    return field_names


# Utility functions for common database operations
def create_db(db_path: str):
    """Create a new database for the project

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


def populate_db(
    conn: sqlite3.Connection, project: Project, local_df: pd.DataFrame, init_key=str
):
    """
    > This function processes and tests the initial csv files compatibility with sql db
    and populates the table of interest

    :param conn: SQL connection object
    :param project: The project object
    :param local_df: a dataframe with the information of the local csv to populate from
    :param init_key: a string of the initial key of the local csv and the name of the db table to populate
    """

    # Process the csv of interest and tests for compatibility with sql table
    local_df = process_test_csv(
        conn=conn, project=project, local_df=local_df, init_key=init_key
    )

    # Only populate the tables if df is not empty
    if not local_df.empty:
        # Add values of the processed csv to the sql table of interest
        add_to_table(
            conn=conn,
            table_name=init_key,
            values=[tuple(i) for i in local_df.values],
            num_fields=len(local_df.columns),
        )


def process_test_csv(
    conn: sqlite3.Connection, project: Project, local_df: pd.DataFrame, init_key=str
):
    """
    > This function process a csv of interest and tests for compatibility with the
    respective sql table of interest

    :param conn: SQL connection object
    :param project: The project object
    :param local_df: a dataframe with the information of the local csv to populate from
    :param init_key: a string corresponding to the name of the initial key of the local csv
    :return: a string of the category of interest and the processed dataframe
    """
    from kso_utils.spyfish_utils import process_spyfish_sites, process_spyfish_movies
    from kso_utils.koster_utils import process_koster_movies_csv
    from kso_utils.sgu_utils import process_sgu_photos_csv

    # Save the category of interest and process the df
    if init_key == "sites":
        # Specify the id of the df of interest
        field_names = ["site_id"]

        # Modify the local_df if Spyfish Aotearoa
        if project.Project_name == "Spyfish_Aotearoa":
            # Rename columns to match schema fields
            local_df = process_spyfish_sites(local_df)

    elif init_key == "movies":
        # Specify the id of the df of interest
        field_names = ["movie_id"]

        # Modify the local_df if Spyfish Aotearoa
        if project.Project_name in ["Spyfish_Aotearoa", "Spyfish_BOPRC"]:
            local_df = process_spyfish_movies(local_df)

        # Modify the local_df if Koster
        if project.Project_name == "Koster_Seafloor_Obs":
            local_df = process_koster_movies_csv(local_df)

        # Reference movies with their respective sites
        sites_df = get_df_from_db_table(conn, "sites")[["id", "siteName"]].rename(
            columns={"id": "site_id"}
        )

        # Merge local (aka movies) and sites dfs
        local_df = pd.merge(local_df, sites_df, how="left", on="siteName")

        # Prevent column_name error
        if "author" not in local_df.columns:
            local_df = local_df.rename(columns={"Author": "author"})

        # Use the filename to set the fpath if it doesn't exist
        if "fpath" not in local_df.columns:
            local_df["fpath"] = local_df["filename"]

    elif init_key == "species":
        # Specify the id of the df of interest
        field_names = ["species_id"]

        # Rename columns to match sql fields
        local_df = local_df.rename(columns={"commonName": "label"})

    elif init_key == "photos":
        # Specify the id of the df of interest
        field_names = ["ID"]

        # Check if the project is the SGU
        if project.Project_name == "SGU":
            # Process the local photos df
            local_df = process_sgu_photos_csv(project)

    else:
        logging.error(
            f"{init_key} has not been processed because the db schema does not have a table for it"
        )

        # create an Empty DataFrame object
        local_df = pd.DataFrame()

        return local_df

    # Rename any columns that start with a capitalise letter to match db schema
    cols = local_df.columns
    local_df.columns = [word[0].lower() + word[1:] for word in cols]

    # Add the names of the basic columns in the sql db
    field_names = field_names + get_column_names_db(conn, init_key)
    field_names.remove("id")

    # Select relevant fields
    local_df = local_df[[c for c in field_names if c in local_df.columns]]

    # Roadblock to prevent empty rows in id_columns
    test_table(local_df, init_key, [local_df.columns[0]])

    return local_df


def check_species_meta(
    csv_paths: dict,
    db_connection: sqlite3.Connection,
):
    """
    > The function `check_species_meta` loads the csv with species information and checks if it is empty

    :param csv_paths: a dictionary with the paths of the csv files with info to initiate the db
    :param db_connection: SQL connection object
    """

    # Load the csv with movies information
    species_df = pd.read_csv(csv_paths["local_species_csv"])

    # Retrieve the names of the basic columns in the sql db
    field_names = get_column_names_db(db_connection, "species")

    # Select the basic fields for the db check
    df_to_db = species_df[[c for c in species_df.columns if c in field_names]]

    # Roadblock to prevent empty lat/long/datum/countrycode
    test_table(df_to_db, "species", df_to_db.columns)

    logging.info("The species dataframe is complete")
