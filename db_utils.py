import os, sys
import sqlite3
import db_starter.schema as schema
import requests
import pandas as pd
import numpy as np

#import csv, json, sys, io
#import operator
#from datetime import datetime

import utils.koster_utils as koster_utils
import utils.spyfish_utils as spyfish_utils
import utils.movie_utils as movie_utils


# Utility functions for common database operations

# Initiate the database
def init_db(db_path: str):
    
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
    
def create_connection(db_file):
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
        print(e)

    return conn


def insert_many(conn, data, table, count):
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


def retrieve_query(conn, query):
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
        print(e)

    rows = cur.fetchall()

    return rows


def execute_sql(conn, sql):
    """Execute multiple SQL statements without return
    :param conn: Connection object
    :param sql: a string of SQL statements
    :return:
    """
    try:
        c = conn.cursor()
        c.executescript(sql)
    except sqlite3.Error as e:
        print(e)


def add_to_table(db_path, table_name, values, num_fields):

    conn = create_connection(db_path)

    try:
        insert_many(
            conn,
            values,
            table_name,
            num_fields,
        )
    except sqlite3.Error as e:
        print(e)

    conn.commit()

    print(f"Updated {table_name}")


def test_table(df, table_name, keys=["id"]):
    try:
        # check that there are no id columns with a NULL value, which means that they were not matched
        assert len(df[df[keys].isnull().any(axis=1)]) == 0
    except AssertionError:
        print(
            f"The table {table_name} has invalid entries, please ensure that all columns are non-zero"
        )
        print(
            f"The invalid entries are {df[df[keys].isnull().any(axis=1)]}"
        )


def get_id(row, field_name, table_name, conn, conditions={"a": "=b"}):

    # Get id from a table where a condition is met

    if isinstance(conditions, dict):
        condition_string = f" AND ".join(
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


def find_duplicated_clips(conn):

    # Retrieve the information of all the clips uploaded
    subjects_df = pd.read_sql_query(
        f"SELECT id, movie_id, clip_start_time, clip_end_time FROM subjects WHERE subject_type='clip'",
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

### Populate sites, movies and species

def add_sites(sites_csv, db_path):

    # Load the csv with sites information
    sites_df = pd.read_csv(sites_csv)
    
    
    # Select relevant fields
    sites_df = sites_df[
        ["site_id", "siteName", "decimalLatitude", "decimalLongitude", "geodeticDatum", "countryCode"]
    ]
    
    # Roadblock to prevent empty lat/long/datum/countrycode
    test_table(
        sites_df, "sites", sites_df.columns
    )

    # Add values to sites table
    add_to_table(
        db_path, "sites", [tuple(i) for i in sites_df.values], 6
    )

    
def add_movies(movies_csv, project_name, db_path):

    # Load the csv with movies information
    movies_df = pd.read_csv(movies_csv)
    
    # Check if the project is the Spyfish Aotearoa
    if project_name == "Spyfish_Aotearoa":
        # Specify the key (path in S3 of the object)
        movies_df["Fpath"] = movies_df["prefix"] + "/" + movies_df["filename"]
        
        # Remove extension from the filename to match the subject metadata from Zoo
        movies_df["filename"] = movies_df["filename"].str.split('.',1).str[0]
            
    # Check if the project is the KSO
    if project_name == "Koster_Seafloor_Obs":
        movies_df = koster_utils.process_koster_movies_csv(movies_df)
    
    # Connect to database
    conn = create_connection(db_path)
    
    # Reference movies with their respective sites
    sites_df = pd.read_sql_query("SELECT id, siteName FROM sites", conn)
    sites_df = sites_df.rename(columns={"id": "Site_id"})

    # Merge movies and sites dfs
    movies_df = pd.merge(
        movies_df, sites_df, how="left", on="siteName"
    )
    
    # Select only those fields of interest
    movies_db = movies_df[
        ["movie_id", "filename", "created_on", "fps", "duration", "survey_start", "survey_end", "Author", "Site_id", "Fpath"]
    ]

    # Roadblock to prevent empty information
    test_table(
        movies_db, "movies", movies_db.columns
    )
    
    # Add values to movies table
    add_to_table(
        db_path, "movies", [tuple(i) for i in movies_db.values], 10
    )


def add_species(species_csv, db_path):

    # Load the csv with species information
    species_df = pd.read_csv(species_csv)
    
    # Select relevant fields
    species_df = species_df[
        ["species_id", "commonName", "scientificName", "taxonRank", "kingdom"]
    ]
    
    # Roadblock to prevent empty information
    test_table(
        species_df, "species", species_df.columns
    )
    
    # Add values to species table
    add_to_table(
        db_path, "species", [tuple(i) for i in species_df.values], 5
    )
    

