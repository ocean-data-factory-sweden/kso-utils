import os, io
import os
import requests
import pandas as pd
import numpy as np
import getpass
import gdown
import zipfile
import boto3
import paramiko
from paramiko import SSHClient
from scp import SCPClient

import utils.tutorials_utils as tutorials_utils
from tqdm import tqdm
from pathlib import Path

# Common utility functions to connect to external servers (AWS, GDrive,...)

def connect_to_server(project_name):
    
    # Get project-specific server info
    server = tutorials_utils.get_project_info(project_name, "server")
    
    # Create an empty dictionary to host the server connections
    server_dict = {}
    
    if server=="AWS":
        # Set aws account credentials
        aws_access_key_id, aws_secret_access_key = aws_credentials()
        
        # Connect to S3
        client = connect_s3(aws_access_key_id, aws_secret_access_key)
        
    if server=="SNIC":
        # Set SNIC credentials
        snic_user, snic_pass = snic_credentials()
        
        # Connect to SNIC
        client = connect_snic(snic_user, snic_pass)
        
    server_dict["client"] = client
        
    return server_dict


def download_csv_from_google_drive(file_url):

    # Download the csv files stored in Google Drive with initial information about
    # the movies and the species

    file_id = file_url.split("/")[-2]
    dwn_url = "https://drive.google.com/uc?export=download&id=" + file_id
    url = requests.get(dwn_url).text.encode("ISO-8859-1").decode()
    csv_raw = io.StringIO(url)
    dfs = pd.read_csv(csv_raw)
    return dfs


def download_init_csv(gdrive_id, db_csv_info):
    
    # Specify the url of the file to download
    url_input = "https://drive.google.com/uc?id=" + str(gdrive_id)
    
    print("Retrieving the file from ", url_input)
    
    # Specify the output of the file
    zip_file = 'db_csv_info.zip'
    
    # Download the zip file
    gdown.download(url_input, zip_file, quiet=False)
    
    # Unzip the folder with the csv files
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(os.path.dirname(db_csv_info))
        
        
    # Remove the zipped file
    os.remove(zip_file)
    
    
def get_db_init_info(project_name, server_dict):
    
    # Define the path to the csv files with initial info to build the db
    db_csv_info = "../db_starter/db_csv_info/"
        
    # Get project-specific server info
    server = tutorials_utils.get_project_info(project_name, "server")
    
    if server == "AWS":
        
        # Names of csv files from AWS to download
        sites_csv = "sites_buv_doc.csv"
        movies_csv = "movies_buv_doc.csv"
        species_csv = "species_buv_doc.csv"
        
        # Create the folder to store the csv files if not exist
        if not os.path.exists(db_csv_info):
            os.mkdir(db_csv_info)
            
        download_object_from_s3(server_dict["client"],
                                bucket='marine-buv',
                                key="init_db_doc_buv/"+sites_csv, 
                                filename=db_csv_info+sites_csv)
        download_object_from_s3(server_dict["client"],
                                bucket='marine-buv',
                                key="init_db_doc_buv/"+movies_csv, 
                                filename=db_csv_info+movies_csv)
        download_object_from_s3(server_dict["client"],
                                bucket='marine-buv',
                                key="init_db_doc_buv/"+species_csv, 
                                filename=db_csv_info+species_csv)
        
        
        db_initial_info = {
            "sites_csv": db_csv_info+sites_csv, 
            "movies_csv": db_csv_info+movies_csv, 
            "species_csv": db_csv_info+species_csv
        }
        
                
    if project_name == "Koster_Seafloor_Obs":
        # Check if the directory db_csv_info exists
        if not os.path.exists(db_csv_info) or len(os.listdir(db_csv_info)) == 0:

            print("There is no folder with initial information about the sites, movies and species.\n Please enter the ID of a Google Drive zipped folder with the inital database information. \n For example, the ID of the template information is: 1PZGRoSY_UpyLfMhRphMUMwDXw4yx1_Fn")

            # Provide ID of the GDrive zipped folder with the init. database information
            gdrive_id = getpass.getpass('ID of Google Drive zipped folder')

            # Download the csv files
            download_init_csv(gdrive_id, db_csv_info)


        # Define the path to the csv files with inital info to build the db
        for file in Path(db_csv_info).rglob("*.csv"):
            if 'sites' in file.name:
                sites_csv = file
            if 'movies' in file.name:
                movies_csv = file
            if 'species' in file.name:
                species_csv = file
            
        db_initial_info = {
            "sites_csv": sites_csv, 
            "movies_csv": movies_csv, 
            "species_csv": species_csv
        }
        
           
    
    return db_initial_info


def update_db_init_info(project_name, csv_to_update):
    
    if project_name == "Spyfish_Aotearoa":
            
        # Start AWS session
        aws_access_key_id, aws_secret_access_key = server_utils.aws_credentials()
        client = server_utils.connect_s3(aws_access_key_id, aws_secret_access_key)


        csv_filename=csv_to_update.name

        upload_file_to_s3(client,
                              bucket='marine-buv',
                              key="init_db_doc_buv/"+csv_filename,
                              filename=str(csv_to_update))
            
            
def snic_credentials():
    # Save your access key for the SNIC server. 
    snic_user = getpass.getpass('Enter your username for SNIC server')
    snic_pass = getpass.getpass('Enter your password for SNIC server')
    
    return snic_user, snic_pass


def connect_snic(snic_user: str, snic_pass: str):
    # Connect to the SNIC server and return SSH client
    client = SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy()) 
    client.load_system_host_keys()
    client.connect(hostname="129.16.125.130", 
                port = 22,
                username=snic_user,
                password=snic_pass)
    return client

            
def aws_credentials():
    # Save your access key for the s3 bucket. 
    aws_access_key_id = getpass.getpass('Enter the key id for the aws server')
    aws_secret_access_key = getpass.getpass('Enter the secret access key for the aws server')
    
    return aws_access_key_id,aws_secret_access_key


def connect_s3(aws_access_key_id, aws_secret_access_key):
    # Connect to the s3 bucket
    client = boto3.client('s3',
                          aws_access_key_id = aws_access_key_id, 
                          aws_secret_access_key = aws_secret_access_key)
    return client
    

def get_matching_s3_objects(client, bucket, prefix="", suffix=""):
    """
    ## Code modified from alexwlchan (https://alexwlchan.net/2019/07/listing-s3-keys/)
    Generate objects in an S3 bucket.

    :param client: S3 client.
    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch objects whose key starts with
        this prefix (optional).
    :param suffix: Only fetch objects whose keys end with
        this suffix (optional).
    """
    
    paginator = client.get_paginator("list_objects_v2")

    kwargs = {'Bucket': bucket}

    # We can pass the prefix directly to the S3 API.  If the user has passed
    # a tuple or list of prefixes, we go through them one by one.
    if isinstance(prefix, str):
        prefixes = (prefix, )
    else:
        prefixes = prefix

    for key_prefix in prefixes:
        kwargs["Prefix"] = key_prefix

        for page in paginator.paginate(**kwargs):
            try:
                contents = page["Contents"]
            except KeyError:
                break

            for obj in contents:
                key = obj["Key"]
                if key.endswith(suffix):
                    yield obj


def get_matching_s3_keys(client, bucket, prefix="", suffix=""):
    """
    ## Code from alexwlchan (https://alexwlchan.net/2019/07/listing-s3-keys/)
    Generate the keys in an S3 bucket.

    :param client: S3 client.
    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    :param suffix: Only fetch keys that end with this suffix (optional).
    """
    
    # Select the relevant bucket
    s3_keys = [obj["Key"] for obj in get_matching_s3_objects(client, bucket, prefix, suffix)]

    # Set the contents as pandas dataframe
    contents_s3_pd = pd.DataFrame(s3_keys, columns = ["Key"])
    
    return contents_s3_pd

def get_koster_movies(client):
    """ 
    Get list of movies from SNIC server using ssh client.
    
    :param client: SSH client (paramiko)
    """
    stdin, stdout, stderr = client.exec_command("ls /cephyr/NOBACKUP/groups/snic2021-6-9/koster_movies/")
    snic_df = pd.DataFrame(stdout.read().decode("utf-8").split('\n'), columns=['spath'])
    return snic_df


# def retrieve_s3_buckets_info(client, bucket, suffix):
    
#     # Select the relevant bucket
#     s3_keys = [obj["Key"] for obj in get_matching_s3_objects(client=client, bucket=bucket, suffix=suffix)]

#     # Set the contents as pandas dataframe
#     contents_s3_pd = pd.DataFrame(s3_keys)
    
#     return contents_s3_pd

    

def check_movies_from_server(movies_df, sites_df, server_i):
    if server_i=="AWS":
        # Set aws account credentials
        aws_access_key_id, aws_secret_access_key = aws_credentials()
        
        # Connect to S3
        client = connect_s3(aws_access_key_id, aws_secret_access_key)
        
        # 
        check_spyfish_movies(movies_df, client)
        
    # Find out files missing from the Server
    missing_from_server = missing_info[missing_info["_merge"]=="right_only"]
    missing_bad_deployment = missing_from_server[missing_from_server["IsBadDeployment"]]
    missing_no_bucket_info = missing_from_server[~(missing_from_server["IsBadDeployment"])&(missing_from_server["bucket"].isna())]
    
    print("There are", len(missing_from_server.index), "movies missing from", server_i)
    print(len(missing_bad_deployment.index), "movies are bad deployments. Their filenames are:")
    print(*missing_bad_deployment.filename.unique(), sep = "\n")
    print(len(missing_no_bucket_info.index), "movies are good deployments but don't have bucket info. Their filenames are:")
    print(*missing_no_bucket_info.filename.unique(), sep = "\n")
    
    # Find out files missing from the csv
    missing_from_csv = missing_info[missing_info["_merge"]=="left_only"].reset_index(drop=True)
    print("There are", len(missing_from_csv.index), "movies missing from movies.csv. Their filenames are:")
    print(*missing_from_csv.filename.unique(), sep = "\n")
    
    return missing_from_server, missing_from_csv

def get_movies_from_aws(client, bucket_i, aws_folder):
        
    # Retrieve info from the bucket
    contents_s3_pd = retrieve_s3_buckets_info(client, bucket_i)

    # Specify the filename of the objects (videos)        
    contents_s3_pd['raw_filename'] = contents_s3_pd['Key'].str.split('/').str[-1]

    # Specify the prefix (directory) of the objects        
    contents_s3_pd['prefix'] = contents_s3_pd['Key'].str.rsplit('/',1).str[0]
    
    # Select only files within the buv-zooniverse-uploads bucket
    zoo_contents_s3_pd = contents_s3_pd[contents_s3_pd['prefix'].str.contains(aws_folder)].reset_index(drop = True)

    # Specify the formats of the movies to select
    movie_formats = tuple(['wmv', 'mpg', 'mov', 'avi', 'mp4', 'MOV', 'MP4'])

    # Select only files of interest (movies)
    zoo_contents_s3_pd_movies = zoo_contents_s3_pd[zoo_contents_s3_pd['raw_filename'].str.endswith(movie_formats)]
    
    return zoo_contents_s3_pd_movies

def download_object_from_s3(client, *, bucket, key, version_id=None, filename):
    """
    Download an object from S3 with a progress bar.

    From https://alexwlchan.net/2021/04/s3-progress-bars/
    """

    # First get the size, so we know what tqdm is counting up to.
    # Theoretically the size could change between this HeadObject and starting
    # to download the file, but this would only affect the progress bar.
    kwargs = {"Bucket": bucket, "Key": key}

    if version_id is not None:
        kwargs["VersionId"] = version_id

    object_size = client.head_object(**kwargs)["ContentLength"]

    if version_id is not None:
        ExtraArgs = {"VersionId": version_id}
    else:
        ExtraArgs = None

    with tqdm(total=object_size, unit="B", unit_scale=True, desc=filename, position=0, leave=True) as pbar:
        client.download_file(
            Bucket=bucket,
            Key=key,
            ExtraArgs=ExtraArgs,
            Filename=filename,
            Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
        )


def upload_file_to_s3(client, *, bucket, key, filename):
    
    # Get the size of the file to upload
    file_size = os.stat(filename).st_size

    with tqdm(total=file_size, unit="B", unit_scale=True, desc=filename, position=0, leave=True) as pbar:
        client.upload_file(
            Filename=filename,
            Bucket=bucket,
            Key=key,
            Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
        )
