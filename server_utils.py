import os, io, sys
import requests
import pandas as pd
import numpy as np
import getpass
import gdown
import zipfile
import boto3
import paramiko
import logging
from paramiko import SSHClient
from scp import SCPClient

import kso_utils.tutorials_utils as tutorials_utils
import kso_utils.spyfish_utils as spyfish_utils
import kso_utils.project_utils as project_utils
from tqdm import tqdm
from pathlib import Path

# Logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

###################################### 
# ###### Common server functions ######
# #####################################

def connect_to_server(project):
    # Get project-specific server info
    server = project.server
    
    # Create an empty dictionary to host the server connections
    server_dict = {}
    
    if server=="AWS":
        # Connect to AWS as a client
        client = get_aws_client()
        
    elif server=="SNIC":
        # Connect to SNIC as a client and get sftp
        client, sftp_client = get_snic_client()
        
    else:
        server_dict = {}
        
    if "client" in vars():
        server_dict['client'] = client
        
    if "sftp_client" in vars():
        server_dict['sftp_client'] = sftp_client
        
    return server_dict

        
def get_db_init_info(project, server_dict):
    
    # Define the path to the csv files with initial info to build the db
    db_csv_info = project.csv_folder
        
    # Get project-specific server info
    server = project.server
    project_name = project.Project_name
    
    # Create the folder to store the csv files if not exist
    if not os.path.exists(db_csv_info):
        os.mkdir(db_csv_info)
            
    if server == "AWS":
            
        # Download csv files from AWS
        db_initial_info = download_csv_aws(project_name, server_dict, db_csv_info)
            
        if project_name == "Spyfish_Aotearoa":
            db_initial_info = spyfish_utils.get_spyfish_choices(server_dict, db_initial_info, db_csv_info)
            
        return db_initial_info
                
    elif server in ["local", "SNIC"]:
        
        csv_folder = db_csv_info
        movie_folder = project.movie_folder
        
        # Define the path to the csv files with initial info to build the db
        if not os.path.exists(csv_folder):
            logging.error("Invalid csv folder specified, please provide the path to the species, sites and movies (optional)")
            
        for file in Path(csv_folder).rglob("*.csv"):
            if 'sites' in file.name:
                sites_csv = file
            if 'movies' in file.name:
                movies_csv = file
            if 'photos' in file.name:
                photos_csv = file
            if 'survey' in file.name:
                surveys_csv = file
            if 'species' in file.name:
                species_csv = file
                
        if "movies_csv" not in vars() and "photos_csv" not in vars() and os.path.exists(csv_folder):
            logging.info("No movies or photos found, an empty movies file will be created.")
            with open(f'{csv_folder}/movies.csv', 'w') as fp:
                pass
                
        try:
            db_initial_info = {
                "local_sites_csv": sites_csv,  
                "local_species_csv": species_csv
            }
            
            if "movies_csv" in vars():
                db_initial_info["local_movies_csv"] = movies_csv
                
            if "photos_csv" in vars():
                db_initial_info["local_photos_csv"] = photos_csv
                
            if "surveys_csv" in vars():
                db_initial_info["local_surveys_csv"] = surveys_csv
            
        except:
            logging.error("Insufficient information to build the database. Please fix the path to csv files.")
            db_initial_info = {}

    else:
        raise ValueError("The server type you have chosen is not currently supported. Supported values are AWS, SNIC and local.")
    return db_initial_info
    


def update_db_init_info(project, csv_to_update):
    
    if server == "AWS":
            
        # Start AWS session
        aws_access_key_id, aws_secret_access_key = server_utils.aws_credentials()
        client = server_utils.connect_s3(aws_access_key_id, aws_secret_access_key)
        bucket = project.bucket
        key = project.key

        csv_filename=csv_to_update.name

        upload_file_to_s3(client,
                              bucket=bucket,
                              key=str(Path(key, csv_filename)),
                              filename=str(csv_to_update))
            
            


#####################
# ## AWS functions ###
# ####################

def aws_credentials():
    # Save your access key for the s3 bucket. 
    aws_access_key_id = getpass.getpass('Enter the key id for the aws server')
    aws_secret_access_key = getpass.getpass('Enter the secret access key for the aws server')
    
    return aws_access_key_id, aws_secret_access_key

                
def connect_s3(aws_access_key_id, aws_secret_access_key):
    # Connect to the s3 bucket
    client = boto3.client('s3',
                          aws_access_key_id = aws_access_key_id, 
                          aws_secret_access_key = aws_secret_access_key)
    return client

def get_aws_client():
    # Set aws account credentials
    aws_access_key_id, aws_secret_access_key = aws_credentials()

    # Connect to S3
    client = connect_s3(aws_access_key_id, aws_secret_access_key)

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

def download_csv_aws(project_path, project_name, server_dict, db_csv_info):
    # Provide bucket and key
    project = project_utils.find_project(project_path, project_name)
    bucket = project.bucket
    key = project.key

    # Create db_initial_info dict
    db_initial_info = {
        "bucket": bucket,
        "key": key
    }

    for i in ['sites', 'movies', 'species', 'surveys']:
        # Get the server path of the csv
        server_i_csv = get_matching_s3_keys(server_dict["client"], 
                                            bucket, 
                                            prefix = key+"/"+i)['Key'][0]

        # Specify the local path for the csv
        local_i_csv = str(Path(db_csv_info,Path(server_i_csv).name))

        # Download the csv
        download_object_from_s3(server_dict["client"],
                            bucket=bucket,
                            key=server_i_csv, 
                            filename=local_i_csv)

        # Save the local and server paths in the dict
        local_csv_str = str("local_"+i+"_csv")
        server_csv_str = str("server_"+i+"_csv")

        db_initial_info[local_csv_str] = Path(local_i_csv)
        db_initial_info[server_csv_str] = server_i_csv

    return db_initial_info
    
    
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
    
    # prvent issues with small files and tqdm
    if file_size > 10000:
        with tqdm(total=file_size, unit="B", unit_scale=True, desc=filename, position=0, leave=True) as pbar:
            client.upload_file(
                Filename=filename,
                Bucket=bucket,
                Key=key,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
            )
    else:
        client.upload_file(
                Filename=filename,
                Bucket=bucket,
                Key=key,
            )
# def retrieve_s3_buckets_info(client, bucket, suffix):
    
#     # Select the relevant bucket
#     s3_keys = [obj["Key"] for obj in get_matching_s3_objects(client=client, bucket=bucket, suffix=suffix)]

#     # Set the contents as pandas dataframe
#     contents_s3_pd = pd.DataFrame(s3_keys)
    
#     return contents_s3_pd
        

##############################
# #######SNIC functions########
# #############################

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

def create_snic_transport(snic_user: str, snic_pass: str):
    # Connect to the SNIC server and return SSH client
    t = paramiko.Transport(("129.16.125.130", 22))
    t.connect(username=snic_user, password=snic_pass)
    sftp = paramiko.SFTPClient.from_transport(t)
    return sftp

def get_snic_client():
    # Set SNIC credentials
    snic_user, snic_pass = snic_credentials()

    # Connect to SNIC
    client = connect_snic(snic_user, snic_pass)
    sftp_client = create_snic_transport(snic_user, snic_pass)

    return client, sftp_client

def get_snic_files(client, folder):
    """ 
    Get list of movies from SNIC server using ssh client.
    
    :param client: SSH client (paramiko)
    """
    stdin, stdout, stderr = client.exec_command(f"ls {folder}")
    snic_df = pd.DataFrame(stdout.read().decode("utf-8").split('\n'), columns=['spath'])
    return snic_df




def download_object_from_snic(sftp_client, remote_fpath: str, local_fpath: str ='.'):
    """
    Download an object from SNIC with progress bar.
    """

    class TqdmWrap(tqdm):
        def viewBar(self, a, b):
            self.total = int(b)
            self.update(int(a - self.n))  # update pbar with increment

    # end of reusable imports/classes
    with TqdmWrap(ascii=True, unit='b', unit_scale=True) as pbar:
        sftp_client.get(remote_fpath, local_fpath, callback=pbar.viewBar)
        
        
def upload_object_to_snic(sftp_client, local_fpath: str, remote_fpath: str):
    """
    Upload an object to SNIC with progress bar.
    """

    class TqdmWrap(tqdm):
        def viewBar(self, a, b):
            self.total = int(b)
            self.update(int(a - self.n))  # update pbar with increment

    # end of reusable imports/classes
    with TqdmWrap(ascii=True, unit='b', unit_scale=True) as pbar:
        sftp_client.put(local_fpath, remote_fpath, callback=pbar.viewBar)
    

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


###################################        
# #######Google Drive functions##### 
# ##################################

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
        
        
