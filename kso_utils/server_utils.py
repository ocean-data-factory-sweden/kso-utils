# base imports
import os
import pandas as pd
import getpass
import gdown
import zipfile
import boto3
import paramiko
import logging
import sys
import ftfy
from tqdm import tqdm
from pathlib import Path
from paramiko import SFTPClient, SSHClient

# util imports
from kso_utils.project_utils import Project

# Logging
logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


######################################
# ###### Common server functions ######
# #####################################


def connect_to_server(project: Project):
    """
    > This function connects to the server specified in the project object and returns a dictionary with
    the client and sftp client

    :param project: the project object
    :return: A dictionary with the client and sftp_client
    """
    # Get project-specific server info
    if project is None or not hasattr(project, "server"):
        logging.error("No server information found, edit projects_list.csv")
        return {}

    # Create an empty dictionary to host the server connections
    server_connection = {}

    if project.server == "AWS":
        # Connect to AWS as a client
        client = get_aws_client()

    elif project.server == "SNIC":
        # Connect to SNIC as a client and get sftp
        client, sftp_client = get_snic_client()

    elif project.server == "TEMPLATE":
        # Connect to SNIC as a client and get sftp
        client = "Wildlife.ai"

    else:
        server_connection = {}

    if "client" in vars():
        server_connection["client"] = client

    if "sftp_client" in vars():
        server_connection["sftp_client"] = sftp_client

    return server_connection


def download_init_csv(project: Project, init_keys: list, server_connection: dict):
    """
    > This function connects to the server of the project and downloads the csv files of interest

    :param project: the project object
    :param init_keys: list of potential names of the csv files
    :param server_connection: A dictionary with the client and sftp_client
    :return: A dictionary with the server paths of the csv files
    """

    # Create empty dictionary to save the server paths of the csv files
    db_initial_info = {}

    if project.server == "AWS":
        # Retrieve a list with all the csv files in the folder with initival csvs
        server_csv_files = get_matching_s3_keys(
            client=server_connection["client"],
            bucket=project.bucket,
            suffix="csv",
        )

        # Select only csv files that are relevant to start the db
        server_csvs_db = [
            s
            for s in server_csv_files
            if any(server_csv_files in s for server_csv_files in init_keys)
        ]

        # Download each csv file locally and store the path of the server csv
        for server_csv in server_csvs_db:
            # Specify the key of the csv
            init_key = [key for key in init_keys if key in server_csv][0]

            # Save the server path of the csv in a dict
            db_initial_info[str("server_" + init_key + "_csv")] = server_csv

            # Specify the local path for the csv
            local_i_csv = str(Path(project.csv_folder, Path(server_csv).name))

            # Download the csv
            download_object_from_s3(
                client=server_connection["client"],
                bucket=project.bucket,
                key=server_csv,
                filename=local_i_csv,
            )

    elif project.server == "TEMPLATE":
        gdrive_id = "1PZGRoSY_UpyLfMhRphMUMwDXw4yx1_Fn"
        download_gdrive(gdrive_id, project.csv_folder)

    elif project.server in ["LOCAL", "SNIC"]:
        logging.info(
            "Running on SNIC or locally so no csv files were downloaded from the server."
        )

    else:
        raise ValueError(
            "The server type you have chosen is not currently supported. Supported values are AWS, SNIC and LOCAL."
        )

    return db_initial_info


def get_ml_data(project: Project):
    """
    It downloads the training data from Google Drive.
    Currently only applies to the Template Project as other projects do not have prepared
    training data.

    :param project: The project object that contains all the information about the project
    :type project: Project
    """
    if project.ml_folder is not None and not os.path.exists(project.ml_folder):
        # Download the folder containing the training data
        if project.server == "TEMPLATE":
            gdrive_id = "1xknKGcMnHJXu8wFZTAwiKuR3xCATKco9"
            ml_folder = project.ml_folder

            # Download template training files from Gdrive
            download_gdrive(gdrive_id, ml_folder)
            logging.info("Template data downloaded successfully")
        else:
            logging.info("No download method implemented for this data")
    else:
        logging.info("No prepared data to be downloaded.")


def update_csv_server(
    project: Project,
    csv_paths: dict,
    server_connection: dict,
    orig_csv: str,
    updated_csv: str,
):
    """
    > This function updates the original csv file with the updated csv file in the server

    :param project: the project object
    :param csv_paths: a dictionary with the paths of the csv files with info to initiate the db
    :param server_connection: a dictionary with the connection to the server
    :param orig_csv: the original csv file name
    :type orig_csv: str
    :param updated_csv: the updated csv file
    :type updated_csv: str
    """
    if project.server == "AWS":
        logging.info("Updating csv file in AWS server")
        # Update csv to AWS
        upload_file_to_s3(
            client=server_connection["client"],
            bucket=project.bucket,
            key=str(csv_paths[orig_csv]),
            filename=str(csv_paths[updated_csv]),
        )

    elif project.server == "TEMPLATE":
        logging.error(
            f"{orig_csv} couldn't be updated. Check writing permisions to the server."
        )

    elif project.server == "SNIC":
        # Specify volume allocated by SNIC
        snic_path = "/mimer/NOBACKUP/groups/snic2021-6-9"
        # TODO: orig_csv and updated_csv as filenames, create full path for upload_object_to_snic with project.csv_folder.
        # Use below definition for production, commented not for development
        # local_fpath = project.csv_folder + updated_csv
        # Special implementation with two dummy folders for SNIC case since local and server
        # are essentially the same for now.
        local_fpath = f"{snic_path}/tmp_dir/local_dir_dev/" + updated_csv
        remote_fpath = f"{snic_path}/tmp_dir/server_dir_dev/" + orig_csv
        upload_object_to_snic(
            sftp_client=server_connection["sftp_client"],
            local_fpath=local_fpath,
            remote_fpath=remote_fpath,
        )

    elif server == "LOCAL":
        logging.error("Updating csv files to the server is a work in progress")

    else:
        logging.error(
            f"{orig_csv} couldn't be updated. Check writing permisions to the server."
        )


def upload_file_server(project: Project, movie_path: str, f_path: str):
    """
    Takes the file path of a file and uploads/saves it to the server.

    :param project: the project object
    :param movie_path: The local path to the movie file you want to upload
    :type movie_path: str
    :param f_path: The server or storage path of the movie you want to upload to
    :type f_path: str

    """
    if project.server == "AWS":
        # Retrieve the key of the movie of interest
        f_path_key = f_path.split("/").str[:2].str.join("/")
        logging.info(
            f"{f_path_key} not uploaded to the server. Uploading to AWS is a WIP"
        )

    elif project.server == "TEMPLATE":
        logging.error(f"{movie_path} not uploaded to the server as project is template")

    elif project.server == "SNIC":
        logging.error("Uploading the movies to the server is a work in progress")

    elif project.server == "LOCAL":
        logging.error(f"{movie_path} not uploaded to the server as project is local")

    else:
        raise ValueError("Specify the server of the project in the project_list.csv.")


#####################
# ## AWS functions ###
# ####################


def aws_credentials():
    # Save your access key for the s3 bucket.
    aws_access_key_id = getpass.getpass("Enter the key id for the aws server")
    aws_secret_access_key = getpass.getpass(
        "Enter the secret access key for the aws server"
    )

    return aws_access_key_id, aws_secret_access_key


def connect_s3(aws_access_key_id: str, aws_secret_access_key: str):
    # Connect to the s3 bucket
    client = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )
    return client


def get_aws_client():
    # Set aws account credentials
    aws_access_key_id, aws_secret_access_key = os.getenv("SPY_KEY"), os.getenv(
        "SPY_SECRET"
    )
    if aws_access_key_id is None or aws_secret_access_key is None:
        aws_access_key_id, aws_secret_access_key = aws_credentials()

    # Connect to S3
    client = connect_s3(aws_access_key_id, aws_secret_access_key)

    return client


def get_matching_s3_objects(
    client: boto3.client, bucket: str, prefix: str = "", suffix: str = ""
):
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

    kwargs = {"Bucket": bucket}

    # We can pass the prefix directly to the S3 API.  If the user has passed
    # a tuple or list of prefixes, we go through them one by one.
    if isinstance(prefix, str):
        prefixes = (prefix,)
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


def get_matching_s3_keys(
    client: boto3.client, bucket: str, prefix: str = "", suffix: str = ""
):
    """
    ## Code from alexwlchan (https://alexwlchan.net/2019/07/listing-s3-keys/)
    Generate the keys in an S3 bucket.

    :param client: S3 client.
    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch keys that start with this prefix (optional).
    :param suffix: Only fetch keys that end with this suffix (optional).
    return a list of the matching objects
    """

    # Select the relevant bucket
    s3_keys = [
        obj["Key"] for obj in get_matching_s3_objects(client, bucket, prefix, suffix)
    ]

    return s3_keys


def download_object_from_s3(
    client: boto3.client,
    *,
    bucket: str,
    key: str,
    version_id: str = None,
    filename: str,
):
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

    with tqdm(
        total=object_size,
        unit="B",
        unit_scale=True,
        desc=filename,
        position=0,
        leave=True,
    ) as pbar:
        client.download_file(
            Bucket=bucket,
            Key=key,
            ExtraArgs=ExtraArgs,
            Filename=filename,
            Callback=lambda bytes_transferred: pbar.update(bytes_transferred),
        )


def upload_file_to_s3(client: boto3.client, *, bucket: str, key: str, filename: str):
    """
    > Upload a file to S3, and show a progress bar if the file is large enough

    :param client: The boto3 client to use
    :param bucket: The name of the bucket to upload to
    :param key: The name of the file in S3
    :param filename: The name of the file to upload
    """

    # Get the size of the file to upload
    file_size = os.stat(filename).st_size

    # Prevent issues with small files (<1MB) and tqdm
    if file_size > 1000000:
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            desc=filename,
            position=0,
            leave=True,
        ) as pbar:
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


def delete_file_from_s3(client: boto3.client, *, bucket: str, key: str):
    """
    > Delete a file from S3.

    :param client: boto3.client - the client object that you created in the previous step
    :type client: boto3.client
    :param bucket: The name of the bucket that contains the object to delete
    :type bucket: str
    :param key: The name of the file
    :type key: str
    """
    client.delete_object(Bucket=bucket, Key=key)


##############################
# #######SNIC functions########
# #############################


def snic_credentials():
    # Save your access key for the SNIC server.
    snic_user = getpass.getpass("Enter your username for SNIC server")
    snic_pass = getpass.getpass("Enter your password for SNIC server")
    return snic_user, snic_pass


def connect_snic(snic_user: str, snic_pass: str):
    # Connect to the SNIC server and return SSH client
    client = SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.load_system_host_keys()
    client.connect(
        hostname="129.16.125.130", port=22, username=snic_user, password=snic_pass
    )
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


def get_snic_files(client: SSHClient, folder: str):
    """
    Get list of movies from SNIC server using ssh client.

    :param client: SSH client (paramiko)
    """
    stdin, stdout, stderr = client.exec_command(f"find {folder} -type f")
    snic_df = pd.DataFrame(stdout.read().decode("utf-8").split("\n"), columns=["fpath"])
    return snic_df


def download_object_from_snic(
    sftp_client: SFTPClient, remote_fpath: str, local_fpath: str = "."
):
    """
    Download an object from SNIC with progress bar.
    """

    class TqdmWrap(tqdm):
        def viewBar(self, a, b):
            self.total = int(b)
            self.update(int(a - self.n))  # update pbar with increment

    # end of reusable imports/classes
    with TqdmWrap(ascii=True, unit="b", unit_scale=True) as pbar:
        sftp_client.get(remote_fpath, local_fpath, callback=pbar.viewBar)


def upload_object_to_snic(sftp_client: SFTPClient, local_fpath: str, remote_fpath: str):
    """
    Upload an object to SNIC with progress bar.
    """

    class TqdmWrap(tqdm):
        def viewBar(self, a, b):
            self.total = int(b)
            self.update(int(a - self.n))  # update pbar with increment

    # end of reusable imports/classes
    with TqdmWrap(ascii=True, unit="b", unit_scale=True) as pbar:
        sftp_client.put(local_fpath, remote_fpath, callback=pbar.viewBar)


def mount_snic(
    client: SSHClient, snic_path: str = "/mimer/NOBACKUP/groups/snic2021-6-9/"
):
    """
    It mounts the remote directory to the local machine

    :param snic_path: The path to the SNIC directory on the remote server, defaults to
            /mimer/NOBACKUP/groups/snic2021-6-9/
    :type snic_path: str (optional)
    :return: The return value is the exit status of the command.
    """
    cmd = "sshfs {}:{} {}".format(
        client.get_transport().get_username(),
        snic_path,
        snic_path,
    )
    stdin, stdout, stderr = client.exec_command(cmd)
    # Print output and errors (if any)
    logging.info("Output:", stdout.read().decode("utf-8"))
    logging.error("Errors:", stderr.read().decode("utf-8"))
    # Verify that the remote directory is mounted
    if os.path.ismount(snic_path):
        logging.info("Remote directory mounted successfully!")
        return 1
    else:
        logging.error("Failed to mount remote directory!")
        return 0


def fix_text_encoding(folder_name):
    """
    This function corrects for text encoding errors, which occur when there is
    for example an ä,å,ö present. It runs through all the file and folder names
    of the directory you give it. It uses the package ftfy, which recognizes
    which encoding the text has based on the text itself, and it encodes/decodes
    it to utf8.
    This function was tested on a Linux and Windows device with package version
    6.1.1. With package version 5.8 it did not work.

    This function can replace the unswedify and reswedify functions from
    koster_utils, but this is not implemented yet.
    """
    dirpaths = []
    for dirpath, dirnames, filenames in os.walk(folder_name):
        for filename in filenames:
            os.rename(
                os.path.join(dirpath, filename),
                os.path.join(dirpath, ftfy.fix_text(filename)),
            )
        dirpaths.append(dirpath)

    for dirpath in dirpaths:
        if sys.platform.startswith("win"):  # windows has different file-path formatting
            index = dirpath.rfind("\\")
        else:  # mac and linux have the same file-path formatting
            index = dirpath.rfind("/")
        old_dir = ftfy.fix_text(dirpath[:index]) + dirpath[index:]
        new_dir = ftfy.fix_text(dirpath)
        os.rename(old_dir, new_dir)


###################################
# #######Google Drive functions#####
# ##################################


def download_gdrive(gdrive_id: str, folder_name: str):
    # Specify the url of the file to download
    url_input = f"https://drive.google.com/uc?&confirm=s5vl&id={gdrive_id}"

    logging.info(f"Retrieving the file from {url_input}")

    # Specify the output of the file
    zip_file = f"{folder_name}.zip"

    # Download the zip file
    gdown.download(url_input, zip_file, quiet=False)

    # Unzip the folder with the files
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(os.path.dirname(folder_name))

    # Remove the zipped file
    os.remove(zip_file)

    # Correct the file names by using correct encoding
    fix_text_encoding(folder_name)
