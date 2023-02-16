import wandb
import requests
import os, json


# Get deposition id, i.e. "id" field from this response and bucket
def get_zenodo_id_bucket(access_key: str):
    headers = {"Content-Type": "application/json"}
    params = {"access_token": access_key}
    r = requests.post(
        "https://zenodo.org/api/deposit/depositions",
        params=params,
        json={},
        # Headers are not necessary here since "requests" automatically
        # adds "Content-Type: application/json", because we're using
        # the "json=" keyword argument
        # headers=headers,
        headers=headers,
    )
    response = r.json()
    return response["id"], response["links"]["bucket"]


def add_file_to_zenodo_upload(access_key: str, bucket_url: str, file_path: str):
    filename = os.path.basename(file_path)
    # The target URL is a combination of the bucket link with the desired filename
    # seperated by a slash.
    params = {"access_token": access_key}
    with open(file_path, "rb") as fp:
        r = requests.put(
            "%s/%s" % (bucket_url, filename),
            data=fp,
            params=params,
        )
    return r.json()


def add_metadata_zenodo_upload(
    access_token: str,
    deposition_id: str,
    title: str,
    description: str,
    creators_dict: dict,
):
    # Add metadata
    data = {
        "metadata": {
            "title": title,
            "upload_type": "software",
            "description": description,
            "creators": [
                {"name": name, "affiliation": affiliation}
                for name, affiliation in creators_dict.items()
            ],
            "communities": [{"identifier": "odf-sweden"}],
            "notes": "Attribution notice: The code used to generate this model can be found "
            "at https://github.com/ocean-data-factory-sweden/koster_data_management",
        }
    }
    headers = {"Content-Type": "application/json"}
    r = requests.put(
        f"https://zenodo.org/api/deposit/depositions/{deposition_id}",
        params={"access_token": access_token},
        data=json.dumps(data),
        headers=headers,
    )
    if r.status_code == 200:
        print("Upload successful")
        r = requests.post(
            f"https://zenodo.org/api/deposit/depositions/{deposition_id}/actions/publish",
            params={"access_token": access_token},
        )
        return r.status_code

    else:
        print("Upload failed")
