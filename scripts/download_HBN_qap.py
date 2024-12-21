import glob
import os
import shutil
import ssl
import zipfile
from urllib.request import urlretrieve

import click
import pandas as pd
import tqdm

ssl._create_default_https_context = ssl._create_unverified_context


def get_archive_url(release_num: int) -> str:
    return (
        f"https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/Data%20Quality/MRI_QAP/release{release_num}_QAP.zip",
        f"release{release_num}_QAP.zip",
    )


@click.command()
@click.option("-s", "--site", default="CBIC")
@click.option("-o", "--output", default=".")
def cli(site: str, output: str):
    tmp_path = os.path.join("notebooks", "tmp")
    archive_path = os.path.join(tmp_path, "qap_archive")
    extract_path = os.path.join(tmp_path, "qap_extract")

    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    os.makedirs(archive_path)
    os.makedirs(extract_path)

    list_qap = []

    for release_num in tqdm.tqdm(range(1, 11)):
        url, filename = get_archive_url(release_num)
        file_path = os.path.join(archive_path, filename)
        urlretrieve(url, os.path.join(file_path))
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            folder = filename.split(".")[0]
            folder_path = os.path.join(extract_path, folder)
            zip_ref.extractall(folder_path)

    csv_path = os.path.join(extract_path, "**", f"MRI_qap_anat_{site}.csv")
    for existing_path in glob.glob(csv_path):
        list_qap.append(pd.read_csv(existing_path))

    shutil.rmtree(tmp_path)
    agg_ds = pd.concat(list_qap)
    agg_ds.to_csv(os.path.join(output, f"{site}_qap.csv"))


if __name__ == "__main__":
    cli()
