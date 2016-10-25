""" Download data files for the Data Science Game 2016 finals from Azure."""
import pkg_resources
import os
from os.path import join
from textwrap import dedent

dependencies = ["azure"]

DSG = """
  / _ \/ __/ ___/ |_  |/ _ <  / __/
 / // /\ \/ (_ / / __// // / / _ \
/____/___/\___/ /____/\___/_/\___/
"""
ACCOUNT = "dsg16codalab"
KEY = "3cvTVXm3S5vnqVtkwS9JJiKBaENEKe+iEIyYK4V69Ls6v+klHh5YDgIR/kqXYmYtL0vu3IHSreQMySnY3MoV8A=="
CONTAINER_NAME = "dsg16challenge"
FILES = ["../data/X_train.csv", "../data/X_test.csv", "../data/Y_train.csv"]


def download_data():

    import azure_helpers.azure_connection as ac

    target = os.getcwd()
    target = join(target, "data")
    if not os.path.exists(target):
        os.makedirs(target)

    print(DSG)
    print("Connecting to azure...")
    conn = ac.AzureClient(ACCOUNT, KEY)
    blob_service = conn.open_blob_service()
    print("Downloading data...")

    download_file = lambda file: conn.download(blob_service=blob_service,
                                               container_name=CONTAINER_NAME,
                                               blob_name=file,
                                               file_path=join(target, file)
                                               )

    [download_file(f) for f in FILES]
    print("Done!")


if __name__ == "__main__":

    print("Checking dependencies...")
    try:
        pkg_resources.require(dependencies)
    except Exception as e:
        print(dedent("""Requirements are not satisfied.
        Please run 'pip install --upgrade azure' to install them.
        Please note that you must use python 3.x to use this script.
         """))
        print(e)

    download_data()
