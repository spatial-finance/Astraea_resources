
import os
import pandas as pd

from google.cloud import bigquery, storage
from google.oauth2 import service_account



def getSceneList( bucket, path ):

    """
    
    """

    scenes = []

    # construct path
    blobs = bucket.list_blobs(prefix=path, delimiter=None)    
    for blob in blobs:

        # split filename on hyphen
        filename = os.path.basename( blob.name )
        basename = filename.split('.', 1)[0]

        # add datetime to list
        tokens = basename.split('-')
        if len( tokens ) == 2 and tokens[ 1 ] not in scenes:
            scenes.append( tokens[ 1 ] )

    return scenes


def main():

    """
    main path of execution
    """

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]='C:/Users/Chris.Williams/.gcs/arkham-255409-c59a52d8653f.json'

    inventory_file = "C:\\Users\\Chris.Williams\\Documents\\ALD\\cement_dataset_v2-update-v2.csv"
    start_record = 0; end_record = 5000

    # load spreadsheet
    df = pd.read_csv( inventory_file )

    # get client connection to s2 bucket
    client = storage.Client()
    bucket = client.get_bucket("eo-ald-update")

    for idx, row in df.iterrows():

        # if record within index constraints
        if idx >= start_record and idx <= end_record:

            if isinstance( row.accuracy, str ) and row.accuracy.lower() == 'exact':

                scenes = getSceneList( bucket, 'cement/exact/l8/chips/{}'.format( row[ 'uid' ] ) )

                getFileList( root_path, scenes )

                
                print ( scenes )



# execute main
if __name__ == '__main__':
    main()

