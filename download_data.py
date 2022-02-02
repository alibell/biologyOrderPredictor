import requests
from progress.bar import Bar
import os
import sys

#
# Script for MIMIC-IV project data download
#

# Getting command line parameter
if len(sys.argv) > 1:
    token = sys.argv[1]

file_path = {
    "mimic-iv.sqlite":[
        "http://chessm2ds.alibellamine.me:5000/",
        3214753792
    ]
}

print("Downloading MIMIC-IV data")

folder = "./data"

for file_name, file_metadata in file_path.items():
    destination = "/".join([folder, file_name])
    
    if os.path.isfile(destination) == False or os.path.getsize(destination) != file_metadata[1]:
        uri = f"{file_metadata[0]}/{token}"
        r = requests.get(uri, stream=True)
        file_size = int(int(r.headers.get('content-length'))/1000)

        with open(destination, "wb") as f:
            with Bar(f"Downloading {file_name}", max=int(file_size)) as bar:
                for chunck in r.iter_content(chunk_size=1024):
                    f.write(chunck)
                    bar.next()