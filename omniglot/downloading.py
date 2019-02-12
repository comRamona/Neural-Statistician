import os
import sys
from urllib.request import urlretrieve
    
def download_file(url, data_dir, file_name):
    sys.stderr.write("Downloading %s".format(file_name))
    file_path = os.path.join(data_dir, file_name)
    if not os.path.exists(file_path):
        urlretrieve(url + file_name, file_path, reporthook)

def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 1e2 / totalsize
        s = "\r%5.1f%% %*d / %d" % (
            percent, len(str(totalsize)), readsofar, totalsize)
        sys.stderr.write(s)
        if readsofar >= totalsize: # near the end
            sys.stderr.write("\n")
    else: # total size is unknown
        sys.stderr.write("read %d\n" % (readsofar,))