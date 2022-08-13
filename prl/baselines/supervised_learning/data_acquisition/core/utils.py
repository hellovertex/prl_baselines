import io
import os
import zipfile


def extract(filename, out_dir):
    z = zipfile.ZipFile(filename)
    for f in z.namelist():
        try:
            os.mkdir(out_dir)
        except FileExistsError:
            pass
        # read inner zip file into bytes buffer
        content = io.BytesIO(z.read(f))
        zip_file = zipfile.ZipFile(content)
        for i in zip_file.namelist():
            zip_file.extract(i, out_dir)
