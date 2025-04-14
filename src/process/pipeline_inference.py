import glob
import os
import re
import shutil

from src.utils.bids import BIDSDirectory


def add_session(directory: BIDSDirectory):
    for sub in directory.get_subjects():

        if not os.path.isdir(os.path.join(directory.base_path, sub)):
            print(f"!! Exclude file : {sub} !!")
            continue

        src = os.path.join(directory.base_path, sub, "anat")

        files = glob.glob("*T1w.nii*", root_dir=src)
        if len(files) > 1:
            print(f"More than one T1w for sub {sub}, try to find acquistion")
            pattern = r"sub-.*_acq-(.*)_T1w"

            acq = []
            for f in files:
                match = re.match(pattern, f)
                if match is None:
                    print("------------")
                    print(f"Error finding modal for file {f}")
                    print("------------")
                else:
                    acq.append(match.group(1))

            for a in acq:
                sesdir = os.path.join(directory.base_path, sub, f"ses-{a}", "anat")

                os.makedirs(sesdir)
                for file in glob.glob(os.path.join(src, f"*{a}_T1w*")):
                    extension = os.path.basename(file).split(".")[1:]
                    extension = ".".join(extension)
                    newname = f"{sub}_ses-{a}_T1w.{extension}"
                    final_path = os.path.join(sesdir, newname)
                    print(final_path)
                    shutil.move(file, final_path)
            if os.path.exists(src):
                shutil.rmtree(src)
        elif len(files) == 1:
            file = os.path.join(directory.base_path, sub, "anat", files[0])
            sesdir = os.path.join(directory.base_path, sub, f"ses-001", "anat")
            os.makedirs(sesdir, exist_ok=True)
            extension = os.path.basename(file).split(".")[1:]
            extension = ".".join(extension)
            newname = f"{sub}_ses-001_T1w.{extension}"
            final_path = os.path.join(sesdir, newname)
            print(final_path)
            shutil.move(file, final_path)

            if os.path.exists(src):
                shutil.rmtree(src)
