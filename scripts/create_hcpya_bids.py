import glob
import os
import shutil
import sys

import tqdm

if __name__ == "__main__":
    base_path = "/home/cbricout/projects/def-sbouix/data/HCP-YA-1200/"
    subjects = os.listdir("/home/cbricout/projects/def-sbouix/data/HCP-YA-1200/")
    new_path = sys.argv[1]
    os.makedirs(new_path)
    for sub in tqdm.tqdm(subjects):
        anat_dir = os.path.join(new_path, f"sub-{sub}", "ses-1", "anat")
        os.makedirs(anat_dir)

        for file in glob.glob(
            os.path.join(
                base_path,
                sub,
                "unprocessed",
                "3T",
                "T1w_MPR1",
                "*_3T_T1w_MPR1.nii.gz",
            )
        ):
            shutil.copy(
                file,
                os.path.join(
                    anat_dir,
                    f"sub-{sub.split('_')[0]}_ses-1_T1w."
                    + ".".join(file.split(".")[1:]),
                ),
            )
