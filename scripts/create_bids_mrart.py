import glob
import os
import shutil

MRART_dir = "/home/at70870/local_scratch/ds004173"

new_mrart = "/home/at70870/local_scratch/MRART_bids"
os.makedirs(new_mrart, exist_ok=True)
sessions = ["standard", "headmotion1", "headmotion2"]
for sub_path in glob.glob(os.path.join(MRART_dir, "sub-*")):
    sub = os.path.basename(sub_path)
    os.makedirs(os.path.join(new_mrart, sub))

    for ses in sessions:
        existing_ses = map(
            lambda x: x.split("-")[2].split("_")[0],
            glob.glob("*.nii.gz", root_dir=os.path.join(sub_path, "anat")),
        )
        if ses in existing_ses:
            os.makedirs(os.path.join(new_mrart, sub, f"ses-{ses}"))

            for ext in ["nii.gz", "json"]:
                shutil.copy(
                    os.path.join(sub_path, "anat", f"{sub}_acq-{ses}_T1w.{ext}"),
                    os.path.join(
                        new_mrart,
                        sub,
                        f"ses-{ses}",
                        f"{sub}_ses-{ses}_acq_T1w.{ext}",
                    ),
                )
