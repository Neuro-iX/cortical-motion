import os
import shutil
import tqdm
import sys

if __name__ == "__main__":
    base_path = (
        "/home/cbricout/projects/ctb-sbouix/datasets/HCPDevelopmentRec/fmriresults01/"
    )
    subjects = os.listdir(
        "/home/cbricout/projects/ctb-sbouix/datasets/HCPDevelopmentRec/fmriresults01/"
    )
    subjects.remove("manifests")

    new_path = sys.argv[1]
    os.makedirs(new_path)
    for sub in tqdm.tqdm(subjects):
        anat_dir = os.path.join(new_path, f"sub-{sub.split('_')[0]}", "ses-1", "anat")
        os.makedirs(anat_dir)
        shutil.copy(
            os.path.join(base_path, sub, "T1w", "T1w_acpc_dc_restore.nii.gz"),
            os.path.join(anat_dir, f"sub-{sub.split('_')[0]}_ses-1_T1w.nii.gz"),
        )
