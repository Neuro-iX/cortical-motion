import glob
import os
import re

import pandas as pd
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt

from src import config

remove_openneuro = [
    "ds000200",
    "ds002799",
    "ds000228",
    "ds002236",
    "ds000120",
    "ds003848",
    "ds003688",
    "ds004219",
    "ds003709",
]


def retrieve_fs_thick(dataset: str, synthetic: bool = False) -> pd.DataFrame:
    """Retrieve non aggregated freesurfer computed thickness for a dataset and combine in a dataframe"""
    if synthetic:
        path_pattern = os.path.join(
            config.DATASET_ROOT,
            dataset,
            "derivatives",
            "sub-*",
            "ses-*",
            "gen-*",
            "stats",
        )
    else:
        path_pattern = os.path.join(
            config.DATASET_ROOT,
            dataset,
            "subjects",
            "derivatives",
            "sub-*",
            "ses-*",
            "stats",
        )

    def find_mean_thick(path: str):
        pattern = r"# Measure Cortex, MeanThickness, Mean Thickness, ([\d\.]*), mm"
        if synthetic:
            identifiers = path.split("/")[6:9]
        else:
            identifiers = path.split("/")[-4:-2]
        with open(path, "r") as file:
            for line in file:
                matches = re.match(pattern, line)
                if matches is not None:
                    return *identifiers, float(matches.group(1))

    group_column = ("sub", "ses")
    if synthetic:
        group_column = ("sub", "ses", "generation")
    rh_files = glob.glob(os.path.join(path_pattern, "rh.aparc.stats"))
    lh_files = glob.glob(os.path.join(path_pattern, "lh.aparc.stats"))
    rh_thick = pd.DataFrame(
        map(find_mean_thick, rh_files),
        columns=(*group_column, "rh_thickness"),
    )
    lh_thick = pd.DataFrame(
        map(find_mean_thick, lh_files),
        columns=(*group_column, "lh_thickness"),
    )
    thick_df = pd.merge(rh_thick, lh_thick, on=group_column)
    thick_df["mean_thickness"] = (
        thick_df["lh_thickness"] + thick_df["rh_thickness"]
    ) / 2
    return thick_df


def retrieve_fs_meas(dataset: str, synthetic: bool = False):
    full_df = None
    ds_path = os.path.join(config.DATASET_ROOT, dataset)
    for hem in ["rh", "lh"]:
        for meas in ["thickness", "area", "volume", "meancurv", "gauscurv"]:
            new = pd.read_csv(os.path.join(ds_path, f"{hem}.{meas}.tsv"), sep="\t")
            new["sub"] = new[f"{hem}.aparc.{meas}"].apply(lambda x: x.split("/")[0])
            new["ses"] = new[f"{hem}.aparc.{meas}"].apply(lambda x: x.split("/")[1])

            if full_df is None:
                full_df = new
            else:
                new = new.drop(
                    columns=[f"{hem}.aparc.{meas}", "BrainSegVolNotVent", "eTIV"]
                )
                full_df = full_df.merge(new, on=("sub", "ses"))
    return full_df


def get_cortical_hcpya_df(model: str):
    thick_df = retrieve_fs_meas("HCP-YA_preproc")
    subject_df = (
        pd.read_csv("article/participants_csv/HCP-YA_subjects.csv")
        .rename(
            columns={
                "Subject": "sub",
                "Age": "age",
                "Gender": "sex",
            }
        )[["sub", "age", "sex"]]
        .iloc[1:]
    )

    subject_df["sub"] = "sub-" + subject_df["sub"].astype(str)
    motion = pd.read_csv(
        f"article/reports/motion_report/HCP-YA_preproc/{model}_report.csv",
        index_col=0,
    )
    full_df = motion.merge(thick_df, on=("sub", "ses"))
    full_df = full_df.merge(subject_df, on=("sub"))
    full_df["model"] = model

    return full_df


def get_cortical_rubic_df(model):
    thick_df = retrieve_fs_meas("HBN_RUBIC_preproc")

    subject_df = pd.read_csv(
        "article/participants_csv/HBN_RUBIC_participants.tsv", sep="\t"
    ).rename(
        columns={
            "participant_id": "sub",
        }
    )
    motion = pd.read_csv(
        f"article/reports/motion_report/HBN_RUBIC_preproc/{model}_report.csv",
        index_col=0,
    )
    full_df = motion.merge(thick_df, on=("sub", "ses"))

    full_df = full_df.merge(subject_df, on=("sub"))

    full_df["Age"] = full_df["Age"].astype(float)
    full_df["Sex"] = full_df["Sex"].astype(float)
    full_df["EHQ_Total"] = full_df["EHQ_Total"].astype(float)

    full_df = full_df.dropna()
    full_df = full_df.reset_index(drop=True)
    full_df = full_df.rename(columns={"Sex": "sex", "Age": "age"})
    full_df["model"] = model
    return full_df


def get_cortical_hcpep_df(model):
    thick_df = retrieve_fs_meas("HCPEP_preproc")

    subject_df = pd.read_csv("article/participants_csv/HCPEP_participants.csv").rename(
        columns={"Age": "age", "Sex": "sex"}
    )
    subject_df["sub"] = "sub-" + subject_df["Session ID"].apply(
        lambda x: x.split("_")[0]
    )
    motion = pd.read_csv(
        f"article/reports/motion_report/HCPEP_preproc/{model}_report.csv",
        index_col=0,
    )
    full_df = motion.merge(thick_df, on=("sub", "ses"))
    full_df = full_df.merge(subject_df, on=("sub"))

    scores = pd.read_csv("article/participants_csv/HCPEP_scores.csv")
    scores["sub"] = scores["Subject ID"].apply(lambda x: "sub-" + x.split("_")[0])
    scores["ses"] = scores["Subject ID"].apply(
        lambda x: "ses-" + x.split("_")[1].removeprefix("MR")
    )
    scores["t1w_scores"] = scores["T1w"]
    scores = scores[scores["T1w"].notna()]

    correct_score = {
        "3/4": "3.5",
        "2/3": "2.5",
        "3/2": "2.5",
        "4/3": "3.5",
        "3 / 4": "3.5",
        "2/ 3": "2.5",
        "2?": "2",
        "4": "4",
        "3": "3",
        "2": "2",
        "1": "1",
    }

    scores["t1w_scores"] = (
        scores["t1w_scores"].apply(lambda x: correct_score[x.strip()]).astype(float)
    )
    # full_df = full_df.merge(subject_df, on=("sub"))

    # full_df["Age"] = full_df["Age"].astype(float)
    # full_df["Sex"] = full_df["Sex"].astype(float)

    # full_df = full_df.dropna()

    full_df = full_df.merge(scores, on=("sub", "ses"), how="left")
    full_df = full_df.reset_index(drop=True)
    # full_df = full_df.rename(columns={"Sex": "sex", "Age": "age"})
    full_df["model"] = model
    return full_df


def get_cortical_mrart_df(model):
    thick_df = retrieve_fs_meas("MRART_preproc")

    subject_df = pd.read_csv(
        "article/participants_csv/MRART_participants.tsv", sep="\t"
    ).rename(
        columns={
            "participant_id": "sub",
        }
    )
    scores = pd.read_csv("article/participants_csv/MRART_scores.tsv", sep="\t")
    scores["sub"] = scores["bids_name"].apply(lambda x: x.split("_")[0])
    scores["ses"] = "ses-" + scores["bids_name"].apply(
        lambda x: x.split("-")[2].split("_")[0]
    )
    motion = pd.read_csv(
        f"article/reports/motion_report/MRART_preproc/{model}_report.csv",
        index_col=0,
    )
    full_df = motion.merge(thick_df, on=("sub", "ses"))

    full_df = full_df.merge(subject_df, on=("sub"))
    full_df = full_df.merge(scores, on=("sub", "ses"))

    full_df = full_df.dropna()
    full_df = full_df.reset_index(drop=True)
    full_df["model"] = model

    return full_df


def get_cortical_openneuro_df(ds, report_model: str):
    thick_df = retrieve_fs_meas(os.path.join("OpenNeuro_preproc", ds))
    subject_path = os.path.join(
        config.DATASET_ROOT, "OpenNeuro", ds, "participants.tsv"
    )
    subject_df = pd.read_csv(subject_path, sep="\t").rename(
        columns={"participant_id": "sub"}
    )
    motion = pd.read_csv(
        f"article/reports/motion_report/OpenNeuro_preproc/{ds}/{report_model}_report.csv",
        index_col=0,
    )
    if "ses" not in motion.columns or "ses" not in thick_df.columns:
        full_df = motion.merge(thick_df, on=("sub"))
    else:
        full_df = motion.merge(thick_df, on=("sub", "ses"))

    if "ses" not in subject_df.columns or "ses" not in full_df.columns:
        full_df = full_df.merge(subject_df, on=("sub"))
    else:
        full_df = full_df.merge(subject_df, on=("sub", "ses"))

    full_df = full_df.rename(
        columns={
            "Age": "age",
            "Gender": "sex",
            "gender": "sex",
            "jsex": "sex",
            "age_ses-T1": "age",
            "ScanAge": "age",
            "participant_age": "age",
        },
        errors="ignore",
    )
    full_df["sex"] = full_df["sex"].replace(
        {
            "FEMALE": "F",
            "MALE": "M",
            1: "M",
            2: "F",
            "female": "F",
            "male": "M",
            "f": "F",
            "m": "M",
        }
    )
    full_df["model"] = report_model

    return full_df


def get_all_openneuro(model):
    all_df = []
    for ds in set(
        os.listdir(os.path.join("article/reports/motion_report/OpenNeuro_preproc/"))
    ).difference(remove_openneuro):
        df = get_cortical_openneuro_df(ds, report_model=model)
        columns = list(filter(lambda x: ("lh" in x and "thickness" in x), df.columns))

        df = df[
            [
                "sub",
                "ses",
                "motion",
                *columns,
                "lh_WhiteSurfArea_area",
                "BrainSegVolNotVent",
                "eTIV",
                "age",
                "sex",
                "model",
            ]
        ]
        if len(df) == 0:
            print(f"Error for ds :{ds}")
        df["dataset"] = ds
        all_df.append(df)
    all_df = pd.concat(all_df)
    return all_df


def fit_one_per_ds_openneuro(full_df):
    all_res = []
    for ds in full_df["dataset"].unique():
        df = full_df[full_df["dataset"] == ds]
        model = smf.gls(
            formula="mean_thickness ~ age + sex + motion",  # Fixed effects
            data=df,  # Your ataFrame
        )
        est2 = model.fit()
        model_df = est2.params
        model_df["model"] = model
        model_df["r2"] = est2.rsquared
        model_df["pvalue"] = est2.pvalues["motion"]
        model_df["log_likelihood"] = est2.llf
        model_df["dataset"] = ds
        model_df["n_volumes"] = len(df)

        all_res.append(model_df)
    all_res = pd.DataFrame(all_res)
    all_res = all_res.sort_values("pvalue", ascending=True)
    all_res = all_res.reset_index(drop=True)
    all_res["rank"] = all_res.index + 1
    return all_res


def fit_one_per_model_openneuro(full_df, var="lh_MeanThickness_thickness"):
    all_res = []
    for model in full_df["model"].unique():
        df = full_df[full_df["model"] == model]
        est = smf.glm(
            formula=f"{var} ~ age + sex + motion",  # Fixed effects
            data=df,  # Your ataFrame
        )
        est2 = est.fit()
        model_df = est2.params
        model_df["model"] = model
        model_df["r2"] = est2.pseudo_rsquared()
        model_df["rse"] = est2.scale**0.5
        model_df["pvalue"] = est2.pvalues["motion"]
        model_df["log_likelihood"] = est2.llf
        model_df["n_volumes"] = len(df)
        all_res.append(model_df)
    all_res = pd.DataFrame(all_res)
    all_res = all_res.sort_values("pvalue", ascending=True)
    all_res = all_res.reset_index(drop=True)
    all_res["rank"] = all_res.index + 1
    return all_res


def fit_one_per_model_per_dataset_openneuro(full_df, var="lh_MeanThickness_thickness"):
    model_df_list = []

    for ds in full_df["dataset"].unique():
        df = full_df[full_df["dataset"] == ds]
        model_ds_df = fit_one_per_model_openneuro(df, var)
        model_ds_df["dataset"] = ds
        model_df_list.append(model_ds_df)

    to_return = pd.concat(model_df_list).reset_index(drop=True)

    return to_return


def fit_one_per_model(full_df, var="lh_MeanThickness_thickness"):
    model_df_list = []
    for model in full_df["model"].unique():
        model_selected_df = full_df[full_df["model"] == model]
        est = smf.glm(
            formula=f"{var} ~ age + sex + motion",
            data=model_selected_df,
        )  # Fixed effects
        est2 = est.fit()
        model_df = est2.params
        model_df["model"] = model
        model_df["r2"] = est2.pseudo_rsquared()
        model_df["rse"] = est2.scale**0.5
        model_df["pvalue"] = est2.pvalues["motion"]
        model_df["log_likelihood"] = est2.llf

        model_df_list.append(model_df)

    to_return = pd.DataFrame(model_df_list)
    to_return = to_return.sort_values("pvalue", ascending=True)
    to_return = to_return.reset_index(drop=True)
    to_return["rank"] = to_return.index + 1
    return to_return


def residual_plot(result):
    plt.figure(figsize=(10, 6))
    plt.scatter(result.fittedvalues, result.resid_response)
    plt.axhline(y=0, color="r", linestyle="--")
    plt.xlabel("Fitted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals vs. Fitted Values")
    plt.show()
