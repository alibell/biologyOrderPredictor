###
#
# MIMIC-IV database creation for Datacamp 2022
#
###

import pandas as pd
import sqlite3
import dask.dataframe as dd
import dask
from io import StringIO

#
# Parameters
#

sqlite_path = "../data/mimic-iv.sqlite"
csv_ed_path = "../data/mimic-iv-ed/mimic-iv-ed-1.0/ed/"
csv_mimic_path = "../data/mimic-iv/mimiciv/1.0/"

# Loading sqlite database

con = sqlite3.connect(sqlite_path)

# Saving CSV to sqlite database

ed_csv = [
    "triage",
    "medrecon",
    "edstays",
    "diagnosis"
]

global_csv = [
    ["core/patients", "subject_id"],
    ["hosp/d_labitems", None]
]

datasets = {}

## ED data

for csv in ed_csv:
    print(f"Writting {csv}")
    datasets[csv] = pd.read_csv(f"{csv_ed_path}/{csv}.csv.gz")
    datasets[csv].to_sql(csv, con, if_exists="replace")

# Getting patients list
patients_list = datasets["edstays"]["subject_id"].unique().tolist()

## Global Mimic Data

for csv in global_csv:
    print(f"Writting {csv[0]}")
    datasets[csv[0]] = pd.read_csv(f"{csv_mimic_path}/{csv[0]}.csv.gz")

    filename = csv[0].split("/")[-1]

    # Filtering
    if csv[1] == "subject_id":
        datasets[csv[0]] = datasets[csv[0]][
            datasets[csv[0]]["subject_id"].isin(patients_list)
        ].reset_index(drop=True)

    datasets[csv[0]].to_sql(filename, con, if_exists="replace")

# Special cases : microbiology and labevents

## microbiology
microbiology_df = pd.read_csv(f"{csv_mimic_path}/hosp/microbiologyevents.csv.gz")

microbiology_df = microbiology_df[
    (microbiology_df["subject_id"].isin(patients_list))
].reset_index(drop=True)

microbiology_df_filter = pd.merge(
            datasets["edstays"][["subject_id","intime", "outtime", "stay_id"]],
            microbiology_df[["microevent_id","subject_id", "charttime"]],
            on="subject_id"
)

microbiology_df_filter = microbiology_df_filter[
            (
                (microbiology_df_filter["charttime"] >= microbiology_df_filter["intime"]) 
                & (microbiology_df_filter["charttime"] <= microbiology_df_filter["outtime"])
            )]

microbiology_df = pd.merge(microbiology_df, microbiology_df_filter[["stay_id", "microevent_id"]], left_on="microevent_id", right_on="microevent_id", how="inner").drop_duplicates("microevent_id")

microbiology_df.to_sql("microbiologyevents", con, if_exists="replace")

## labevents
### The database is too big to be processed once, dealing with batchs

n_lines = int(1e9)

# Dropping table
cursor = con.cursor()
cursor.execute("DROP TABLE IF EXISTS labevents")
cursor.close()

with open(f"{csv_mimic_path}/hosp/labevents.csv", "r") as f:
    header = f.readline()

    i = 0
    while True:
        print(f"Writting rows {i*n_lines} to {(i+1)*n_lines}")
        i += 1

        lines = f.readlines(n_lines)

        if f.closed or len(lines) == 0:
            break

        temp_file = StringIO("\n".join([header]+lines))
        temp_df = pd.read_csv(temp_file)

        # Filtering intesresting results
        temp_df = temp_df[
            (temp_df["subject_id"].astype("int64").isin(patients_list))
        ].reset_index(drop=True)

        # Filtering results to the accurate date
        temp_df_filter = pd.merge(
            datasets["edstays"][["subject_id","intime", "outtime", "stay_id"]],
            temp_df[["labevent_id","subject_id", "charttime"]],
            on="subject_id"
        )
        temp_df_filter = temp_df_filter[
            (
                (temp_df_filter["charttime"] >= temp_df_filter["intime"]) 
                & (temp_df_filter["charttime"] <= temp_df_filter["outtime"])
            )]

        temp_df = pd.merge(temp_df, temp_df_filter[["stay_id", "labevent_id"]], left_on="labevent_id", right_on="labevent_id", how="inner")

        # Writting to database
        temp_df.to_sql("labevents", con, if_exists="append")

# Creating index
indexes = {
    "triage":["subject_id","stay_id"],
    "medrecon":["subject_id","stay_id"],
    "edstays":["subject_id","stay_id","hadm_id"],
    "diagnosis":["subject_id","stay_id"],
    "patients":["subject_id"],
    "d_labitems":["itemid"],
    "microbiologyevents":["microevent_id","subject_id","stay_id","hadm_id","micro_specimen_id"],
    "labevents":["labevent_id","subject_id","stay_id","hadm_id","specimen_id","itemid"],
}

cursor = con.cursor()
for table_name, indexes_columns in indexes.items():
    for indexes_column in indexes_columns:
        indexes_query = f"CREATE INDEX {table_name}_{indexes_column} ON {table_name} ({indexes_column})"

        cursor.execute(indexes_query)

cursor.close()