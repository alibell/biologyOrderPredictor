"""
    This script the preprocessing functions
"""

import sqlite3
import pandas as pd

def generate_features_dataset(database, get_drugs=True, get_diseases=True):

    """
        Generate features dataset according to the data

        Parameters
        ----------
        database: str, path of the database sqlite file
        get_drugs: boolean, if true the drug history is returned,
        get_diseases: boolean, if true the disease history is returned
    """

    to_merge = []
    
    # Sqlite connection
    conn = sqlite3.connect("./data/mimic-iv.sqlite")

    ## Getting the features
    features = pd.read_sql(f"""
        SELECT 
            s.stay_id,
            s.intime intime,
            p.gender gender,
            p.anchor_age age,
            t.temperature,
            t.heartrate,
            t.resprate,
            t.o2sat,
            t.sbp,
            t.dbp,
            t.pain,
            t.chiefcomplaint
        FROM edstays s
        LEFT JOIN patients p
            ON p.subject_id = s.subject_id
        LEFT Join triage t
            ON t.stay_id = s.stay_id
    """, conn)

    ## Additional features
    ### Last visit
    last_visit = pd.read_sql(f"""
        SELECT DISTINCT
            s1.stay_id,
            CAST(MAX((julianday(s1.intime)-julianday(s2.intime))) <= 7 AS INT) last_7,
            CAST(MAX((julianday(s1.intime)-julianday(s2.intime))) <= 30 AS INT) last_30
        FROM edstays s1
        INNER JOIN edstays s2
            ON s1.subject_id = s2.subject_id
                AND s1.stay_id != s2.stay_id
                AND s1.intime >= s2.intime
        WHERE (julianday(s1.intime)-julianday(s2.intime)) <= 30
        GROUP BY s1.stay_id 
    """, conn)
    to_merge.append(last_visit)

    ### Past diagnosis
    if get_diseases:
        past_diagnosis = pd.read_sql(f"""
            SELECT 
                s1.stay_id,
                d.icd_code,
                d.icd_version,
                COUNT(1) n
            FROM edstays s1
            INNER JOIN diagnosis d
                ON d.subject_id = s1.subject_id
            INNER JOIN edstays s2
                ON d.stay_id = s2.stay_id
            WHERE 
                s1.intime >= s2.intime
                AND s1.stay_id != s2.stay_id
            GROUP BY 
                s1.stay_id,
                d.icd_code,
                d.icd_version
        """, conn)
        past_diagnosis = pd.pivot_table(
            past_diagnosis.groupby(["stay_id","icd_version"])["icd_code"].agg(lambda x: x.tolist()) \
                    .reset_index(),
                index="stay_id",
                columns="icd_version",
                values="icd_code",
                aggfunc=lambda x: x
        ).reset_index().rename(columns={
            9:"icd9",
            10:"icd10"
        })
        to_merge.append(past_diagnosis)

    ### Drugs
    if get_drugs:
        drugs = pd.read_sql(f"""
            SELECT stay_id, gsn, 1 n
            FROM medrecon
        """, conn)
        drugs = drugs.groupby("stay_id")["gsn"].agg(lambda x: x.tolist()).reset_index()
        to_merge.append(drugs)

    ### Merging all together
    for df_to_merge in to_merge:
        features = pd.merge(
            features,
            df_to_merge,
            left_on="stay_id",
            right_on="stay_id",
            how="left"
        )

    features = features.sort_values("stay_id").reset_index(drop=True)

    return features
    

def generate_labels_dataset(database, lab_dictionnary):

    """
        Generate features dataset according to the data

        Parameters
        ----------
        database: str, path of the database sqlite file
        lab_dictionnary: dictionnary containing the id (keys) and label (value) of the biological exams to predict
    """

    to_merge = []
    
    # Sqlite connection
    conn = sqlite3.connect("./data/mimic-iv.sqlite")

    # Getting biological values
    lab_dictionnary_pd = pd.DataFrame.from_dict(lab_dictionnary, orient="index").reset_index()
    lab_dictionnary_list = [str(x) for x in lab_dictionnary.keys()]

    ## Let's create an index to speed up queries
    conn.execute("CREATE INDEX IF NOT EXISTS biological_index ON labevents (stay_id, itemid)")

    # 1. Generating features

    ## Getting list of stay_id
    stays = pd.read_sql(
        "SELECT DISTINCT stay_id FROM edstays",
        conn
    )

    ## Getting the features
    labs = pd.read_sql(f"""
        SELECT 
            le.stay_id,
            le.itemid item_id
        FROM labevents le
        WHERE le.itemid IN ('{"','".join(lab_dictionnary_list)}')
        GROUP BY
            le.stay_id,
            le.itemid
    """, conn)

    labs_deduplicate = pd.merge(
        lab_dictionnary_pd.rename(columns={0:"label"}),
        labs,
        left_on="index",
        right_on="item_id"
    ) \
    .drop_duplicates(["stay_id", "label"])[["stay_id","label"]] \
    .reset_index(drop=True)

    labs_deduplicate_pivot = pd.pivot_table(
        labs_deduplicate.assign(value=1),
        index="stay_id",
        columns="label",
        values="value"
    ).fillna(0)

    labs_deduplicate_pivot_final = labs_deduplicate_pivot.join(
        stays[["stay_id"]].set_index("stay_id"),
        how="right"
    ).fillna(0).astype("int8").reset_index()

    labels = labs_deduplicate_pivot_final.sort_values("stay_id").reset_index(drop=True)

    return labels