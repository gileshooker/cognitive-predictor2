import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


train = pd.read_csv("data_split/train_data.csv")
test = pd.read_csv("data_split/test_data.csv")

train["trailbmb7c_adjusted"] = train["trailbmb7c"].copy()
test["trailbmb7c_adjusted"] = test["trailbmb7c"].copy()

non_english_completed = train[
    (train["langmb7c"] != 1) & (train["trailbmb7c"] < 300)
]["trailbmb7c"]

if len(non_english_completed) > 0:
    impute_value = non_english_completed.median()

    train.loc[
        (train["trailbmb7c"] == 300) & (train["langmb7c"] != 1),
        "trailbmb7c_adjusted"
    ] = impute_value

    test.loc[
        (test["trailbmb7c"] == 300) & (test["langmb7c"] != 1),
        "trailbmb7c_adjusted"
    ] = impute_value
else:
    impute_value = np.nan

# Trail B ceiling indicator:
# 1 if Trail B hit the ceiling of 300, else 0.
train["trailb_ceiling"] = (train["trailbmb7c"] == 300).astype(int)
test["trailb_ceiling"] = (test["trailbmb7c"] == 300).astype(int)



label_map = {"NI": 0, "MCI": 1, "PD": 1, "CC": 1}

y_train = train["cog_dxmb7"].map(label_map)
y_test = test["cog_dxmb7"].map(label_map)

train = train.loc[y_train.dropna().index].copy()
test = test.loc[y_test.dropna().index].copy()
y_train = y_train.dropna().astype(int)
y_test = y_test.dropna().astype(int)



demographics = [
    "age7c", "gender1", "race1c", "educ1", "phx_income7", "curjob7"
]

raw_scores = [
    "craftursmb7c", "craftdremb7c", "dgtformb7c", "dgtbckmb7c",
    "dsymscrmb7c", "trailamb7c", "trailbmb7c_adjusted", "udsverfcmb7c",
    "vegmb7c", "animalsmb7c", "avlt_delayed_totalmb7c",
    "avlt_t1_totalmb7c", "avlt_t6_totalmb7c", "avlt_total_correctmb7c",
    "wrat5mb7c", "mocatotsmb7c", "craftdvrmb7c", "craftvrsmb7c",
    "udsbentcmb7c", "udsbentdmb7c", "avlt_lotmb7c", "avlt_listb_totalmb7c",
    "trailb_ceiling", "casisummb7c"
]

z_scores = [
    "memory_immed_domainmb7c", "memory_delay_domainmb7c",
    "lang_semantic_domainmb7c", "phonemic_domainmb7c",
    "attn_process_domainmb7c", "executive_domainmb7c",
    "visuo_domainmb7c", "lang_phonemic_domainmb7c"
]

reduced_core = [
    "executive_domainmb7c",
    "memory_delay_domainmb7c",
    "trailbmb7c_adjusted",
    "casisummb7c",
    "dsymscrmb7c",
    "mocatotsmb7c"
]

full_features = demographics + raw_scores + z_scores
reduced_with_demo = demographics + reduced_core
reduced_without_demo = reduced_core.copy()


all_needed = set(full_features + reduced_with_demo + reduced_without_demo)
missing = [c for c in all_needed if c not in train.columns]
if missing:
    raise ValueError(f"These required columns are missing from the data: {missing}")



def find_threshold(y_true, y_probs, target=0.9):
    thresholds = np.quantile(y_probs, np.linspace(1, 0, 50))

    last_sens, last_spec = 0.0, 0.0
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        cm = confusion_matrix(y_true, preds)

        if cm.size != 4:
            continue

        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        last_sens, last_spec = sens, spec

        if sens >= target:
            return t, sens, spec

    return 0.5, last_sens, last_spec


def train_and_save(feature_list, suffix):
    X_train = train[feature_list].fillna(train[feature_list].mean())
    X_test = test[feature_list].fillna(train[feature_list].mean())

    rf = RandomForestClassifier(
        n_estimators=1000,
        class_weight="balanced",
        random_state=42
    )
    rf.fit(X_train, y_train)

    train_probs = rf.predict_proba(X_train)[:, 1]
    test_probs = rf.predict_proba(X_test)[:, 1]

    thresh, sens, spec = find_threshold(y_train, train_probs, target=0.9)

    print(f"{suffix}:")
    print(f"  n_features = {len(feature_list)}")
    print(f"  threshold  = {thresh:.4f}")
    print(f"  train sens = {sens:.4f}")
    print(f"  train spec = {spec:.4f}")
    print(f"  test mean risk = {test_probs.mean():.4f}")

    joblib.dump(rf, f"model_{suffix}.pkl")
    joblib.dump(thresh, f"threshold_{suffix}.pkl")
    joblib.dump(feature_list, f"features_{suffix}.pkl")
    joblib.dump(X_train.mean(), f"feature_means_{suffix}.pkl")

train_and_save(full_features, "full")
train_and_save(reduced_with_demo, "reduced_demo")
train_and_save(reduced_without_demo, "reduced_nodemo")

print("All models saved.")