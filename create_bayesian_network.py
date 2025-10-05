"""
    Learn a Bayesian Network or BN Classifier from the German credit dataset.
    Preprocess the data, learn the structure and parameters, and save the resulting BN.
"""
import argparse
import pandas as pd
import pyagrum as gum
import pyagrum.skbn as skbn

# Shared preprocessing used by both BN learner and BNClassifier

def load_and_preprocess(input_path: str) -> pd.DataFrame:
    columns = [
        "CheckingAccount", "Duration", "CreditHistory", "Purpose", "CreditAmount",
        "SavingsAccount", "EmploymentSince", "InstallmentRate", "PersonalStatusSex", "OtherDebtors",
        "ResidenceSince", "Property", "Age", "OtherInstallmentPlans", "Housing",
        "ExistingCredits", "Job", "LiablePeople", "Telephone", "ForeignWorker",
        "CreditRisk"
    ]

    data = pd.read_csv(input_path, sep=" ", header=None, names=columns)

    # Put numeric variables into bins
    data['Age'] = pd.cut(data['Age'], bins=3, labels=["Young", "Middle", "Old"])
    data['CreditAmount'] = pd.cut(data['CreditAmount'], bins=3, labels=["Low", "Medium", "High"])
    data['Duration'] = pd.cut(data['Duration'], bins=3, labels=["Short", "Medium", "Long"])

    # Translate to readable category names
    _checking_account_map = {
        "A11": "lt0",
        "A12": "0to200",
        "A13": "ge200",
        "A14": "No Checking"
    }
    data['CheckingAccount'] = data['CheckingAccount'].map(_checking_account_map).fillna(data['CheckingAccount'])

    _credit_history_map = {
        "A30": "No credits taken / all paid duly",
        "A31": "All credits at this bank paid duly",
        "A32": "Existing credits paid duly till now",
        "A33": "Delay in paying off in the past",
        "A34": "Critical account / other credits elsewhere",
    }
    data['CreditHistory'] = data['CreditHistory'].map(_credit_history_map).fillna(data['CreditHistory'])

    _purpose_map = {
        "A40": "car (new)",
        "A41": "car (used)",
        "A42": "furniture/equipment",
        "A43": "radio/television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "others",
    }
    data['Purpose'] = data['Purpose'].map(_purpose_map).fillna(data['Purpose'])

    _savings_map = {
        "A61": "lt100 DM",
        "A62": "100to500 DM",
        "A63": "500to1000 DM",
        "A64": "ge1000 DM",
        "A65": "unknown / no savings",
    }
    data['SavingsAccount'] = data['SavingsAccount'].map(_savings_map).fillna(data['SavingsAccount'])

    _employment_map = {
        "A71": "unemployed",
        "A72": "lt1 year",
        "A73": "1to4 years",
        "A74": "4to7 years",
        "A75": "ge7 years",
    }
    data['EmploymentSince'] = data['EmploymentSince'].map(_employment_map).fillna(data['EmploymentSince'])

    _ps_map = {
        "A91": "male: divorced/separated",
        "A92": "female: divorced/separated/married",
        "A93": "male: single",
        "A94": "male: married/widowed",
        "A95": "female: single",
    }
    data['PersonalStatusSex'] = data['PersonalStatusSex'].map(_ps_map).fillna(data['PersonalStatusSex'])

    _od_map = {
        "A101": "none",
        "A102": "co-applicant",
        "A103": "guarantor",
    }
    data['OtherDebtors'] = data['OtherDebtors'].map(_od_map).fillna(data['OtherDebtors'])

    _property_map = {
        "A121": "real estate",
        "A122": "building society savings / life insurance",
        "A123": "car or other",
        "A124": "unknown / no property",
    }
    data['Property'] = data['Property'].map(_property_map).fillna(data['Property'])

    _oip_map = {
        "A141": "bank",
        "A142": "stores",
        "A143": "none",
    }
    data['OtherInstallmentPlans'] = data['OtherInstallmentPlans'].map(_oip_map).fillna(data['OtherInstallmentPlans'])

    _housing_map = {
        "A151": "rent",
        "A152": "own",
        "A153": "for free",
    }
    data['Housing'] = data['Housing'].map(_housing_map).fillna(data['Housing'])

    _job_map = {
        "A171": "unemployed/unskilled - non-resident",
        "A172": "unskilled - resident",
        "A173": "skilled employee/official",
        "A174": "management/self-employed/highly qualified/officer",
    }
    data['Job'] = data['Job'].map(_job_map).fillna(data['Job'])

    _tel_map = {
        "A191": "none",
        "A192": "yes (registered)",
    }
    data['Telephone'] = data['Telephone'].map(_tel_map).fillna(data['Telephone'])

    _fw_map = {
        "A201": "yes",
        "A202": "no",
    }
    data['ForeignWorker'] = data['ForeignWorker'].map(_fw_map).fillna(data['ForeignWorker'])

    data['CreditRisk'] = data['CreditRisk'].map({1: "Good", 2: "Bad"})

    return data


def learn_bn(data: pd.DataFrame, score: str = "aic", smoothing: int = 1) -> gum.BayesNet:
    learner = gum.BNLearner(data)
    score = score.lower()
    if score == "aic":
        learner.useScoreAIC()
    elif score == "bic":
        learner.useScoreBIC()
    elif score == "log":
        learner.useScoreLog2Likelihood()
    else:
        raise ValueError(f"Unsupported score '{score}'. Try: aic|bic|log")

    learner.useSmoothingPrior(smoothing)
    bn = learner.learnBN()
    return bn


def learn_classifier(data: pd.DataFrame, method: str = "TAN", target: str = "CreditRisk") -> gum.BayesNet:

    clf = skbn.BNClassifier(learningMethod=method)
    clf.fit(data=data, targetName=target)
    return clf.bn


def main():
    parser = argparse.ArgumentParser(description="Learn a Bayesian Network or BN Classifier from the German credit dataset.")
    parser.add_argument("mode", choices=["learner", "classifier"], help="Choose 'learner' for generic BN structure learning or 'classifier' for BNClassifier")
    parser.add_argument("--input", default="statlog+german+credit+data/german.data", help="Path to input raw data file")
    parser.add_argument("--output", default="credit_risk.bif", help="Output filename for the learned BN (extension should be supported by pyAgrum)")
    parser.add_argument("--score", default="aic", help="Scoring method for learner: aic|bic|log")
    parser.add_argument("--smoothing", type=int, default=1, help="Smoothing prior for learner")
    parser.add_argument("--method", default="TAN", help="Classifier learning method: Naive|TAN|ChowLiu|Tabu|GHC")
    parser.add_argument("--save-processed", default="", help="If set, save the processed data to specified CSV file")

    args = parser.parse_args()

    data = load_and_preprocess(args.input)
    if args.save_processed != "":
        data.to_csv(args.save_processed, index=False)

    if args.mode == "learner":
        bn = learn_bn(data, score=args.score, smoothing=args.smoothing)
    else:
        bn = learn_classifier(data, method=args.method)

    # Save BN
    gum.saveBN(bn, args.output, True)
    print(f"Saved BN to: {args.output}")


if __name__ == "__main__":
    main()
