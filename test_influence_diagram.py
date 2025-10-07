"""
    Evaluate total utility of a learned Influence Diagram on a dataset
"""

import pyagrum as gum
import pandas as pd
import re

# --- USER SETTINGS ---
ID_PATH = "credit_decision_influence_diagram_ghc.xml"
DATA_PATH = "processed_credit.csv"
DECISION_NODE = "ApproveLoan"
UTILITY_NODE = "Utility"
OUTCOME_NODE = "CreditRisk"


def normalize_label(label):
    """Normalize a label to match pyAgrum’s internal naming rules."""
    if pd.isna(label):
        return label
    s = str(label)
    # Replace *sequences* of non-word characters with a single underscore
    s = re.sub(r"[^\w]+", "_", s)

    # If starts with a number, prepend underscore
    if re.match(r"^\d", s) and not s.isnumeric():
        s = "_" + s

    return s


def build_label_map(id):
    """
    Build mapping from raw label -> normalized label for each variable in the ID.
    Example: {'CreditRisk': {'Good': 'Good', 'Bad': 'Bad'}}
    """
    mapping = {}
    for node_name in id.names():
        var = id.variable(node_name)
        label_map = {}
        for i in range(var.domainSize()):
            raw_label = var.label(i)
            normalized_label = normalize_label(raw_label)
            label_map[raw_label] = normalized_label
            label_map[normalized_label] = normalized_label  # self-map
        mapping[node_name] = label_map
    return mapping


def map_evidence(evidence, label_map):
    """Convert evidence labels to match ID’s internal normalized labels."""
    mapped = {}
    for var, val in evidence.items():
        if var in label_map:
            mapped[var] = label_map[var].get(val, normalize_label(val))
        else:
            mapped[var] = normalize_label(val)
    return mapped


def evaluate_influence_diagram(id_path=ID_PATH, data_path=DATA_PATH):
    print(f"Loading Influence Diagram: {id_path}")
    id = gum.loadID(id_path)

    print(f"Loading dataset: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")

    # Build label normalization map
    label_map = build_label_map(id)

    # Setup inference engine
    ie = gum.ShaferShenoyLIMIDInference(id)
    ie.makeInference()

    total_utility = 0
    total_cases = 0

    for index, row in df.iterrows():
        # Collect evidence (exclude outcome, decision, utility)
        evidence = {}
        for col in df.columns:
            if col in id.names() and col not in [OUTCOME_NODE, DECISION_NODE, UTILITY_NODE]:
                evidence[col] = str(row[col])

        # Normalize evidence
        evidence = map_evidence(evidence, label_map)

        # Enter evidence
        ie.setEvidence(evidence)
        ie.makeInference()

        # Estimate the posterior over CreditRisk given current evidence
        predicted_risk = ie.posterior(OUTCOME_NODE).argmax()
        best_decision = predicted_risk[0][0].get('CreditRisk')

        true_risk = str(row[OUTCOME_NODE])

        # Retrieve realized utility
        realized_utility = 0
        if true_risk == "Bad" and best_decision == 1:
            realized_utility = -5
        elif true_risk == "Good" and best_decision == 0:
            realized_utility = -1

        total_utility += realized_utility
        total_cases += 1

        ie.eraseAllEvidence()

        #if index < 1:  # Print first 1 cases for verification
        #    print(f"\nCase {index + 1}:")
        #   print(f" Evidence: {evidence}")
        #    print(f" Optimal Decision: {best_decision}")
        #    print(f" True Credit Risk: {true_risk}")
        #    print(f" Realized Utility: {realized_utility}")

    avg_utility = total_utility / total_cases if total_cases else 0
    print("\n=== Evaluation Results ===")
    print(f"Total Utility: {total_utility:.2f}")
    print(f"Average Utility per case: {avg_utility:.2f}")

    return total_utility, avg_utility


if __name__ == "__main__":
    evaluate_influence_diagram()
