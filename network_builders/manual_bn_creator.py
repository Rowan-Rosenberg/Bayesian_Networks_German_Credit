import pyagrum as gum
import pandas as pd
import numpy as np

# For observed: Build fastVariable string from df unique values, with label sanitization for BIF compatibility
def build_fast_var_string(series, var_name):
    unique_vals = sorted(series.dropna().unique())
    if not unique_vals:
        print(f"No unique values found for {var_name}, defaulting to 'unknown'.")
        return f"{var_name}{{unknown}}"
    
    # Check if all values are integers (handles numpy int types)
    try:
        int_vals = [int(v) for v in unique_vals]
        vals_str = '|'.join(str(v) for v in sorted(int_vals))
        result = f"{var_name}{{{vals_str}}}"
        print(f"Integer values detected for {var_name}: {result}")
        return result
    except (ValueError, TypeError):
        pass
    
    # Check if all values are floats (handles numpy float types)
    try:
        float_vals = [float(v) for v in unique_vals]
        vals_str = '|'.join(str(v) for v in sorted(float_vals))
        # print(f"Float values detected for {var_name}: {vals_str}")
        return f"{var_name}{{{vals_str}}}"
    except (ValueError, TypeError):
        pass
    
    # Otherwise, treat as strings and sanitize
    cleaned_vals = []
    for v in unique_vals:
        s = str(v)
        # Replace invalid characters with underscores or safe alternatives
        s = s.replace(' ', '_').replace('-', '_').replace('<', 'lt').replace('>', 'gt').replace('>=', 'ge').replace('<=', 'le')
        s = s.replace('(', '_').replace(')', '_').replace('[', '_').replace(']', '_').replace('/', '_').replace('\\', '_')
        s = s.replace(':', '_').replace(';', '_').replace(',', '_').replace('.', '_')
        s = s.replace('__', '_')  # Avoid double underscores
        s = s.strip('_')  # Remove leading/trailing underscores
        # If starts with digit, prefix with underscore (for strings only)
        if s and s[0].isdigit():
            s = '_' + s
        # Ensure not empty
        if not s:
            s = 'unknown'
        cleaned_vals.append(s)
    vals_str = '|'.join(cleaned_vals)
    return f"{var_name}{{{vals_str}}}"

# variable information 
vars_df = pd.read_csv('Dataset/processed_credit.csv')
bn = gum.BayesNet('GermanCreditManualBN')

nodes = [
    "CheckingAccount", "SavingsAccount", "Property", "Housing",
    "EmploymentSince", "Job", "Income", "ResidenceSince",
    "CreditHistory", "ExistingCredits", "OtherInstallmentPlans", "OtherDebtors",
    "CreditAmount", "Duration", "InstallmentRate", "Purpose",
    "Age", "PersonalStatusSex", "LiablePeople", "ForeignWorker",
    "Capacity", "Character", "Capital", "Collateral", "Terms",
    "CreditRisk"
]

bn_vars = set(nodes)
df_vars = set(vars_df.columns)
observed_vars = df_vars.intersection(bn_vars)
unobserved_vars = bn_vars - observed_vars

# Proposed fastVariable strings for unobserved
unobserved_fast_vars = {
    "Capacity": "Capacity{low|medium|high}",
    "Character": "Character{bad|good}",
    "Capital": "Capital{low|medium|high}",
    "Collateral": "Collateral{none|some|good}",
    "Terms": "Terms{short|medium|long}",
    "Income": "Income{low|medium|high}"
}

# Update BN variables using fastVariable``
for var_name in bn_vars:
    if var_name in observed_vars:
        # Build from df
        fast_str = build_fast_var_string(vars_df[var_name], var_name)
        print(f"Built fastVariable for {var_name}: {fast_str}")
        new_var = gum.fastVariable(fast_str)
        bn.add(new_var)
    else:
        # Use proposed
        fast_str = unobserved_fast_vars[var_name]
        new_var = gum.fastVariable(fast_str)
        bn.add(new_var)

print("Domains updated using fastVariable format.")

bn.addArcs([
    ("CheckingAccount", "Capital"),
    ("SavingsAccount", "Capital"),
    ("Housing", "Capital"),
    ("Housing", "ResidenceSince"),
    
    ("ExistingCredits", "CreditHistory"),
    ("OtherInstallmentPlans", "CreditHistory"),
    
    ("CreditHistory", "Character"),
    ("EmploymentSince", "Character"),
    ("ResidenceSince", "Character"),
    ("OtherInstallmentPlans", "Character"),
    ("Purpose", "Character"),


    ("EmploymentSince", "Income"),
    ("Job", "Income"),
    ("Income", "CheckingAccount"),
    ("Income", "SavingsAccount"),

    ("Income", "Capacity"),
    ("ExistingCredits", "Capacity"),
    ("OtherDebtors", "Capacity"),
    ("LiablePeople", "Capacity"),


    ("CreditAmount", "Terms"),
    ("Duration", "Terms"),
    ("InstallmentRate", "Terms"),
    

    ("Age", "EmploymentSince"),
    ("Age", "ResidenceSince"),
    ("Age", "PersonalStatusSex"),
    ("Age", "OtherDebtors"),
    ("Age", "Job"),

    ("ForeignWorker", "Job"),
    ("PersonalStatusSex", "LiablePeople"),
    

    ("Property", "Collateral"),
    ("Housing", "Collateral"),

    ("Character", "CreditRisk"),
    ("Capacity", "CreditRisk"),
    ("Capital", "CreditRisk"),
    ("Collateral", "CreditRisk"),
    ("Terms", "CreditRisk")
])

gum.saveBN(bn, "GermanCreditManual.bif")
