import pyagrum as gum

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

for node in nodes:
    bn.add(node)


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