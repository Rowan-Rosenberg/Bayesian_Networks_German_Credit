"""
    Convert a learned Bayesian Network into an Influence Diagram
"""

import argparse
import pyagrum as gum
import pyagrum.lib.notebook as gnb

def create_credit_influence_diagram(bn_path: str):

    # Load the learned Bayesian Network
    print(f"Loading Bayesian Network from: {bn_path}")
    bn = gum.loadBN(bn_path)

    # Create Influence Diagram
    id = gum.InfluenceDiagram()

    # Add all chance nodes from the BN to the Influence Diagram
    for node_name in bn.names():
        id.addChanceNode(bn.variable(node_name))
        
    # Add arcs from the BN to the Influence Diagram
    for parent_id, child_id in bn.arcs():
        parent_name = bn.variable(parent_id).name()
        child_name = bn.variable(child_id).name()
        id.addArc(parent_name, child_name)

    # Copy the CPTs from the BN to the Influence Diagram
    for node_name in bn.names():
        id.cpt(node_name).fillWith(bn.cpt(node_name))


    # Decision Node: whether to approve or reject the loan
    decision_name = "ApproveLoan"
    decision_node = gum.LabelizedVariable(decision_name, "Approve or reject the loan", 2)
    decision_node.changeLabel(0, "Approve")
    decision_node.changeLabel(1, "Reject")
    id.addDecisionNode(decision_node)

    # Add arcs: decision affects utility, and credit risk affects both decision outcome and utility
    id.addArc("CreditRisk", decision_name)

    # Define Utility Node
    utility_name = "Utility"
    utility_node = gum.LabelizedVariable(utility_name, "Financial outcome of decision", 1)
    id.addUtilityNode(utility_node)

    # Arcs: both the decision and the actual credit risk determine the utility
    id.addArc(decision_name, utility_name)
    id.addArc("CreditRisk", utility_name)

    # Define Utility Table based on suggestions in dataset documentation
    util = id.utility(utility_name)
    util["Good", "Approve"] = 0     
    util["Bad", "Approve"] = -5     
    util["Good", "Reject"] = -1     
    util["Bad", "Reject"] = 0           
    id.utility(utility_name).fillWith(util)

    print("\n Influence Diagram constructed:")

    return id


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Create an Influence Diagram from a learned Bayesian Network.")
    parser.add_argument("--bn-path", default="credit_risk.bif", help="Path to the learned Bayesian Network file (BIF format)")
    parser.add_argument("--id-path", default="credit_decision_influence_diagram.xml", help="Path to save the Influence Diagram (XML format)")
    args = parser.parse_args()

    BN_PATH = args.bn_path
    ID_PATH = args.id_path
    
    id = create_credit_influence_diagram(BN_PATH)
    gum.saveID(id, ID_PATH)
