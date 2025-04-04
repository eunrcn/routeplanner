import pandas as pd
import random
import gurobipy as gp
from gurobipy import GRB

# Load the Excel file with the first column as the index
file_name = "data.xlsx"  # Ensure the file is in the correct location
Weather = pd.read_excel(file_name, sheet_name="Weather exposure", index_col=0)
Safety = pd.read_excel(file_name, sheet_name="Safety", index_col=0)
Time = pd.read_excel(file_name, sheet_name="Travel time", index_col=0)

ws = 0.3
we = 0.5
wt = 0.3

m = gp.Model("SinglePathFlow")

# Create variables
x = {}
f = {}
for i in Safety.index:
    for j in Safety.columns:
        x[(i,j)] = m.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")
        f[(i,j)] = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"f_{i}_{j}")

# 1) Objective on x_{ij}, NOT on f_{ij}
m.setObjective(
    gp.quicksum(
        (ws * Safety.loc[i,j] + we * Weather.loc[i,j] + wt * Time.loc[i,j]) * x[(i,j)]
        for i in Safety.index
        for j in Safety.columns
    ),
    GRB.MINIMIZE
)

# 2) Coupling: flow only on edges that are "activated"
for i in Safety.index:
    for j in Safety.columns:
        m.addConstr(f[(i,j)] <= x[(i,j)], name=f"flow_coupling_{i}_{j}")

# 3) Flow constraints
# Let A=1, B=14
A, B = 12, 5

# Exactly 1 unit of flow leaves A
m.addConstr(
    gp.quicksum(f[(A,j)] for j in Safety.columns if j != A)
    - gp.quicksum(f[(i,A)] for i in Safety.index if i != A)
    == 1,
    name="flow_out_of_A"
)

# Exactly 1 unit of flow arrives at B
m.addConstr(
    gp.quicksum(f[(B,j)] for j in Safety.columns if j != B)
    - gp.quicksum(f[(i,B)] for i in Safety.index if i != B)
    == -1,
    name="flow_into_B"
)

# Flow conservation for other nodes
for i in Safety.index:
    if i not in [A, B]:
        m.addConstr(
            gp.quicksum(f[(i,j)] for j in Safety.columns if j != i)
            - gp.quicksum(f[(k,i)] for k in Safety.index if k != i)
            == 0,
            name=f"flow_balance_{i}"
        )

# Optimize
m.optimize()

if m.status == GRB.OPTIMAL:
    print("Optimal solution cost:", m.objVal)
    chosen_edges = [(i,j) for (i,j) in x if x[(i,j)].X > 0.5]
    print("Edges used:")
    f = 0
    for edge in chosen_edges:
        if f == 0:
            for edg in chosen_edges:
                if edg[0] == A:
                    print(edg)
                    C = edg[1]
                    f += 1
                else:
                    pass
        else:
            for edg in chosen_edges:
                if edg[0] == C:
                    print(edg)
                    C = edg[1]
                else:
                    pass                 
else:
    print("No optimal solution found. Status:", m.status)
    
