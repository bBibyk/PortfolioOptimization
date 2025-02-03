import pyscipopt
import numpy as np
import pandas as pd
import os
import time

script_folder = os.path.dirname(os.path.abspath(__file__))

model = pyscipopt.Model("portfolio")

df = pd.read_csv(script_folder+"/data.csv").fillna(0)
l = len(df)

take = np.empty(l, dtype=object)
part = np.empty(l, dtype=object)


for i in range(0, l):
    part[i] = model.addVar(vtype="I", name="x", lb=0, ub=100)
    take[i] = model.addVar(vtype="B", name="y")

r = np.empty(l)
ret = df["last_three_years"]
min_r = min(ret)
max_r = max(ret)
c = np.empty(l)
min_c = min(df["valor"])
max_c = max(df["valor"])
d = np.empty(l)
min_d = min(df["number_of_holdings"])
max_d = max(df["number_of_holdings"])
v = np.empty(l)
min_v = min(df["last_three_years_volatility"])
max_v = max(df["last_three_years_volatility"])
for i in range(l):
    r[i] = (ret[i]-min_r)/(max_r-min_r)
    c[i] = (df["valor"][i]-min_c)/(max_c-min_c)
    d[i] = (df["number_of_holdings"][i]-min_d)/(max_d-min_d)
    v[i] = (df["last_three_years_volatility"][i]-min_v)/(max_v-min_v)


for i in range(0, l):
    model.addCons(part[i]<=take[i]*100)
    model.addCons(take[i]<=part[i])

model.addCons(pyscipopt.quicksum(take[i] for i in range(l)) == 10) #mod

model.addCons(pyscipopt.quicksum(part[i] for i in range(l)) == 100)

model.addCons(pyscipopt.quicksum(part[i]*(df["asset_class"][i]=="Equity") for i in range(l)) == 75) #mod
model.addCons(pyscipopt.quicksum(part[i]*(df["asset_class"][i]=="Money Market") for i in range(l)) == 2) #mod
model.addCons(pyscipopt.quicksum(part[i]*(df["asset_class"][i]=="Cryptocurrencies") for i in range(l)) == 5) #mod
model.addCons(pyscipopt.quicksum(part[i]*(df["asset_class"][i]=="Precious Metals") for i in range(l)) == 5) #mod
model.addCons(pyscipopt.quicksum(part[i]*(df["asset_class"][i]=="Bonds") for i in range(l)) == 3) #mod
model.addCons(pyscipopt.quicksum(part[i]*(df["asset_class"][i]=="Real Estate") for i in range(l)) == 5) #mod
model.addCons(pyscipopt.quicksum(part[i]*(df["asset_class"][i]=="Commodities") for i in range(l)) == 5) #mod


model.setObjective(pyscipopt.quicksum(part[i]*(r[i]+c[i]+d[i]-v[i]) for i in range(l)), "maximize")


model.printStatistics()
model.hideOutput()
model.optimize()

if model.getStatus() == "optimal":
    model.printSol()
    print(f"Optimal Objective Value: {model.getObjVal()}\n")
    print("Selected Investments:\n")
    
    for i in range(l):
        take_val = model.getVal(take[i])
        part_val = model.getVal(part[i])
        
        if take_val > 0.5:  # Binary variable, check if it is taken
            print(f"Part : {part_val}, ISIN : {df["isin"][i]}, Class : {df['asset_class'][i]}, Name : {df["name"][i]}")
else:
    print("No optimal solution found.")