from pyomo.environ import *
import pandas as pd
import math
import numpy as np

#d(i,j,t)leri d(i,t) olarak değiştirdim

def yeni_model2(teams,days,results_df ,M=1000):


    # Create ConcreteModel
    model = ConcreteModel()
    # Variables
    model.x = Var(teams, teams, days, within=Binary)
    model.p = Param(teams, teams, within = Integers, mutable = True)
    model.d1 = Var(teams, days, within=NonNegativeIntegers)
    model.y = Var(teams, teams, days, within = NonNegativeReals)
    model.z = Var(teams, teams, days, within = NonNegativeReals)
    
    # Constraints
    model.constraints = ConstraintList()


        
    for i in teams:
        for j in teams:
            model.p[i, j] = results_df.iloc[i-1, j-1]


    for t in days:
        for i in teams:
            model.constraints.add(model.x[i,i,t] == 0)

    for i in teams:
        for j in teams:
            if i<j:
                model.constraints.add(sum((model.x[i,j,t] + model.x[j,i,t]) for t in days) == 1)

    for i in teams:
        for t in days:
            model.constraints.add(sum((model.x[i,j,t] + model.x[j,i,t]) for j in teams if j!=i ) == 1)

    for i in teams:

        for t in days:
            model.constraints.add(model.d1[i, t] >= 0)


    for i in teams:  
        for t in days:
            model.constraints.add( expr= model.d1[i, t] ==
            sum(
                sum(model.p[i, j_prime] * model.x[i, j_prime, w] for w in range(1, t)) for j_prime in teams 
            )
            +sum(
                sum(model.p[i, j_prime] * model.x[j_prime, i, w] for w in range(1, t)) for j_prime in teams 
            )

            )
 
    for i in teams:
        for j in teams:
            for t in days:
                model.constraints.add(-model.y[i,j,t]<= (model.d1[i,t] - model.d1[j,t])*model.x[i,j,t])
    for i in teams:
        for j in teams:
            for t in days:
                model.constraints.add((model.d1[i,t] - model.d1[j,t])*model.x[i,j,t] <= model.y[i,j,t])

    def rule_of(model):
        return sum(sum(sum((model.y[i,j,t]) for t in days) for j in teams) for i in teams)


    model.obj = Objective(rule=rule_of, sense=minimize)

    
    # Solve the model
    solver = SolverFactory('gurobi') 
    results= solver.solve(model, tee=True)
    obj_value = value(model.obj())
 # Adjust TimeLimit as needed

    #model.constraints.pprint()
    #model.write(f"my_model2.lp", io_options={'symbolic_solver_labels': True})

    solution_df = pd.DataFrame(columns=['Day', 'Team 1', 'Team 2'])
    for t in days:
        for i in teams:
            for j in teams:
                if model.x[i, j, t].value == 1:
                    new_row = pd.DataFrame({'Day': [t], 'Team 1': [i], 'Team 2': [j]})
                    solution_df = pd.concat([solution_df, new_row]).reset_index(drop=True)

    if results.solver.termination_condition == TerminationCondition.optimal:
        print("Optimal solution found.")
        for t in days:
            for i in teams:
                for j in teams:
                    if value(model.x[i, j, t]) == 1:
                        print(f"Team {i} vs Team {j} in Slot {t}")
  

    else:
        print("No optimal solution found.")
            

    l2=[]
    l=[]
    for t in days:
        for i in teams:
            for j in teams:
                if value(model.x[i,j,t] == 1):
                    #print(f"{i} vs {j} slot {t} d'nin değeri= ")
                    #print(abs(value(model.y[i,j,t]-model.z[i,j,t])))
                    l.append(abs(value(model.y[i,j,t]-model.z[i,j,t])))

    y_df = pd.DataFrame(columns=['Day', 'Team 1', 'Value'])
    for t in days:
        for i in teams:
            for j in teams:
                if value(model.x[i,j,t] == 1):
                    new_row1 = pd.DataFrame({'Day': [t], 'Team 1': [i], 'Value': [value(model.y[i,j,t])]})
                    y_df = pd.concat([y_df, new_row1]).reset_index(drop=True)
                    #print(f"{i} vs {j} slot {t} y'nin değeri= ")
                    print({value(model.y[i,j,t])})

    z_df = pd.DataFrame(columns=['Day', 'Team 2', 'Value'])
    for t in days:
        for i in teams:
            for j in teams:
                if value(model.x[i,j,t] == 1):
                    new_row1 = pd.DataFrame({'Day': [t], 'Team 2': [j], 'Value': [value(model.z[i,j,t])]})
                    z_df = pd.concat([z_df, new_row1]).reset_index(drop=True)
                    #print(f"{i} vs {j} slot {t} z'nin değeri= ")
                    print({value(model.z[i,j,t])})



  

    return solution_df, y_df, z_df, obj_value, l, l2
"""
for t in days:
        for i in teams:
            for j in teams:
                if j<i:
                    model.constraints.add(model.x[i,j,t] == 0)

            for t in days:
        print(f"round {t}:")
        for i in teams:
            print(f"{i}. takimin kazandiği puani:")
            print(value(model.p[i,t]))
    """
