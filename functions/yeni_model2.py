from pyomo.environ import *
import pandas as pd
import math
import numpy as np

#d(i,j,t)leri d(i,t) olarak değiştirdim

def yeni_model2(teams,days,results_df ,first_round_solution, M=1000):


    # Create ConcreteModel
    model = ConcreteModel()
    # Variables
    model.x = Var(teams, teams, days, within=Binary)
    model.p = Param(teams, teams, within = Integers, mutable = True)
    model.d1 = Var(teams, days, within=Reals)
    model.d2 = Var(teams, days, within=Reals)
    model.y = Var(teams, teams, days, within = Reals)
    model.z = Var(teams, teams, days, within = Reals)
    
    # Constraints
    model.constraints = ConstraintList()

    
    for i in teams:
        for j in teams:
            model.p[i, j] = results_df.iloc[i-1, j-1]

    for _, row in first_round_solution.iterrows():
            day = row['Day']
            team1 = row['Team 1']
            team2 = row['Team 2']
            model.x[team1, team2, day].fix(1)

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

    for j in teams:

        for t in days:
            model.constraints.add(model.d2[j, t] >= 0)

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
 
    for j in teams:
        for t in days:
            model.constraints.add( expr= model.d2[j, t] ==
            sum (
                sum(model.p[j, j_prime] * model.x[j, j_prime, w] for w in range(1, t)) for j_prime in teams
                )
            +sum(
                sum(model.p[j, j_prime] * model.x[j_prime, j, w] for w in range(1, t)) for j_prime in teams
                )
            )
    for i in teams:
        for j in teams:
            for t in days:
                model.constraints.add(model.y[i, j, t] <= M * model.x[i, j, t])
    for i in teams:
        for j in teams:
            for t in days:
                model.constraints.add(model.y[i, j, t] <= model.d1[i, t])

    for i in teams:
        for j in teams:
            for t in days:
                model.constraints.add(model.z[i, j, t] <= M * model.x[i, j, t])
    for i in teams:
        for j in teams:
            for t in days:
                model.constraints.add(model.z[i, j, t] <= model.d2[j, t])

    for i in teams:
        for j in teams:
            for t in days:
                model.constraints.add(expr= model.y[i, j, t] + M >= model.d1[i, t] + M * model.x[i, j, t])

    for i in teams:
        for j in teams:
            for t in days:
                model.constraints.add(expr= model.z[i, j, t] + M >= model.d2[j, t] + M * model.x[i, j, t])

    for i in teams:
        for j in teams:
            for t in days:
                model.constraints.add(model.y[i, j, t] >= 0)

    for i in teams:
        for j in teams:
            for t in days:
                model.constraints.add(model.z[i, j, t] >= 0)

    def rule_of(model):
        return sum(sum(sum(((model.y[i, j, t] - model.z[i, j, t])**2) for t in days) for j in teams) for i in teams)


    model.obj = Objective(rule=rule_of, sense=minimize)

    
    # Solve the model
    solver = SolverFactory('gurobi')
    results= solver.solve(model)
    obj_value = value(model.obj())
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
            
    for t in days:
        for i in teams:
            for j in teams:
                if value(model.x[i,j,t] == 1):
                    print(f"{i} vs {j} slot {t} y'nin değeri= ")
                    print({value(model.y[i,j,t])})

    for t in days:
            for i in teams:
                for j in teams:
                    if value(model.x[i,j,t] == 1):
                        print(f"{i} vs {j} slot {t} z'nin değeri= ")
                        print({value(model.z[i,j,t])})
    
    for t in days:
        for i in teams:
            for j in teams:
                if value(model.x[i,j,t] == 1):
                    print(f"{i} vs {j} slot {t} d1'nin değeri= ")
                    print({value(model.d1[i,t])})
    for t in days:
        for i in teams:
            for j in teams:
                if value(model.x[i,j,t] == 1):
                    print(f"{j} vs {i} slot {t} d2'nin değeri= ")
                    print({value(model.d2[j,t])})


    return solution_df, obj_value
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