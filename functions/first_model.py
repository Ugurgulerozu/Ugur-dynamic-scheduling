import random
from pyomo.environ import ConcreteModel, Objective, ConstraintList, Var, Binary, SolverFactory, TerminationCondition, SolverStatus,value
import pandas as pd


def first_model(teams,days):

    n = len(teams)
    model = ConcreteModel()

    model.x = Var(teams, teams, days, within=Binary)

    # Constraints
    model.constraints = ConstraintList()


    for t in days:
        for i in teams:
            model.constraints.add(model.x[i,i,t] == 0)

    for t in days:
        for i in teams:
            for j in teams:
                if j<i:
                    model.constraints.add(model.x[i, j, t] == 0)

    for i in teams:
        for j in teams:
            if i<j:
                model.constraints.add(sum(model.x[i,j,t] + model.x[j,i,t] for t in days) == 1)

    for i in teams:
        for t in days:
            model.constraints.add(sum(model.x[i,j,t] + model.x[j,i,t] for j in teams if j!=i ) == 1)

    solver = SolverFactory('gurobi')
    results = solver.solve(model, tee=True)

    
    # Print the results
    if results.solver.termination_condition == TerminationCondition.optimal:
        print("Optimal solution found.")
        for t in days:
            for i in teams:
                for j in teams:
                        if value(model.x[i, j, t]) == 1:
                            print(f"Team {i} vs Team {j} in Slot {t}")
            

    else:
        print("No optimal solution found.")



    solution_df = pd.DataFrame(columns=['Day', 'Team 1', 'Team 2'])

    for t in days:
        for i in teams:
            for j in teams:
                    if model.x[i, j, t].value == 1:
                        solution_df = pd.concat([solution_df, pd.DataFrame({'Day': [t], 'Team 1': [i], 'Team 2': [j]})],
                                            ignore_index=True)
    
  
    return solution_df
