from pyomo.environ import *
import pandas as pd
import math
import numpy as np

#Tankut hocanın yazdığı modeli kodladım

def yeni_model_nlp(teams,days,results_df,M=1000):


    # Create ConcreteModel
    model = ConcreteModel()
    # Variables
    model.x = Var(teams, teams, days, within=Binary)
    model.p = Param(teams, teams, within = Integers, mutable = True)
    model.d1 = Var(teams, teams, days, within=NonNegativeIntegers)
    model.d2 = Var(teams, teams, days, within=NonNegativeIntegers)
    model.y = Var(teams, teams, days, within=Reals)

    # Constraints
    model.constraints = ConstraintList()


    for i in teams:
        for j in teams:
            model.p[i, j] = results_df.iloc[i-1, j-1]



    for i in teams:
        for j in teams:
            for t in days:
                if i>=j:
                    model.constraints.add(model.x[i,j,t] == 0) 

    for i in teams:
        for j in teams:
            if i<j:
                model.constraints.add(sum(model.x[i,j,t] for t in days) == 1)

    for i in teams:
        for t in days:
            model.constraints.add(sum(model.x[i,j,t] + model.x[j,i,t] for j in teams if i!=j) == 1)



    for i in teams:
        for j in teams:
            if i<j:
                for t in days:
                    model.constraints.add(model.d1[i, j, t] - model.d2[i, j, t] ==
                    sum(
                        sum(model.p[i, j_prime] * model.x[i, j_prime, w] for w in range(1, t)) for j_prime in teams 
                    )
                    +sum(
                        sum(model.p[i, j_prime] * model.x[j_prime, i, w] for w in range(1, t)) for j_prime in teams 
                    )
                    -sum (
                        sum(model.p[j, j_prime] * model.x[j, j_prime, w] for w in range(1, t)) for j_prime in teams
                        )
                    -sum(
                        sum(model.p[j, j_prime] * model.x[j_prime, j, w] for w in range(1, t)) for j_prime in teams
                        )
                    )
    

    for t in days:
        for i in teams:
            for j in teams:
                model.constraints.add(model.d1[i,j,t] <= M)

    for t in days:
        for i in teams:
            for j in teams:
                model.constraints.add(model.d2[i,j,t] <= M)


    def rule_of(model):
        return sum(sum(sum(((model.d1[i,j,t] + model.d2[i,j,t])*model.x[i,j,t]) for t in days) for j in teams if i<j) for i in teams)


    model.obj = Objective(rule=rule_of, sense=minimize)


    
    # Solve the model
    solver = SolverFactory('gurobi', options={'MIPFocus':2, 'TimeLimit':2000}) 
    results= solver.solve(model, tee=True)
    obj_value = value(model.obj())
    #model.write(f"my_model10.lp", io_options={'symbolic_solver_labels': True})
    model.constraints.pprint()




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

     
    l=[]
    for t in days:
        for i in teams:
            for j in teams:
                if value(model.x[i,j,t] == 1):
                    print(f"{i} vs {j} slot {t} d'nin değeri= ")
                    #print(abs(value(model.d1[i,j,t]-model.d2[i,j,t])))
                    l.append(abs(value(model.d1[i,j,t]-model.d2[i,j,t])))
 




    return solution_df, obj_value, l
""""
    for _, row in real_solution.iterrows():
        day = row['Day']
        team1 = row['Team 1']
        team2 = row['Team 2']
        model.x[team1, team2, day].fix(1)

            for t in days:
        for i in teams:
            for j in teams:
                model.constraints.add(-model.y[i,j,t]<= (model.d1[i,j,t] - model.d2[i,j,t]))
    for t in days:
            for i in teams:
                for j in teams:
                    model.constraints.add((model.d1[i,j,t] - model.d2[i,j,t]) <= model.y[i,j,t])
        """