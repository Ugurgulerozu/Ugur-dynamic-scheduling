from pyomo.environ import *
import pandas as pd

def dynamic_model(teams,days, rating_df, solution_keeper, r):


    # Create ConcreteModel
    model = ConcreteModel()
    # Variables
    model.x = Var(teams, teams, days, within= Binary)
    model.u = Param(teams, teams, within=NonNegativeReals, mutable=True)
    model.max = Var(within = NonNegativeReals)



    # Constraints
    model.constraints = ConstraintList()


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


    # Read data from the file and assign it to model.u
    for i in teams:
        for j in teams:
            model.u[i, j] = rating_df.iloc[i - 1, j - 1]

    # Fiy variables based on the provided solution schedule
    #previous_round = []
    #previous_round.append(day)
    for _, row in solution_keeper.iterrows():
        day = row['Day']
        team1 = row['Team 1']
        team2 = row['Team 2']
        model.x[team1, team2, day].fix(1)

    
    def rule_of(model):
        return sum(sum(sum( model.u [i, j] * model.x[i, j, t] for t in r) for i in teams) for j in teams)




    model.obj = Objective(rule=rule_of, sense=minimize)

    
    # Solve the model
    solver = SolverFactory('gurobi')
    results= solver.solve(model)
    obj_value = round(model.obj(),0)
    #model.write(f"my_model{r[0]}.lp", io_options={'symbolic_solver_labels': True})

    solution_df = pd.DataFrame(columns=['Day', 'Team 1', 'Team 2'])
    for t in r:
        for i in teams:
            for j in teams:
                if model.x[i, j, t].value == 1:
                    new_row = pd.DataFrame({'Day': [t], 'Team 1': [i], 'Team 2': [j]})
                    solution_df = pd.concat([solution_df, new_row]).reset_index(drop=True)

    if results.solver.termination_condition == TerminationCondition.optimal:
        print("Optimal solution found.")
        for t in r:
            for i in teams:
                for j in teams:
                    if value(model.x[i, j, t]) == 1:
                        print(f"Team {i} vs Team {j} in Slot {t}")
        

    else:
        print("No optimal solution found.")


    return solution_df, obj_value

"""model.x[4,1,1].fix(1)
    model.x[6,12,1].fix(1)
    model.x[7,3,1].fix(1)
    model.x[8,5,1].fix(1)
    model.x[10,2,1].fix(1)
    model.x[15,11,1].fix(1)
    model.x[16,9,1].fix(1)
    model.x[18,17,1].fix(1)
    model.x[19,13,1].fix(1)
    model.x[20,14,1].fix(1)
    model.x[1,6,2].fix(1)
    model.x[5,2,2].fix(1)
    model.x[7,8,2].fix(1)
    model.x[9,15,2].fix(1)
    model.x[10,3,2].fix(1)
    model.x[12,4,2].fix(1)
    model.x[13,14,2].fix(1)
    model.x[20,17,2].fix(1)
    model.x[19,11,2].fix(1)
    model.x[18,16,2].fix(1)
    model.x[3,1,3].fix(1)
    model.x[4,7,3].fix(1)
    model.x[6,8,3].fix(1)
    model.x[10,5,3].fix(1)
    model.x[11,17,3].fix(1)
    model.x[12,2,3].fix(1)
    model.x[13,9,3].fix(1)
    model.x[14,16,3].fix(1)
    model.x[15,19,3].fix(1)
    model.x[20,18,3].fix(1)
    model.x[17,1,1].fix(1)
    model.x[18,2,1].fix(1)
    model.x[3,13,1].fix(1)
    model.x[4,16,1].fix(1)
    model.x[5,15,1].fix(1)
    model.x[6,12,1].fix(1)
    model.x[7,8,1].fix(1)
    model.x[14,10,1].fix(1)
    model.x[19,11,1].fix(1)
    model.x[20,9,1].fix(1)"""
