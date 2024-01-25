from pyomo.environ import ConcreteModel, ConstraintList, Var, Param, Objective,TerminationCondition,minimize, Binary, NonNegativeReals, SolverFactory, value
import pandas as pd

def first_round_model(teams,days, data_file, round_number):

    # Create ConcreteModel
    model = ConcreteModel()
    # Variables
    model.x = Var(teams, teams, days, within=Binary)
    model.u = Param(teams, teams, within=NonNegativeReals, mutable=True)


    # Constraints
    model.constraints = ConstraintList()
    
    for t in days:
        for i in teams:
            model.constraints.add(model.x[i,i,t] == 0)


    for i in teams:
        for j in teams:
            if i<j:
                model.constraints.add(sum(model.x[i,j,t] + model.x[j,i,t] for t in days) == 1)

    for i in teams:
        for t in days:
            model.constraints.add(sum(model.x[i,j,t] + model.x[j,i,t] for j in teams if j!=i ) == 1)


    # Read data from the file and assign it to model.u
    data = data_file
    for i in teams:
        for j in teams:
            model.u[i, j] = data.iloc[i - 1, j - 1]

    # Fix variables based on the provided solution schedule


    def rule_of(model):
        return ((sum(sum(sum((model.u[i, j] * model.x[i, j, t]) for t in round_number) for j in teams )for i in teams)))


    model.obj = Objective(rule=rule_of, sense=minimize)
    
    # Solve the model
    solver = SolverFactory('gurobi')
    results= solver.solve(model)
    obj_value = round(model.obj(),0)
    model.write("my_model2.lp", io_options={'symbolic_solver_labels': True})

    solution_df = pd.DataFrame(columns=['Day', 'Team 1', 'Team 2'])
    for t in round_number:
        for i in teams:
            for j in teams:
                if model.x[i, j, t].value == 1:
                    new_row = pd.DataFrame({'Day': [t], 'Team 1': [i], 'Team 2': [j]})
                    solution_df = pd.concat([solution_df, new_row]).reset_index(drop=True)

    if results.solver.termination_condition == TerminationCondition.optimal:
        print("Optimal solution found.")
        for t in round_number:
            for i in teams:
                for j in teams:
                    if value(model.x[i, j, t]) == 1:
                        print(f"Team {i} vs Team {j} in Slot {t}")
        

    else:
        print("No optimal solution found.")


    return solution_df, obj_value