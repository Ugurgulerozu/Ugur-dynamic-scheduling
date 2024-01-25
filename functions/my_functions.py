import random
from pyomo.environ import *
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
 

def Colleys_rating(input_file, teams):

    results_df = pd.read_excel(input_file)
    
    C_matrix = (np.ones((len(teams), len(teams)))) * -1
    C_matrix[np.diag_indices_from(C_matrix)] = 12

    # Get all unique team identifiers from the DataFrame
    all_teams = set(results_df['Team 1']) | set(results_df['Team 2'])

    # Create empty dictionaries to store the wins and losses for each team
    wins = {team: 0 for team in all_teams}
    losses = {team: 0 for team in all_teams}

    # Iterate over each row in the DataFrame
    for _, row in results_df.iterrows():
        team1 = row['Team 1']
        team2 = row['Team 2']
        winner = row['Winner']
        loser = row['Loser']

        # Count the wins for the winning team
        if winner in wins:
            wins[winner] += 1
        else:
            wins[winner] = 1

        # Count the losses for the losing team
        if loser in losses:
            losses[loser] += 1
        else:
            losses[loser] = 1


    wins = {k: wins[k] for k in sorted(wins)}
    losses = {k: losses[k] for k in sorted(losses)}


    n = len(teams) 
    b_i = np.zeros(n) 


    for i in range(1, n + 1):
        num_wins = wins.get(i, 0)
        num_losses = losses.get(i, 0)
        b_i[i - 1] = 1 + ((num_wins - num_losses) / 2)

    r_i = np.linalg.solve(C_matrix, b_i)
    r_i =np.array(r_i)
  
  
    n = len(teams)  

    U_ij = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            U_ij[i, j] = round((r_i[i] - r_i[j]) ** 2, 3)

    ratings_df = pd.DataFrame(U_ij, columns=teams)

    return ratings_df


def round_generate(days_list, p):
    if not days_list or p <= 0:
        return "Invalid input"
    
    separated_rounds = []
    num_days = len(days_list)
    
    # Create the first round with only day 1
    first_round = [1]
    separated_rounds.append(first_round)
    
    remaining_days = days_list[1:]  # Exclude day 1
    
    cumulative_days = [1]  # Initialize with day 1
    
    for i in range(num_days - 1):  # Iterate through the remaining days
        cumulative_days.append(remaining_days[i])  # Add one more day each time
        separated_rounds.append(cumulative_days.copy())  # Append a copy of the cumulative days

    return separated_rounds


def generate_results(teams):

    n = len(teams)
    matches = []

    for i in range(n):
        for j in range(i + 1, n):
            team1 = teams[i]
            team2 = teams[j]

            result = random.choice([team1, team2, 'tie'])

            matches.append({'Team 1': team1, 'Team 2': team2, 'Result': result})

    result_df = pd.DataFrame(matches)
    

    return result_df

def point_rating(old_ratings, teams, solution_df, result_df, r, p):
    all_teams = set(teams)
    n = len(all_teams)
    ratings = old_ratings
    # Ratings of English teams
    #ratings = {1:2035,2:2025,3:1899,4:1867,5:1851,6:1811,7:1783,8:1779,9:1760,10:1757,11:1754,12:1741,13:1725,14:1714,15:1703,16:1699}

    sublists = []

    # Define the number of elements in each sublist (except the last one)
    elements_per_sublist = p

    # Calculate the number of sublists needed
    num_sublists = (len(r) + elements_per_sublist - 1) // elements_per_sublist
    # Create the sublists
    for i in range(num_sublists):
        sublist = r[i * elements_per_sublist : (i + 1) * elements_per_sublist]
        sublists.append(sublist)
    for i in sublists:
        h=i
    relevant_matches = pd.DataFrame(columns=['Team 1', 'Team 2', 'Result'])
    for _, row in solution_df.iterrows():
        team1 = row['Team 1']
        team2 = row['Team 2']
        day = row['Day']
        
        for _, row in result_df.iterrows():
            team11 = row ['Team 1']
            team22 = row ['Team 2']
            result = row ['Result']

            for i in h:
                if day == i:
                    if team1 == team11 and team2 == team22 or team1 == team22 and team2 == team11 :
                        relevant_matches = pd.concat([relevant_matches, pd.DataFrame({'Team 1': [team1], 'Team 2': [team2], 'Result': [result]})],
                                                    ignore_index=True)
    print(relevant_matches)
    # Iterate over each row in the DataFrame
    for _, row in relevant_matches.iterrows():
        team1 = row['Team 1']
        team2 = row['Team 2']
        winner = row['Result']

        if winner == team1:
            ratings[team1] += 3  
        elif winner == team2:
            ratings[team2] += 3
        else:
            ratings[team1] += 1
            ratings[team2] += 1

    ratings_array = np.array(list(ratings.values()))

    U_ij = np.zeros((n, n)) 

    for i in range(n):
        for j in range(n):
            U_ij[i, j] = abs((ratings_array[i] - ratings_array[j]))

    output_ratings = {index: value for index, value in enumerate(ratings_array, start=1)}
    print(output_ratings)
    ratings_df = pd.DataFrame(U_ij, columns=teams, index=teams)

    return output_ratings, ratings_df



def match_result_prediction(solution_df, ratings, result_df):
    results = []
    
    merged_df = pd.merge(solution_df, result_df)

    # Filter rows where 'result' is NaN (indicating a difference)
    difference_df = merged_df[merged_df['result'].isna()]

    # Reset the index of the resulting dataframe
    difference_df.reset_index(drop=True, inplace=True)

    # Display the difference dataframe
    print(difference_df)
    
    relevant_matches = pd.DataFrame(columns=['Team 1', 'Team 2', 'Result'])
    for _, row in solution_df.iterrows():
        team1 = row['Team 1']
        team2 = row['Team 2']

        for _, row in result_df.iterrows():
            team11 = row ['Team 1']
            team22 = row ['Team 2']
            result = row ['Result']

    
            if team1 != team11 or team2 != team11:
                relevant_matches = pd.concat([relevant_matches, pd.DataFrame({'Team 1': [team1], 'Team 2': [team2], 'Result': [result]})],
                                            ignore_index=True)
    print(relevant_matches)
    # Iterate over each row in the DataFrame
    for _, row in difference_df.iterrows():
        team1 = row['Team 1']
        team2 = row['Team 2']
        

        expected_score1 = 1 / (1 + 10 ** ((ratings[team2] - ratings[team1]*10) / 400))
        expected_score2 = 1 / (1 + 10 ** ((ratings[team1] - ratings[team2]*10) / 400))

        # Assign the expected scores as probabilities
        probability_team1_wins = expected_score1 - 0.15
        probability_team2_wins = expected_score2 - 0.15
        probability_draw = 1 - probability_team1_wins - probability_team2_wins

        # Generate a random number between 0 and 1
        random_number = random.random()

        # Compare with probabilities to determine the match result
        if random_number < expected_score1:
            result = team1
        elif random_number < expected_score2 + expected_score1:
            result = team2
        else:
            result = 'tie'

        result_df = pd.concat([result_df, pd.DataFrame({'Team 1': [team1], 'Team 2': [team2], 'Result': [result]})],
                                            ignore_index=True)
        results.append(result)


    return result_df


def r_i_rating(teams, solution_df, result_df, r):

    all_teams = set(teams)
    n = len(all_teams)


    relevant_matches = pd.DataFrame(columns=['Team 1', 'Team 2', 'Result'])
    for _, row in solution_df.iterrows():
        team1 = row['Team 1']
        team2 = row['Team 2']
        day = row['Day']

        for _, row in result_df.iterrows():
            team11 = row ['Team 1']
            team22 = row ['Team 2']
            result = row ['Result']

    
            if team1 == team11 and team2 == team22 or team1 == team22 and team2 == team11 :
                relevant_matches = pd.concat([relevant_matches, pd.DataFrame({'Team 1': [team1], 'Team 2': [team2], 'Result': [result]})],
                                            ignore_index=True)
    team_wins = {i: 0 for i in range(1, 21)}

    # Iterate through the rows of the DataFrame
    for _, row in relevant_matches.iterrows():
        team1 = row['Team 1']
        team2 = row['Team 2']
        result = row['Result']

        if team1 == result:
            team_wins.update({team1:team_wins[team1]+1})
        elif team2 == result:
            team_wins.update({team2:team_wins[team2]+1})

    

    r_i_list=[]
    for i in range(1,21):
        w_i = {i: team_wins[i] for i in range(1, 21)}
        ow_i = {i: sum(team_wins[j] for j in range(1,21) if j != i) for i in range(1,21)}
        oow_i = {i: sum(sum(team_wins[k] for k in range(1,21) if k != j) for j in range(1,21)  if i != j) for i in range(1, 21)}
        
        r_i =  200/3* w_i[i]/day+ 100/3* ow_i[i]/day
        r_i_list.append(r_i)
        print(r_i)
    ratings_array = np.array(r_i_list)
    U_ij = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            U_ij[i, j] = abs((ratings_array[i] - ratings_array[j]))
  
    output_ratings = {index: value for index, value in enumerate(ratings_array, start=1)}
    print(output_ratings)
    rating_df = pd.DataFrame(U_ij, columns=teams, index=teams)
    
    return output_ratings, rating_df



def Elo_rating(old_ratings, teams, solution_df, result_df, r, k=20):

    all_teams = set(teams)
    n = len(all_teams)

    # Ratings of English teams
    #ratings = {1:2035,2:2025,3:1899,4:1867,5:1851,6:1811,7:1783,8:1779,9:1760,10:1757,11:1754,12:1741,13:1725,14:1714,15:1703,16:1699}
    # Ratings of Spanish teams

    ratings = old_ratings


    relevant_matches = pd.DataFrame(columns=['Team 1', 'Team 2', 'Result'])
    for _, row in solution_df.iterrows():
        team1 = row['Team 1']
        team2 = row['Team 2']
        day = row['Day']

        for _, row in result_df.iterrows():
            team11 = row ['Team 1']
            team22 = row ['Team 2']
            result = row ['Result']

            if r[-1] == day:
                if team1 == team11 and team2 == team22 or team1 == team22 and team2 == team11 :
                    relevant_matches = pd.concat([relevant_matches, pd.DataFrame({'Team 1': [team1], 'Team 2': [team2], 'Result': [result]})],
                                               ignore_index=True)
    print(relevant_matches)
    # Iterate over each row in the DataFrame
    for _, row in relevant_matches.iterrows():
        team1 = row['Team 1']
        team2 = row['Team 2']
        winner = row['Result']

        # Calculate expected scores for each team
        expected_score1 = 1 / (1 + 10 ** ((ratings[team2] - ratings[team1]) / 400))
        expected_score2 = 1 / (1 + 10 ** ((ratings[team1] - ratings[team2]) / 400))

        # Update ratings based on match outcome
        if winner == team1:
            ratings[team1] += k * (1 - expected_score1)  # Winning team gains
            ratings[team2] += k * (0 - expected_score2)  # Losing team loses
        elif winner == team2:
            ratings[team1] += k * (0 - expected_score1)  # Winning team loses
            ratings[team2] += k * (1 - expected_score2)  # Losing team gains
        else:
            # For draws, update both teams' ratings
            ratings[team1] += k * (0.5 - expected_score1)  # Both teams gain/lose equally
            ratings[team2] += k * (0.5 - expected_score2)  # Both teams gain/lose equally
    
    ratings_array = np.array(list(ratings.values()))
    #keyler = np.array(list(ratings.keys()))
    #for i in range(n):
    #    ratings_array[i]= round(ratings_array[i],0)

    U_ij = np.zeros((n, n)) 

    for i in range(n):
        for j in range(n):
            U_ij[i, j] = abs((ratings_array[i] - ratings_array[j]))

    output_ratings = {index: value for index, value in enumerate(ratings_array, start=1)}
    print(output_ratings)
    ratings_df = pd.DataFrame(U_ij, columns=teams, index=teams)

    return output_ratings, ratings_df


def first_model(teams,days):


    model = ConcreteModel()

    model.x = Var(teams, teams, days, within=Binary)

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

    solver = SolverFactory('gurobi')
    results = solver.solve(model, tee=True)

    
    # Print the results
    if results.solver.termination_condition == TerminationCondition.optimal:
        print("Optimal solution found.")
        if t in days:
            for i in teams:
                for j in teams:
                    if value(model.x[i, j, t]) == 1:
                        print(f"Team {i} vs Team {j} in Slot {t}")
        

    else:
        print("No optimal solution found.")



    solution_df = pd.DataFrame(columns=['Day', 'Team 1', 'Team 2'])

    if t == 1:
        for i in teams:
            for j in teams:
                if model.x[i, j, t].value == 1:
                    solution_df = pd.concat([solution_df, pd.DataFrame({'Day': [t], 'Team 1': [i], 'Team 2': [j]})],
                                        ignore_index=True)
    
  
    return solution_df








def dynamic_model(teams,days, data_file, solution_file, round_number):


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
                model.constraints.add(sum((model.x[i,j,t] + model.x[j,i,t]) for t in days) == 1)

    for i in teams:
        for t in days:
            model.constraints.add(sum((model.x[i,j,t] + model.x[j,i,t]) for j in teams if j!=i ) == 1)


    # Read data from the file and assign it to model.u
    data = data_file
    for i in teams:
        for j in teams:
            model.u[i, j] = data.iloc[i - 1, j - 1]

    # Fiy variables based on the provided solution schedule
    oku = solution_file
    previous_round = []
    for _, row in oku.iterrows():
        day = row['Day']
        team1 = row['Team 1']
        team2 = row['Team 2']
        model.x[team1, team2, day].fix(1)
        previous_round.append(day)


    previous_round = set(previous_round)
    current_round = set(round_number)

    round_difference = current_round.difference(previous_round)
    round_difference = list(round_difference)


    def rule_of(model):
        return (sum(sum(sum((model.u[i, j] * model.x[i, j, t]) for t in round_difference) for j in teams)for i in teams))


    model.obj = Objective(rule=rule_of, sense=minimize)

    
    # Solve the model
    solver = SolverFactory('gurobi')
    results= solver.solve(model)
    obj_value = round(model.obj(),0)

    solution_df = pd.DataFrame(columns=['Day', 'Team 1', 'Team 2'])
    for t in round_number:
        for i in teams:
            for j in teams:
                if model.x[i, j, t].value == 1:
                    new_row = pd.DataFrame({'Day': [t], 'Team 1': [i], 'Team 2': [j]})
                    solution_df = pd.concat([solution_df, new_row]).reset_index(drop=True)

    if results.solver.termination_condition == TerminationCondition.optimal:
        print("Optimal solution found.")
        for t in round_difference:
            for i in teams:
                for j in teams:
                    if value(model.x[i, j, t]) == 1:
                        print(f"Team {i} vs Team {j} in Slot {t}")
        

    else:
        print("No optimal solution found.")


    return solution_df, obj_value


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