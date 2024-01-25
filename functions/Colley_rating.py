import numpy as np 
import pandas as pd

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

def asfasf(df: pd.DataFrame):
    pass
