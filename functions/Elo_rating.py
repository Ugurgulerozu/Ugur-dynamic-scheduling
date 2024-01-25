import numpy as np 
import pandas as pd

def Elo_rating(ratings, teams, solution_df, result_df, r, k=32):

    n = len(teams)

    relevant_matches = pd.DataFrame(columns=['Team 1', 'Team 2', 'Result'])

   

    for _, row in solution_df.iterrows():
        team1 = row['Team 1']
        team2 = row['Team 2']
        day = row['Day']
        
        for _, row in result_df.iterrows():
            team11 = row ['Team 1']
            team22 = row ['Team 2']
            result = row ['Result']
            for i in r:
                if i==day:
                    if team1 == team11 and team2 == team22 or team1 == team22 and team2 == team11 :
                        relevant_matches = pd.concat([relevant_matches, pd.DataFrame({'Team 1': [team1], 'Team 2': [team2], 'Result': [result]})],
                                                    ignore_index=True)
    relevant_matches = relevant_matches.drop_duplicates()           
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


    ratings_df = pd.DataFrame(U_ij, columns=teams, index=teams)

    return ratings, ratings_df