import pandas as pd
import numpy as np

def point_rating(ratings, teams, solution_df, result_df, r):

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
            U_ij[i, j] = (abs(ratings_array[i] - ratings_array[j]))
    


    ratings_df = pd.DataFrame(U_ij, columns=teams, index=teams)

    return ratings, ratings_df