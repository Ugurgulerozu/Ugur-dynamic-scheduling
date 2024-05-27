import pandas as pd
import random
import numpy as np

def generate_results(teams):

    n = len(teams)
    matches = []
    weights = [0.368, 0.368, 0.264]
    #weights = [0, 0, 1]
    for i in range(n):
        for j in range(i + 1, n):
            team1 = teams[i]
            team2 = teams[j]

            result = random.choices([team1, team2, 'tie'], weights=weights, k=1)[0]
            #result = random.choice([team1, team2, 'tie'])
            #result =random.choice([team1])

            matches.append({'Team 1': team1, 'Team 2': team2, 'Result': result})

    result_df = pd.DataFrame(matches)
    result_matrix = np.zeros((n, n), dtype=int)


    for _, row in result_df.iterrows():
        result = row['Result']
        team1 = row['Team 1']
        team2 = row['Team 2']

        index_team1 = teams.index(team1)
        index_team2 = teams.index(team2)


        if team1 == result:
            result_matrix[index_team1, index_team2] = 3
        elif team2 == result:
            result_matrix[index_team2, index_team1] = 3
        else:
            result_matrix[index_team1, index_team2] = 1
            result_matrix[index_team2, index_team1] = 1
    

    return result_df, result_matrix