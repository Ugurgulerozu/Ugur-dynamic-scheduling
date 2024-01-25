import pandas as pd
import random

def generate_results(teams):

    n = len(teams)
    matches = []

    for i in range(n):
        for j in range(i + 1, n):
            team1 = teams[i]
            team2 = teams[j]

            #result = random.choice([team1, team2, 'tie'])
            result =random.choice([team1, team2, 'tie'])

            matches.append({'Team 1': team1, 'Team 2': team2, 'Result': result})

    result_df = pd.DataFrame(matches)
    

    return result_df