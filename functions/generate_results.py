import pandas as pd
import random

def generate_results(teams):

    n = len(teams)
    matches = []
    weights = [0.37, 0.37, 0.26]
    #weights = [0, 0, 1]
    for i in range(n):
        for j in range(i + 1, n):
            team1 = teams[i]
            team2 = teams[j]

            #if i <= 8: # Teams 1 to 7 win against teams 8 and beyond
            #    result = team1
            
            #else:
            #    result = 'tie'
            #result = random.choices([team1, team2, 'tie'], weights=weights, k=1)[0]
            #result = random.choice([team1, team2, 'tie'])
            result =random.choice([team1])

            matches.append({'Team 1': team1, 'Team 2': team2, 'Result': result})

    result_df = pd.DataFrame(matches)
    

    return result_df