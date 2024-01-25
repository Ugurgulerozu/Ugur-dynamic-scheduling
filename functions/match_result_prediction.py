import numpy as np 
import pandas as pd
import random

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