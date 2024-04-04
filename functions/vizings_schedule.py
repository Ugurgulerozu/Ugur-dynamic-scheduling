# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:35:18 2024

@author: ug033207
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor
This is a script file.

FUNCTIONS
---------
canonical() generates a single round robin tournament (SRRT) using the circle method
vizing() generates a random vizing schedule, SRRT
findcoe() calculates the coe value of a given SRRT (stored in edgeSchedule)
rndAllRounds() randomly shuffles the rounds of a given schedule
checkSchedule() checks whether a generated schedule is a round robin tournament or not
rs() swaps two given rounds in the schedule
writeScheduleToFile() writes a schedule to the specified file

DATA STRUCTURES
---------------
The basic data structure used in the code is a two-dimensional  NumPy array with rows and columns corresponding to teams. 
The value of each cell of the array tells in which round the two teams are playing each other. 
The array is called edgeSchedule and is declared with one extra row and column where index 0 is redundant.
An edgeSchedule corresponds to a SRRT.
A useful information is the opponent of a team in a given round and it is stored under another two-dimensional NumPy array called adj where rows are teams and columns are rounds.  
Index 0 is redundant. The adj array can be extracted by applying the findadj() function to a given edgeSchedule.
v plays adj(v,d) in round d 

In Python, when an array is passed to a function as an argument, changes applied to the array in the function
are preserved after the return from the function. Thus, one should be careful about function calls with array 
arguments especially in loops. copy() function creates a copy of an array, and thus may be needed before
a function call when one does not want to change the original array. 
"""

#randomly shuffle all weeks
def rndAllRounds(edgeSchedule, nColors):
    swap = rnd.sample(range(1, nColors + 1), nColors) #randomize weeks
    for i in range(nColors): 
        edgeSchedule[edgeSchedule == swap[i]] =101 + i  
    edgeSchedule = np.where(edgeSchedule !=0, edgeSchedule - 100, 0) #np.where creates a copy and does not change original
    return edgeSchedule

#check whether the generated schedule is round robin
def checkSchedule(edgeSchedule, nTeams, which):
    ok = True
    nRounds = nTeams - 1    

    for round in range(1, nRounds + 1):
        countCol = np.count_nonzero(edgeSchedule == round, axis = 0)
        countRow = np.count_nonzero(edgeSchedule == round, axis = 1)
        #print("round = ", round)
        #print(countCol[1:])
        #print(countRow[1:])
        if not(np.all(countCol[1:] == 1)): ok = False #does not check the 0th column, those should be 0
        if not(np.all(countRow[1:] == 1)): ok = False #does not check the 0th row, those should be 0 
        if not(ok):
            print("NOT A VALID SCHEDULE!",which)
            break
    return ok

#swap two weeks
def rs(edgeSchedule, color1, color2):
    edgeSchedule[edgeSchedule == color1] = -1
    edgeSchedule[edgeSchedule == color2] = color1
    edgeSchedule[edgeSchedule == -1] = color2
    return 0

def findadj(edgeSchedule, nVertices, nColors):
    adj = np.zeros((nVertices + 1, nColors + 1), dtype=int) 
    for vertex1 in range(1, nVertices + 1):
        for vertex2 in range(1, nVertices + 1):
            if vertex2 != vertex1: adj[vertex1][edgeSchedule[vertex1][vertex2]] = vertex2
    return adj

#find the coe value of a given schedule
def findcoe(edgeSchedule, nTeams):   
    #debugging, some tars schedules come with numbers in row 0 or column 0
    #error was in findTransferAgent()
    #edgeSchedule = edgeScheduleIn.copy()
    #edgeSchedule[0] = 0
    #edgeSchedule[:,0] = 0
    #print(edgeSchedule)
    #checkRowColumn(edgeSchedule,"findcoe")
    nRounds = nTeams - 1
    coe = np.zeros((nTeams + 1, nTeams + 1), dtype=int) 
    for round in range(1, nRounds):
        #print("round=",round)        
        i,j = np.where(edgeSchedule == round)
        #print(i,j)
        k,l = np.where(edgeSchedule == round + 1)  
        #print(k,l)
        coe[i,l[j-1]] += 1 #indexing starts at 0 for i,j,k,l
    #print(coe)
    i,j = np.where(edgeSchedule == nRounds)
    k,l = np.where(edgeSchedule == 1)
    coe[i,l[j-1]] += 1 #indexing starts at 0 for i,j,k,l
    #print(coe)
    return np.sum(np.square(coe))

#needed in vizing
def findFree1Random(free, w, v0, nColors):
    free1Random = -1
    count = 0
    for color in range(1, nColors+1):
        if free[v0, color] > 0: 
            if free[w, color] > 0:
                count = count + 1
    
    if count > 0:
        choose = rnd.randrange(1, count+1)
        count = 0
        for color in range(1, nColors+1):
            if free[v0, color] > 0: 
                if free[w, color] > 0:
                    count = count + 1
                    if count == choose:
                        free1Random = color
                        break
    return free1Random

#needed in vizing
def findFree2Random(free, v0, taboo, nColors):
    free2Random = -1
    count = 0
    for color in range(1, nColors+1):
        if color != taboo:
            if free[v0, color] > 0:
                count = count + 1
    
    if count > 0:
        choose = rnd.randrange(1, count+1)
        count = 0
        for color in range(1, nColors+1):
            if color != taboo:
                if free[v0, color] > 0:
                    count = count + 1
                    if count == choose:
                        free2Random = color
                        break
    return free2Random

#needed in vizing
def findFree3Random(free, w, nColors):
    free3Random = -1
    count = 0
    for color in range(1, nColors+1):
        if free[w, color] > 0:
            count = count + 1
            
    if count > 0:
        choose = rnd.randrange(1, count+1)
        count = 0
        for color in range(1, nColors+1):
            if free[w, color] > 0:
                count = count + 1
                if count == choose:
                    free3Random = color
                    break
    return free3Random

#generates a random schedule using the vizing method
def vizing(nTeams):
#generate a random vizing schedule
    nVertices = nTeams - 1    
    nColors = nTeams - 1 #max degree + 1

#indexing in python starts at 0 i chose not to use the zeroeth index row 0 and column 0 redundant
    free = np.zeros((nVertices + 1, nColors + 1),dtype=int)
    edgeColor = np.zeros((nVertices + 2, nVertices + 2),dtype=int) #last team will be added later
    
    #First apply Vizing to K(n-1) with n-1 colors (n-2+1). Teams are vertices, edges games. Colors are rounds
    taboo = -1 #line 1
    
    for vertex in range(1, nVertices+1):  #line 2
        for color in range(1, nColors):
            free[vertex,color] = color
        free[vertex,nColors] = -1 #+1 additional color not available at the beginning
    
    nUnColored = nVertices * (nVertices - 1) / 2 #undirected
    #This is a complete graph, therefore all edges (games) should be colored

    while nUnColored > 0: #line 3
        #print("nUnColored= ",nUnColored)
        if (taboo < 0): #no edge was uncolored in the previous iteration
            choose = rnd.randrange(1, nUnColored+1)
            count = 0
            cik = -1
            for node1 in range(1, nVertices): #find an uncolored edge
                for node2 in range (node1+1, nVertices+1):
                   if edgeColor[node1, node2] == 0:
                       count = count + 1
                       if (count == choose):
                           w = node1
                           v0 = node2
                           cik = 1
                           break
                if cik > 0:
                    break
            
            #if an edge was uncolored in the previous iteration w and v0 will set during uncoloring
        psi = findFree1Random(free, w, v0, nColors) #line 6
        if psi > 0:  #there is a common free color
            edgeColor[w, v0] = psi
            edgeColor[v0, w] = psi
            free[w, psi] = -1
            free[v0, psi] = -1
            taboo = -1
            nUnColored = nUnColored - 1
        else:
            psi = findFree2Random(free, v0, taboo, nColors)
            if psi < 0: #line 10
                for vertex in range(1, nVertices+1): #line 11
                    free[vertex, nColors] = nColors
                edgeColor[w, v0] = nColors #line 12
                edgeColor[v0, w] = nColors #line 12
                free[w, nColors] = -1 #line 12
                free[v0, nColors] = -1 #line 12
                nUnColored = nUnColored - 1
                taboo = -1 #line 13
            else:
                alpha0 = psi #line 15
                if taboo < 0: #line 16
                    beta = findFree3Random(free, w, nColors) #line 17    
                #findMaximal
#alternate ediyor ama maximal olmasi garanti degil herhalde. ama renkler distinct olursa zaten her yerde 1 gidilecek yer var
                cont = 1
                nextVertex = v0
                nextColor = beta
                endVertex = v0
                starte1 = v0
                countEdges = 0
                while cont > 0:
                    found = -1
                    for vertex in range(1, nVertices+1):
                        if (edgeColor[nextVertex, vertex] == nextColor):
                            found = 1
                            countEdges = countEdges + 1
                            foundVertex = vertex
                            break
                    if found < 0:
                        cont = -1
                        endVertex = nextVertex
                    else:
                        if nextColor == beta:
                            nextColor = alpha0
                        else:
                            nextColor = beta
                        starte1 = nextVertex
                        nextVertex = foundVertex
                if endVertex != w: #line 19
                    cont = 1
                    nextVertex = v0
                    prevVertex = v0
                    nextColor = beta
                    while cont > 0:
                        found = -1
                        for vertex in range(1, nVertices+1):
                            if (edgeColor[nextVertex, vertex] == nextColor):
                                if (vertex != prevVertex):
                                    if nextColor == beta:
                                        edgeColor[nextVertex, vertex] = alpha0
                                        edgeColor[vertex, nextVertex] = alpha0
                                        nextColor = alpha0
                                    else:
                                        edgeColor[nextVertex, vertex] = beta
                                        edgeColor[vertex, nextVertex] = beta
                                        nextColor = beta
                                    prevVertex = nextVertex
                                    nextVertex = vertex
                                    found = 1
                                    break
                        if found < 0:
                            cont = -1
                    free[v0, beta] = beta
                    free[v0, alpha0] = -1
                    if countEdges % 2 == 0: #intermediates will have both colors on different ends swapped, nothing will be freed
                        free[endVertex, beta] = -1
                        free[endVertex, alpha0] = alpha0
                    else:
                        free[endVertex, beta] = beta
                        free[endVertex, alpha0] = -1
                    edgeColor[w, v0] = beta #line 21
                    edgeColor[v0, w] = beta #line 21
                    free[v0, beta] = -1
                    free[w, beta] = -1
                    taboo = -1
                    nUnColored = nUnColored - 1
                else: #endVertex = w line 23
                    free[starte1, edgeColor[starte1, w]] = edgeColor[starte1, w]
                    free[w, edgeColor[starte1, w]] = edgeColor[starte1, w]
                    edgeColor[starte1, w] = 0
                    edgeColor[w, starte1] = 0
                    edgeColor[w, v0] = alpha0
                    edgeColor[v0, w] = alpha0
                    free[w, alpha0] = -1
                    free[v0, alpha0] = -1
                    v0 = starte1 #e0 in diger ucu yine w
                    taboo = alpha0
   
    for vertex in range(1, nVertices+1):
        for color in range(1, nColors+1):
            if (free[vertex, color] > 0): #should be unique
                edgeColor[vertex, nTeams] = color
                edgeColor[nTeams, vertex] = color
                break
    nVertices = nVertices + 1 #add the last team
    #print("VIZING FINISHED")
    return edgeColor

#generates a schedule using the circle method top left fixed counterclockwise
def canonical(nTeams):
    nGames = int(nTeams/2)
    edgeColor = np.zeros((nTeams + 1, nTeams + 1), dtype=int) 
    team1 = np.zeros(nGames,dtype=int) 
    team2 = np.zeros(nGames,dtype=int) 
    #team3 = np.zeros(nGames,dtype=int) 
    #team4 = np.zeros(nGames,dtype=int)     
    
    game = 0
    for team in range(1,nTeams+1,2):
        team1[game] = team
        team2[game] = team + 1
        game = game + 1        
    team3 = np.copy(team1)    
    team4 = np.copy(team2)

    #for game in range(nGames):
    #    team3[game] = team1[game] #kopya
    #    team4[game] = team2[game] #kopya

    #print("Week 1")
    #print(team1)
    #print(team2)    
    
    for game in range(nGames):
        edgeColor[team1[game]][team2[game]] = 1
        edgeColor[team2[game]][team1[game]] = 1

    for week in range(2, nTeams):
        #rotate
        team1[nGames-1] = team4[nGames-1]
        team2[0] = team3[1]
        for game in range(1,nGames - 1): team1[game] = team3[game + 1]
        for game in range(1,nGames): team2[game] = team4[game - 1]    
        for game in range(nGames):
            edgeColor[team1[game]][team2[game]] = week
            edgeColor[team2[game]][team1[game]] = week
        team3 = np.copy(team1) #copy    
        team4 = np.copy(team2) #copy
        
        #for game in range(nGames):
        #    team3[game] = team1[game] #kopya
        #    team4[game] = team2[game] #kopya
        #print("Week ", week)
        #print(team1)
        #print(team2)  
    return edgeColor

def writeScheduleToFile(edgeSchedule,nTeams,OFValue,fileName):
    filehandle = open(fileName, 'w') #file with the best schedule   
    filehandle.write("%d\n" %(OFValue))
    for team1 in range(1, nTeams):
        for team2 in range(team1 + 1, nTeams + 1):
            filehandle.write("%d%c%d%c%d\n" % (team1, ".", team2, " ", edgeSchedule[team1][team2]))
    filehandle.close()
    
    
def generate_results(teams):
    #rnd.seed(1)
    n = len(teams)
    matches = []

    for i in range(n):
        for j in range(i + 1, n):
            team1 = teams[i]
            team2 = teams[j]

            #result = random.choice([team1, team2, 'tie'])
            result =rnd.choice([team1, team2, 'tie'])
            #result =rnd.choice([team1, team2])

            matches.append({'Team 1': team1, 'Team 2': team2, 'Result': result})
    result_df = pd.DataFrame(matches)
    result_array = np.zeros((n, n), dtype=int)
    for _, row in result_df.iterrows():
        result = row['Result']
        team1 = row['Team 1']
        team2 = row['Team 2']

        index_team1 = teams.index(team1)
        index_team2 = teams.index(team2)

        if team1 == result:
            result_array[index_team1, index_team2] = 3
        elif team2 == result:
            result_array[index_team2, index_team1] = 3
        else:
            result_array[index_team1, index_team2] = 1
            result_array[index_team2, index_team1] = 1

    return result_df, result_array

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

def calculate_obj_value2(n, timetable, result_df):
    teams = list(range(1,n+1))
    days = list(range(1,n))
    non_zero_entries = timetable[1:, 1:]

    new_format_list = []

    # Iterate through the timetable matrix to extract day, team1, and team2 information
    for team1 in range(n):
        for team2 in range(n):
            day = non_zero_entries[team2, team1]
            new_format_list.append([day, team1, team2])

    # Convert the list to a NumPy array
    new_format_array = np.array(new_format_list)
    new_format_array[:, 1:] += 1
    new_format_array = new_format_array[new_format_array[:, 0].argsort()]
    new_format_array = new_format_array[new_format_array[:, 2] < new_format_array[:, 1]]
    print(new_format_array)
    solution_df = pd.DataFrame(new_format_array, columns=['Day', 'Team 1', 'Team 2'])

    initial_ratings = {i: 0 for i in range(1, n+1)}
    U_ij = np.zeros((n, n)) 
    ratings_array = np.array(list(initial_ratings.values()))
    for i in range(n):
        for j in range(n):
            U_ij[i, j] = abs((ratings_array[i] - ratings_array[j]))
    ratings_df = pd.DataFrame(U_ij, columns=teams, index=teams)

    objective_values = []


    ratings = initial_ratings

    for d in days:

        obj_count = 0
        for _, row in solution_df.iterrows():
            team1 = row['Team 1']
            team2 = row['Team 2']
            day = row['Day']
            if d == day:
                obj_count += ratings_df.iloc[team1-1, team2-1]

        d= [d]
        ratings, ratings_df = point_rating(ratings, teams, solution_df, result_df, d)
        objective_values.append(obj_count)
        a = sum(objective_values)
    return a

def calculate_obj_value(n, timetable, result_matrix):
    teams = list(range(1,n+1))
    days = list(range(1,n))
    non_zero_entries = timetable[1:, 1:]

    new_format_list = []

    # Iterate through the timetable matrix to extract day, team1, and team2 information
    for team1 in range(n):
        for team2 in range(n):
            day = non_zero_entries[team2, team1]
            new_format_list.append([day, team1, team2])

    # Convert the list to a NumPy array
    new_format_array = np.array(new_format_list)
    new_format_array[:, 1:] += 1
    new_format_array = new_format_array[new_format_array[:, 0].argsort()]
    new_format_array = new_format_array[new_format_array[:, 2] < new_format_array[:, 1]]
    timetable = new_format_array
    ratings = {i: 0 for i in teams}

    objective_values = []

    
    for d in days:
        obj_count =0
        for row in timetable:
            day,team1,team2 = row
            if d==day:
                obj_count +=(abs(ratings[team1] - ratings[team2]))
                ratings[team1] += result_matrix[team1-1,team2-1]
                ratings[team2] += result_matrix[team2-1,team1-1]
        
        objective_values.append(obj_count)
    a = sum(objective_values)
    return a

def invert_schedule(schedule):
    num_teams = schedule.shape[0] - 1
    num_days = num_teams - 1

    inverted_schedule = np.zeros((num_teams + 1, num_teams + 1), dtype=int)

    for i in range(1, num_teams + 1):
        for j in range(1, num_teams + 1):
            day = schedule[i, j]
            if day != 0:
                new_day = num_days - day + 1
                inverted_schedule[i, j] = new_day

    return inverted_schedule


import numpy as np
import random as rnd
import pandas as pd

import sys
import time
import json

#rnd.seed(31) #provide a seed if you want to get the same results for purposes such as debugging
runNTeams = {4:0, 6:1, 8:1, 10:1, 12:1, 14:1, 16:1, 18:1, 20:1, 22:0, 24:0, 26:0, 28:0, 30:0, 40:0} #dictionary decides for which nTeams to run the experiments
nInst =  {4:100, 6:100, 8:100, 10:100, 12:1000, 14:100, 16:1000, 18:1000, 20:1000, 22:5, 24:5, 26:5, 28:5, 30:5, 40:0} #dictionary decides how many runs you run for each number of teams



'''
Ornek edgeSchedule, ilk satir ve sutun kullanilmiyor, 
i takimi ile j takimi hangi hafta oynuyor onu gosteriyor
        edgeSchedule=np.array([
       [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0], 
       [ 0,  0,  8, 13,  2, 10,  3, 12,  6,  4, 11,  9,  1,  5,  7],
       [ 0,  8,  0,  9, 11, 12,  5, 13,  4,  3,  2,  1,  7, 10,  6],
       [ 0, 13,  9,  0, 12,  7,  1, 10, 11,  8,  6,  2,  4,  3,  5],
       [ 0,  2, 11, 12,  0, 13,  6,  4,  1,  9,  5,  8, 10,  7,  3],
       [ 0, 10, 12,  7, 13,  0,  8,  3,  2,  6,  1,  4,  5,  9, 11],
       [ 0,  3,  5,  1,  6,  8,  0, 11, 10,  7, 12, 13,  9,  4,  2],
       [ 0, 12, 13, 10,  4,  3, 11,  0,  8,  1,  7,  5,  2,  6,  9],
       [ 0,  6,  4, 11,  1,  2, 10,  8,  0,  5,  9,  7,  3, 13, 12],
       [ 0,  4,  3,  8,  9,  6,  7,  1,  5,  0, 10, 11, 12,  2, 13],
       [ 0, 11,  2,  6,  5,  1, 12,  7,  9, 10,  0,  3, 13,  8,  4],
       [ 0,  9,  1,  2,  8,  4, 13,  5,  7, 11,  3,  0,  6, 12, 10],
       [ 0,  1,  7,  4, 10,  5,  9,  2,  3, 12, 13,  6,  0, 11,  8],
       [ 0,  5, 10,  3,  7,  9,  4,  6, 13,  2,  8, 12, 11,  0,  1],
       [ 0,  7,  6,  5,  3, 11,  2,  9, 12, 13,  4, 10,  8,  1,  0]])

for nTeams in range(8,10,2): #make sure runNTeams has been set up correctly and is equal to 1 for the experiments you want to run
    if runNTeams[nTeams] == 1:        
        print("\n",nTeams, nTeams, nTeams, nTeams, nTeams)       
        nColors = nTeams - 1
      
        if nTeams % 2 != 0:
            print("ERROR: Number of teams (nTeams) should be even!")
            sys.exit(0)    
        
        #canonicalSchedule = canonical(nTeams)
        best = 100000
        for inst in range(1, nInst[nTeams] + 1):             
            #edgeSchedule = rndAllRounds(canonicalSchedule.copy(), nColors)
            edgeSchedule = vizing(nTeams)
            #checkSchedule(edgeSchedule, nTeams, "edgeSchedule")            
            OFValue = findcoe(edgeSchedule,nTeams)    
            print("INSTANCE",inst,"OFValue",OFValue)
            if OFValue < best:
                best = OFValue            
                bestSchedule = edgeSchedule        
        print("nTeams",nTeams,"BEST",best)       
'''