def round_generate(days,p):

    separate_pieces = {}
    round_names = []
    round1= [days[0]]
    round_names.append(round1)
    for i in range(0, len(days)-1, p):
        separate_pieces[f'round{i//p+1}'] = days[:i+1 + p]

    for i in range(1, len(separate_pieces) + 1):
        ro= globals()[f"round{i}"] = separate_pieces[f'round{i}']
        round_names.append(ro)
    
  
    return round_names

