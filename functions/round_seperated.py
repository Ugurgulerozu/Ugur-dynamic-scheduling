import math

def round_seperated(days, p):
    n = len(days)
    k = math.ceil((n - 1) / p)
    sublists = [days[i*p+1:((i+1)*p)+1] for i in range(k)] 
    return sublists

