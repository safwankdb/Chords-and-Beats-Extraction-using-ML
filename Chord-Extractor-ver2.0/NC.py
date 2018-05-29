N_to_C={1:'A',2:'Am',3:'Bm',4:'C',5:'D',6:'Dm',7:'E',8:'Em',9:'F',10:'G'}
C_to_N = {v: k for k, v in N_to_C.items()}
def NtoC(n) :
    if n in range(1,11):
        return N_to_C[n]
def CtoN(c) :
    if c in C_to_N.keys:
        return C_to_N[c]
