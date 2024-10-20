import numpy as np
def markov(rho, A, nmax, rng) :
    N = A.shape[0]
    assert (rho.shape[0] == N)
    assert np.all(rho>=0) #toutes les probabilités doivent être >= 0
    assert (rho.sum(0)==1) # vérifier que la somme des probabilités = 1
    assert (A.shape[0] == A.shape[1]) #vérifier que c'est une matrice carré
    #Vérifier que c'est une matrice stochastique
    assert np.all(A>=0) # tous les éléments de A doivent être >=0
    assert np.all(A.sum(1) == np.ones((N,))) #la somme de chaque ligne de A est égale à 1 
    
    states = np.arange(N)
    X = [rng.choice(states,p=rho)]
    for _ in range(nmax-1) :
        current_state = X[-1]
        next_state = rng.choice(states,p=A[current_state])
        X.append(next_state)
    return X