import numpy as np
from mdp import MDP

# 4 actions can take (up, down, left, right)
# With a action, 0.8 chance to go straight, 0.1 for left and right
actions= [  [ [0.8,  1,  0], [0.1, 0, 1], [0.1, 0, -1] ],
            [ [0.8,  0,  1], [0.1, 1, 0], [0.1, -1, 0] ],
            [ [0.8, -1,  0], [0.1, 0, 1], [0.1, 0, -1] ],
            [ [0.8,  0, -1], [0.1, 1, 0], [0.1, -1, 0] ]  ]

##### PARS #####
lx= 20
ly= 20
Niter= 30
Nprint= 1 
##### PARS #####

# Initialize the game
game= MDP(lx, ly, 1, -1, actions, -0.04, 0.8, 20, 30, 30)
#print('Init conf')
#print(game.utility)
OFILE= open('utility_0', 'w')
for i in range(lx):
    for j in range(ly):
        if np.isnan(game.utility[i,j])==None: 
            print(i, j, 0, file=OFILE)
        else:
            print(i, j, game.utility[i,j], file=OFILE)
    print(file=OFILE)

# Playing
for i in range(Niter):
    game.update()
#    print('Iteration:', i+1)
#    print(game.utility)
    if (i+1)%Nprint==0:
        OFILE= open('utility_'+str(i+1), 'w')
        for i in range(lx):
            for j in range(ly):
                if np.isnan(game.utility[i,j]): 
                    print(i, j, 0, file=OFILE)
                else:
                    print(i, j, game.utility[i,j], file=OFILE)
            print(file=OFILE)
