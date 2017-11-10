# Solving bellman eqn with iterations
import numpy as np

class MDP(object):
    def __init__(self, lx, ly, score_win, score_lose, actions, reward, gamma, cell_block, cell_win, cell_lose):
        self.lx= lx
        self.ly= ly

        self.score_win=  score_win
        self.score_lose= score_lose

        self.actions= actions # 3D list: [action_1:[possible_1:[prob, dx, dy], ...], ...]
        # assume same actions for all positions
        # i.e., moving against boundaries is valid and will bound back to origin
        
        self.reward= reward # reward for the position
        self.gamma= gamma   # discount for future reward
        
        if (type(cell_block) is set) and (type(cell_win) is set) and (type(cell_lose) is set):
            self.cell_block= cell_block # obstacle
            self.cell_win=   cell_win   # win and get postivie points
            self.cell_lose=  cell_lose  # lose and get negative points
        elif (type(cell_block) is int) and (type(cell_win) is int) and (type(cell_lose) is int):
            self.cell_block, self.cell_win, self.cell_lose= self._random_init_(cell_block, cell_win, cell_lose)
        else:
            print('Err: block, win, lose should be set() of cells or int() for random init')
            exit('init: type error for cell_block, cell_win, or cell_lose')

        self.utility= self._init_utility_()

    
    def _random_init_(self, Nblock, Nwin, Nlose):
        if (Nblock+Nwin+Nlose) > self.lx*self.ly:
            exit('Err - init: (random_init) block, win, lose cells are more than the world size')

        cell_block= set()
        while Nblock != 0:
            pos= ( np.random.randint(self.lx), np.random.randint(self.ly) )
            if pos not in cell_block:
                cell_block.add(pos)
                Nblock -= 1
        
        cell_win= set()
        while Nwin != 0:
            pos= ( np.random.randint(self.lx), np.random.randint(self.ly) )
            if (pos not in cell_block) and (pos not in cell_win):
                cell_win.add(pos)
                Nwin -= 1
        
        cell_lose= set()
        while Nlose != 0:
            pos= ( np.random.randint(self.lx), np.random.randint(self.ly) )
            if (pos not in cell_block) and (pos not in cell_win) and (pos not in cell_lose):
                cell_lose.add(pos)
                Nlose -= 1

        return cell_block, cell_win, cell_lose

    
    def _init_utility_(self):
        utility= np.zeros( (self.lx, self.ly) )
        for x, y in self.cell_block:
            utility[x,y]= None
        for x, y in self.cell_win:
            utility[x,y]= self.score_win
        for x, y in self.cell_lose:
            utility[x,y]= self.score_lose
        return utility


    def _final_loc_(self, x, y, dx, dy):
        # with position and a move, calculate final location
        x2, y2= (x+dx, y+dy)
        if (x2, y2) in self.cell_block: return x, y
        elif x2<0 or x2>=self.lx: return x, y
        elif y2<0 or y2>=self.ly: return x, y
        else: return x2, y2


    def _update_utility_(self, x, y):
        # utility: current utility (2D numpy array)
        # pos: coordinates of the position being updated (x,y)
    
        # Calculate future rewards
        rewards_future= []
        for a in self.actions: # loop tho all actions
            rewards_future.append(0)
            for prob, dx, dy in a:
                x2, y2= self._final_loc_(x, y, dx, dy)
                rewards_future[-1] += prob * self.utility[x2,y2]
    
        return self.reward + self.gamma * max(rewards_future)

    
    def update(self):
        utility_updated= np.zeros( (self.lx, self.ly) )
        for x in range(self.lx):
            for y in range(self.ly):
                if (x,y) in self.cell_block:
                    utility_updated[x,y]= None
                elif (x,y) in self.cell_win:
                    utility_updated[x,y]= self.score_win
                elif (x,y) in self.cell_lose:
                    utility_updated[x,y]= self.score_lose
                else:
                    utility_updated[x,y]= self._update_utility_(x, y)
        
        self.utility= utility_updated
