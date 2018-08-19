import numpy as np;
'''
    This is the RL_Environment which creates a set of
    all the possible states and actions available in 
    the environment.
    
    The object takes a model. The model must have functions:
    - model.getAllStates() ->   This should return all the
                                possible states with are the
                                permitable actions in form of
                                Q[s][a] = value and the entire
                                state space.

'''

class RL_Environment(object):
    def __init__(self, model):
        # Q is is a dict which corresponds to Q[s][a] = value;
        self.model = model;
        self.stateSpace, self.actionSpace, self.Q = self.model.getAllStates();
        self.statesVisited = [];
        self.updateCounts = {};
        self.updateCounts_sa = {};
        for state in self.stateSpace:
            self.updateCounts_sa[state] = {};
            for action in self.actionSpace:
                self.updateCounts_sa[state][action] = 1.0;

    def Print(self):
        print self.Q;

    def getPolicy(self):
        s = "";
        for r in range(self.model.rows):
            for c in range(self.model.cols):
                if self.model.isTerminal(r,c):
                    s += "\t\t"+str(self.model.Value(r,c));
                elif self.model.isObstacle(r,c):
                    s += "\t\tX";
                elif (r,c) not in self.statesVisited:
                    s += "\t\t*";
                else:
                    best_action = None;
                    best_val = None;
                    for action in self.actionSpace:
                        if action in self.Q[(r,c)] and best_val < self.Q[(r,c)][action]:
                            best_val = self.Q[(r,c)][action];
                            best_action = action;
                    s += "\t\t"+ best_action;
            s += "\n"
        return s;




class GridWorld(object):
    def __init__(self, rows, cols, init_number = None, negitive_random = False, obstacles = [], rewards = [], actions = ["N", "S", "E", "W"]):
        self.rows = rows;
        self.cols = cols;
        self.r_mask = np.zeros((rows,cols));
        self.actions = actions;
        self.obstacle_mask = -1;
        self.reward_mask = -2;
        self.simple_mask = 0;
        if init_number is None:
            self.grid = np.random.rand(rows, cols);
            if negitive_random:
                self.grid = -1 * self.grid;
        else:
            self.grid = init_number * np.ones((rows,cols));

        for i in obstacles:
            if not len(i) == 2:
                raise Exception("Invalid obstacle position");
            r = i[0];
            c = i[1];
            self.r_mask[r,c] = self.obstacle_mask;
            
        for i in rewards:
            if not len(i) == 3:
                raise Exception("Invalid rewards position and value");  
            r = i[0];
            c = i[1];
            self.r_mask[r,c] = self.reward_mask;
            self.grid[r,c] = i[2];     

    def isObstacle(self, row, col):
        if self.isBoundary(row,col) or (col > -1 and row > -1 and self.r_mask[row, col] == self.obstacle_mask):
            return True;
        else:
            return False;

    def isTerminal(self, row, col):
        if col > -1 and row > -1 and self.r_mask[row, col] == self.reward_mask:
            return True;
        else:
            return False;

    def isBoundary(self, row, col):
        if row < self.rows and col < self.cols and col > -1 and row > -1:
            return False;
        else:
            return True;

    def Value(self, row, col):
        if self.isBoundary(row, col) or self.isObstacle(row,col):
            return None;
        else:
            return self.grid[row, col];

    def setValue(self, row, col, value):
        if self.isBoundary(row, col) or self.isObstacle(row,col) or self.isTerminal(row,col):
            raise Exception("Invalid row or column");
        else:
            self.grid[row, col] = value;


    def getAllStates(self):
        Q = {};
        stateSpace = [];
        for r in range(0,self.rows):
            for c in range(0,self.cols):
                if self.isBoundary(r,c) or self.isObstacle(r,c):
                    continue;
                if self.isTerminal(r,c):
                    Q[(r,c)] = {};
                    continue;
                Q[(r,c)] = self.getAllPermitableActions(r,c);
                stateSpace.append((r,c));
        return stateSpace, self.actions, Q;                

    '''
        Returns a list of all permitable actions
        and their values at the current row and
        col. (permitable_action, value)
    '''
    def getAllPermitableActions(self, current_row, current_col):
        A = {};
        for action in self.actions:
            av = self.getActionValue(current_row, current_col, action);
            if av is not None:
                A[action] = av;
        return A;

    def getActionLocation(self, r, c, action):
        if action == "N": r -= 1;
        elif action == "S": r += 1;
        elif action == "E": c += 1;
        elif action == "W": c -= 1;
        return (r, c);

    def getActionValue(self, r, c, action):
        (r,c) = self.getActionLocation(r,c,action);
        if self.isBoundary(r,c) or self.isObstacle(r,c):
            return None;
        else:
            return self.grid[r,c];

    # GridWorld UTILS
    def __str__(self):
        s = "";
        for c in range(0,self.rows):
            for c1 in range(0, self.cols):
                if self.r_mask[c,c1] == self.obstacle_mask:
                    s_val = "X";
                elif self.r_mask[c,c1] == self.reward_mask:
                    s_val = "|" + "{:.{}f}".format( self.grid[c,c1], 3 ) + "|";
                else:
                    s_val = "{:.{}f}".format( self.grid[c,c1], 3 );
                s += "\t" + s_val;
            s += "\n";
        return s;