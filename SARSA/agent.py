import random;

class Agent(object):
    def __init__(self, environment):
        self.environment = environment;
        self.currentRow = 0;
        self.currentCol = 0;

    def setState(self, row, col):
        self.currentCol = col;
        self.currentRow = row;
        return ((row, col), self.environment.model.Value(row, col));

    def randomizeState(self):
        while True:
            r = random.choice(range(0,self.environment.model.rows));
            c = random.choice(range(0,self.environment.model.cols));
            if ((r,c) in self.environment.Q) and not (self.environment.model.isTerminal(r,c) or self.environment.model.isObstacle(r,c)):
                return self.setState(r,c);

    def Move(self, epsilon = 0.1):
        (action, nextLoc, nextVal) = self.getAction();
        self.setState(nextLoc[0], nextLoc[1]);
        return (action, nextLoc, nextVal);

    def currentState(self):
        return (self.currentRow, self.currentCol);

    def isGameOver(self):
        return len(self.environment.Q[(self.currentRow, self.currentCol)]) < 1 

    def getAction(self, epsilon = 0.1):
        if random.uniform(0,1) < epsilon:
            while True:
                action = random.choice(self.environment.actionSpace);
                if self.isGameOver(): return (None, None, None);
                if action in self.environment.Q[(self.currentRow, self.currentCol)]:
                    return (action, self.environment.model.getActionLocation(self.currentRow, self.currentCol, action), self.environment.model.getActionValue(self.currentRow, self.currentCol, action));
        best_action = None;
        best_val = None;
        best_loc = None;
        for action in self.environment.actionSpace:
            if action in self.environment.Q[(self.currentRow, self.currentCol)]:
                av = self.environment.model.getActionValue(self.currentRow, self.currentCol, action);
                if best_val < av:
                    best_val = av;
                    best_action = action;
                    best_loc = self.environment.model.getActionLocation(self.currentRow, self.currentCol, best_action)
        return (best_action, best_loc, best_val);