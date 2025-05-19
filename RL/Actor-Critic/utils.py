class DiscountCumsum:
    def __init__(self, rewards, gamma):
        self.G, self.out = 0, []
        self.rewards = rewards
        self.gamma = gamma
    
    def __call__(self):
        for i in reversed(self.rewards):
            self.G = self.gamma * self.G + i
            self.out.append(self.G)
        
        return list(reversed(self.out)) 
    
    
    