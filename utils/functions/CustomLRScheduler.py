class CustomLRScheduler:
    def __init__(self, optimizer, thresholds, factors):
        self.optimizer = optimizer
        self.thresholds = thresholds
        self.factors = factors
        self.current_threshold_index = 0

    def step(self, loss):
        if loss < self.thresholds[self.current_threshold_index]:
            self.optimizer.param_groups[0]['lr'] *= self.factors[self.current_threshold_index]
            self.current_threshold_index += 1