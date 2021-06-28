class Averager(object):
    def __init__(self, alpha=0.98):
        self.alpha = alpha
        self.value = None

    def reset(self):
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.value * self.alpha + (1.0 - self.alpha) * new_value

    def get_value(self):
        return self.value
