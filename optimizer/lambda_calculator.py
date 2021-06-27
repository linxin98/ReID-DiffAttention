class LambdaCalculator(object):
    def __init__(self, milestones, warmup=True):
        self.milestones = milestones
        self.warmup = warmup

    def get_lambda(self, epoch):
        true_epoch = epoch + 1
        lam = 1
        if self.warmup:
            if true_epoch <= 10:
                lam *= true_epoch / 10
        for milestone in self.milestones:
            if true_epoch > milestone:
                lam *= 0.1
        return lam


def get_lambda_calculator(milestones, warmup=True):
    calculator = LambdaCalculator(milestones=milestones, warmup=warmup)
    return calculator.get_lambda
