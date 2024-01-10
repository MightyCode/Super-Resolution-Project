class StatsManager():
    """
    A class meant to track the loss during a neural network learning experiment.

    Though not abstract, this class is meant to be overloaded to compute and
    track statistics relevant for a given task. For instance, you may want to
    overload its methods to keep track of the accuracy, top-5 accuracy,
    intersection over union, PSNR, etc, when training a classifier, an object
    detector, a denoiser, etc.
    """

    def __init__(self, upscale_factor_list):
        self.upscale_factor_list = upscale_factor_list
        self.init()

    def __repr__(self):
        """Pretty printer showing the class name of the stats manager. This is
        what is displayed when doing ``print(stats_manager)``.
        """
        return self.__class__.__name__

    def init(self):
        """Initialize/Reset all the statistics"""
        self.number_update = {}

        self.running_loss = {}

        for upscale_factor in self.upscale_factor_list:
            self.number_update[upscale_factor] = 0
            self.running_loss[upscale_factor] = {"loss" : 0}

    def accumulate(self, loss, x, y, d, upscale_factor):
        """Accumulate statistics

        Though the arguments x, y, d are not used in this implementation, they
        are meant to be used by any subclasses. For instance they can be used
        to compute and track top-5 accuracy when training a classifier.

        Arguments:
            loss (tuple): the loss obtained during the last update.
            x (Tensor): the input of the network during the last update.
            y (Tensor): the prediction of by the network during the last update.
            d (Tensor): the desired output for the last update.
            upscale_factor (int): the upscale factor for the last update.
        """
        self.number_update[upscale_factor] += 1

        for key in loss.keys():
            if key not in self.running_loss[upscale_factor].keys():
                self.running_loss[upscale_factor][key] = 0

            self.running_loss[upscale_factor][key] += loss[key]

    def summarize(self):
        """Compute statistics based on accumulated ones"""

        for upscale_factor in self.upscale_factor_list:
            for special_loss in self.running_loss[upscale_factor].keys():
                self.running_loss[upscale_factor][special_loss] /= self.number_update[upscale_factor]

        return self.running_loss
