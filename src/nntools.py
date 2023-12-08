"""
Neural Network tools developed for UCSD ECE285 MLIP.

Copyright 2019. Charles Deledalle, Sneha Gupta, Anurag Paul, Inderjot Saggu.
"""

import os
import time
import torch
import torch.utils.data as td
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
import torchvision
import datetime
import json

# Get info for training info file
import platform, subprocess, re, socket
import psutil
import sys

def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command ="sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).decode().strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub( ".*model name.*:", "", line,1)
    return ""

class StatsManager():
    """
    A class meant to track the loss during a neural network learning experiment.

    Though not abstract, this class is meant to be overloaded to compute and
    track statistics relevant for a given task. For instance, you may want to
    overload its methods to keep track of the accuracy, top-5 accuracy,
    intersection over union, PSNR, etc, when training a classifier, an object
    detector, a denoiser, etc.
    """

    def __init__(self):
        self.init()

    def __repr__(self):
        """Pretty printer showing the class name of the stats manager. This is
        what is displayed when doing ``print(stats_manager)``.
        """
        return self.__class__.__name__

    def init(self):
        """Initialize/Reset all the statistics"""
        self.running_loss = 0
        self.number_update = 0

    def accumulate(self, loss, x=None, y=None, d=None):
        """Accumulate statistics

        Though the arguments x, y, d are not used in this implementation, they
        are meant to be used by any subclasses. For instance they can be used
        to compute and track top-5 accuracy when training a classifier.

        Arguments:
            loss (float): the loss obtained during the last update.
            x (Tensor): the input of the network during the last update.
            y (Tensor): the prediction of by the network during the last update.
            d (Tensor): the desired output for the last update.
        """
        self.running_loss += loss
        self.number_update += 1

    def summarize(self):
        """Compute statistics based on accumulated ones"""
        return self.running_loss / (self.number_update+1e-9)


class Experiment():
    """
    A class meant to run a neural network learning experiment.

    After being instantiated, the experiment can be run using the method
    ``run``. At each epoch, a checkpoint file will be created in the directory
    ``output_dir``. Two files will be present: ``checkpoint.pth.tar`` a binary
    file containing the state of the experiment, and ``config.txt`` an ASCII
    file describing the setting of the experiment. If ``output_dir`` does not
    exist, it will be created. Otherwise, the last checkpoint will be loaded,
    except if the setting does not match (in that case an exception will be
    raised). The loaded experiment will be continued from where it stopped when
    calling the method ``run``. The experiment can be evaluated using the method
    ``evaluate``.

    Attributes/Properties:
        epoch (integer): the number of performed epochs.
        history (list): a list of statistics for each epoch.
            If ``perform_validation_during_training``=False, each element of the
            list is a statistic returned by the stats manager on training data.
            If ``perform_validation_during_training``=True, each element of the
            list is a pair. The first element of the pair is a statistic
            returned by the stats manager evaluated on the training set. The
            second element of the pair is a statistic returned by the stats
            manager evaluated on the validation set.

    Arguments:
        net (NeuralNetork): a neural network.
        train_set (Dataset): a training data set.
        val_set (Dataset): a validation data set.
        stats_manager (StatsManager): a stats manager.
        output_dir (string, optional): path where to load/save checkpoints. If
            None, ``output_dir`` is set to "experiment_TIMESTAMP" where
            TIMESTAMP is the current time stamp as returned by ``time.time()``.
            (default: None)
        batch_size (integer, optional): the size of the mini batches.
            (default: 16)
        perform_validation_during_training (boolean, optional): if False,
            statistics at each epoch are computed on the training set only.
            If True, statistics at each epoch are computed on both the training
            set and the validation set. (default: False)
    """

    INFO_VERSION = 1.0

    def __init__(self, net, train_set, val_set, optimizer, stats_manager, device, criterion,
                 output_dir=None, batch_size=16, perform_validation_during_training=False, tensor_board=False, use_lpips_loss=True):

        self.net = net
        self.train_set = train_set
        self.val_set = val_set
        self.tensor_board = tensor_board
        if self.train_set is not None:
            self.train_set_len = train_set.__len__()
        else:
            self.train_set_len = 0

        if self.train_set is not None:
            self.val_set_len = val_set.__len__()
        else:
            self.val_set_len = 0

        # Init for info file
        self.nb_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
        self.version = Experiment.INFO_VERSION
        self.trains = None
        
        self.goal_epoch = 0
        self.start_epoch = 0
        self.current_epoch = 0

        self.current_training_time = 0
        self.training_start_time = 0
        self.batch_size = batch_size

        # Define data loaders
        if train_set is not None and batch_size is not None:
            self.train_loader = td.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                        drop_last=True)

        if val_set is not None and batch_size is not None:
            self.val_loader = td.DataLoader(val_set, batch_size=batch_size, shuffle=False,
                                    drop_last=True)

        self.optimizer = optimizer
        self.stats_manager = stats_manager
        self.device = device
        self.output_dir = output_dir
        self.perform_validation_during_training = perform_validation_during_training

        # Initialize history
        self.history = []

        self.criterion = criterion
        
        if use_lpips_loss:
            self.lpips = LPIPS(net_type='vgg').to(self.device)
            self.coef = 1/1000
        else:
            self.lpips, self.coef = lambda x,y:0, 0

        # Define checkpoint paths
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            self.checkpoint_path = os.path.join(output_dir, "checkpoint.pth.tar")
            self.state_path = os.path.join(output_dir, "state.txt")
            self.info_path = os.path.join(output_dir, "info.json")

        # Transfer all local arguments/variables into attributes
        locs = {k: v for k, v in locals().items() if k != 'self'}
        self.__dict__.update(locs)

        if output_dir is not None:
            # Load checkpoint and check compatibility
            if os.path.isfile(self.state_path):
                with open(self.state_path, 'r') as f:
                    state_file = f.read().strip()
                    inner_state = self.state().strip()

                    # Don't take into account the last character of the file, \n    
                    if state_file != inner_state:
                        raise ValueError(
                            "Cannot create this experiment: "
                            "I found a checkpoint conflicting with the current setting.")
                self.load()
            else:
                self.save()

    @property
    def epoch(self):
        """Returns the number of epochs already performed."""
        return len(self.history)

    def info(self):
        """Returns the setting of the experiment."""
        result = {
            "Version": self.version,
            "NumberParameters": self.nb_param,
            "GoalEpoch": self.goal_epoch,
            "Trains": []
        }

        if self.trains is not None:
            for train in self.trains:
                if train["StartEpoch"] != self.start_epoch:
                    result["Trains"].append(train)

        train_info = {}
        train_info['DeviceName'] = platform.node()
        train_info['SocketName'] = socket.gethostname()
        train_info['CPU'] = get_processor_name()
        train_info['TorchDevice'] = str(self.device)

        if torch.cuda.is_available():
            train_info['GPU'] = torch.cuda.get_device_name()
        train_info['RAM'] = str(round(psutil.virtual_memory().total / (1024.0 **3), 2)) + " GB"
        train_info['Python'] = sys.version
        train_info['StartEpoch'] = self.start_epoch
        train_info['EndEpoch'] = self.current_epoch
        train_info['BatchSize'] = self.batch_size
        train_info['TrainingSize'] = self.train_set_len
        train_info['ValidationSize'] = self.val_set_len
        train_info['TrainingStartDate'] = str(self.training_start_time)
        train_info['TrainingTime'] = f'{str(round(self.current_training_time, 2))} s'

        result["Trains"].append(train_info)

        return result

    def state(self):
        return "Net({})\n".format(self.net) + \
                "Optimizer({})\n".format(self.optimizer)

    def checkpoint_dict(self):
        """Returns the current state of the experiment."""
        return { 
            'Net': self.net.state_dict(),
            'Optimizer': self.optimizer.state_dict(),
            'History': self.history 
        }

    def save(self):
        """Saves the experiment on disk, i.e, create/update the last checkpoint."""
        if self.output_dir is not None:
            # Save checkpoint
            torch.save(self.checkpoint_dict(), self.checkpoint_path)

            # Save state
            with open(self.state_path, 'w') as f:
                print(self.state(), file=f)

            # Save config
            with open(self.info_path, 'w') as f:
                json.dump(self.info(), f, indent=4)


    def load_checkpoint_dict(self, checkpoint):
        """Loads the experiment from the input checkpoint."""
        self.net.load_state_dict(checkpoint['Net'])
        self.optimizer.load_state_dict(checkpoint['Optimizer'])
        self.history = checkpoint['History']

        # The following loops are used to fix a bug that was
        # discussed here: https://github.com/pytorch/pytorch/issues/2830
        # (it is supposed to be fixed in recent PyTorch version)
        """for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self.device)"""

    def load(self):
        # if checkpoint_path exists

        if os.path.isfile(self.checkpoint_path):
            """Loads the experiment from the last checkpoint saved on disk."""
            checkpoint = torch.load(self.checkpoint_path,
                                    map_location=self.device)
            
            self.load_checkpoint_dict(checkpoint)

            del checkpoint

        if os.path.isfile(self.info_path):
            with open(self.info_path, 'r') as f:
                info = json.load(f)
                
                self.version = info['Version']
                self.nb_param = info['NumberParameters']
                self.trains = info['Trains']
                self.goal_epoch = info['GoalEpoch']

    def __repr__(self):
        """Pretty printer showing the setting of the experiment. This is what
        is displayed when doing ``print(experiment)``. This is also what is
        saved in the ``config.txt`` file.
        """
        string = ''
        for key, val in self.info().items():
            string += '{} : {}\n'.format(key, val)

        return string

    def run(self, num_epochs, plot=None):
        """Runs the experiment, i.e., trains the network using backpropagation
        based on the optimizer and the training set. Also performs statistics at
        each epoch using the stats manager.

        Arguments:
            num_epoch (integer): the number of epoch to perform.
            plot (func, optional): if not None, should be a function taking a
                single argument being an experiment (meant to be ``self``).
                Similar to a visitor pattern, this function is meant to inspect
                the current state of the experiment and display/plot/save
                statistics. For example, if the experiment is run from a
                Jupyter notebook, ``plot`` can be used to display the evolution
                of the loss with ``matplotlib``. If the experiment is run on a
                server without display, ``plot`` can be used to show statistics
                on ``stdout`` or save statistics in a log file. (default: None)
        """

        self.net.train()

        if self.stats_manager is not None:
            self.stats_manager.init()
        

        self.start_epoch = self.epoch
        self.goal_epoch = num_epochs

        if self.start_epoch < self.goal_epoch and self.tensor_board:
            #initialize tensorboard writer

            self.x_tensorboard, self.d_tensorboard = next(iter(self.train_loader))
        
            self.writer = SummaryWriter(self.output_dir)
            self.writer.add_graph(self.net, self.x_tensorboard.to(self.device))


        print("Start/Continue training from epoch {}".format(self.start_epoch))
        
        if plot is not None:
            plot(self)

        self.current_training_time = 0
        self.training_start_time = datetime.datetime.now()

        for current_epoch in range(self.start_epoch, self.goal_epoch):
            self.current_epoch = current_epoch
            s = time.time()
            self.stats_manager.init()

            for x, d in self.train_loader:
                x, d = x.to(self.device), d.to(self.device)
                self.optimizer.zero_grad()
                y = self.net.forward(x)
                loss = self.criterion(y, d, self.lpips, self.coef)
                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    self.stats_manager.accumulate(loss.item(), x, y, d)
                
            if not self.perform_validation_during_training:
                self.history.append(self.stats_manager.summarize())
            else:
                self.history.append(
                    (self.stats_manager.summarize(), self.evaluate()))
            
            if self.tensor_board:
                self.add_metrics_to_tensorboard()
                if self.current_epoch % 5 == 0:
                    self.add_image_to_tensorboard()
                
            if self.perform_validation_during_training:
                print("Epoch {} (Time: {:.2f}s) Loss: {:.5f} psnr: {} ssim: {}".format(self.epoch, time.time() - s, self.history[-1][0]['loss'], self.history[-1][1]['psnr'], self.history[-1][1]['ssim']))
            else:
                print("Epoch {} (Time: {:.2f}s) Loss: {:.5f} psnr: {} ssim: {}".format(self.epoch, time.time() - s, self.history[-1]['loss'], self.history[-1]['psnr'], self.history[-1]['ssim']))
            
            self.current_training_time += (time.time() - s)

            self.save()
            
            if plot is not None:
                plot(self)

        print("Finish training for {} epochs".format(self.goal_epoch))

    def evaluate(self):
        """Evaluates the experiment, i.e., forward propagates the validation set
        through the network and returns the statistics computed by the stats
        manager.
        """
        self.stats_manager.init()
        self.net.eval()

        with torch.no_grad():
            for x, d in self.val_loader:
                x, d = x.to(self.device), d.to(self.device)
                y = self.net.forward(x)

                loss = self.criterion(y, d, self.lpips, self.coef)

                self.stats_manager.accumulate(loss.item(), x, y, d)

            if self.tensor_board:
                if self.current_epoch % 5 == 0:
                    self.add_image_to_tensorboard('val')

        self.net.train()

        return self.stats_manager.summarize()
    

    def add_metrics_to_tensorboard(self):
        if not self.perform_validation_during_training:
            self.writer.add_scalar('Loss/train', self.history[-1]['loss'], self.current_epoch)
            self.writer.add_scalar('PSNR/train', self.history[-1]['psnr'], self.current_epoch)
            self.writer.add_scalar('SSIM/train', self.history[-1]['ssim'], self.current_epoch)
        else:
            self.writer.add_scalar('Loss/train', self.history[-1][0]['loss'], self.current_epoch)
            self.writer.add_scalar('PSNR/train', self.history[-1][0]['psnr'], self.current_epoch)
            self.writer.add_scalar('SSIM/train', self.history[-1][0]['ssim'], self.current_epoch)
            self.writer.add_scalar('Loss/val', self.history[-1][1]['loss'], self.current_epoch)
            self.writer.add_scalar('PSNR/val', self.history[-1][1]['psnr'], self.current_epoch)
            self.writer.add_scalar('SSIM/val', self.history[-1][1]['ssim'], self.current_epoch)

    
    def add_image_to_tensorboard(self, mode='train'):
        if mode == 'val':
            x,d = next(iter(self.val_loader))
        else:
            x,d = self.x_tensorboard, self.d_tensorboard
        x,d = x.to(self.device), d.to(self.device)
        with torch.no_grad():
            y = self.net.forward(x)
        grid = torch.cat((d,y), dim=0)
        grid = torchvision.utils.make_grid(grid, nrow=2, padding=2, normalize=True)
        self.writer.add_image('Image/{}'.format(mode), grid, self.current_epoch)
            