"""
Neural Network tools developed for UCSD ECE285 MLIP.

Copyright 2019. Charles Deledalle, Sneha Gupta, Anurag Paul, Inderjot Saggu.
"""

from src.models.Model import Model
from src.models.StatsManager import StatsManager
from src.models.ModelUtils import ModelUtils

from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

import os
import time
import torch
import torch.utils.data as td
import torchvision

import datetime
import json

# Get info for training info file
import platform, socket
import psutil
import sys

class Criterion:
    def __init__(self) -> None:
        pass

    def compute(self, y, d, lpips, coef):
        pass

    def itemize(self, loss):
        for key in loss.keys():
            loss[key] = loss[key].item()

    """
    Return only the sum loss
    """
    def __call__(self, y, d, pips, coef):
        return self.compute(y, d, pips, coef)["loss"]


class Trainer(Model):
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
    RESULTS_PATH = "results/"

    TRAINING = "training"
    VALIDATION = "validation"

    def __init__(self, net, train_set, val_set, optimizer, stats_manager: StatsManager, device, criterion,
                 output_dir=None, batch_size=16, perform_validation_during_training=False, 
                 tensor_board=False, use_lpips_loss=True):

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
        self.version = Trainer.INFO_VERSION
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

        self.output_path = Trainer.RESULTS_PATH + output_dir
        
        self.perform_validation_during_training = perform_validation_during_training

        # Initialize history
        self.history = {
            Trainer.TRAINING : [],
            Trainer.VALIDATION : []
        }

        self.criterion = criterion
        
        if use_lpips_loss:
            self.lpips = LPIPS(net_type='vgg').to(self.device)
            self.coef = 1/300
        else:
            self.lpips, self.coef = lambda x,y:0, 0

        # Define checkpoint paths
        if self.output_path is not None:
            os.makedirs(self.output_path, exist_ok=True)
            self.checkpoint_path = os.path.join(self.output_path, "checkpoint.pth.tar")
            self.state_path = os.path.join(self.output_path, "state.txt")
            self.info_path = os.path.join(self.output_path, "info.json")

        # Transfer all local arguments/variables into attributes
        locs = {k: v for k, v in locals().items() if k != 'self'}
        self.__dict__.update(locs)

        if self.output_path is not None:
            # Load checkpoint and check compatibility
            if os.path.isfile(self.state_path):
                with open(self.state_path, 'r') as f:
                    state_file = f.read().strip()
                    inner_state = self.state().strip()

                    print(state_file, inner_state)
                    # Don't take into account the last character of the file, \n    
                    if state_file != inner_state:
                        raise ValueError(
                            "Cannot create this experiment: "
                            "I found a checkpoint conflicting with the current setting.")
                self.load()
            else:
                self.save()

        self.net.eval()

    @property
    def epoch(self):
        """Returns the number of epochs already performed."""
        return len(self.history[Trainer.TRAINING])

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
        train_info['CPU'] = ModelUtils.get_processor_name()
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
        if self.output_path is not None:
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


        print(checkpoint)

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

        self.net.eval()

        if self.stats_manager is not None:
            self.stats_manager.init()
        
        if plot is not None and len(self.history[Trainer.TRAINING]) > 0:
            plot(self)

        self.start_epoch = self.epoch
        self.goal_epoch = num_epochs

        if self.start_epoch < self.goal_epoch and self.tensor_board:
            #initialize tensorboard writer
            for low_res_patches, high_res in self.train_loader:
                self.x_tensorboard = low_res_patches
                self.d_tensorboard = high_res
                break

            for i, low_res in enumerate(self.x_tensorboard):
                upscale = self.train_set.get_upscale_factor(i)

                self.writer = SummaryWriter(self.output_path)
                self.net.set_upscale_mode(upscale)
                self.writer.add_graph(self.net, low_res.to(self.device))

        print("Start/Continue training from epoch {}".format(self.start_epoch))
        
        """if plot is not None:
            plot(self)"""

        self.current_training_time = 0
        self.training_start_time = datetime.datetime.now()

        for current_epoch in range(self.start_epoch, self.goal_epoch):
            self.current_epoch = current_epoch
            s = time.time()
            self.stats_manager.init()

            self.net.train()

            for low_res_patches, high_res in self.train_loader:
                # For upscale in patches
                for i, low_res in enumerate(low_res_patches):
                    self.net.set_upscale_mode(self.train_set.get_upscale_factor(i))

                    low_res, high_res = low_res.to(self.device), high_res.to(self.device)
                    self.optimizer.zero_grad()
                    
                    prediction = self.net.forward(low_res)

                    losses = self.criterion.compute(prediction, high_res, self.lpips, self.coef)

                    total_loss = losses["loss"]
                    total_loss.backward()

                    losses = self.criterion.itemize(losses)

                    self.optimizer.step()

                    with torch.no_grad():
                        self.stats_manager.accumulate(
                            losses, 
                            low_res, 
                            prediction, 
                            high_res, 
                            self.train_set.get_upscale_factor(i)
                    )
        
            self.history[Trainer.TRAINING].append(self.stats_manager.summarize())

            self.net.eval()
            if self.perform_validation_during_training:
                self.history[Trainer.VALIDATION].append(self.evaluate())

            if self.tensor_board:
                self.add_metrics_to_tensorboard()
                if self.current_epoch % 5 == 0:
                    self.add_image_to_tensorboard()
                
            """
            Do a good print, taking into account all tested categories, all losses, all metrics for all upscale factors
            """

            print(f"Epoch {self.epoch} (Time : {round(time.time() - s, 2)}s) ")

            current_history = {}
            for categories in self.history.keys():
                current_history[categories] = self.history[categories][-1]

            for category in current_history.keys():
                for upscale_factor in self.stats_manager.upscale_factor_list:
                    print(f"\t {category[0:4]}.. x{upscale_factor} =>", end="")
                    for loss_name in current_history[categories][upscale_factor]['loss'].keys():
                        loss = round(current_history[category][upscale_factor]['loss'][loss_name], 4)
                        print(f"{loss_name} : {loss}", end=" | ")
            
                    for metric_name in self.stats_manager.metrics:
                        metric = round(current_history[category][upscale_factor]['metric'][metric_name], 2)
                        print(f"{metric_name} : {metric}", end=" ")
                    
                    print()
                

            """if self.perform_validation_during_training:
                print("Epoch {} (Time: {:.2f}s) Loss: {:.5f} psnr: {} ssim: {}".format(
                    self.epoch, 
                    time.time() - s, 
                    self.history[Trainer.TRAINING][-1]['loss'], 
                    self.history[Trainer.TRAINING][-1]['psnr'], 
                    self.history[Trainer.TRAINING][-1]['ssim']))
            else:
                print("Epoch {} (Time: {:.2f}s) Loss: {:.5f} psnr: {} ssim: {}".format(
                    self.epoch, 
                    time.time() - s, 
                    self.history[Trainer.TRAINING][-1]['loss'], 
                    self.history[Trainer.TRAINING][-1]['psnr'], 
                    self.history[Trainer.TRAINING][-1]['ssim']))"""
            
            self.current_training_time += (time.time() - s)

            self.save()
            
        if plot is not None and len(self.history[Trainer.TRAINING]) > 0 and self.goal_epoch != self.start_epoch:
            plot(self)

        print("Finish training for {} epochs".format(self.goal_epoch))
        self.net.eval()

    def evaluate(self):
        """Evaluates the experiment, i.e., forward propagates the validation set
        through the network and returns the statistics computed by the stats
        manager.
        """
        self.stats_manager.init()
        self.net.eval()

        with torch.no_grad():
            for low_res_patches, high_res in self.val_loader:
                # For upscale in patches
                for i, low_res in enumerate(low_res_patches):
                    self.net.set_upscale_mode(self.val_set.get_upscale_factor(i))
                    
                    low_res, high_res = low_res.to(self.device), high_res.to(self.device)
                    y = self.net.forward(low_res)

                    losses = self.criterion.compute(y, high_res, self.lpips, self.coef)

                    losses = self.criterion.itemize(losses)

                    self.stats_manager.accumulate(losses, 
                                                  low_res, y, high_res, 
                                                  self.val_set.get_upscale_factor(i))

            if self.tensor_board:
                if self.current_epoch % 5 == 0:
                    self.add_image_to_tensorboard('val')

        return self.stats_manager.summarize()
    

    def add_metrics_to_tensorboard(self):
        train_history = self.history[Trainer.TRAINING][-1]
        
        for upscale_factor in self.stats_manager.upscale_factor_list:
            self.writer.add_scalar(f"Loss/train-x{upscale_factor}", 
                                   train_history[upscale_factor]['loss']['loss'], self.current_epoch)
            
            self.writer.add_scalar(f"PSNR/train-x{upscale_factor}", 
                                   train_history[upscale_factor]['metric']['psnr'], self.current_epoch)
            
            self.writer.add_scalar(f"SSIM/train-x{upscale_factor}", 
                                   train_history[upscale_factor]['metric']['ssim'], self.current_epoch)

            if self.perform_validation_during_training:
                val_history = self.history[Trainer.VALIDATION][-1]
                self.writer.add_scalar(f"Loss/val-x{upscale_factor}", 
                                       val_history[upscale_factor]['loss']['loss'], self.current_epoch)

                self.writer.add_scalar(f"PSNR/val-x{upscale_factor}", 
                                       val_history[upscale_factor]['metric']['psnr'], self.current_epoch)

                self.writer.add_scalar(f"SSIM/val-x{upscale_factor}", 
                                       val_history[upscale_factor]['metric']['ssim'], self.current_epoch)
    
    def add_image_to_tensorboard(self, mode='train'):
        with torch.no_grad():
            for low_res_patches, high_res in self.val_loader:
                if mode == 'train':
                    low_res_patches, high_res = self.x_tensorboard, self.d_tensorboard

                for i, low_res in enumerate(low_res_patches):
                    upscale = self.train_set.get_upscale_factor(i)
                    self.net.set_upscale_mode(upscale)

                    low_res, high_res = low_res.to(self.device), high_res.to(self.device)
                    
                    y = self.net.forward(low_res)
                    grid = torch.cat((high_res,y), dim=0)
                    grid = torchvision.utils.make_grid(grid, nrow=2, padding=2, normalize=True)
                    self.writer.add_image('Image/{}{}'.format(mode, self.train_set.get_upscale_factor(i)), grid, self.current_epoch)

                break

def Experiment(net, train_set, val_set, optimizer, stats_manager, device, criterion,
                output_dir=None, batch_size=16, perform_validation_during_training=False, tensor_board=False, use_lpips_loss=False):
    print("WARNING /!\ Class experiment is deprecated. Please use class Trainer instead.")
    exp = Trainer(net, train_set, val_set, optimizer, stats_manager, device, criterion,
                output_dir, batch_size, perform_validation_during_training, tensor_board, use_lpips_loss)
    return exp


if __name__ == "__main__":

    from UpscaleNN import UpscaleNN

    up = UpscaleNN()

    mod = Model(up, output_dir="results/smallbatchexperiment-upscale")

    print(mod.info())
    print(mod.architecture())
    print(mod.get_weight().keys())
    print(mod)