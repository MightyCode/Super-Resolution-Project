from src.models.ModelUtils import ModelUtils

import os, platform, socket, sys
import psutil
import torch

class Model:
    RESULTS_PATH = "results/"

    def __init__(self, net, device='cpu', output_dir=None):
        self.net = net
        self.device = device
        self.nb_param = sum(p.numel() for p in net.parameters() if p.requires_grad)
        self.output_dir = Model.RESULTS_PATH + output_dir

        if output_dir is not None and os.path.isfile(self.output_dir):
                checkpoint = torch.load(self.output_dir, map_location=self.device)
                self.load_checkpoint_dict(checkpoint)
                del checkpoint
        elif output_dir is not None and os.path.isdir(self.output_dir):
            checkpoint_path = os.path.join(self.output_dir, "checkpoint.pth.tar")
            if os.path.isfile(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.load_checkpoint_dict(checkpoint)
                del checkpoint
        elif output_dir is not None:
            raise ValueError("Cannot find the weights file. " + str(output_dir))

    def __call__(self, X):
        return self.forward(X)


    def forward(self, X):
        return self.net(X)

    def info(self):
        """Returns the setting of the experiment."""
        result = {
            "NumberParameters": self.nb_param,
        }

        train_info = {}
        train_info['DeviceName'] = platform.node()
        train_info['SocketName'] = socket.gethostname()
        train_info['CPU'] = ModelUtils.get_processor_name()
        train_info['TorchDevice'] = str(self.device)

        if torch.cuda.is_available():
            train_info['GPU'] = torch.cuda.get_device_name()
        train_info['RAM'] = str(round(psutil.virtual_memory().total / (1024.0 **3), 2)) + " GB"
        train_info['Python'] = sys.version

        return result

    def architecture(self):
        return "Net({})\n".format(self.net)

    def get_weight(self):
        return self.net.state_dict()

    def load_checkpoint_dict(self, checkpoint):
        """Loads the experiment from the input checkpoint."""
        self.net.load_state_dict(checkpoint['Net'])

    def __repr__(self):
        """Pretty printer showing the setting of the experiment. This is what
        is displayed when doing ``print(experiment)``. This is also what is
        saved in the ``config.txt`` file.
        """
        string = ''
        for key, val in self.info().items():
            string += '{} : {}\n'.format(key, val)

        return string
