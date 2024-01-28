from src.models.UpscaleNN import UpscaleNN
from src.models.UpscaleResidualNN import UpscaleResidualNN
from src.models.UpscaleResidual2NN import UpscaleResidual2NN
from src.models.UpscaleResidual3NN import UpscaleResidual3NN
from src.models.rdn import RDN

import src.models.Trainer as nt

import torch

class InitModel:
    """
    Create a model architecture from name
    DO NOT LOAD ANY WEIGHTS !!!!
    """
    @staticmethod
    def create_model(model_name: str,
                     model_hyperparameters: dict, 
        super_res_factor: float, device) -> torch.nn.Module:

        # to lower
        r = None
        if "upscale" in model_name.lower():
            if "residual3" in model_name.lower() or "resid3" in model_name.lower():
                MODEL_INIT = UpscaleResidual3NN
            elif "residual2" in model_name.lower() or "resid2" in model_name.lower():
                MODEL_INIT = UpscaleResidual2NN
            elif "residual" in model_name.lower() or "resid" in model_name.lower():
                MODEL_INIT = UpscaleResidualNN
            else:
                MODEL_INIT = UpscaleNN

            channel_positions = 3
            if "channel_positions" in model_hyperparameters:
                channel_positions = len(model_hyperparameters["channel_positions"])
            
            channel_interpolations = None
            if "channel_interpolations" in model_hyperparameters:
                channel_interpolations = model_hyperparameters["channel_interpolations"]

            r = MODEL_INIT(default_upscale_factor=super_res_factor, 
                           num_channel=channel_positions, 
                            channel_interpolation=channel_interpolations,
                           old_version=("old" in model_name.lower())) 
        elif "rdn" in model_name.lower():
            r = RDN(C=model_hyperparameters["C"], D=model_hyperparameters["D"], 
                    G=model_hyperparameters["G"], G0=model_hyperparameters["G0"], 
                    scaling_factor=super_res_factor, 
                    kernel_size=model_hyperparameters["kernelSize"], 
                    upscaling='shuffle', weights=None)
        
        if r is None:
            raise Exception("The model name is not correct")
        
        r.to(device)

        return r


    """
    Create a model from name and load weights from path
    The model cannot be trained
    """
    @staticmethod
    def create_model_static(
        model_name: str, model_weights_path : str, 
        model_hyperparameters: dict, 
        super_res_factor: float, device) -> torch.nn.Module:

        r = InitModel.create_model(model_name, model_hyperparameters, super_res_factor, device)
        r.eval()

        exp = nt.Model(r, device, model_weights_path)

        return exp
