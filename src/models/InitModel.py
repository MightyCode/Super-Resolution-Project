from src.models.UpscaleNN import UpscaleNN
from src.models.UpscaleResidualNN import UpscaleResidualNN
from src.models.rdn import RDN

import src.models.Trainer as nt

import torch

class InitModel:
    @staticmethod
    def create_model(model_name: str,
                     model_hyperparameters: dict, 
        super_res_factor: float, device) -> torch.nn.Module:

        # to lower
        r = None
        if "upscale" in model_name.lower():
            print(model_name.lower())
            if "residual" in model_name.lower() or "resid" in model_name.lower():
                MODEL_INIT = UpscaleResidualNN
            else:
                MODEL_INIT = UpscaleNN

            r = MODEL_INIT(default_upscale_factor=super_res_factor, 
                           num_channel=len(model_hyperparameters["channels"]), 
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

    @staticmethod
    def create_model_static(
        model_name: str, model_weights_path : str, 
        model_hyperparameters: dict, 
        super_res_factor: float, device) -> torch.nn.Module:

        r = InitModel.create_model(model_name, model_hyperparameters, super_res_factor, device)

        exp = nt.Model(r, device, model_weights_path)

        return exp
