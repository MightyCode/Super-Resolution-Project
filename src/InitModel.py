from src.UpscaleNN import UpscaleNN, UpscaleResidualNN
from src.rdn import RDN

import src.nntools as nt

import torch

class InitModel:
    @staticmethod
    def create_model(
        model_name: str, model_weights_path : str, model_hyperparameters: dict, 
        super_res_factor: float, device) -> torch.nn.Module:

        # to lower
        r = None
        if "upscale" in model_name.lower():
            print(model_name.lower())
            if "residual" in model_name.lower() or "resid" in model_name.lower():
                MODEL_INIT = UpscaleResidualNN
            else:
                MODEL_INIT = UpscaleNN

            r = MODEL_INIT(super_res_factor=super_res_factor, old_version=("old" in model_name.lower())) 
        elif "rdn" in model_name.lower():
            r = RDN(C=model_hyperparameters["C"], D=model_hyperparameters["D"], 
                    G=model_hyperparameters["G"], G0=model_hyperparameters["G0"], 
                    scaling_factor=super_res_factor, 
                    kernel_size=model_hyperparameters["kernelSize"], 
                    upscaling='shuffle', weights=None)
        
        if r is None:
            raise Exception("The model name is not correct")
        
        r.to(device)

        exp = nt.Model(r, device, model_weights_path)

        return exp
