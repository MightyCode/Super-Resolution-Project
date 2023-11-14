[1mdiff --git a/compare_methods.ipynb b/compare_methods.ipynb[m
[1mindex 37bbf59..127d5ea 100644[m
[1m--- a/compare_methods.ipynb[m
[1m+++ b/compare_methods.ipynb[m
[36m@@ -354,13 +354,6 @@[m
     "\n",[m
     "plt.show()\n"[m
    ][m
[31m-  },[m
[31m-  {[m
[31m-   "cell_type": "code",[m
[31m-   "execution_count": null,[m
[31m-   "metadata": {},[m
[31m-   "outputs": [],[m
[31m-   "source": [][m
   }[m
  ],[m
  "metadata": {[m
[1mdiff --git a/src/test_rdn.py b/src/test_rdn.py[m
[1mindex fb4776f..ff4ebdf 100644[m
[1m--- a/src/test_rdn.py[m
[1m+++ b/src/test_rdn.py[m
[36m@@ -25,7 +25,7 @@[m [mfrom matplotlib import pyplot as plt[m
 [m
 [m
 if __name__ == '__main__':[m
[31m-    img = Image.open('resources/pokemon_jpg/pokemon_jpg/1.jpg')[m
[32m+[m[32m    img = Image.open('resources/pokemon/sugimori/train/low_res/1.png')[m
     plt.imshow(img)[m
     plt.show()[m
     img = torchvision.transforms.ToTensor()(img)[m
[1mdiff --git a/src/train_rdn.ipynb b/src/train_rdn.ipynb[m
[1mindex 224dcb1..79c35fb 100644[m
[1m--- a/src/train_rdn.ipynb[m
[1m+++ b/src/train_rdn.ipynb[m
[36m@@ -16,14 +16,14 @@[m
   },[m
   {[m
    "cell_type": "code",[m
[31m-   "execution_count": 4,[m
[32m+[m[32m   "execution_count": 2,[m
    "metadata": {},[m
    "outputs": [[m
     {[m
      "name": "stdout",[m
      "output_type": "stream",[m
      "text": [[m
[31m-      "cpu\n"[m
[32m+[m[32m      "cuda\n"[m
      ][m
     }[m
    ],[m
[36m@@ -52,7 +52,7 @@[m
   },[m
   {[m
    "cell_type": "code",[m
[31m-   "execution_count": 5,[m
[32m+[m[32m   "execution_count": 3,[m
    "metadata": {},[m
    "outputs": [],[m
    "source": [[m
[36m@@ -104,25 +104,18 @@[m
   },[m
   {[m
    "cell_type": "code",[m
[31m-   "execution_count": 7,[m
[32m+[m[32m   "execution_count": 4,[m
    "metadata": {},[m
    "outputs": [[m
     {[m
[31m-     "ename": "AttributeError",[m
[31m-     "evalue": "'RDN' object has no attribute '_parameters'",[m
[32m+[m[32m     "ename": "NameError",[m
[32m+[m[32m     "evalue": "name 'train_set' is not defined",[m
      "output_type": "error",[m
      "traceback": [[m
       "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",[m
[31m-      "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",[m
[31m-      "Cell \u001b[0;32mIn[7], line 4\u001b[0m\n\u001b[1;32m      2\u001b[0m r \u001b[38;5;241m=\u001b[39m RDN(C\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m6\u001b[39m, D\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m4\u001b[39m, G\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m32\u001b[39m, G0\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m64\u001b[39m, scaling_factor\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m2\u001b[39m, kernel_size\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m3\u001b[39m, c_dims\u001b[38;5;241m=\u001b[39m\u001b[38;5;241m3\u001b[39m, upscaling\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mups\u001b[39m\u001b[38;5;124m'\u001b[39m, weights\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mNone\u001b[39;00m)\n\u001b[1;32m      3\u001b[0m \u001b[38;5;66;03m#r = r.to(device)\u001b[39;00m\n\u001b[0;32m----> 4\u001b[0m adam \u001b[38;5;241m=\u001b[39m \u001b[43mtorch\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43moptim\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mAdam\u001b[49m\u001b[43m(\u001b[49m\u001b[43mr\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mparameters\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mlr\u001b[49m\u001b[38;5;241;43m=\u001b[39;49m\u001b[43mlr\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m      5\u001b[0m stats_manager \u001b[38;5;241m=\u001b[39m SuperResolutionStatsManager()\n\u001b[1;32m      6\u001b[0m exp1 \u001b[38;5;241m=\u001b[39m nt\u001b[38;5;241m.\u001b[39mExperiment(r, train_set, val_set, adam, stats_manager, device, batch_size\u001b[38;5;241m=\u001b[39mB,\n\u001b[1;32m      7\u001b[0m                      output_dir\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdenoising1\u001b[39m\u001b[38;5;124m\"\u001b[39m, perform_validation_during_training\u001b[38;5;241m=\u001b[39m\u001b[38;5;28;01mTrue\u001b[39;00m)\n",[m
[31m-      "File \u001b[0;32m~/enseirb/2A/TD/machineDeepLearning/mldl/lib/python3.8/site-packages/torch/optim/adam.py:137\u001b[0m, in \u001b[0;36mAdam.__init__\u001b[0;34m(self, params, lr, betas, eps, weight_decay, amsgrad, foreach, maximize, capturable, differentiable, fused)\u001b[0m\n\u001b[1;32m    132\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mInvalid weight_decay value: \u001b[39m\u001b[38;5;132;01m{}\u001b[39;00m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;241m.\u001b[39mformat(weight_decay))\n\u001b[1;32m    133\u001b[0m defaults \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mdict\u001b[39m(lr\u001b[38;5;241m=\u001b[39mlr, betas\u001b[38;5;241m=\u001b[39mbetas, eps\u001b[38;5;241m=\u001b[39meps,\n\u001b[1;32m    134\u001b[0m                 weight_decay\u001b[38;5;241m=\u001b[39mweight_decay, amsgrad\u001b[38;5;241m=\u001b[39mamsgrad,\n\u001b[1;32m    135\u001b[0m                 maximize\u001b[38;5;241m=\u001b[39mmaximize, foreach\u001b[38;5;241m=\u001b[39mforeach, capturable\u001b[38;5;241m=\u001b[39mcapturable,\n\u001b[1;32m    136\u001b[0m                 differentiable\u001b[38;5;241m=\u001b[39mdifferentiable, fused\u001b[38;5;241m=\u001b[39mfused)\n\u001b[0;32m--> 137\u001b[0m \u001b[38;5;28;43msuper\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mAdam\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[38;5;28;43mself\u001b[39;49m\u001b[43m)\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[38;5;21;43m__init__\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mparams\u001b[49m\u001b[43m,\u001b[49m\u001b[43m \u001b[49m\u001b[43mdefaults\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m    139\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m fused:\n\u001b[1;32m    140\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m differentiable:\n",[m
[31m-      "File \u001b[0;32m~/enseirb/2A/TD/machineDeepLearning/mldl/lib/python3.8/site-packages/torch/optim/optimizer.py:59\u001b[0m, in \u001b[0;36mOptimizer.__init__\u001b[0;34m(self, params, defaults)\u001b[0m\n\u001b[1;32m     56\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mstate \u001b[38;5;241m=\u001b[39m defaultdict(\u001b[38;5;28mdict\u001b[39m)\n\u001b[1;32m     57\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mparam_groups \u001b[38;5;241m=\u001b[39m []\n\u001b[0;32m---> 59\u001b[0m param_groups \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;43mlist\u001b[39;49m\u001b[43m(\u001b[49m\u001b[43mparams\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m     60\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mlen\u001b[39m(param_groups) \u001b[38;5;241m==\u001b[39m \u001b[38;5;241m0\u001b[39m:\n\u001b[1;32m     61\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mValueError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124moptimizer got an empty parameter list\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",[m
[31m-      "File \u001b[0;32m~/enseirb/2A/TD/machineDeepLearning/mldl/lib/python3.8/site-packages/torch/nn/modules/module.py:1710\u001b[0m, in \u001b[0;36mModule.parameters\u001b[0;34m(self, recurse)\u001b[0m\n\u001b[1;32m   1688\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mparameters\u001b[39m(\u001b[38;5;28mself\u001b[39m, recurse: \u001b[38;5;28mbool\u001b[39m \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mTrue\u001b[39;00m) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m Iterator[Parameter]:\n\u001b[1;32m   1689\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124mr\u001b[39m\u001b[38;5;124;03m\"\"\"Returns an iterator over module parameters.\u001b[39;00m\n\u001b[1;32m   1690\u001b[0m \n\u001b[1;32m   1691\u001b[0m \u001b[38;5;124;03m    This is typically passed to an optimizer.\u001b[39;00m\n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m   1708\u001b[0m \n\u001b[1;32m   1709\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[0;32m-> 1710\u001b[0m     \u001b[38;5;28;01mfor\u001b[39;00m name, param \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mnamed_parameters(recurse\u001b[38;5;241m=\u001b[39mrecurse):\n\u001b[1;32m   1711\u001b[0m         \u001b[38;5;28;01myield\u001b[39;00m param\n",[m
[31m-      "File \u001b[0;32m~/enseirb/2A/TD/machineDeepLearning/mldl/lib/python3.8/site-packages/torch/nn/modules/module.py:1737\u001b[0m, in \u001b[0;36mModule.named_parameters\u001b[0;34m(self, prefix, recurse)\u001b[0m\n\u001b[1;32m   1714\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124mr\u001b[39m\u001b[38;5;124;03m\"\"\"Returns an iterator over module parameters, yielding both the\u001b[39;00m\n\u001b[1;32m   1715\u001b[0m \u001b[38;5;124;03mname of the parameter as well as the parameter itself.\u001b[39;00m\n\u001b[1;32m   1716\u001b[0m \n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m   1732\u001b[0m \n\u001b[1;32m   1733\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[1;32m   1734\u001b[0m gen \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_named_members(\n\u001b[1;32m   1735\u001b[0m     \u001b[38;5;28;01mlambda\u001b[39;00m module: module\u001b[38;5;241m.\u001b[39m_parameters\u001b[38;5;241m.\u001b[39mitems(),\n\u001b[1;32m   1736\u001b[0m     prefix\u001b[38;5;241m=\u001b[39mprefix, recurse\u001b[38;5;241m=\u001b[39mrecurse)\n\u001b[0;32m-> 1737\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m elem \u001b[38;5;129;01min\u001b[39;00m gen:\n\u001b[1;32m   1738\u001b[0m     \u001b[38;5;28;01myield\u001b[39;00m elem\n",[m
[31m-      "File \u001b[0;32m~/enseirb/2A/TD/machineDeepLearning/mldl/lib/python3.8/site-packages/torch/nn/modules/module.py:1680\u001b[0m, in \u001b[0;36mModule._named_members\u001b[0;34m(self, get_members_fn, prefix, recurse)\u001b[0m\n\u001b[1;32m   1678\u001b[0m modules \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mnamed_modules(prefix\u001b[38;5;241m=\u001b[39mprefix) \u001b[38;5;28;01mif\u001b[39;00m recurse \u001b[38;5;28;01melse\u001b[39;00m [(prefix, \u001b[38;5;28mself\u001b[39m)]\n\u001b[1;32m   1679\u001b[0m \u001b[38;5;28;01mfor\u001b[39;00m module_prefix, module \u001b[38;5;129;01min\u001b[39;00m modules:\n\u001b[0;32m-> 1680\u001b[0m     members \u001b[38;5;241m=\u001b[39m \u001b[43mget_members_fn\u001b[49m\u001b[43m(\u001b[49m\u001b[43mmodule\u001b[49m\u001b[43m)\u001b[49m\n\u001b[1;32m   1681\u001b[0m     \u001b[38;5;28;01mfor\u001b[39;00m k, v \u001b[38;5;129;01min\u001b[39;00m members:\n\u001b[1;32m   1682\u001b[0m         \u001b[38;5;28;01mif\u001b[39;00m v \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;129;01mor\u001b[39;00m v \u001b[38;5;129;01min\u001b[39;00m memo:\n",[m
[31m-      "File \u001b[0;32m~/enseirb/2A/TD/machineDeepLearning/mldl/lib/python3.8/site-packages/torch/nn/modules/module.py:1735\u001b[0m, in \u001b[0;36mModule.named_parameters.<locals>.<lambda>\u001b[0;34m(module)\u001b[0m\n\u001b[1;32m   1713\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m \u001b[38;5;21mnamed_parameters\u001b[39m(\u001b[38;5;28mself\u001b[39m, prefix: \u001b[38;5;28mstr\u001b[39m \u001b[38;5;241m=\u001b[39m \u001b[38;5;124m'\u001b[39m\u001b[38;5;124m'\u001b[39m, recurse: \u001b[38;5;28mbool\u001b[39m \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mTrue\u001b[39;00m) \u001b[38;5;241m-\u001b[39m\u001b[38;5;241m>\u001b[39m Iterator[Tuple[\u001b[38;5;28mstr\u001b[39m, Parameter]]:\n\u001b[1;32m   1714\u001b[0m \u001b[38;5;250m    \u001b[39m\u001b[38;5;124mr\u001b[39m\u001b[38;5;124;03m\"\"\"Returns an iterator over module parameters, yielding both the\u001b[39;00m\n\u001b[1;32m   1715\u001b[0m \u001b[38;5;124;03m    name of the parameter as well as the parameter itself.\u001b[39;00m\n\u001b[1;32m   1716\u001b[0m \n\u001b[0;32m   (...)\u001b[0m\n\u001b[1;32m   1732\u001b[0m \n\u001b[1;32m   1733\u001b[0m \u001b[38;5;124;03m    \"\"\"\u001b[39;00m\n\u001b[1;32m   1734\u001b[0m     gen \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_named_members(\n\u001b[0;32m-> 1735\u001b[0m         \u001b[38;5;28;01mlambda\u001b[39;00m module: \u001b[43mmodule\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43m_parameters\u001b[49m\u001b[38;5;241m.\u001b[39mitems(),\n\u001b[1;32m   1736\u001b[0m         prefix\u001b[38;5;241m=\u001b[39mprefix, recurse\u001b[38;5;241m=\u001b[39mrecurse)\n\u001b[1;32m   1737\u001b[0m     \u001b[38;5;28;01mfor\u001b[39;00m elem \u001b[38;5;129;01min\u001b[39;00m gen:\n\u001b[1;32m   1738\u001b[0m         \u001b[38;5;28;01myield\u001b[39;00m elem\n",[m
[31m-      "File \u001b[0;32m~/enseirb/2A/TD/machineDeepLearning/mldl/lib/python3.8/site-packages/torch/nn/modules/module.py:1269\u001b[0m, in \u001b[0;36mModule.__getattr__\u001b[0;34m(self, name)\u001b[0m\n\u001b[1;32m   1267\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m name \u001b[38;5;129;01min\u001b[39;00m modules:\n\u001b[1;32m   1268\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m modules[name]\n\u001b[0;32m-> 1269\u001b[0m \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mAttributeError\u001b[39;00m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;132;01m{}\u001b[39;00m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m object has no attribute \u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;132;01m{}\u001b[39;00m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;241m.\u001b[39mformat(\n\u001b[1;32m   1270\u001b[0m     \u001b[38;5;28mtype\u001b[39m(\u001b[38;5;28mself\u001b[39m)\u001b[38;5;241m.\u001b[39m\u001b[38;5;18m__name__\u001b[39m, name))\n",[m
[31m-      "\u001b[0;31mAttributeError\u001b[0m: 'RDN' object has no attribute '_parameters'"[m
[32m+[m[32m      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",[m
[32m+[m[32m      "\u001b[1;32m/media/maxence/DATA/Autres/cours/enseirb/3A/Super-Resolution-Project/src/train_rdn.ipynb Cell 8\u001b[0m line \u001b[0;36m7\n\u001b[1;32m      <a href='vscode-notebook-cell:/media/maxence/DATA/Autres/cours/enseirb/3A/Super-Resolution-Project/src/train_rdn.ipynb#X10sZmlsZQ%3D%3D?line=4'>5</a>\u001b[0m adam \u001b[39m=\u001b[39m torch\u001b[39m.\u001b[39moptim\u001b[39m.\u001b[39mAdam(r\u001b[39m.\u001b[39mparameters(), lr\u001b[39m=\u001b[39mlr)\n\u001b[1;32m      <a href='vscode-notebook-cell:/media/maxence/DATA/Autres/cours/enseirb/3A/Super-Resolution-Project/src/train_rdn.ipynb#X10sZmlsZQ%3D%3D?line=5'>6</a>\u001b[0m stats_manager \u001b[39m=\u001b[39m SuperResolutionStatsManager()\n\u001b[0;32m----> <a href='vscode-notebook-cell:/media/maxence/DATA/Autres/cours/enseirb/3A/Super-Resolution-Project/src/train_rdn.ipynb#X10sZmlsZQ%3D%3D?line=6'>7</a>\u001b[0m exp1 \u001b[39m=\u001b[39m nt\u001b[39m.\u001b[39mExperiment(r, train_set, val_set, adam, stats_manager, device, batch_size\u001b[39m=\u001b[39mB,\n\u001b[1;32m      <a href='vscode-notebook-cell:/media/maxence/DATA/Autres/cours/enseirb/3A/Super-Resolution-Project/src/train_rdn.ipynb#X10sZmlsZQ%3D%3D?line=7'>8</a>\u001b[0m                      output_dir\u001b[39m=\u001b[39m\u001b[39m\"\u001b[39m\u001b[39msuperresol\u001b[39m\u001b[39m\"\u001b[39m, perform_validation_during_training\u001b[39m=\u001b[39m\u001b[39mTrue\u001b[39;00m)\n",[m
[32m+[m[32m      "\u001b[0;31mNameError\u001b[0m: name 'train_set' is not defined"[m
      ][m
     }[m
    ],[m
[36m@@ -154,7 +147,7 @@[m
    "name": "python",[m
    "nbconvert_exporter": "python",[m
    "pygments_lexer": "ipython3",[m
[31m-   "version": "3.8.10"[m
[32m+[m[32m   "version": "3.10.13"[m
   }[m
  },[m
  "nbformat": 4,[m
