## Quickstart

First, create the conda environment 
```
conda env create -f environment.yaml
```
Then, modify the hyperparameters in the respective config yaml file. An example file has been provided for you. You can use the `--help` flag to get descriptions of the configuration parameters or check the config dataclasses in the implementation. Finally, run `latent_icm.py` with the path to the config file. For example:
```
python latent_icm.py --config-file './configs/example_config.yaml' 
```
