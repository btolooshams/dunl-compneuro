[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


# Deconvolutional Unrolled Neural Learning (DUNL) for Computational Neuroscience

This code is wrriten for this paper [https://www.cell.com/neuron/abstract/S0896-6273(25)00119-9](https://www.cell.com/neuron/abstract/S0896-6273(25)00119-9) published at Neuron.

Learning locally low-rank temporal representation from neural time series data.

### PATH

For any scripts to run, make sure you are in `dunl` directory.

### Configuration

Check `config` for detailed parameters of each experiment.

You should provide all detailed information about the model as a yaml file.

See `instrcutions.yaml` for information on important parameters.

### Data generation

Go to `dunl/preprocess_script`. Create a script similar to `generate_simulated_data_dopamine_spiking_into_dataformat.py`.

### Data preparation

Go to `dunl/preprocess_script`. Create a data dictionary that has format similar to those in preprocess files `preprocess_data_whisker_into_dataformat.pt`.

Run `prepare_data_and_save.py` which take a datafolder with numpy files created from last step (e.g., `...general_format_processed.npy`) and create a `..._trainready.py` data file.

The key module to load data can be found in `dunl/dataloader.py` which is `DUNLdataset`. Seethe Module for more info on the data.

### Tensorboard

`dunl/boardfunc.py` contain preliminary functions that are being used during training to report train progress onto a board.

### Training

See `dunl/train_sharekernels_acrossneurons.py` for an example training script.

See `dunl/train_sharekernels_acrossneurons_groupneuralfirings.py` for using group sparsity across neurons.






