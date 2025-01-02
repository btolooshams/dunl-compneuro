[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)


# Deconvolutional Unrolled Neural Learning (DUNL) for Computational Neuroscience

This repo is in progress.

This code is wrriten for this paper [https://pmc.ncbi.nlm.nih.gov/articles/PMC10802267](https://pmc.ncbi.nlm.nih.gov/articles/PMC10802267/).

Learning locally low-rank temporal representation from neural time series data.

### PATH

For any scripts to run, make sure you are in `src` directory.

### Configuration

Check `config` for detailed parameters of each experiment.

You should provide all detailed information about the model as a yaml file.

See `instrcutions.yaml` for information on important parameters.

### Data generation

Go to `src/preprocess_script`. Create a script similar to `generate_simulated_data_dopamine_spiking_into_dataformat.py`.

### Data preparation

Go to `src/preprocess_script`. Create a data dictionary that has format similar to those in preprocess files `preprocess_data_whisker_into_dataformat.pt`.

Run `prepare_data_and_save.py` which take a datafolder with numpy files created from last step (e.g., `...general_format_processed.npy`) and create a `..._trainready.py` data file.

The key module to load data can be found in `src/dataloader.py` which is `DUNLdataset`. Seethe Module for more info on the data.

### Tensorboard

`src/boardfunc.py` contain preliminary functions that are being used during training to report train progress onto a board.

### Training

See `src/train_sharekernels_acrossneurons.py` for an example training script.

See `src/train_sharekernels_acrossneurons_groupneuralfirings.py` for using group sparsity across neurons.






