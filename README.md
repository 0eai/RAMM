# RAMM

This repository provides the code for the paper [will be updated soon](https://willbeupdated.com).

## Introduction
This repository contains code for a multimodal transformer model, designed for the MuSe 2024 challenge. The model supports various tasks such as humor detection and perception analysis. It allows for flexible combinations of features and labels. It also incorporates a Unimodal Model to handle single feature input.

## Usage

To train or evaluate the model, run the `main.py` script with appropriate arguments. Here's an example:

```bash
python main.py --task humor --features w2v-msp bert-multilingual vit-fer \
               --epochs 100 --batch_size 256 \
               --result_csv results.csv
```

This command trains a model on the humor task using w2v-msp, bert-multilingual, and vit-fer features.

## Arguments

The script accepts various arguments to control its behavior:

* `--task`: Specify the task to perform (e.g., `humor`, `perception`).
* `--features`: Specify a list of features to use (e.g., `w2v-msp`, `bert-multilingual`, `vit-fer`).
* `--label_dim`: Specify the label dimension for the perception task.
* `--normalize`: Enables feature normalization.
* `--epochs`: Sets the number of training epochs.
* `--batch_size`: Sets the batch size for training and evaluation.
* `--result_csv`: Appends results to a CSV file.
* `--predict`: Makes predictions on the test set and saves them.
* `--eval_model`: Evaluates an existing model instead of training.
* `--device`: Specifies the device to use (e.g., `cpu`, `cuda`).

Refer to the script's argument parser for a complete list of options and their descriptions.

## Key Features

* **Multimodal and Unimodal:** The model handles both multiple input modalities (features) through a transformer architecture, and single feature input using the Unimodal Model.
* **Flexible:** Supports various tasks and allows for custom feature and label combinations.
* **Normalization:** Optionally normalizes features for better performance.
* **Caching:** Caches preprocessed data to speed up subsequent runs.
* **Evaluation and Prediction:** Can be used for both model evaluation and generating predictions on new data.

## Dependencies

* Python 3.x
* PyTorch
* Pandas
* NumPy
* Scikit-learn
* tqdm
* ... (other common machine learning libraries)

## Project Structure

* `config.py`: Configuration settings for paths, tasks, features, etc.
* `data_parser.py`: Functions to load and preprocess data.
* `dataset.py`: Defines the `MuSeDataset` class for handling data loading and batching.
* `eval.py`: Evaluation functions and metrics.
* `model.py`: Defines the core transformer model architecture and the Unimodal Model.
* `train.py`: Training and validation functions.
* `main.py`: The main script for running the model.
* `utils.py`: Helper functions for logging and other utilities.

## Notes

* Ensure that you have the necessary data files in the paths specified in `config.py`.
* Adjust the configuration settings and model hyperparameters as needed for your specific use case.

Please feel free to contribute or report any issues!

## Contact
Feel free to reach out to me at [ankit<at>soongsil<dot>ac<dot>kr](mailto:ankit<at>soongsil<dot>ac<dot>kr) if you encounter any issues or have more exciting ideas related to this project.

## Citation

If you use this code or methodology in your research, please cite our paper:

    Wiil be updated.