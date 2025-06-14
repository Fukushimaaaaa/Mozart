# Shakuhachi Tablature Prototype

This directory contains a minimal pipeline for optical music recognition of
vertical shakuhachi notation. It reuses the preprocessing and classifier logic
from the main project but applies a custom segmenter designed for the vertical
box layout typically found in shakuhachi scores. Runtime settings are stored in
`config.py` using [Pydantic](https://docs.pydantic.dev/).

## Training

The recognition step relies on a neural network classifier trained on isolated
symbols. Training data should be placed under `train_data/shakuhachi` where each
symbol has its own subdirectory containing sample images in PNG format:

```
train_data/
    shakuhachi/
        ro/
        tsu/
        re/
        chi/
        ...
```

To train the classifier run `train.py` in this folder:

```
python train.py
```

The trained model will be written to
`trained_models/shakuhachi_model.sav`.

## Usage

Once a model has been trained you can process a folder of score images using
`main.py`:

```
python main.py <input_folder> <output_folder> \
    --model-path trained_models/shakuhachi_model.sav \
    --dataset-path train_data/shakuhachi
```

Each output file contains the recognized shakuhachi symbols mapped to simple
MIDI note names as defined in `SHAKU_MAP` in `main.py`.
