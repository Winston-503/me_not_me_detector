# Real-time 'me-not_me' Face Detector

Real-time face detector with Python, TensorFlow/Keras and OpenCV. This is program, that do real-time face detection on webcam image and also can distinguish me from other people.

You can read detailed tutorial [here](). There you can find problem statement, some theory explanations and code analysis.

## Results

[![Real-time 'me-not_me' Face Detector](https://img.youtube.com/vi/j_cn1p6_B7I/0.jpg)](https://www.youtube.com/watch?v=j_cn1p6_B7I)

If for some reason you can't see the video above here is the link - [Real-time 'me-not_me' Face Detector on Youtube](https://www.youtube.com/watch?v=j_cn1p6_B7I).

## Setup

To run this code, you must have *tensorflow* and *opencv* libraries installed.

You should create virtual environment, activate it and run `pip install -r requirements.txt`. You can also do it with conda - create virtual environment, activate it and run following commands (they are listed in `requirements.txt` file too):

```
conda install -c conda-forge numpy
conda install -c conda-forge opencv
conda install -c conda-forge tensorflow
```

## Project Structure

The project has the following structure:

```
me_not_me_detector
├───article
├───datasets
├───models
│   .gitignore
│   data_augmentation.ipynb
│   me_not_me_classifier.ipynb
│   me_not_me_classifier_model_comparison.ipynb
│   me_not_me_detector.ipynb
│   README.md
└── requirements.txt
```

Let's talk about folders.
- The `article` folder contains the data for the tutorial.
- The `datasets` folder contains datasets, each of them has two classes - *me* and *not_me*. To know more about the dataset see `data_augmentation.ipynb`.
- The `models` folder contains the trained models for their test and further use.

Now let's talk about the code files - jupyter notebooks. 
- `data_augmentation.ipynb` file creates an augmented dataset from an initial one and provides some information about the dataset.
- `me_not_me_classifier_model_comparison.ipynb` file contains code to train and test five different models.
- `me_not_me_classifier.ipynb` file does the same thing, but for one particular model. You can use it as an example to build your own classifier.
- `me_not_me_detector.ipynb` file uses the OpenCV library and turns the classifier into a real-time detector.

