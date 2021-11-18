# Real-time 'me-not_me' Face Detector

Real-time face detector built using Python, TensorFlow/Keras and OpenCV. 

This is a program, that does real-time face detection on webcam image and also can distinguish me from other people.

| ![preview.jpg](article/img/preview.jpg) |
|:--:|
| <b>Real-time 'me-not_me' Face Detector</b>|

## Tutorial

You can read detailed tutorial [on Towards Data Science](https://towardsdatascience.com/how-to-create-real-time-face-detector-ff0e1f81925f) or directly [on GitLab](https://gitlab.com/Winston-90/me_not_me_detector/-/blob/main/article/article.md). There you can find problem statement, some theory explanations and code analysis.

## Results

[![Real-time 'me-not_me' Face Detector](https://img.youtube.com/vi/MtEcbV5hdhQ/0.jpg)](https://www.youtube.com/watch?v=MtEcbV5hdhQ)

If for some reason you can't see the video above here is the link - [Real-time 'me-not_me' Face Detector on Youtube](https://www.youtube.com/watch?v=MtEcbV5hdhQ).

## Setup

To run this code, you must have *tensorflow* and *opencv* libraries installed.

You should create a virtual environment, activate it and run `pip install -r requirements.txt`. You can also do it with conda - create virtual environment, activate it and run the following commands (they are listed in the `requirements.txt` file too):

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
│   ├───face_dataset_test_images
│   │   ├───me      # this folder contains TEST images for ME class
│   │   └───not_me  # this folder contains TEST images for NOT_ME class
│   ├───face_dataset_train_aug_images
│   │   ├───me      # this folder contains augmented TRAIN images for ME class (optional)
│   │   └───not_me  # this folder contains augmented TRAIN images for NOT_ME class (optional)
│   └───face_dataset_train_images
│       ├───me      # this folder contains TRAIN images for ME class
│       └───not_me  # this folder contains TRAIN images for NOT_ME class
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
- The `models` folder contains trained models for their test and further use.
- The `datasets` folder contains three folders - for a train set, test set and augmented train set (optional). Each of them contains two subfolders for two classes - *me* and *not_me*. In the general case, it contains N subfolders for N classes.

Now let's talk about the code files - jupyter notebooks. 
- `data_augmentation.ipynb` file creates an augmented dataset from an initial one and provides some information about the dataset.
- `me_not_me_classifier_model_comparison.ipynb` file contains code to train and test five different models.
- `me_not_me_classifier.ipynb` file does the same thing, but for one particular model. You can use it as an example to build your own classifier.
- `me_not_me_detector.ipynb` file uses the OpenCV library and turns the classifier into a real-time detector.

