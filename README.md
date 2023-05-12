# Plankton Classifier with Keras

This repository contains the code for a plankton classifier built with the Keras deep learning library. The model is trained to identify different species of plankton from images, using a dataset of 27,900 RGB FlowCam images with a resolution of 120x120x3. This work is based on a bachelor thesis by Gabriel Torland, Kim Svalde, and Andreas Primstad.

## Data

The dataset under study is comprised of FlowCam images acquired during voyages in the North Sea over the course of 2013 to 2018, specifically during the months of September and December. These images were systematically organized within a directory wherein individual folders were labelled according to the year and month corresponding to their respective acquisition dates. A provided CSV file, which contained classifications for each image, served as a key for sorting these images into designated folders based on their classification. This organization was facilitated by the script located at `ds_utils/ds_parser.py`. This script also incorporates functionalities that allow for the identification of corrupted images and the partitioning of the dataset into training, testing, and validation subsets. If an alternative split ratio is required, please employ the `ds_utils/ds_padder.py` script directly. It is important to note that the padding script is not flawless and may occasionally generate anomalies, as observed in the dataset used for this project. Therefore, an additional script was developed to detect such irregularities (i.e., `ds_utils/ds_abnormal_bg.py`). However, a manual validation is necessary post-detection to prevent the accidental deletion of normal images. In this project, this process was conducted alongside a comprehensive visual inspection of the entire dataset. To streamline the process, a bash/bat script is available. This script eliminates the need to execute each script individually and aids in structuring the output directory. The outcome of executing these scripts yields the baseline dataset, which is employed in the first benchmark of this project.

In the second benchmark of this project, the script `ds_utils/ds_balancer_v1.py` was employed to achieve full dataset balance through a combination of undersampling and oversampling techniques. In contrast, the third benchmark solely utilized oversampling, which the script `ds_utils/ds_balancer_v2.py` were responsible for.

To generate plots depicting the distribution of the different datasets, use the `ds_utils/ds_overview.py` script.

## Model & Training

The scripts responsible for training the respective models in this project are located within the subdirectories of the `models` directory (e.g., `models/resnet`). Each subdirectory is named to denote the corresponding network undergoing training. Predominantly, these scripts are in Jupyter Notebook format, enhancing the readability and interpretability of the code.

## Evaluation

The performance of the model is evaluated on the test set using the `result_utils/main.py` script.
