# Plankton Classification on a Novel Dataset

This repository contains the code utilized in a bachelor's thesis undertaken by Gabriel Torland, Kim Svalde, and Andreas Primstad. 

## Project Objective

The aim of this project is to create a novel plankton dataset and assess its efficacy through the classification performance of established convolutional neural networks. 

To achieve this objective, the project will compare the performance of known convolutional neural networks, and a research review will be conducted to determine which networks to include based on their prevalence in this domain. Furthermore, the project will explore strategies to mitigate dataset imbalances through various augmentation techniques, and evaluate how this affects classification performance on different plankton classes. Next, the difference between transfer learning and fine-tuning will be assessed through experimentation, and the difference between the various network architectures will be evaluated. Performance metrics, including F1-scores, Precision, Recall, Top-1 error, Top-5 error, and area under the curve (AUC-PR), will be analyzed for each network configuration.  

By systematically evaluating the different architectures, augmentation strategies, and training approaches, this project aims to provide insight into the efficacy of known convolutional neural networks on a novel dataset of plankton classes.

## Dataset

The dataset under study is comprised of FlowCam images acquired during voyages in the North Sea over the course of 2013 to 2018, specifically during the months of September and December. These images were systematically organized within a directory wherein individual folders were labelled according to the year and month corresponding to their respective acquisition dates. A provided CSV file, which contained classifications for each image, served as a key for sorting these images into designated folders based on their classification. This organization was facilitated by the script located at `ds_utils/ds_parser.py`. This script also incorporates functionalities that allow for the identification of corrupted images and the partitioning of the dataset into training, testing, and validation subsets. If an alternative split ratio is required, please employ the `ds_utils/ds_padder.py` script directly. It is important to note that the padding script is not flawless and may occasionally generate anomalies, as observed in the dataset used for this project. Therefore, an additional script was developed to detect such irregularities (i.e., `ds_utils/ds_abnormal_bg.py`). However, a manual validation is necessary post-detection to prevent the accidental deletion of normal images. In this project, this process was conducted alongside a comprehensive visual inspection of the entire dataset. To streamline the process, a bash/bat script is available. This script eliminates the need to execute each script individually and aids in structuring the output directory. The outcome of executing these scripts yields the baseline dataset, which is employed in the first benchmark of this project.

In the second benchmark of this project, the script `ds_utils/ds_balancer_v1.py` was employed to achieve full dataset balance through a combination of undersampling and oversampling techniques. In contrast, the third benchmark solely utilized oversampling, which the script `ds_utils/ds_balancer_v2.py` were responsible for.

To generate plots depicting the distribution of the different datasets, use the `ds_utils/ds_overview.py` script.

## Model & Training

The scripts responsible for training the respective models in this project are located within the subdirectories of the `models` directory (e.g., `models/resnet`). Each subdirectory is named to denote the corresponding network undergoing training. Predominantly, these scripts are in Jupyter Notebook format, enhancing the readability and interpretability of the code.

## Evaluation

The performance of the model is evaluated on the test set using the `result_utils/main.py` script.

# Setup
First, clone the project to your local system:

```bash
git clone https://github.com/username/project.git
cd project
```
Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

Now you should be able to execute any script in the repository!

# Licence

This project is licensed under the MIT License.

The MIT License is a short and simple permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code.

Here's a summary of the key points:

- **Use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software**: You can do almost anything with the software, from downloading it and using it for personal projects, to incorporating it into your own software and distributing it, even commercially.

- **Inclusion of copyright and license notices**: If you use this software in your own work, you need to include the original copyright and license notice.

- **The software is provided 'as is', without any warranty or conditions**: The author of the software is not responsible if the software doesn't work, or if it causes any damage. You're using it at your own risk.

For the exact terms, see the [LICENSE](LICENSE) file in the project root.

# Contact
If you have any questions about the project, please feel free to reach out:

**Gabriel Torland**  
[<img align="left" alt="gmail" width="22px" src="https://cdn-icons-png.flaticon.com/512/281/281769.png" />](mailto:gabri.torland@gmail.com) gabri.torland@gmail.com  
[<img align="left" alt="github" width="22px" src="https://cdn-icons-png.flaticon.com/512/25/25231.png" />](https://github.com/GabrielTorland) @GabrielTorland

**Kim Andre Svalde**  
[<img align="left" alt="gmail" width="22px" src="https://cdn-icons-png.flaticon.com/512/281/281769.png" />](mailto:kimsvalde@gmail.com) kimsvalde@gmail.com  
[<img align="left" alt="github" width="22px" src="https://cdn-icons-png.flaticon.com/512/25/25231.png" />](https://github.com/kimdal1) @kimdal1

**Andreas Primstad**  
[<img align="left" alt="gmail" width="22px" src="https://cdn-icons-png.flaticon.com/512/281/281769.png" />](mailto:andreaspri@live.no) andreaspri@live.no  
[<img align="left" alt="github" width="22px" src="https://cdn-icons-png.flaticon.com/512/25/25231.png" />](https://github.com/Andreaspri) @Andreaspri


