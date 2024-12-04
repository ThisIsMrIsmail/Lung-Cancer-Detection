# Lung Cancer Detection
<ul>
<li>Built a CNN Model that classifies patients into 3 classes according to Breast CT-scanned Images.</li>
<li>Developed using python with its libraries Tensorflow, Keras, NumPy, and Pandas.</li>
<li>Used the Deep learning model Convolutional Neural Network with some optimization methods.</li>
</ul>

https://github.com/ThisIsMrIsmail/Lung-Cancer-Detection/assets/96701820/74ff4d15-8c15-4275-ac6e-65d1654f9f62

## Getting the Dataset

1. **Download the dataset:**
    You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/adityamahimkar/iqothnccd-lung-cancer-dataset) or any other reliable source that provides lung cancer CT scan images.

2. **Extract the dataset:**
    After downloading, extract the dataset into the `data` directory within the project folder.

    ```bash
    mkdir -p data
    unzip /path/to/downloaded/dataset.zip -d data
    ```

3. **Verify the dataset structure:**
    Ensure that the dataset is structured correctly, with images organized into appropriate subdirectories for each class.

    ```
    Data/
    ├── The IQ-OTHNCCD lung cancer dataset/
    │   ├── Bengin cases/
    │   ├── Malignant cases/
    │   └── Normal cases/
    └── ...
    ```

Make sure you have the necessary permissions to download and use the dataset.


## How to Run the Code

1. **Clone the repository:**
    ```bash
    git clone https://github.com/ThisIsMrIsmail/Lung-Cancer-Detection.git
    ```
2. **Navigate to the project directory:**
    ```bash
    cd Lung-Cancer-Detection
    ```
3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Run the backend server:**
    ```bash
    python api/app.py
    ```
5. **Open the web app url on browser, and start testing:**
    ```bash
    http://127.0.0.1:5000
    ```

Make sure you have Python and pip installed on your system.