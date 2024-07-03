# Diseases Classfication


## Installation

### Using pip
1. Clone the repository:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    ```

3. Activate the virtual environment:
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```bash
        source venv/bin/activate
        ```

4. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Using Conda
1. Clone the repository:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-directory>
    ```

2. Create a conda environment:
    ```bash
    conda create --name myenv python=3.10
    conda activate myenv
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```


### Download Weights
Download all weights from the following Google Drive link and place them in the appropriate directory:
[Google Drive Link](https://drive.google.com/drive/folders/1--0S27oZGIuTxfIa0bsX2XHDAm7KwM-L?usp=sharing)

## Running the Project
To launch the project, run the following command in the conda prompt:
```bash
streamlit run app.py
```