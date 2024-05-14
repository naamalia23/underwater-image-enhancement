# Underwater Image Enhancement

This project was carried out for the final assignment of a computer vision course. We chose the theme of underwater image enhancement.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Introduction

Over the past few decades, underwater image processing has become a major concern due to its challenging and important nature. The quality of underwater images tends to be lower than images taken in the air, often appearing foggy and blurry. To improve image visualization, various image enhancement techniques such as equalization
histograms, contrast enhancement, and so on have been proposed by various researchers.

In this final project we tried to use several image enhancement methods: CLAHE, Unsharp Masking, and Fusion.

The quality measurement matrices that we use are UIQM (Underwater Image Quality Measurement) and UCIQE (Underwater Color Image Quality Evaluation).

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/naamalia23/underwater-image-enhancement.git
    ```

2. Navigate into the project directory:

    ```bash
    cd underwater-image-enhancement
    ```

3. Create and activate a virtual environment:

    * create virtual environment
    ```bash
    python -m venv venv
    ```
    
    * create virtual environment
    ```bash
    # Windows command prompt
    .venv\Scripts\activate.bat

    # Windows PowerShell
    .venv\Scripts\Activate.ps1

    # macOS and Linux
    source .venv/bin/activate
    ```

4. Install the required dependencies:

    ```bash
    pip install numpy
    pip install opencv-python
    pip install Pillow
    pip install streamlit
    ```

## Usage

You can run the main file that has been integrated with Streamlit using the following command:

```bash
streamlit run main.py
```

Supporting our experiments also allows working on code separate from streamlit that can be explored further. Code can be executed with command:

```bash
python filename.py
```

## Dependencies

- Python (>=3.6)
- [Streamlit](https://streamlit.io/)
- Library Numpy, CV2, Image