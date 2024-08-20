## Table of Contents

- Introduction
- Installation
- Usage example
## Introduction
This repository build an demo of Images Retrieval.

We will use CLIP model to predct with different measure L1, L2, Cosine Similarity and Correlation Coefficient.
Based on dataset we will query image use the measurements in turn to see the difference result. 
## Installation
### 1. Clone the repository  
```
git clone https://github.com/thehaodev/IMAGE-RETRIEVAL.git
```
### 2. Create and activate a virtual environment 
```
py -m venv .venv
.venv\Scripts\activate
```
### 3. Install the required dependencies 
```
pip install -r requirements.txt
```
## Usage example
In query.py there is two function to query image, one is use clip and the other just use pure measure  L1, L2, Cosine Similarity and Correlation Coefficient.

This is example result
![image](https://github.com/user-attachments/assets/f8b92c54-be60-4067-a614-df636178adbb)


Note: This is sample demo so it's not have UI yet. I will update later. 


