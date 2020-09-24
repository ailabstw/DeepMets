# DeepMets&reg;

This repository contains the inference code for DeepMets on Python3 and Pytorch. This project is recently co-developed by [Taiwan AI Labs](https://ailabs.tw/) and Taipei Veterans General Hospital. DeepMets trained on 1029 in-house T1 contrast-enhanced MRI dataset generates segmentation mask for brain metastasis.

## How to obtain license and model weights 

If you wish to obtain the license, model weights, data or other further information for DeepMets, please contact us ([contact@taimedimg.tw](contact@taimedimg.tw)).

## How to run the code

    python main.py --dataset <DATA_FILE> --checkpoint <CKPT_FILE> --license <LICENSE_FILE> --output-path <OUTPUT_FOLDER>
    
- **DATA_FILE:** A .csv or .txt file that contains paths of folder (with multiple dicom files inside) that you want to inference.
- **CKPT_FILE:** Path of checkpoint file.
- **LICENSE_FILE:** Path of license file.
- **OUTPUT_FOLDER:** Path to save inference results.