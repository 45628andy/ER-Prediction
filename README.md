Multi-Modal Emergency Department Clinical Outcome Prediction Project
=========================

Python code for implementing the multi-modal prediction model based on MIMIC-IV datasets. Note: some files are from [https://github.com/nliulab/mimic4ed-benchmark/tree/main], the benchmark that this project is based on. 

## Table of contents
* [Dataset](#Dataset)
* [Additional Preprocessing](#Additional-Preprocessing)
* [Reproducing Result](#Reproducing-Result)
* [Citation](#citation)

## Dataset

First, MIMIC-IV datasets need to be obtained. These datasets are required and are not provided in this repoistory. Also, the datasets need to be preprocessed. Please follow [https://github.com/nliulab/mimic4ed-benchmark/tree/main] for preprocessing.

After preprocessing the `master_dataset.csv` file needs to be pasted to `processed` folder, and `train.csv` and `test.csv` need to be pasted to `processed_train_test` folder.


## Additional Preprocessing

For X-Ray notes, download `mimic-cxr-reports.zip`, unzip it, and paste it to `processed` folder. Run `data_add_notes.py` in interactive shell to add the embedding.

For med embedding, download MIMIC-IV ED dataset and paste it to `mimic-iv-ed-2.2` folder. Run `data_add_medrecon.py` in interactive shell to add the embedding.

## Reproducing Result

To reproduce the result, follow `Notebook_models_wtih_cxr.ipynb`.


## Citation

Xie F, Zhou J, Lee JW, Tan M, Li SQ, Rajnthern L, Chee ML, Chakraborty B, Wong AKI, Dagan A, Ong MEH, Gao F, Liu N. Benchmarking emergency department prediction models with machine learning and public electronic health records. Scientific Data 2022 Oct; 9: 658. <https://doi.org/10.1038/s41597-022-01782-9>