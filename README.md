# Decentralized Projected Riemannian Gradient Descent (DPRGD)
This repo contains the code for the paper *Decentralized Online Riemannian Optimization with Dynamic Environments*. 
- Our simulation studies cover hyperbolic spaces and the space of symmetric prositive definite (SPD) matrices. 
- Our data applications involves environmental monitoring using the FLUXNET2015 dataset.

## Getting Started

Create and activate conda environments and install necessary dependencies.

```
conda create --name opt python=3.10
conda activate opt
pip install -r requirements.txt
```

Download [FLUXNET2015 dataset](https://fluxnet.org/data/fluxnet2015-dataset/) into the folder data/zip. 

Run the following code to unzip, select, and save raw data into data/raw.  

```
python src/data/data_loader.py
```

Next, we compute weekly correlation/covariance matrices and save the processed data into data/processed.

```
python src/data/data_processor.py
```

## Files

- simulation.ipynb contains the simulation experiments
- data.ipynb contains real data analysis using the [FLUXNET2015 dataset](https://fluxnet.org/data/fluxnet2015-dataset/)

## Citation

```
@article{chen2024decentralized,
  title={Decentralized Online Riemannian Optimization with Dynamic Environments},
  author={Chen, Hengchao and Sun, Qiang},
  journal={arXiv preprint arXiv:2410.05128},
  year={2024}
}
```
