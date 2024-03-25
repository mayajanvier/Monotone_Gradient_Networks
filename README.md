# Monotone_Gradient_Networks
[Maya Janvier](https://github.com/mayajanvier) and [Margot Boyer](https://github.com/MargotBoyer)  

Implementation of [Learning Gradients of Convex Functions with Monotone Gradient Networks (Chaudhari et al. (2023))](https://arxiv.org/abs/2301.10862) [1]


# Reproducing the results
The main code is in the `experiments.ipynb` notebook. You will be able to reproduce the following experiments:
- Implementation of models: `models.py` and section 1 notebook
- Gradient field experiment from [1] : section 2 notebook
- Optimal coupling: section 3 notebook, 3.a Wasserstein loss, 3.b KL-divergence loss, 3.c CP-Flow experiment (see setup)
- Color domain adaptation: section 4 notebook

  
# Setup
## Running CP-Flow experiment (part 3.c)
- Clone [CP-Flow](https://github.com/CW-Huang/CP-Flow) to be able to run the corresponding experiment in the notebook
- Move `models.py`,`train_ot_coupling.py` into the main folder of `CP-Flow`
- Move into CP-Flow folder and run `pip install -r requirements.txt`
- Run `python3 train_ot_coupling.py`

## Color adaptation experiment
If you want to run on more images, you can follow these steps: 
- Download the [Dark Zurich dataset](https://www.trace.ethz.ch/publications/2019/GCMA_UIoU/) validation set.
- Get the folder `/Dark_Zurich_val_anon/rgb_anon/val_ref/day/GOPR0356_ref` and add new pictures to the dark_zurich folder !

