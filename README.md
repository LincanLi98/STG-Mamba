# STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model

This is the official repository of our work `STG-Mamba`, the paper is currently posted on ArXiV.

❗️❗️This is an Ongoing Project, the full code will be released by Mid May.


## Requirements

- PyTorch==1.11.0
- Python==3.8.10
- numpy==1.22.4
- pandas==2.0.3
- einops==0.7.0
- argparse
- dataclasses
- typing
- time
- math




## Datasets

(1) PEMS04: `PEMS04` dataset is among the most popular benchmark in ST Data Mining & Traffic Forecasting. You can find the source data at [here](https://github.com/MengzhangLI/STFGNN/tree/master/data).

(2) KnowAir: `KnowAir` is an open-sourced weather dataset introduced by [Shuo Wang et al.](https://dl.acm.org/doi/abs/10.1145/3397536.3422208) at SIGSPATIAL' 20. We've already put the knowair data in this repository, with the extracted Graph Adjacency Matrix based on weather station's geographical location. You can also access the raw data at [here](https://drive.google.com/file/d/1R6hS5VAgjJQ_wu8i5qoLjIxY0BG7RD1L/view).  

(3) HZ-Metro: `HZ-Metro` is an open-sourced Metro Crowd-Flow dataset collected from 80 Metro Stations in HangZhou, China. You can find the raw data at [here](https://github.com/HCPLab-SYSU/PVCGN). Also, there is a [reference link](https://github.com/skyzh/Meteor/blob/master/station_line.csv) of the 80 Metro Stations' ID and their Name in real-world.



## Model Training/Testing

Using the following commands to Train/Test STG_Mamba model on `KnowAir`, `PEMS04`, `HZ_Metro` respectively. You can also optionally change the model parameters in the file `main.py`.

```bash
# KnowAir
python main.py -dataset=know_air -model=STGmamba

```

```bash
# PEMS04
python main.py -dataset=pems04 -model=STGmamba

```

```bash
#HZ_Metro
python main.py -dataset=hz_metro -model=STGmamba
```




## Citation

If you find this repository useful in your own research, please cite the following poster paper:
```
@misc{li2024stgmamba,
      title={STG-Mamba: Spatial-Temporal Graph Learning via Selective State Space Model}, 
      author={Lincan Li and Hanchen Wang and Wenjie Zhang and Adelle Coster},
      year={2024},
      eprint={2403.12418},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
