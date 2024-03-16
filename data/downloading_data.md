# Downloading data

1. Download `Extended SMPL+H model` from https://mano.is.tue.mpg.de/download.php
2. Download `DMPLs compatible with SMPL` from https://smpl.is.tue.mpg.de/download.php
3. Download AMASS dataset, i.e., all `SMPL+H G` files from https://amass.is.tue.mpg.de/download.php
4. Down `KIT Motion-Language Dataset` from https://drive.google.com/drive/folders/1MnixfyGfujSP-4t8w_2QvjtTVpEKr97t
4. Clone `HumanML3D` repository from https://github.com/EricGuo5513/HumanML3D?tab=readme-ov-file
5. run scripts
    1. `raw_pose_processing.ipynb` (takes a couple of hours)
    2. `motion_representation.ipynb`
    3. `cal_mean_variance.ipynb`
    4. (Optional. Run it if you need animations) `animation.ipynb`. *Need to change from p to maplotlib 3d.*




## Citations
% MOYO (not included in original release of AMASS)
@inproceedings{tripathi2023ipman,
  title     =  {{3D} Human Pose Estimation via Intuitive Physics}, 
  author    =  {Tripathi, Shashank and M{\"u}ller, Lea and Huang, Chun-Hao P. and Taheri Omid and Black, Michael J. and Tzionas, Dimitrios}, 
  booktitle =  {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  month     =  {June}
  year      =  {2023}
}

% AMASS
@inproceedings{AMASS:2019,
  title={AMASS: Archive of Motion Capture as Surface Shapes},
  author={Mahmood, Naureen and Ghorbani, Nima and F. Troje, Nikolaus and Pons-Moll, Gerard and Black, Michael J.},
  booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
  year={2019},
  month = {Oct},
  url = {https://amass.is.tue.mpg.de},
  month_numeric = {10}
}

% KIT Motion-Language Dataset
@article{Plappert2016,
    author = {Matthias Plappert and Christian Mandery and Tamim Asfour},
    title = {The {KIT} Motion-Language Dataset},
    journal = {Big Data}
    publisher = {Mary Ann Liebert Inc},
    year = 2016,
    month = {dec},
    volume = {4},
    number = {4},
    pages = {236--252},
    url = {http://dx.doi.org/10.1089/big.2016.0028},
    doi = {10.1089/big.2016.0028},
}

% HumanML3D
@InProceedings{Guo_2022_CVPR,
    author    = {Guo, Chuan and Zou, Shihao and Zuo, Xinxin and Wang, Sen and Ji, Wei and Li, Xingyu and Cheng, Li},
    title     = {Generating Diverse and Natural 3D Human Motions From Text},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {5152-5161}
}

% SMPL
@article{SMPL:2015,
    author = {Loper, Matthew and Mahmood, Naureen and Romero, Javier and Pons-Moll, Gerard and Black, Michael J.},
    title = {{SMPL}: A Skinned Multi-Person Linear Model},
    journal = {ACM Trans. Graphics (Proc. SIGGRAPH Asia)},
    month = oct,
    number = {6},
    pages = {248:1--248:16},
    publisher = {ACM},
    volume = {34},
    year = {2015}
}

% MANO
@article{MANO:SIGGRAPHASIA:2017,
    title = {Embodied Hands: Modeling and Capturing Hands and Bodies Together},
    author = {Romero, Javier and Tzionas, Dimitrios and Black, Michael J.},
    journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH Asia)},
    volume = {36},
    number = {6},
    series = {245:1--245:17},
    month = nov,
    year = {2017},
    month_numeric = {11}
}

% DMPL
@article{Dyna:SIGGRAPH:2015,
    title = {Dyna: A Model of Dynamic Human Shape in Motion},
    author = {Pons-Moll, Gerard and Romero, Javier and Mahmood, Naureen and Black, Michael J.},
    month = aug,
    number = {4},
    volume = {34},
    pages = {120:1--120:14}, 
    journal = {ACM Transactions on Graphics, (Proc. SIGGRAPH)},
    year = {2015}
}