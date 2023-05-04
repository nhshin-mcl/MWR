[CVPR 2022] Moving Window Regression: A Novel Approach to Ordinal Regression
=============================================================================
Official Pytorch Implementation of the CVPR 2022 paper, "Moving Window Regression: A Novel Approach to Ordinal Regression."

Paper
-----------------------------------------------------------------------------
<!--[Moving Window Regression: A Novel Approach to Ordinal Regression]()-->
A novel ordinal regression algorithm, called moving window regression (MWR), is proposed in this paper. First, we propose the notion of relative rank (ρ-rank), which is a new order representation scheme for input and reference instances. Second, we develop global and local relative regressors (ρ-regressors) to predict ρ-ranks within entire and specific rank ranges, respectively. Third, we refine an initial rank estimate iteratively by selecting two reference instances to form a search window and then estimating the ρ-rank within the window. 

The full paper can be found via the [link](https://arxiv.org/abs/2203.13122) above.

<!--Please cite our paper if you use our code or dataset:-->

Datasets
-----------------------------------------------------------------------------
* [MORPH II](https://uncw.edu/oic/tech/morph.html)
* [CLAP2015](https://chalearnlap.cvc.uab.cat/dataset/18/description/)
* [CACD](https://bcsiriuschen.github.io/CARC/)
* [UTK](https://susanqq.github.io/UTKFace/)

Dependencies
-----------------------------------------------------------------------------
* Python 3
* Pytorch

Preprocessing
-----------------------------------------------------------------------------
We use [MTCNN](https://github.com/ipazc/mtcnn) for face detection and face alignment code provided from [pyimagesearch](https://pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/) for face alignment.

Pretrained Models and Reference Lists
-----------------------------------------------------------------------------
You can download models, IMDB-WIKI pretrained model, and reference lists [here](https://drive.google.com/drive/folders/1voOLN-_V6zzZTOmb8zKkT_2zBPjpliRm?usp=sharing).

Train
-----------------------------------------------------------------------------
To train MWR, run the script in train_code folder. 
```c
python train_code/train.py
```

Test
-----------------------------------------------------------------------------
Use the following command for evaluation.
```c
python test_code/op.py --dataset Dataset --regression Regression_type --experiment_setting Experimental_setting --im_path Image_path
```

