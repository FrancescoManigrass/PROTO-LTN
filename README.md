# PROTOtypical Logic Tensor Networks (PROTO-LTN) for Zero Shot Learning


- The repository contains the code to implement ProtoLTN, the grounded ontology and the script to evaluate the models on Awa2,CUB,aPY and SUN  datasets.
- Logic Tensor Network framework is cloned from the paper Learning and Reasoning in Logic Tensor Networks: Theory and Application for Semantic Image Interpretation by Serafini Luciano, Donadello Ivan, d'Avila Garcez Artur
- All datasets are cloned from https://github.com/lzrobots/DeepEmbeddingModel_ZSL.git
  
- All the material in the repository is the implementation of the paper accepted for publication *PROTOtypical Logic Tensor Networks (PROTO-LTN) for Zero Shot Learning* 
  by Simone Martone, Manigrasso Francesco , Lamberti Fabrizio, Morra Lia.
- Download the repository with, for example, `git clonehttps://github.com/Frankkkko/PROTO-LTN.git`.


## Data

- `APY_data` it contains the dataset from *Describing objects by their attributes* by A. Farhadi, I. Endres, D. Hoiem and D. Forsyth
- `Awa2_data` it contains the dataset from *Zero-Shot Learning -- A Comprehensive Evaluation of the Good, the Bad and the Ugly* by Yongqin Xian, Christoph H. Lampert, Bernt Schiele, Zeynep Akata
- `CUB_data` it contains the dataset from *Caltech-UCSD Birds-200-2011 (CUB-200-2011)* by Catherine Wah1 , Steve Branson1 , Peter Welinder2 , Pietro Perona2 , Serge Belongie

  
  

## Models

- `FasterCnn`: it contains the code to implement Faster CNN and Faster 
    - `imageset.npy`: a numpy array that contains imageset divisions in training and validation from trainval.txt set with a ratio of 80:20 respectively
    - `keras_frcnn`: the files related to the Faster CNN keras implementation and LTN block
    - `input`: the folder contains the output of the prediction of Faster CNN model
  
- `FasterLTN`: it contains the code to implement Faster CNN and Faster 
    - `imageset.npy`: a numpy array that contains imageset divisions in training and validation from trainval.txt set with a ratio of 80:20 respectively
    - `keras_frcnn`: the files related to the Faster CNN keras implementation and LTN block
    - `input`: the folder contains the output of the prediction of Faster CNN model

- `Weights`: the weights of the trained models can be accessed [here](https://www.dropbox.com/sh/5ww40hzcf31voyl/AABeia-QqLiGTi3fHp5Mk0Eha?dl=0) 
## How to train a FasterCnn 

```sh
$ python train_frcnn.py --train_path `(PASCAL VOC/PART FOLDER)`
```

## How to test a FasterCnn 
```sh
$ python test_frcnn.py --train_path `(PASCAL VOC/PART FOLDER)`
```

## How to train a FasterLTN

```sh
$ python train_frcnnltn.py --train_path `(PASCAL VOC/PART FOLDER)` --background (to include background predicate) --alpha ( to include alpha parameter) --half ( to use only half training dataset)
```



## How to test the grounded theories

```sh
$ python test_frcnnltn.py
```
- Results are in `reports/model_evaluation/types_partOf_models_evaluation.csv`
- More detailed results are in `reports/model_evaluation/training_without_constraints` and in `reports/model_evaluation/training_with_constraints`

## How to evaluate the model

```sh
$ python validation.py --train_path `(PASCAL VOC/PART FOLDER)` --background (to include background predicate) 
```
- Results of the detection are in `input` folder
- Result of map are in `results` folder, the map script is 
- Repository cloned from https://github.com/Cartucho/mAP.git

# Citation

If you make use of the dataset in your research, please cite our paper:

Manigrasso Francesco,Filomeno Davide Miro ,Morra Lia, Lamberti Fabrizio, 
["Faster-LTN: a neuro-symbolic, end-to-end object detection architecture"](), 
Proc. The 30th International Conference on Artificial Neural Networks.(ICANN 2021). 





# Contributors & Maintainers
Francesco Manigrasso,Lia Morra,  and Fabrizio Lamberti
