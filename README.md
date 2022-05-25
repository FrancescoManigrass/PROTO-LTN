# PROTOtypical Logic Tensor Networks (PROTO-LTN) for Zero Shot Learning


- The repository contains the code to implement ProtoLTN, the grounded ontology and the script to evaluate the models on Awa2,CUB,aPY and SUN  datasets.
- Logic Tensor Network framework is cloned from the paper Learning and Reasoning in Logic Tensor Networks: Theory and Application for Semantic Image Interpretation by Serafini Luciano, Donadello Ivan, d'Avila Garcez Artur
- All datasets are cloned from "https://github.com/lzrobots/DeepEmbeddingModel_ZSL.git"
  
- All the material in the repository is the implementation of the paper accepted for publication *PROTOtypical Logic Tensor Networks (PROTO-LTN) for Zero Shot Learning* 
  by Simone Martone, Manigrasso Francesco , Lamberti Fabrizio, Morra Lia.
- Download the repository with, for example, `git clonehttps://github.com/Frankkkko/PROTO-LTN.git`.


## Data

- `APY_data` it contains the dataset from *Describing objects by their attributes* by A. Farhadi, I. Endres, D. Hoiem and D. Forsyth
- `Awa2_data` it contains the dataset from *Zero-Shot Learning -- A Comprehensive Evaluation of the Good, the Bad and the Ugly* by Yongqin Xian, Christoph H. Lampert, Bernt Schiele, Zeynep Akata
- `CUB_data` it contains the dataset from *Caltech-UCSD Birds-200-2011 (CUB-200-2011)* by Catherine Wah1 , Steve Branson1 , Peter Welinder2 , Pietro Perona2 , Serge Belongie
- `SUN_data` it contains the dataset from *SUN Database: Large-scale Scene Recognition from Abbey to Zoo* by Jianxiong Xiao; James Hays; Krista A. Ehinger; Aude Oliva; Antonio Torralba

  
  

## logictensornetworks
-  it contains the code to implement Logic tensornetwork Framework cloned from "https://github.com/ivanDonadello/logictensornetworks.git"
   

## How to train/evaluate ProtoLTN 

```sh
$ python main.py --learning_rate 0.01 --alpha 0.001 --regularization_parameter 0.00001 --negation_axioms
```


# Citation

If you make use of the dataset in your research, please cite our paper:

Simone Martone, Manigrasso Francesco,Morra Lia, Lamberti Fabrizio, 
[],






# Contributors & Maintainers
Francesco Manigrasso,Lia Morra,  and Fabrizio Lamberti
