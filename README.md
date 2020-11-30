# DGST

This code corresponding to the paper: **DGST: a Dual-Generator Network for Text Style Transfer (EMNLP2020)**.

The main website is here https://xiao.ac/proj/dgst.

This code is based on

```
python 3.7
pytorch (version >= 1.4.0)
torchvision (version >= 0.4.1)
fasttext (version >= 0.8.4)
nltk
tqmd
```

### Training

Please run *Exp_DGST.py* to train the model like:

```console
> python3 Exp_DGST.py --dataset Yelp
```
Or
```console
> python3 Exp_DGST.py --dataset Imdb
```
to train DEST model on Yelp or Imdb dataset.

You can use the parameter *-pg* to show the training progress, e.g.

```console
> python3 Exp_DGST.py --dataset Yelp -pg
```

The trained model will be saved in *./model_save/* , and the outcomes will be in *./outputs/* .

### Ablation Study

There are ablation study types named: 
1. full-model
2. no-rec
3. no-tran
4. rec-no-noise
5. tran-no-noise
6. pre-noise

For details please see the [paper](https://www.aclweb.org/anthology/2020.emnlp-main.578/).

To run the ablation study, just run the file *Exp_ablation_study.py* .
```console
> python3 Exp_ablation_study.py
```
Then the file will let you choose a ablation type.

You can also use the parameter *-pg* if you want to show the training progress, e.g.

```console
> python3 Exp_ablation_study.py -pg
```

## Paper and Citation

This work has been published in EMNLP2020. [Here is the paper](https://www.aclweb.org/anthology/2020.emnlp-main.578/). If you find MSP interesting, please consider citing:

> &nbsp;
> @inproceedings{li-etal-2020-dgst,
    title = "{DGST}: a Dual-Generator Network for Text Style Transfer",
    author = "Li, Xiao  and  Chen, Guanyi  and  Lin, Chenghua  and  Li, Ruizhe",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)", month = nov, year = "2020", address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.emnlp-main.578",
}
> &nbsp;

## Acknowledgement

This work is supported by the award made by the UK Engineering and Physical SciencesResearch Council (Grant number: EP/P011829/1).
