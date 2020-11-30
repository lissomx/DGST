import os
import fasttext
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
import math
import numpy as np
from collections import Counter


def evaluate(source_texts, transfer_texts, other_source_texts, origin_senti):
    source_texts_ = [i.split(' ') for i in source_texts]
    transfer_texts_ = [i.split(' ') for i in transfer_texts]
    other_source_texts_ = [i.split(' ') for i in other_source_texts]

    # corpus bleu
    # https://kite.com/python/docs/nltk.translate.bleu_score.corpus_bleu
    cbleu4 = corpus_bleu( [[i] for i in source_texts_], transfer_texts_) * 100
    n_sents = len(source_texts_)
    all_bleu_scores3 = 0.0
    all_bleu_scores4 = 0.0
    for sou, tran in zip(source_texts_,transfer_texts_):
        all_bleu_scores4 += sentence_bleu([sou], tran)
    bleu4 = all_bleu_scores4 / n_sents * 100.0

    # transfer
    labels = fasttext_classifier.predict(transfer_texts)
    truth = str(1 - origin_senti)
    transfer = float(sum([truth in l for ll in labels[0] for l in ll])) / n_sents * 100.0

    # ground_true_bleu
    n_sents = len(other_source_texts_)
    all_bleu_scores = 0.0
    for i in range(len(other_source_texts_)):
        sou = other_source_texts_[i]
        tran = transfer_texts_[i]
        all_bleu_scores += sentence_bleu([sou], tran)
    gBleu = all_bleu_scores / n_sents * 100.0

    print(f'GroungTrueBleu:{gBleu:4.4f} | SelfBleu4: {bleu4:4.4f} | SelfBleu4(c): {cbleu4:4.4f} | Transfer Acc: {transfer:4.4f}')

    return (gBleu,bleu4,cbleu4,transfer)


fasttext_classifier = None
def init_fasttext(dataset_base, corpus_name, model_save):
    global fasttext_classifier
    # train / load
    fasttext_doc_path = f"{model_save}/fasttext.{corpus_name}.pk"
    if os.path.exists(fasttext_doc_path):
        fasttext_classifier = fasttext.load_model(model_save+f"fasttext.{corpus_name}.pk")
    else:
        fasttext_classifier = fasttext.train_supervised(f'{dataset_base}/{corpus_name}/fasttest.train', label='__label__', epoch=20, lr=1)
        fasttext_classifier.save_model(fasttext_doc_path)
        result = fasttext_classifier.test(f'{dataset_base}/{corpus_name}/fasttest.test')
        print('P@1:', result[1])
        print('R@1:', result[2])
        print('Number of examples:', str(result[0]))