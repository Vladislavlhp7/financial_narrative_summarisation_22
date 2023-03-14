from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
from preprocessing import preprocess
import os
from tqdm import tqdm
import torch


def format_french_model(lang, sents):
    txt_formatted = [">>{}<< {}".format(lang, s) for s in sents]
    return txt_formatted


def perform_translation_single_pass_fr(sents, model, tokenizer, device: str = 'cpu', lang="fr"):
    txt_formatted = format_french_model(lang, sents)
    txt_translated_encoded = model.generate(**tokenizer(txt_formatted, return_tensors="pt", padding=True).to(device))
    txt_translated_decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in txt_translated_encoded]
    return txt_translated_decoded


def perform_backtranslation_fr(sents, lang_original, lang_tmp, file_path, device: str = 'cpu', batch_size: int = 64,
                               save: bool = True):
    model1_name = 'Helsinki-NLP/opus-mt-en-fr'
    model1 = MarianMTModel.from_pretrained(model1_name).to(device)
    model1_tkn = MarianTokenizer.from_pretrained(model1_name)

    model2_name = 'Helsinki-NLP/opus-mt-fr-en'
    model2 = MarianMTModel.from_pretrained(model2_name).to(device)
    model2_tkn = MarianTokenizer.from_pretrained(model2_name)

    new_sents = []
    if save and os.path.exists(file_path):
        os.remove(file_path)
    for i in tqdm(range(0, len(sents), batch_size), desc=f'Back-translating {lang_original}-{lang_tmp}'):
        batch_sents = sents[i:i + batch_size]
        batch_pass1 = perform_translation_single_pass_fr(batch_sents, model1, model1_tkn, device=device, lang=lang_tmp)
        batch_pass2 = perform_translation_single_pass_fr(batch_pass1, model2, model2_tkn, device=device,
                                                         lang=lang_original)
        new_sents.extend(batch_pass2)
    # ensure text quality is consistent
    new_sents = [preprocess(s)[0] for s in new_sents]
    if save:
        with open(file_path, 'a') as f:
            for s in new_sents:
                f.write(s + '\n')
    return new_sents


def backtranslate(sents, lang_original, lang_tmp, device: str = 'cpu', save: bool = True):
    new_sents = []
    file_path = f'../tmp/back_translated_summary_{lang_original}_{lang_tmp}.txt'
    if lang_tmp == 'fr':
        new_sents = perform_backtranslation_fr(sents, lang_original, lang_tmp, save=save, device=device,
                                               file_path=file_path)
    return new_sents


def main():
    lang_original = 'en'
    lang_tmp = 'fr'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    df = pd.read_csv('../tmp/training_corpus_2023-02-07 16-33.csv')
    sents = df.loc[df.label == 1].sent.tolist()
    new_sents = backtranslate(sents, lang_original, lang_tmp, device)
    return new_sents


main()
