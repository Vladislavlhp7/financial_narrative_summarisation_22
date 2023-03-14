from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
from preprocessing import preprocess


def format_french_model(lang, sents):
    txt_formatted = [">>{}<< {}".format(lang, s) for s in sents]
    return txt_formatted


def perform_translation_single_pass_fr(sents, model, tokenizer, lang="fr"):
    txt_formatted = format_french_model(lang, sents)
    txt_translated_encoded = model.generate(**tokenizer(txt_formatted, return_tensors="pt", padding=True))
    txt_translated_decoded = [tokenizer.decode(t, skip_special_tokens=True) for t in txt_translated_encoded]
    return txt_translated_decoded


def perform_backtranslation_fr(sents, lang_original, lang_tmp):
    model1_name = 'Helsinki-NLP/opus-mt-en-fr'
    model1_tkn = MarianTokenizer.from_pretrained(model1_name)
    model1 = MarianMTModel.from_pretrained(model1_name)

    model2_name = 'Helsinki-NLP/opus-mt-fr-en'
    model2_tkn = MarianTokenizer.from_pretrained(model2_name)
    model2 = MarianMTModel.from_pretrained(model2_name)

    pass1 = perform_translation_single_pass_fr(sents, model1, model1_tkn, lang=lang_tmp)
    pass2 = perform_translation_single_pass_fr(pass1, model2, model2_tkn, lang=lang_original)
    return pass2


def backtranslate(sents, lang_original, lang_tmp, save: bool = True):
    new_sents = []
    if lang_tmp == 'fr':
        new_sents = perform_backtranslation_fr(sents, lang_original, lang_tmp)
    # ensure text quality is consistent
    new_sents = [preprocess(s)[0] for s in new_sents]
    if save:
        with open(f'../tmp/back_translated_summary_{lang_original}_{lang_tmp}.txt', 'w') as f:
            for s in new_sents:
                f.write(s)
    return new_sents


def main():
    lang_original = 'en'
    lang_tmp = 'fr'
    df = pd.read_csv('../tmp/training_corpus_2023-02-07 16-33.csv')
    sents = df.loc[df.label == 1].sent.tolist()
    new_sents = backtranslate(sents, lang_original, lang_tmp)
    return new_sents


main()
