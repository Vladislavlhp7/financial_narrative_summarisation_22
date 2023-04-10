from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

from metrics import calc_rouge
from query import get_report, get_summary
from statistics import calc_rouge_agg_from_gold_summaries

baseline_summarizers = {
    'lex': LexRankSummarizer,
    'lsa': LsaSummarizer,
    'luhn': LuhnSummarizer,
    'textrank': TextRankSummarizer
}


def get_baseline_summary(text: str, max_sents: int = 1000, method: str = 'lex'):
    my_parser = PlaintextParser.from_string(text, Tokenizer('english'))
    baseline_summarizer = baseline_summarizers[method]()
    baseline_summary = baseline_summarizer(my_parser.document, sentences_count=max_sents)

    # Assembling the summary
    baseline_summary_str = ''
    for sentence in baseline_summary:
        baseline_summary_str += str(sentence)
    return baseline_summary_str


def main():
    file_id = '17'
    # training = True
    report = get_report(file_id)[:100]
    summary = get_baseline_summary(report)

    # Calculate Rouge with a random gold summary
    summary_gold = get_summary(file_id)
    calc_rouge(txt1=summary_gold, txt2=summary, stats=['f'], verbose=True)

    # Calculate aggregate Rouge on all gold summaries
    rouge_df = calc_rouge_agg_from_gold_summaries(summary=summary, file_id=file_id)
    print(rouge_df)

# main()
