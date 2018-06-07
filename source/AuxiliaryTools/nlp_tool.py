import string
from nltk.tokenize import TweetTokenizer
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
import editdistance


def sentence_edit_distance(s1, s2):
    s1 = s1.split() if type(s1) is str else s1
    s2 = s2.split() if type(s2) is str else s2
    if type(s1) is list and type(s2) is list:
        return editdistance.eval(s1, s2)
    else:
        print("Error: Only str and list is supported, got", type(s1), type(s2))
        raise TypeError


def num_there(s):
    return any(i.isdigit() for i in s)


def low_case_tokenizer(sentence, tree_bank=True):
    if tree_bank:
        return treebank_tokenizer(sentence)
    else:
        tkn = TweetTokenizer(preserve_case=False)
        return tkn.tokenize(sentence)


def treebank_tokenizer(sentence):
    # split 's but also split <>, wait to use in further work
    t = TreebankWordTokenizer()
    word_lst = t.tokenize(sentence.lower().replace("<", "LAB_").replace(">", "_RAB"))
    ret = []
    for w in word_lst:
         ret.append(w.replace("LAB_", "<").replace("_RAB", ">"))
    return ret


def treebank_detokenizer(tokens):
    d = TreebankWordDetokenizer()
    return d.tokenize(tokens)


def convert_to_word_lst(sentence, lower=True):
    sentence = filter(lambda x: x in string.printable, sentence)
    exclude = '''!"#$%&\'()*+,:;<=>?@[\\]^_`{|}~-/\t''' + '\n'
    for e in exclude:
        sentence = sentence.replace(e, ' ')
    if lower:
        sentence = sentence.lower()
    word_sq = sentence.split(' ')
    ret = []
    for ind, w in enumerate(word_sq):
        if '.' in w:
            if num_there(w):
                try:  # detect whether there is a float number
                    float_word = float(w)
                    if w[len(w) - 1] == '.':
                        w.replace('.', '')
                    ret.append(w)
                except ValueError:
                    if w[len(w) - 1] == '.':
                        w.replace('.', '')
                    ret.append(w)
            else:
                ret.extend(w.split('.'))
        else:
            ret.append(w)
    return filter(lambda x: x.strip(), ret)


if __name__ == "__main__":
    print("Testing")
    s = "show<O> me<O> the<O> flights<O> from<O> san<B-fromloc.city_name> diego<I-fromloc.city_name> to<O> newark<B-toloc.city_name>"
    # s = "xxx's a-b_c-d (b-d) <aaa_ddd> yyyyy<sdfsd_bsdb> yyyyy:<sdfsd_bsdb>"
    print(low_case_tokenizer(s))
