# coding:utf-8
import math
import editdistance


def sentence_edit_distance(s1, s2):
    s1 = s1.split() if type(s1) is str else s1
    s2 = s2.split() if type(s2) is str else s2
    if type(s1) is list and type(s2) is list:
        return editdistance.eval(s1, s2)
    else:
        print("Error: Only str and list is supported, got", type(s1), type(s2))
        raise TypeError


def diverse_score(s, t):
    """
    calculate pairing score
    :param s: target str
    :param t: candidate str
    :return: score, edit distance, length penalty
    """
    lst_s = s.split()
    lst_t = t.split()
    length_penalty = math.exp(-abs((len(lst_s) - len(lst_t))/len(lst_s)))
    # length_penalty = math.exp(-abs((len(lst_s) - len(lst_t))/max(len(lst_s), len(lst_t))))
    e_d = sentence_edit_distance(lst_t, lst_s)
    # print(e_d * length_penalty, e_d, length_penalty, '\n', s, '\n', t)
    return e_d * length_penalty


if __name__ == '__main__':
    s = 'tell me the flights from <fromloc.city_name> to <toloc.city_name>'
    s1 = 'please show flights arriving in <toloc.city_name> from <fromloc.city_name>'
    s2 = "okay i would like to fly from <fromloc.city_name> to <toloc.city_name>"

    s3 = 'show me all flights from <fromloc.city_name> to <toloc.city_name> with prices'
    s4 = "what are all the flights between <fromloc.city_name> and <toloc.city_name>"
    print(s, diverse_score(s, s))
    print(s1, diverse_score(s, s1), sentence_edit_distance(s, s1))
    print(s2, diverse_score(s, s2), sentence_edit_distance(s, s2))
    print(s3, diverse_score(s, s3), sentence_edit_distance(s, s3))
    print(s4, diverse_score(s, s4), sentence_edit_distance(s, s4))
