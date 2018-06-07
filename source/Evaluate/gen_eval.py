import json
import argparse
import random
import re
# from nlp_tool import sentence_edit_distance  # this is worked out of pycharm
from source.AuxiliaryTools.nlp_tool import sentence_edit_distance


# # ==== load config =====
# with open('../../config.json', 'r') as con_f:
#     CONFIG = json.load(con_f)
#
#
# parser = argparse.ArgumentParser()
# tn = 'navigate_labeled'
# # cm = 'intent-slot'
# cm = 'slot'
# parser.add_argument("--appr_check", help="check gen res's appearance within one file from another file", action="store_true")
# parser.add_argument("--appr_check_in", help="specific the file to check in", type=str,
#                     default=CONFIG['path']["OnmtData"] + f"SourceData/train_{tn}_{cm}1_src.txt")
# parser.add_argument("--appr_check_target", help="specific the file containing utterance to check", type=str,
#                     default=CONFIG['path']["OnmtData"] + f"Result/{tn}_{cm}1_pred.txt")
#
# parser.add_argument("--top_x", help="specific top x in checking appr", type=int, default=10)
# args = parser.parse_args()


def find_closest_sent(tgt_s, sent_set):
    closest_s = ''
    min_distance = len(tgt_s.split())
    for s in sent_set:
        tmp_d = sentence_edit_distance(tgt_s, s)
        if tmp_d < min_distance:
            closest_s = s
            min_distance = tmp_d
    return closest_s, min_distance


def get_1_2_gram_set(line):
    word_lst = line.split()
    bi_gram_set = set()
    uni_gram_set = set(word_lst)
    for ind, word in enumerate(word_lst):
        if ind + 1 < len(word_lst):
            temp_bi_gram = word_lst[ind] + ' ' + word_lst[ind + 1]
            bi_gram_set.add(temp_bi_gram)
    return word_lst, uni_gram_set, bi_gram_set


def get_distinct_score(target_set):
    all_uni_gram_set = set()
    all_bi_gram_set = set()
    total_word_num = 0
    for line in target_set:
        word_lst, uni_gram_set, bi_gram_set = get_1_2_gram_set(line)
        all_uni_gram_set = all_uni_gram_set | uni_gram_set
        all_bi_gram_set = all_bi_gram_set | bi_gram_set
        total_word_num += len(word_lst)

    distinct_1 = 100.0 * len(all_uni_gram_set) / total_word_num  # convert to percent
    distinct_2 = 100.0 * len(all_bi_gram_set) / total_word_num  # convert to percent
    return distinct_1, distinct_2, len(all_uni_gram_set), len(all_bi_gram_set), total_word_num


def appearance_check(result_file, test_what_file, in_what_file, top_x=1):
    total = 0
    not_appeared = 0
    unique_set = {}
    unique = 0
    unique_new = 0

    try:
        with open(test_what_file, 'r') as test_file, \
                open(in_what_file, 'r') as source_file, \
                open(result_file, 'w') as log_f:
            source_lines = []
            for line in source_file.readlines():
                source_lines.append(re.sub('\s<\d*>', '', line))
            source_lines = set(source_lines)
            test_lines = test_file.readlines()
            line_count = 0
            distance_sum = 0
            length_sum = 0

            # ========== min edit distance evaluation ==========
            for line in test_lines:
                if line_count % 10 < top_x:
                    if line not in unique_set:
                        if line not in source_lines:
                            not_appeared += 1
                            if line not in unique_set:
                                unique_new += 1
                                tmp_closest, tmp_d = find_closest_sent(line, source_lines)
                                log_f.write("=== %d %d === \n\tgen: %s\n\tori: %s\n" % (
                                    tmp_d, len(line.split()), line.replace('\n', ''), tmp_closest.replace('\n', '')
                                ))
                                distance_sum += tmp_d
                                length_sum += len(line.split())
                        unique += 1
                        unique_set[line] = True
                    total += 1
                line_count += 1

            # ========= distinct evaluation ===========
            test_new_lines = set(unique_set.keys())
            # augmented_data = test_new_lines
            augmented_data = source_lines | test_new_lines
            augmented_distinct_1, augmented_distinct_2, augmented_unigram, augmented_bigram, augmented_total_word = get_distinct_score(augmented_data)
            source_distinct_1, source_distinct_2, source_unigram, source_bigram, source_total_word = get_distinct_score(source_lines)

            # ======== edit distance evaluation on source ========
            source_distance_sum = 0
            for line in source_lines:
                temp_source_lines = source_lines.copy()
                temp_source_lines.remove(line)
                tmp_closest, tmp_d = find_closest_sent(line, temp_source_lines)
                source_distance_sum += tmp_d
            ave_source_d = source_distance_sum / len(source_lines)

            # ======== edit edit distance evaluation on augmented ==========
            augmented_distance_sum = 0
            for line in augmented_data:
                temp_augmented_data = augmented_data.copy()
                temp_augmented_data.remove(line)
                tmp_closest, tmp_d = find_closest_sent(line, temp_augmented_data)
                augmented_distance_sum += tmp_d
            ave_augmented_d = augmented_distance_sum / len(augmented_data)

        ave_d = 0 if unique_new == 0 else distance_sum / unique_new
        ave_l = 0 if unique_new == 0 else length_sum / unique_new
        eval_results = {
            "Not Appeared": not_appeared,
            "Total": total,
            "Unique": unique,
            "Unique New": unique_new,
            "Avg. Distance for new": ave_d,
            "Avg. Distance for augmented": ave_augmented_d,
            "Avg. Distance for source": ave_source_d,
            'Avg. Length': ave_l,

            'source_distinct_1': source_distinct_1,
            'source_distinct_2': source_distinct_2,
            'source_unigram': source_unigram,
            'source_bigram': source_bigram,
            'source_total_word': source_total_word,

            'augmented_distinct_1': augmented_distinct_1,
            'augmented_distinct_2': augmented_distinct_2,
            'augmented_unigram': augmented_unigram,
            'augmented_bigram': augmented_bigram,
            'augmented_total_word': augmented_total_word,

            'source_size': len(source_lines),
            'generated_new_size': len(test_new_lines),
            'augmented_size': len(augmented_data),
        }
        return eval_results
    except FileNotFoundError as e:
        return {'no_file': e}
        # return {'no_file': (result_file, test_what_file, in_what_file)}


# if __name__ == "__main__":
#     if args.appr_check and args.appr_check_target and args.appr_check_in:
#         appearance_check(result_file_path, args.appr_check_target, args.appr_check_in, top_x=args.top_x)

