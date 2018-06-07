# coding:utf-8
import json
import argparse
import random
import re
from nlp_tool import sentence_edit_distance  # this is worked out of pycharm
# from source.AuxiliaryTools.nlp_tool import sentence_edit_distance


# ==== load config =====
with open('../../config.json', 'r') as con_f:
    CONFIG = json.load(con_f)


def show_cluster_result(result_file):
    with open(CONFIG['path']['ClusteringResult'] + result_file, 'r') as reader:
        json_data = json.load(reader)
    sampled_keys = random.sample(json_data.keys(), min(20, len(json_data.keys())))
    while True:
        print('There are %d clusters in total' % len(json_data.keys()))
        for ind, key in enumerate(sampled_keys):
            print("id:%d\t#%d   %s" % (ind, len(json_data[key]), key))
        kid = input("select a 'common slot' by id: \ninput -1 to quit \ninput -2 to re-sample\ninput:")
        try:
            kid = int(kid)
            if kid == -1:
                break
            elif kid == -2:
                sampled_keys = random.sample(json_data.keys(), min(20, len(json_data.keys())))
                for ind, key in enumerate(sampled_keys):
                    print("id:%d\t#%d   %s" % (ind, len(json_data[key]), key))
                continue
            else:
                c_slot = sampled_keys[kid]
                print('==== %s ====' % c_slot)
                for item in json_data[c_slot]:
                    print(item['user_say'], '||', item['user_temp'])
                print('==== %s ====' % c_slot)
        except TypeError:
            print("Error: select is not integer", int(kid))


def find_closest_sent(tgt_s, sent_set):
    closest_s = ''
    min_distance = len(tgt_s.split())
    for s in sent_set:
        tmp_d = sentence_edit_distance(tgt_s, s)
        if tmp_d < min_distance:
            closest_s = s
            min_distance = tmp_d
    return closest_s, min_distance


def appearance_check(test_what_file, in_what_file, top_x=1):
    total = 0
    not_appeared = 0
    unique_set = {}
    unique = 0
    unique_new = 0
    with open(test_what_file, 'r') as test_file, open(in_what_file, 'r') as source_file, \
            open('./data_tool.log', 'w') as log_f:
        target_lines = []
        for line in source_file.readlines():
            target_lines.append(re.sub('\s<\d*>', '', line))
        target_lines = set(target_lines)
        test_lines = test_file.readlines()
        line_count = 0
        distance_sum = 0
        length_sum = 0
        for line in test_lines:
            if line_count % 10 < top_x:
                if line not in unique_set:
                    unique += 1
                    unique_set[line] = True
                    if line not in target_lines:
                        not_appeared += 1
                        if unique_new not in unique_set:
                            unique_new += 1
                            tmp_closest, tmp_d = find_closest_sent(line, target_lines)
                            log_f.write("=== %d %d === \n\tgen: %s\n\tori: %s\n" % (
                                tmp_d, len(line.split()), line.replace('\n', ''), tmp_closest.replace('\n', '')
                            ))
                            distance_sum += tmp_d
                            length_sum += len(line.split())
                total += 1
            line_count += 1
    ave_d = 0 if unique_new == 0 else distance_sum / unique_new
    ave_l = 0 if unique_new == 0 else length_sum / unique_new
    log_str = "%s %f %s %f %s %f %s %f \n%s %f %s %f " % (
        "Not Appeared", not_appeared,
        "Total", total,
        "Unique", unique,
        "Unique New", unique_new,
        "New: Average Distance", ave_d,
        'New: Average Length', ave_l,
    )
    print(log_str)


def remove_appeared(test_what_file, in_what_file, source_file_path, result_file):
    total = 0
    appeared = 0
    pure = []
    with open(test_what_file, 'r') as test_file, \
            open(in_what_file, 'r') as target_file, \
            open(source_file_path, 'r') as source_file:
        target_lines = target_file.readlines()
        test_lines = test_file.readlines()
        source_lines = source_file.readlines()
        line_count = 0
        for line in test_lines:
            if line_count % 10 < 10:
                if line not in target_lines:
                    pure.append(str(line_count % 10) + " " + line)
                    appeared += 1
                total += 1
            if line_count % 10 == 0:
                # to show source sentence in result.
                pure.append('\n==== %s\n' % source_lines[int(line_count / 10)])
            line_count += 1
    print("Appeared", appeared, "Total", total)
    with open(result_file, 'w') as writer:
        for line in pure:
            writer.write(line)


if __name__ == '__main__':
    tn = 'navigate_labeled'
    # cm = 'intent-slot'
    cm = 'slot'
    parser = argparse.ArgumentParser()
    parser.add_argument("--c_res", help="show clustering result of arg result file", type=str)
    parser.add_argument("--appr_check", help="check utterance appearance within one file from another file", action="store_true")
    parser.add_argument("--appr_remove", help="remove same utterance within one file from another file", action="store_true")
    parser.add_argument("--appr_check_in", help="specific the file to check in", type=str,
                        default=CONFIG['path']["OnmtData"] + f"SourceData/train_{tn}_{cm}1_src.txt")
    # parser.add_argument("--appr_check_in", help="specific the file to check in", type=str,
    #                     default="/users4/ythou/Projects/TaskOrientedDialogue/data/Backup/SourceData_circle_wise/" + "train_navigate_src.txt")
    parser.add_argument("--appr_check_target", help="specific the file containing utterance to check", type=str,
                        default=CONFIG['path']["OnmtData"] + f"Result/{tn}_{cm}1_pred.txt")
    parser.add_argument("--appr_remove_res", help="specific the result file for appearance remove", type=str,
                        default=CONFIG['path']["OnmtData"] + f"Result/{tn}_{cm}1_pred_remove-appr.txt")
    parser.add_argument("--appr_source_file", help="specific the source file for appearance remove", type=str,
                        default=CONFIG['path']["OnmtData"] + "SourceData/test_navigate_src.txt")
    parser.add_argument("--top_x", help="specific top x in checking appr", type=int, default=10)
    args = parser.parse_args()
    if args.c_res:
        show_cluster_result(args.c_res)

    if args.appr_check and args.appr_check_target and args.appr_check_in:
        appearance_check(args.appr_check_target, args.appr_check_in, top_x=args.top_x)

    if args.appr_remove:
        remove_appeared(args.appr_check_target, args.appr_check_in, args.appr_source_file, args.appr_remove_res)
