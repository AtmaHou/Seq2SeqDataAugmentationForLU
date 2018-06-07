# coding:utf-8
import argparse
import json
from source.Evaluate.slot_filling import prepare_data_to_dukehan, prepare_data_to_conll_format
from source.Evaluate.slot_filling_data_processor import cook_slot_filling_data
from set_config import refresh_config_file
import copy
import subprocess

# ============ Args Process ==========
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", type=str, default='atis_labeled', help="choose task: atis_labeled, navigate_labeled, schedule_labeled, weather_labeled, navigate, schedule, weather")
parser.add_argument("-gd", "--gen_data", type=str, default='xiaoming', help="generate slot_filling data to Xiaoming's model or DukeHan's: xiaoming, dukehan")
parser.add_argument("-cd", "--cook_data", action='store_true', help="cook data for Xiaoming's model")
parser.add_argument("-trn", "--train", action='store_true', help="train Xiaoming's slot filling model")
parser.add_argument("-rfo", "--refill_only", action='store_true', help="generate data for refill only data")

parser.add_argument("--topx", type=int, default=10, help="select using topx of generated sentences")
parser.add_argument("-s", "--seed", type=int, default=100, help="select random seed")
# parser.add_argument("-bs", "--batch_size", type=int, default=16, help="select batch size")
parser.add_argument("-sr", "--split_rate", type=float, default=0.1, help="select different train set size")
parser.add_argument("-cm", "--cluster_method", type=str, default='_intent', help="choose cluster method: '_intent', '_slot', '_intent_slot'")
parser.add_argument("-et", "--extend", action='store_true', help="choose whether use expand train data")


# Deep Customize
parser.add_argument('--config', default='./config.json', help="specific a config file by path")
args = parser.parse_args()

# ============ Refresh Config ==========
refresh_config_file(args.config)


# ============ Settings ==========
with open(args.config, 'r') as con_f:
    CONFIG = json.load(con_f)


if __name__ == "__main__":
    print('debug:', args.task)
    if args.gen_data == 'xiaoming':
        for split_rate in [4478]:
        # for split_rate in CONFIG['experiment']['train_set_split_rate']:
            prepare_data_to_conll_format(CONFIG, args.task, split_rate, '_leak-gan', use_topx=args.topx, refilled_only=args.refill_only, pair_mod='', no_index='', no_filter_str='')
            # prepare_data_to_conll_format(CONFIG, args.task, split_rate, '_intent-slot', use_topx=args.topx, refilled_only=args.refill_only, pair_mod='', no_index='', no_filter_str='_nf')
            # prepare_data_to_conll_format(CONFIG, args.task, split_rate, '_intent-slot', use_topx=args.topx, refilled_only=args.refill_only, pair_mod='_full_connect', no_index='_ni')
            # prepare_data_to_conll_format(CONFIG, args.task, split_rate, '_intent-slot', use_topx=args.topx, refilled_only=args.refill_only, pair_mod='', no_index='_ni')
            # prepare_data_to_conll_format(CONFIG, args.task, split_rate, '_nc', use_topx=args.topx, refilled_only=args.refill_only, pair_mod='_circle', no_index='_ni')
    else:
        print('Warning: not generating any data')

    # if args.cook_data:
    #     cook_slot_filling_data(config=CONFIG, task_name=args.task, refill_only=args.refill_only)
    #
    # if args.train:
    #
    #     # for split_rate in [0.08, 0.1, 0.2, 0.5, 1]:
    #     #     for cluster_method in CONFIG['experiment']['cluster_method']:
    #         # for cluster_method in ['_intent', '_slot']:
    #         for cluster_method in ['_intent-slot']:
    #             param_replace_table = {
    #                 '<TASK_NAME>': args.task,
    #                 '<EXTEND>': '-et' if args.extend else '',
    #                 '<REFILL_ONLY>': '-rfo' if args.refill_only else '',
    #                 '<SPLIT_RATE>': str(split_rate),
    #                 '<CLUSTER_METHOD>': cluster_method,
    #                 '<SEED>': str(args.seed),
    #             }
    #             print('Debug', param_replace_table)
    #             param_config = dress_param_with_config(CONFIG['slot_filling'], param_replace_table)
    #             run_slot_filling(f"{args.task}{cluster_method}{split_rate}{'-et' if args.extend else ''}{'-rfo' if args.refill_only else ''}", param_config['train_and_test'])
    #             # if not args.extend:
                #     break  # ori training set is irrelevant to cluster method, so only run it once for each rate
    print('Notice! task option affect all: gd, cd ,trn! \nNotice! Extend option will only affect train & test\nrfo effect all')
    # print("!!!!!!!!!!!!!!!!!! run in debug mode !!!!!!!!!!!!!!")
    print('Use rfo in full model!!!!!')
    # print('Not use rfo in full model!!!!!')
    print('Notice: rfo results came from thesaurus method!')
