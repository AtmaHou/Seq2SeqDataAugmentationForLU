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
parser.add_argument("-t", "--task", type=str, default='weather_labeled', help="choose task: atis_labeled, navigate_labeled, schedule_labeled, weather_labeled, navigate, schedule, weather")
parser.add_argument("-gd", "--gen_data", type=str, default='none', help="generate slot_filling data to Xiaoming's model or DukeHan's: xiaoming, dukehan")
parser.add_argument("-cd", "--cook_data", action='store_true', help="cook data for Xiaoming's model")
parser.add_argument("-trn", "--train", action='store_true', help="train Xiaoming's slot filling model")
parser.add_argument("-rfo", "--refill_only", action='store_true', help="generate data for refill only data")

parser.add_argument("--topx", type=int, default=5, help="select using topx of generated sentences")
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


def dress_param_with_config(param_config, replace_table):
    # replace "data_mark, data_dir, result_dir" slots in param
    ret = copy.deepcopy(param_config)
    for key in ret:
        param_str = " ".join(ret[key])
        for slot_name in replace_table:
            param_str = param_str.replace(slot_name, replace_table[slot_name])
        param_lst = param_str.split()
        ret[key] = param_lst
    return ret


def run_slot_filling(task_name, param):
    print('========================== Call BI-LSTM for: %s ==========================' % task_name)
    print('==========================       Param       ==========================\n%s' % ' '.join(param))
    print('==========================  BI-LSTM Output  ========================== \n')
    proc = subprocess.Popen(param, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    with open('log/' + 'slot-filling' + task_name + 'log', 'w') as writer:
        for line in proc.stdout.readlines():
            print(line)
            writer.write(line.rstrip().decode("utf-8") + '\n')
            if b'error' in line.lower() and b'check_error' not in line.lower():
                raise RuntimeError


if __name__ == "__main__":
    print('debug:', args.task)
    if args.gen_data == 'dukehan':
        for split_rate in CONFIG['experiment']['train_set_split_rate']:
            prepare_data_to_dukehan(CONFIG, args.task, split_rate, use_topx=10)
    elif args.gen_data == 'xiaoming':
        for split_rate in CONFIG['experiment']['train_set_split_rate']:
        #     for cluster_method in CONFIG['experiment']['cluster_method']:
        # for split_rate in [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.1, 0.2, 0.5, 1]:
        #     for cluster_method in ['_intent', '_slot']:
            for cluster_method in ['_intent-slot']:
                prepare_data_to_conll_format(CONFIG, args.task, split_rate, cluster_method, use_topx=args.topx, refilled_only=args.refill_only)
    else:
        print("Wrong args!")

    if args.cook_data:
        print("")

    if args.train:
        print("")
