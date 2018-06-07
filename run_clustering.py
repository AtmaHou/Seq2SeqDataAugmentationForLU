# coding:utf-8
from source.Cluster import clustering
from source.Cluster import conll_format_clustering
# from source.Cluster.clustering import slot_clustering_and_dump_dict
import argparse
import json
from set_config import refresh_config_file


# ============ Args Process ==========
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", type=str, default='stanford_labeled', help="choose target dataset: stanford, stanford_labeled, atis")
parser.add_argument("-cm", "--cluster_mode", type=str, default='all', help="select cluster mode: slot, intent, slot-intent, all, no_clustering")
parser.add_argument('--config', default='./config.json', help="specific a config file by path")
args = parser.parse_args()

# ============ Refresh Config ==========
refresh_config_file(args.config)

# ============ Settings ==========
with open(args.config, 'r') as con_f:
    CONFIG = json.load(con_f)


def run_clustering():
    if args.data == "stanford":
        clustering.slot_clustering_and_dump_dict(config=CONFIG, train_set_split_rate_lst=CONFIG['experiment']['train_set_split_rate'])
    elif args.data == "stanford_labeled":
        conll_format_clustering.clustering_and_dump_dict(
            data_dir=CONFIG['path']['RawData']['stanford_labeled'],
            config=CONFIG,
            cluster_mode=args.cluster_mode,
            train_set_split_rate_lst=CONFIG['experiment']['train_set_split_rate'])
    elif args.data == 'atis':
        conll_format_clustering.clustering_and_dump_dict(
            data_dir=CONFIG['path']['RawData']['atis'],
            config=CONFIG,
            cluster_mode=args.cluster_mode,
            train_set_split_rate_lst=CONFIG['experiment']['train_set_split_rate'])
    else:
        print("Error: Wrong dataset args.")


if __name__ == "__main__":
    run_clustering()
