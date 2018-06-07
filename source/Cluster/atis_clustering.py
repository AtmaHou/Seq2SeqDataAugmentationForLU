# coding:utf-8
from source.Cluster.clustering import Cluster

# wait to construct


class AtisCluster(Cluster):
    def __init__(self, input_dir, output_dir):
        Cluster.__init__(self, input_dir, output_dir)

    def unpack_and_cook_raw_data(self, raw_data):
        pass

if __name__ == "__main__":
    print("Hi, there!!")
