# coding:utf-8
from source.AuxiliaryTools.ConfigTool import update_config


def refresh_config_file(config_path='./config.json'):
    print('Config Position:', config_path)
    # For my linux server setting
    update_config("/users4/ythou/Projects/TaskOrientedDialogue/data/", config_path=config_path)
    # For my windows setting
    # update_config("E:/Projects/Research/TaskOrientedDialogue/data/", config_path='./config.json')

if __name__ == "__main__":
    refresh_config_file()
