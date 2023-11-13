from omegaconf import OmegaConf


# 从yaml文件中读取配置，返回OmegaConf对象
def load_yaml_with_omegaconf(path: str):
    return OmegaConf.load(path)
