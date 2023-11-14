import torch
import math
from pathlib2 import Path
from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm, TQDMProgressBar
import sys


class EarlyStopping(object):
    """
    根据待监测指标和次数决定是否提前终止训练，当监测连续超过指定次数没有更优时，则提前终止
    """

    def __init__(self, monitor: str = 'val_loss', mode: str = 'min', patience: int = 1):
        """
        Parameters:
            monitor: 要监测的指标，只有传入指标字典才会生效
            mode: 监测指标的模式，min 或 max
            patience: 最大容忍次数
        """
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.__value = -math.inf if mode == 'max' else math.inf
        self.__times = 0

    def state_dict(self) -> dict:
        """
        保存状态，以便下次加载恢复
        torch.save(state_dict, path)
        """
        return {
            'monitor': self.monitor,
            'mode': self.mode,
            'patience': self.patience,
            'value': self.__value,
            'times': self.__times
        }

    def load_state_dict(self, state_dict: dict):
        """
        加载状态

        Parameters:
            state_dict: 保存的状态
        """
        self.monitor = state_dict['monitor']
        self.mode = state_dict['mode']
        self.patience = state_dict['patience']
        self.__value = state_dict['value']
        self.__times = state_dict['times']

    def reset(self):
        """
        重置次数
        """
        self.__times = 0

    def step(self, metrics) -> bool:
        """
        Parameters:
            metrics: 指标字典或数值标量

        Returns:
            返回bool标量，True表示触发终止条件
        """
        if isinstance(metrics, dict):
            metrics = metrics[self.monitor]

        if (self.mode == 'min' and metrics <= self.__value) or (
                self.mode == 'max' and metrics >= self.__value):
            self.__value = metrics
            self.__times = 0
        else:
            self.__times += 1
        if self.__times >= self.patience:
            return True
        return False


class ModelCheckpoint(object):
    """
    训练时自动保存需要保存的数据
    """

    def __init__(self, filepath: str = 'checkpoint.pth', monitor: str = 'val_loss',
                 mode: str = 'min', save_best_only: bool = False, save_freq: int = 1):
        """
        Parameters:
            filepath: 文件名或文件夹名，需要保存的位置，如果为文件夹，则保存的检查点数量可能不止一个
            monitor: 监测指标
            mode: 监测模式，min 或 max
            save_best_only: 是否只保存指标最好的检查点， True 或 False
            save_freq: 保存的频率，只有 save_best_only=False 时有效
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_freq = save_freq
        self.__times = 1
        self.__value = -math.inf if mode == 'max' else math.inf

    @staticmethod
    def save(filepath: str, times: int = None, **kwargs):
        """
        保存检查点

        Parameters:
            filepath: 文件名或文件夹名，需要保存的位置，如果为文件夹，则保存的检查点数量可能不止一个
            times: 当前保存的次数，只用于保存路径为文件夹的情况，只用于命名文件
            kwargs: 所有需要保存的内容
        """
        path = Path(filepath)
        if path.is_dir():
            if not path.exists():
                path.mkdir(parents=True)
            path.joinpath(f'checkpoint-{times}.pth')

        # 保存所有的内容
        torch.save(kwargs, str(path))

    def state_dict(self):
        """
        保存状态，以便下次加载恢复
        torch.save(state_dict, path)
        """
        return {
            'filepath': self.filepath,
            'monitor': self.monitor,
            'save_best_only': self.save_best_only,
            'mode': self.mode,
            'save_freq': self.save_freq,
            'times': self.__times,
            'value': self.__value
        }

    def load_state_dict(self, state_dict: dict):
        """
        加载状态

        Parameters:
            state_dict: 保存的状态
        """
        self.filepath = state_dict['filepath']
        self.monitor = state_dict['monitor']
        self.save_best_only = state_dict['save_best_only']
        self.mode = state_dict['mode']
        self.save_freq = state_dict['save_freq']
        self.__times = state_dict['times']
        self.__value = state_dict['value']

    def reset(self):
        """
        重置次数
        """
        self.__times = 1

    def step(self, metrics, **kwargs):
        """
        Parameters:
            metrics: 监测指标，字典或数值标量
            kwargs: 要保存的指标
        """
        if isinstance(metrics, dict):
            metrics = metrics[self.monitor]

        flag = False

        if self.save_best_only:
            if (self.mode == 'min' and metrics <= self.__value) or (
                    self.mode == 'max' and metrics >= self.__value):
                self.__value = metrics
                self.save(self.filepath, self.__times, **kwargs)
                flag = True
        else:
            if self.__times % self.save_freq == 0:
                self.save(self.filepath, self.__times, **kwargs)
                flag = True

        self.__times += 1
        return flag


class LightningTQDMProgressBar(TQDMProgressBar):
    """
    重写TQDMProgressBar，修复其在验证时进度条不断换行显示的问题
    """

    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate, process_position)

    def init_validation_tqdm(self) -> Tqdm:
        bar = Tqdm(
            desc=self.validation_description,
            position=0,  # 避免验证时进度条不断换行显示
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar
