from utils.engine import ClassificationLightningModel
from argparse import ArgumentParser
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
import lightning as L


# 创建回调函数
# TODO: 创建更多的回调函数，但会显著增加预设配置文件的参数量，因此后续用到什么回调函数再创建什么回调函数
def create_callbacks(params):
    callbacks = []
    if params.ModelCheckpoint.used:
        if "params" in params.ModelCheckpoint:
            model_checkpoint = ModelCheckpoint(**params.ModelCheckpoint.params)
            callbacks.append(model_checkpoint)
        else:
            raise KeyError("ModelCheckpoint params not found")

    if params.EarlyStopping.used:
        if "params" in params.EarlyStopping:
            early_stopping = EarlyStopping(**params.EarlyStopping.params)
            callbacks.append(early_stopping)
        else:
            raise KeyError("EarlyStopping params not found")

    if params.TQDMProgressBar.used:
        if "params" in params.TQDMProgressBar:
            tqdm_progress_bar = TQDMProgressBar(**params.TQDMProgressBar.params)
        else:
            tqdm_progress_bar = TQDMProgressBar()
        callbacks.append(tqdm_progress_bar)

    if len(callbacks) == 0:
        callbacks = None
    return callbacks


# 创建日志记录器
# TODO: 根据日志名称创建指定的日志记录器，而不是只有 WandbLogger
def create_logger(params):
    logger = None
    if params.used:
        logger = WandbLogger(**params.params)
    return logger


# 创建训练器
# TODO: 拆分 Trainer 的 strategy 参数，因为 strategy 可以通过类传递
def create_trainer(omega_conf):
    # 创建回调函数
    callbacks = create_callbacks(omega_conf.fit.callbacks)
    # 创建日志记录器
    logger = create_logger(omega_conf.fit.logger)
    logger.log_hyperparams(omega_conf)  # 保存超参数
    # 创建训练器
    trainer = L.Trainer(
        callbacks=callbacks,
        logger=logger,
        **omega_conf.fit.Trainer
    )
    return trainer


# 创建参数解析器，用于命令行参数解析
def parser_args(return_args=False):
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/default.yaml",
                        help="the config used to build a lightning Trainer")
    args = parser.parse_args()
    if return_args:
        return args
    return parser
