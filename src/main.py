import torch
from loguru import logger
import argparse
from utils.utils import init_logger, show_config, set_seed
from data_utils import get_data_loader
from train.mrc_trainer import MrcTrainer
from model import MrcBertModel
from accelerate import Accelerator


# from eval import eval_func


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="The arguments of MRC for NER")

    # parameters of all path
    parser.add_argument("--train", default="", type=str)
    parser.add_argument("--dev", default="", type=str)
    parser.add_argument("--test", default="", type=str)
    parser.add_argument("--log_path", default="../log", type=str)
    parser.add_argument("--save_model_dir", default="", type=str)
    parser.add_argument("--bert_dir", default="/home/zhliu/plm/chinese_roberta_wwm_ext_large")

    # parameters of training
    parser.add_argument("--num_epoch", default=50, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--optimizer", default="AdamW", type=str)
    parser.add_argument("--grad_accum_steps", default=1, type=int)
    parser.add_argument("--logging_steps", default=20, type=int)
    parser.add_argument("--warmup", default=0.1, type=float)
    parser.add_argument("--clip_value", default=5, type=float)
    parser.add_argument("--lr", default=1e-5, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--gpu", default=0, type=int, help="set -1 if don't use gpu")
    parser.add_argument("--max_len", default=128, type=int)
    parser.add_argument("--batch_first", default=True, type=bool)

    # parameters of model
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--hidden_size", default=1024, type=int)
    parser.add_argument("--flat", action="store_true", help="is flat ner")
    parser.add_argument(
        "--span_loss_candidates",
        choices=["all", "pred_and_gold", "gold"],
        default="all",
        help="Candidates used to compute span loss",
    )

    # parameters of other

    return parser


if __name__ == "__main__":

    # 获取配置参数
    parser = get_parser()
    config = parser.parse_args()
    config_dict = config.__dict__
    show_config(config_dict)

    # 初始化配置
    init_logger(config_dict["log_path"])
    logger.info(config_dict)
    set_seed(config_dict["seed"])
    accelerator = Accelerator()
    device = accelerator.device

    device = torch.device("cuda") if torch.cuda.is_available() and config_dict["gpu"] >= 0 else torch.device("cpu")

    # 加载数据
    train_loader = get_data_loader(config_dict, "train")
    dev_loader = get_data_loader(config_dict, "dev")
    test_loader = get_data_loader(config_dict, "test")

    # 初始化模型
    model = MrcBertModel(config_dict)
    model.to(device)

    # 训练
    trainer = MrcTrainer(model, config_dict, accelerator)
    trainer.train(train_loader, dev_loader, test_loader)
    print("=" * 20 + "Training End!" + "=" * 20)
