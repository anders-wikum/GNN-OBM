from params import TRAIN_CONFIGS, REGIMES
from gnn_library.util import train, save, gen_train_input
from torch.nn import Module

def train_base_model(
    regime_key: str,
    train_config_key: str,
    name: str,
    args: dict,
    seed: int = 0
) -> Module:
    print("Generating the dataset")
    train_loader, val_loader = gen_train_input(
        REGIMES[regime_key],
        TRAIN_CONFIGS[train_config_key],
        args,
        seed=seed
    )
    print("Starting training")
    _, _, _, GNN, _ = train(train_loader, val_loader, args)
    save(GNN, args, name)
    return GNN

def train_meta_model(
    base_models: list[Module],
    name: str,
    args: dict,
    seed: int = 0
) -> Module:
    
    print("Generating the dataset")
    train_loader, val_loader = gen_train_input(
        REGIMES['META_TRAIN'],
        TRAIN_CONFIGS['META'],
        args,
        base_models=base_models,
        seed=seed
    )
    print("Starting training")
    _, _, _, META_GNN, _ = train(train_loader, val_loader, args)
    save(META_GNN, args, name)

