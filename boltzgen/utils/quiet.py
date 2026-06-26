import logging, warnings


def quiet_startup() -> None:
    warnings.filterwarnings("ignore", message=r".*predict_dataloader.*num_workers.*")
    warnings.filterwarnings(
        "ignore", message=r".*tensorboardX.*removed as a dependency.*"
    )
    warnings.filterwarnings(
        "ignore",
        message=r"The pynvml package is deprecated",
        category=FutureWarning,
        module=r"torch\.cuda",
    )
    
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
        module=r"pytorch_lightning\.utilities\._pytree",
    )
    
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.utils.data")


    warnings.filterwarnings(
        "ignore",
        message=r"Non-SM100f kernel expects bias to be float32",
        category=UserWarning,
        module=r"cuequivariance_ops_torch\.triangle_attention",
    )

    warnings.filterwarnings(
        "ignore",
        message=r"The argument 'device' of Tensor\.(pin_memory|is_pinned)\(\) is deprecated",
        category=DeprecationWarning,
    )
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
