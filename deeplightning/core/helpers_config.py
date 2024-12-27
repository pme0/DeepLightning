from omegaconf import ListConfig, DictConfig


TypesLabels = {
    int: "int",
    float: "float",
    DictConfig: "omegaconf.dictconfig.DictConfig",
    ListConfig: "omegaconf.dictconfig.ListConfig",
}


def _print(s, indent=2):
    print(" "*indent + s)


def _assert_type(typpe, subcfg, transf: str, param: str = None):
    target = subcfg[param] if param else subcfg
    assert isinstance(target, typpe), (
        f"Parameter '{transf}.{param}' must be type {TypesLabels[typpe]}, "
        f"found {type(target)}."
    ) 


def _assert_len(length: int, subcfg: DictConfig, transf: str, param: str = None):
    if length:
        target = subcfg[param] if param else subcfg
        p = param if param else ""
        assert len(target) == length, (
            f"Parameter '{transf}.{p}' must be length {length}, "
            f"found {len(target)}."
        )


def _assert_equal_len(subcfg: DictConfig, transf: str, params: str):
    lengths = [len(subcfg[x]) for x in params]
    p = "{" + ",".join(f'"{item}"' for item in params) + "}"
    assert len(set(lengths)) == 1, (
        f"Parameters '{transf}.{p}' must be same length, "
        f"found lengthts {lengths}."
    )