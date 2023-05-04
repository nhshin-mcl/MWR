from network.model_mwr import MWR


def build_model(cfg):
    net = eval(cfg.model_name)(cfg)
    return net

