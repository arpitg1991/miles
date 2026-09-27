from model_args_utils import load_sibling_model_args


def model_args() -> str:
    # Real layers 0-4: KDA + dense x3, DSA + MoE, KDA + MoE (the kdatp parity slice).
    return load_sibling_model_args(__file__, "glm5.3-flash", nlayers=5)
