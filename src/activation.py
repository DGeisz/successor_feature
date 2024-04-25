import einops
from IPython.display import display
import circuitsvis as cv

from src.model import model
from src.encoder import encoder


def get_linear_feature_activation(layer_nine_z, averaged=True, feature_i=18):
    layer_nine_z = einops.rearrange(
        layer_nine_z, "batch seq n_heads d_head -> batch seq (n_heads d_head)"
    )[:, -1, :].squeeze(1)
    feature = encoder.W_enc[:, feature_i]

    linear_feature_activation = (
        einops.einsum(
            layer_nine_z - encoder.b_dec, feature, "batch d_model, d_model -> batch"
        )
        + encoder.b_enc[feature_i]
    )

    if averaged:
        return linear_feature_activation.mean(dim=0)
    else:
        return linear_feature_activation


def get_linear_feature_activation_from_cache(cache, averaged=True, feature_i=18):
    layer_nine_z = cache["z", 9]

    return get_linear_feature_activation(layer_nine_z, averaged, feature_i)


def show_stats_for_string(seq: str):
    _, cache = model.run_with_cache(seq)

    z = cache["z", 9]
    z = einops.rearrange(z, "batch seq n_heads d_head -> batch seq (n_heads d_head)")

    acts = encoder(z)[2]

    feature_acts = acts[0, :, 18]

    display(
        cv.tokens.colored_tokens(
            tokens=model.to_str_tokens(seq), values=feature_acts, max_value=2
        )
    )

    print("Max Activation:", feature_acts.max().item())
    print("All Activations: ", feature_acts.tolist())
    print(
        "Largest Activations:", feature_acts.sort(descending=True).values[:5].tolist()
    )
