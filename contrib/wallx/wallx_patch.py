"""Runtime workaround for a Wall-X (commit ce2a1ae) checkpoint-loading bug. No repo file is modified.

Problem: Qwen2_5_VLMLP pads the vision MLP intermediate size 3420 -> 3456 and expects its own
`_load_from_state_dict` hook to zero-pad the *gate* and *up* halves of the fused `gate_up_proj`
separately. But the serving wrapper first calls `reshape_compatible_state_dict(ckpt, model.state_dict())`,
and `model.state_dict()` reports the padded shapes (the `_save_to_state_dict` un-padding never fires because
the child Linear's keys are not in `destination` yet). The generic helper then tail-pads the fused tensor,
which shifts the whole `up` half by 36 rows and leaves random init in the tails -> vision MLP output is wrong.

Fix: let those specific tensors pass through with their checkpoint shape so the model's own hook pads them.
Disable with WALLX_AGENT_NO_PATCH=1.
"""
import os

_ALIGN = 128


def _aligned(n: int) -> int:
    return (n + _ALIGN - 1) // _ALIGN * _ALIGN


def _is_vl_mlp_key(name, param, target) -> bool:
    if ".mlp." not in name or not name.startswith("visual."):
        return False
    if name.endswith("mlp.gate_up_proj.weight") or name.endswith("mlp.gate_up_proj.bias"):
        return param.shape[0] % 2 == 0 and target.shape[0] == 2 * _aligned(param.shape[0] // 2) and tuple(param.shape[1:]) == tuple(target.shape[1:])
    if name.endswith("mlp.down_proj.weight"):
        return param.ndim == 2 and target.shape[1] == _aligned(param.shape[1]) and param.shape[0] == target.shape[0]
    return False


def apply() -> bool:
    if os.environ.get("WALLX_AGENT_NO_PATCH") == "1":
        print("[wallx_patch] disabled by WALLX_AGENT_NO_PATCH=1", flush=True)
        return False
    from wall_x._vendor.harrix.serving._wallx_infer import model_wrapper as mw

    orig = mw.reshape_compatible_state_dict
    if getattr(orig, "_wallx_agent_patched", False):
        return True

    def patched(state_dict, model_sd, log_fn=None):
        passthrough, rest = {}, {}
        for name, param in state_dict.items():
            target = model_sd.get(name)
            if target is not None and param.shape != target.shape and _is_vl_mlp_key(name, param, target):
                passthrough[name] = param
            else:
                rest[name] = param
        out = orig(rest, model_sd, log_fn=log_fn)
        out.update(passthrough)
        print(f"[wallx_patch] {len(passthrough)} vision-MLP tensors left for the model's own zero-padding hook", flush=True)
        return out

    patched._wallx_agent_patched = True
    mw.reshape_compatible_state_dict = patched
    return True
