# /// script
# requires-python = "==3.12.*"
# dependencies = []
# ///

from pathlib import Path
from textwrap import dedent


def _naive_attention_source() -> str:
    return dedent(
        """\
        import math
        import torch


        def naive_attention(q, k, v):
            '''Simple scaled dot-product attention.

            Args:
                q, k, v: [batch, heads, seq_len, head_dim]
            Returns:
                out: attention output
                m: per-row max logits
                l: per-row sum exp logits
            '''
            scale = 1.0 / math.sqrt(q.size(-1))
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            m = scores.max(dim=-1).values
            scores_exp = torch.exp(scores - m.unsqueeze(-1))
            l = scores_exp.sum(dim=-1)
            p = scores_exp / l.unsqueeze(-1)
            out = torch.matmul(p, v)
            return out, m, l
        """
    )


def main() -> None:
    env_dir = Path("env_data")
    env_dir.mkdir(parents=True, exist_ok=True)
    target_path = env_dir / "naive_attention.py"
    target_path.write_text(_naive_attention_source(), encoding="utf-8")


if __name__ == "__main__":
    # Our focus is only on attention kernel, so we can disregard W_o in this task.
    # We can just set it to identity and focus the agent on optimizing the attention computation itself.
    main()
