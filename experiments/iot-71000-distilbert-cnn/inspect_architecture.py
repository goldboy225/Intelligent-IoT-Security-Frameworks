"""Structural and local CPU checks only; random inputs are NOT dataset results."""
import json
import platform
import time
from pathlib import Path
import torch
from model import DualHead,multitask_loss


def main():
    torch.manual_seed(42);torch.set_num_threads(4)
    model=DualHead(470,initialization='random');x=torch.randn(64,470)
    model.eval()
    with torch.inference_mode():
        model(x);times=[]
        for _ in range(5):
            start=time.perf_counter();model(x);times.append(time.perf_counter()-start)
    model.train();model.zero_grad(set_to_none=True)
    binary,attack=model(x)
    loss=multitask_loss(binary,attack,torch.ones(64,dtype=torch.long),torch.arange(64)%5,torch.ones(5))
    loss.backward()
    qk=model.fusion.in_proj_weight.grad[:2304]
    result=dict(status='structural_check_only_not_accuracy_reproduction',
        initialization='random, no pretrained checkpoint or real samples used',
        torch=torch.__version__,python=platform.python_version(),cpu_threads=4,cuda_available=torch.cuda.is_available(),
        dimensions=dict(input_features=470,projection_output=768,transformer_tokens=1,
            transformer_layers=model.encoder.config.n_layers,transformer_width=model.encoder.config.dim,
            fusion_width=1152,fusion_tokens=1,fusion_heads=8,binary_logits=2,attack_logits=5),
        parameters=sum(p.numel() for p in model.parameters()),
        full_width_backward_loss_finite=bool(torch.isfinite(loss)),
        fusion_query_key_gradient_max_abs=float(qk.abs().max()),
        cpu_forward_seconds_five_warmed_batches=times,
        caveat='Timings are this untrained assumed implementation on random 64x470 inputs. They neither validate the manuscript accuracy nor establish real-time gateway suitability.',
        shape_result='Eq.4 gives z in R^768. Reshaping into L x 768 without adding values forces L=1. CNN kernel 5 over one transformer token sees only the center and zero padding.',
        attention_result='For one fusion token, softmax of a 1x1 attention score is exactly 1; query/key weighting cannot adaptively select among feature sources. This finding is conditional on the literal concatenated-vector interpretation, not proof of what the undocumented author code did.')
    assert result['full_width_backward_loss_finite']
    assert result['fusion_query_key_gradient_max_abs']<1e-7
    target=Path(__file__).parent/'results/architecture_checks.json';target.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
