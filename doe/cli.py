import argparse
import json
from pathlib import Path

from .data import generate_dataset
from .engine import TrainConfig, evaluate, predict, train


def main(argv=None):
    parser = argparse.ArgumentParser(description="Four-plane neural DOE synthesis, single-FFT Fresnel optics")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Generate independent procedural train/validation/test quartets")
    generate.add_argument("--output", required=True)
    generate.add_argument("--train", type=int, default=128)
    generate.add_argument("--validation", type=int, default=24)
    generate.add_argument("--test", type=int, default=24)
    generate.add_argument("--seed", type=int, default=2026)
    fit = sub.add_parser("train", help="Train with differentiable Fresnel reconstruction, without DOE labels")
    fit.add_argument("--data", required=True)
    fit.add_argument("--output", required=True)
    fit.add_argument("--initial-checkpoint", help="Fine-tune existing weights; optimizer starts fresh")
    fit.add_argument("--objective",choices=("physics","perceptual"),default="physics")
    fit.add_argument('--architecture',choices=('fixed','wavefront','adaptive'),default='fixed')
    fit.add_argument('--initializer-width',type=int,default=8)
    fit.add_argument('--initializer-depth',type=int,choices=(3,4),default=3)
    for flag, kind, default in (("epochs", int, 24), ("batch-size", int, 4), ("stages", int, 6),
                                ("width", int, 12), ("learning-rate", float, 0.005), ("seed", int, 42)):
        fit.add_argument("--" + flag, type=kind, default=default)
    infer = sub.add_parser("predict", help="Export DOE, four amplitude PNGs and metrics for exactly four input PNGs")
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--images", nargs=4, required=True, metavar="PNG")
    infer.add_argument("--output", required=True)
    assess = sub.add_parser("evaluate", help="Held-out comparison against untrained network and GS references")
    assess.add_argument("--data", required=True)
    assess.add_argument("--checkpoint", required=True)
    assess.add_argument("--output", required=True)
    assess.add_argument("--batch-size", type=int, default=4)
    assess.add_argument("--gs-steps", type=int, default=80)
    for p in (fit, infer, assess):
        p.add_argument("--device", default="auto")
        p.add_argument("--threads", type=int, default=2)
    for p in (infer,assess):
        p.add_argument('--refine-steps',type=int,default=0,help='Optional per-input optimizer; zero preserves pure neural inference')
        p.add_argument('--refine-lr',type=float,default=0.08)
    args = vars(parser.parse_args(argv))
    command = args.pop("command")
    if command == "generate":
        generate_dataset(args.pop("output"), **args)
    elif command == "train":
        data, output = args.pop("data"), args.pop("output")
        train(data, output, TrainConfig(**args))
    elif command == "predict":
        report = predict(**args)
        print(json.dumps({"mean_pearson": report["mean_pearson"], "mean_ssim": report["mean_ssim"]}, indent=2))
    else:
        evaluate(**args)


if __name__ == "__main__":
    main()
