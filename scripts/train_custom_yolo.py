"""OPTIONAL project template: fine-tune YOLO on your OWN annotated dataset.
Needs a real dataset with sufficient images. CPU training can take a long time.
Example: python scripts/train_custom_yolo.py --data path/to/data.yaml --epochs 10
Do not treat the tiny illustrative toy dataset as a meaningful training benchmark.

Dice project (model A, dice-type detector) — run 1 baseline, from the repository root:
    .venv\\Scripts\\python.exe scripts/train_custom_yolo.py
        --data datasets/dice_type/data.yaml
        --epochs 50 --project runs/dice_type --name run1_baseline

Outputs (in <project>/<name>/): weights/best.pt (best model on validation),
results.png (loss + metric curves), confusion_matrix.png, PR curves, results.csv.
"""
import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True,help='Path to YOUR exported data.yaml')
    # One epoch = the model sees every training image once.
    parser.add_argument('--epochs',type=int,default=10)
    # Images are resized to imgsz x imgsz pixels before entering the network.
    # Bigger = small objects keep more detail, but training is slower.
    parser.add_argument('--imgsz',type=int,default=416)
    # Pretrained weights to start from (transfer learning), e.g. yolo11n.pt / yolo11s.pt.
    parser.add_argument('--model',default='yolo11n.pt')
    # Added for the dice project — defaults keep the template's original behaviour.
    # Number of images processed together before each weight update.
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--device',default='cpu',help="'cpu', or '0' for the first NVIDIA GPU")
    # Where results are saved: <project>/<name>/. Naming each run (run1_baseline,
    # run2_...) keeps runs separate and easy to compare in the presentation.
    parser.add_argument('--project',default=None,help='Output folder for runs (default: Ultralytics runs/detect)')
    parser.add_argument('--name',default=None,help='Name of this run, e.g. run1_baseline')
    args=parser.parse_args()
    if not Path(args.data).is_file():
        parser.error('Could not find data.yaml: '+args.data)
    model=YOLO(args.model)
    # project must be an absolute path, otherwise Ultralytics puts it under its
    # own runs/detect/ folder instead of where we asked.
    project=str(Path(args.project).resolve()) if args.project else None
    model.train(data=args.data,epochs=args.epochs,imgsz=args.imgsz,device=args.device,batch=args.batch,
                project=project,name=args.name)

    # Print the final validation metrics of the best model (also in results.csv).
    # model.trainer.metrics is a dict like {'metrics/mAP50(B)': 0.93, ...}
    print('\nBest model saved to:',model.trainer.best)
    for metric_name,value in model.trainer.metrics.items():
        print(f'  {metric_name:25s} {value:.3f}')

if __name__=='__main__':
    main()
