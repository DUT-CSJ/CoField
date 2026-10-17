# CoField: Globally Coordinated Local Neural Fields for Medical Lesion Segmentation

PyTorch implementation of CoField and CoField-V.

This work has been submitted to IEEE Transactions on Image Processing (TIP). Hope everything goes well.

## Setup

Use Python 3.10+ and install a suitable PyTorch build, then run:

```bash
pip install -r requirements.txt
```

## Images

Set `classes` to the number of foreground classes. Binary masks use zero for background and nonzero for foreground; multiclass masks use integer labels `0..classes`. Paths in the split JSON are relative to `data_root`:

```json
{
  "folds": {
    "fold0": {
      "train": [["images/train.png", "masks/train.png"]],
      "val": [["images/val.png", "masks/val.png"]],
      "test": [["images/test.png", "masks/test.png"]]
    }
  }
}
```

Provide `fold0` through `fold3` for four-fold evaluation, preserving patient/case/video groups. Train and evaluate each fold separately, then average its metrics:

```bash
python train.py --config configs/image.json --fold 0
python evaluate.py --checkpoint "<CHECKPOINT>" --data-root "<DATA_ROOT>" --split-file "<SPLIT_JSON>" --fold 0 --split val
```

## Video Segmentation

SUN-SEG: Use the [official prepared layout](https://github.com/GewelsJI/VPS/blob/main/docs/DATA_PREPARATION.md): `TrainDataset/{Frame,GT}/<clip>/` and `Test{Easy,Hard}Dataset/{Seen,Unseen}/{Frame,GT}/<clip>/`. 

TED is available from the [official dataset page](https://humanheart-project.creatis.insa-lyon.fr/ted.html).

```bash
python train.py --config configs/sunseg.json
python evaluate.py --checkpoint "<VIDEO_CHECKPOINT>" --data-root "<SUN_SEG_ROOT>" --output "<RESULTS_JSON>"
```
