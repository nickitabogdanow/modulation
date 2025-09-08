PYTHON ?= python

DATASET_TAG ?= v1
EXAMPLES ?= 200
SEED ?= 42

.PHONY: data train-ml train-dl eval lint

data:
	$(PYTHON) dataset_builder.py --examples $(EXAMPLES) --use-multipath --save-dir data --tag $(DATASET_TAG) --seed $(SEED)

train-ml:
	$(PYTHON) train_ml.py --train data/train_$(DATASET_TAG).npz --val data/val_$(DATASET_TAG).npz --test data/test_$(DATASET_TAG).npz --seed $(SEED)

train-dl:
	$(PYTHON) train_dl.py --mode 1d --train data/train_$(DATASET_TAG).npz --val data/val_$(DATASET_TAG).npz --test data/test_$(DATASET_TAG).npz --epochs 30 --batch-size 128 --seed $(SEED)

eval:
	$(PYTHON) evaluate_compare.py --train data/train_$(DATASET_TAG).npz --val data/val_$(DATASET_TAG).npz --test data/test_$(DATASET_TAG).npz

lint:
	ruff check .

