# Dataset notice

`diagnosed_cbc_data_v4.csv` is redistributed unmodified from the Kaggle dataset
**Anemia Types Classification** by Ehab Aboelnaga:
https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification

Licence: Apache License 2.0 (full text in `LICENSE-APACHE-2.0.txt` in this folder). The dataset
author owns the data. This repository only redistributes it, so the notebooks run without a Kaggle
account.

SHA-256 of the file: f7dee8d4fe500aec6e8e4bfbeb6df6b83bc86bef03d540721e0aeaad2759fcdb (125,590 bytes).
`python src/check_dataset.py` verifies it.

The file contains all 1,281 rows, including the 247 held-out test rows of this project. No script or
notebook reads the test rows before the final evaluation (`load_test()` in `src/data.py` refuses).
