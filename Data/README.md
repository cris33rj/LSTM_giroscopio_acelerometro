# Dados

As bases são públicas e não ficam no repositório. Baixe cada uma e organize-as na pasta `Data/` (esta pasta), na raiz do projeto, como mostrado abaixo. Esses são os caminhos que os notebooks esperam.

```
Data/
├── Raw/
│   ├── watch/accel/        # WISDM 2019: data_16xx_accel_watch.txt
│   ├── watch/gyro/         # WISDM 2019: data_16xx_gyro_watch.txt
│   ├── phone/accel/        # (opcional) usados em experiments/
│   └── phone/gyro/
├── UCI-HAPT/
│   ├── RawData/            # acc_exp*_user*.txt, gyro_exp*_user*.txt, labels.txt
│   ├── Train/              # X_train.txt, y_train.txt (apenas experiments/)
│   └── Test/
├── KUHAR/
│   └── 3.Time_domain_subsamples/KU-HAR_time_domain_subsamples_20750x300.csv
├── UNIMIB-SHAR/
│   ├── adl_data.npy
│   └── adl_labels.npy
├── FORTH_TRACE_DATASET-master/
└── processed/              # gerado pelos notebooks (CSVs intermediários)
```

| Base | Download |
|---|---|
| WISDM Smartphone and Smartwatch Activity and Biometrics (2019) | https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset |
| UCI-HAPT: Smartphone-Based Recognition of Human Activities and Postural Transitions | https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions |
| KU-HAR | https://data.mendeley.com/datasets/45f952y38r/5 |
| UniMiB-SHAR | http://www.sal.disco.unimib.it/technologies/unimib-shar/ |
| FORTH-TRACE | https://github.com/spl-icsforth/FORTH_TRACE_DATASET |

O notebook `notebooks/00_wisdm_preprocessamento.ipynb` lê `Data/Raw/watch/` e gera `Data/processed/activity_recognition_data_watch_concat_by_comparacao_grupos.csv`, que os notebooks do WISDM usam.
