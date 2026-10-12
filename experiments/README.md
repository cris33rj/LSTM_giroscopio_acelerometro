# Experiments

History of the experiments that led to the models in [`notebooks/`](../notebooks/). They are kept to document the process. They are not needed to reproduce the main results, and some depend on intermediate CSVs produced by earlier versions of the pipeline. The notebooks were written in Portuguese.

| Family | Files | What it investigates |
|---|---|---|
| Sensor concatenation (WISDM) | `concat_*`, `data_prep`, `activity_recognition_data_concatenation_types_comparison_by_visualization` | Ways to synchronize accelerometer and gyroscope: by index (participant, activity, timestamp) vs. by size-equalized groups; watch vs. phone |
| LSTM baselines (WISDM) | `activity_recognition_tuner_*`, `sliding_window_*`, `pre_saved_*` | LSTM with L2, dropout and batch norm; activity subsets; 6, 22 and 27 features (with composite functions); 70/30 holdout vs. k-fold by participant groups |
| Main model variants (WISDM) | `wisdm_6_ativ_*`, `activity_recognition_*only_participantes*` | With and without sliding window, 6 vs. 18 classes, only participants with all activities |
| Other architectures | `*_lstm_bidirectional`, `*_convlstm_*` | Bidirectional LSTM and ConvLSTM2D (watch + phone, 12 channels, 18 activities) |
| Streaming inference | `realtime_cognition_*` | Window-by-window classification simulating real-time use |
| Biometrics | `*_following_weiss_paper` | Genuine vs. impostor authentication from movement patterns (inspired by Weiss et al., 2019) |
| UCI-HAPT, KU-HAR, UniMiB-SHAR | `uci_hapt_*`, `kuhar_*`, `unimib_shar_*` | Variants without sliding window, with precomputed features, linear interpolation and class weights |
| Augmentation ablation (FORTH-TRACE) | `har_forth_trace_*_only_*`, `*_class_weight`, `*_crop_120`, `*_interpolation`, `*_synthetic_transitions_*` | Isolated effect of each data augmentation technique and of synthetic transitions |
