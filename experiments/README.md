# Experimentos

Histórico dos experimentos que levaram aos modelos de [`notebooks/`](../notebooks/). Foram mantidos para documentar o processo. Não são necessários para reproduzir os resultados principais e podem depender de CSVs intermediários gerados em versões anteriores do pipeline.

| Família | Arquivos | O que investiga |
|---|---|---|
| Concatenação dos sensores (WISDM) | `concat_*`, `data_prep`, `activity_recognition_data_concatenation_types_comparison_by_visualization` | Formas de sincronizar acelerômetro e giroscópio: por índice (participante, atividade, timestamp) vs. por grupos de tamanho equalizado; relógio vs. celular |
| Baselines LSTM (WISDM) | `activity_recognition_tuner_*`, `sliding_window_*`, `pre_saved_*` | LSTM com L2, dropout e batch norm; subconjuntos de atividades; 6, 22 e 27 features (com funções compostas); holdout 70/30 vs. k-fold por grupos de participantes |
| Variações do modelo principal (WISDM) | `wisdm_6_ativ_*`, `activity_recognition_*only_participantes*` | Com e sem janela deslizante, 6 vs. 18 classes, participantes com todas as atividades |
| Outras arquiteturas | `*_lstm_bidirectional`, `*_convlstm_*` | LSTM bidirecional e ConvLSTM2D (relógio + celular, 12 canais, 18 atividades) |
| Inferência em fluxo | `realtime_cognition_*` | Classificação janela a janela simulando uso em tempo real |
| Biometria | `*_following_weiss_paper` | Autenticação genuíno vs. impostor a partir do padrão de movimento (inspirado em Weiss et al., 2019) |
| UCI-HAPT, KU-HAR, UniMiB-SHAR | `uci_hapt_*`, `kuhar_*`, `unimib_shar_*` | Versões sem janela deslizante, com features prontas, interpolação linear e class weights |
| Ablação de augmentation (FORTH-TRACE) | `har_forth_trace_*_only_*`, `*_class_weight`, `*_crop_120`, `*_interpolation`, `*_synthetic_transitions_*` | Efeito isolado de cada técnica de aumento de dados e de transições sintéticas |
