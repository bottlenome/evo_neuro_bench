# 研究状況メモ (2025-10-17)

## コードベースの整理状況
- Google Colab の単一ノートブックからの移管が完了し、`src/evo_neuro_bench/` 以下にモデル・アダプタ・タスクがモジュールとして整理されている。特に `benchmark.py` から各タスクごとに `ModelAdapter` を介して学習を呼び出す構成になっている。 

## ModelAdapter に関するメモ
- `ModelAdapter` に `prefer_base_logits` フラグを追加し、ベースモデルが `out["abstract_logits"]` を提供していて `n_actions` と整合する場合はそれを優先的に返すようにした。キーが存在しない、または次元が一致しない場合は従来通り `motor` 経路に付随する線形ヘッドを遅延初期化して利用するため、既存の運動タスクは後方互換。
- すべてのベースモデルに軽量な抽象ヘッドを追加し、`N_ABSTRACT_ACTIONS (=8)` 次元のロジットを返すよう統一した。ヒトモデルでは前頭前野ワーキングメモリ、魚/頭足類では中央統合表現など、認知側の表象から抽象ヘッドを生成している。

## タスク分類と抽象ロジット設計
- **Embodied / Motor タスク**: `local_reflex`, `peristalsis`, `rpm_mini`, `detour` などは従来通り `motor` 出力にアダプタの線形ヘッドを乗せて使用する。
- **Abstract / Reasoning タスク**: `arc_mini`, `grid_firststep`, `reversal`（選択肢比較型）は `n_actions = N_ABSTRACT_ACTIONS` を指定し、`prefer_base_logits=True` でベースの `abstract_logits` を直接学習させる。分類数が異なるタスクでは、必要に応じてベース側の `abstract_dim` 引数をオーバーライドすることで対応可能。
