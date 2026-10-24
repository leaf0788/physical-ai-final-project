# Physical AI 応用1 最終課題 - 模倣学習でハートを描くロボット

## 概要
2輪移動ロボット（unicycle model）を、模倣学習で学習したニューラルネットワーク方策で制御し、ハート型の軌道を描かせます。
AIは現在姿勢と次の目標点の相対関係から、線速度 `v` と角速度 `w` を推論します。

## AIを使っている箇所
`train_policy.py` が、ルールベースのexpert controllerから状態-行動ペアを生成し、2層MLPを教師あり学習します。
実行時は `run_sim.py` / `ros2_turtlesim_heart_node.py` が学習済みモデルを読み込み、毎ステップ推論してロボットを制御します。

## 実行手順（スタンドアロン版）
```bash
pip install -r requirements.txt
python src/train_policy.py
python src/run_sim.py
```

生成物:
- `output/heart_policy.pt`
- `output/training_curve.png`
- `output/final_trajectory.png`
- `output/PAI_final_heart_robot.mp4`

## ROS2 turtlesim版
ROS 2環境で以下を実行します。

```bash
ros2 run turtlesim turtlesim_node
python3 src/ros2_turtlesim_heart_node.py
```

ノードは `/turtle1/pose` をsubscribeし、`/turtle1/cmd_vel` に推論した速度指令をpublishします。

## 工夫点
- AI利用が動画上で分かるよう、推論中の速度値と現在waypointを表示。
- expert controllerの出力をニューラルネットワークに模倣させ、講義で扱った模倣学習の考え方を簡略化して再現。
- スタンドアロン版とROS2版を同じ学習済み方策で動かせる構成。