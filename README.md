# Statistical Learning and Attentional Suppression: A Computational Model

## Question
Can a simulated agent learn to suppress a location where distractors appear
frequently (70% of trials)?

## Model
Priority map = salience - beta * learned distractor probability + noise.
Learning: delta rule (rate alpha). Control: alpha = 0 (no learning).

## How to run
pip install -r requirements.txt
python scripts/run_simulation.py
python scripts/run_control.py
python scripts/analyze.py
python scripts/run_sweep.py
python scripts/plot_sweep.py

## Key results

![Capture rate, learning agents](results/capture_by_block.png)
![Capture rate, control](results/control_capture_by_block.png)
![Parameter sweep](results/sweep_heatmap.png)

In learning agents, distractor capture at the high-probability location falls
across blocks. In control agents (alpha = 0), it stays flat. The sweep shows
suppression emerges when both alpha and beta are above zero.