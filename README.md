# Foosball RL Environment

This project implements a Reinforcement Learning environment for Foosball using PyBullet.

## Running the Training

To train the foosball agent, run the `train.py` script. You can specify the training stage using the `--stage` argument.

```bash
python train.py --stage 1
```

If you do not specify a stage, the script will run all stages sequentially.

```bash
python train.py
```

## Viewing Logs with TensorBoard

The training script logs data to the `logs/` directory. You can view these logs using TensorBoard.

1.  Install TensorBoard:
    ```bash
    pip install tensorboard
    ```

2.  Run TensorBoard and point it to the log directory:
    ```bash
    tensorboard --logdir logs/
    ```

3.  Open your web browser and navigate to the URL provided by TensorBoard (usually `http://localhost:6006/`).

You will be able to see the training metrics, including `rollout/goals_scored` and `rollout/goals_conceded`, in the TensorBoard interface.