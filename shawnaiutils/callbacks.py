from keras import callbacks
import os
from datetime import datetime

def standard_callbacks(run_name: str = 'default_run', monitor: str = 'val_loss', patience: int = 7) -> list:
    """
    Create a standard set of Keras callbacks for model training.

    Includes:
      - ModelCheckpoint: saves best model weights
      - EarlyStopping: stops when validation metric plateaus
      - ReduceLROnPlateau: lowers LR when progress stalls
      - TensorBoard: logs metrics for visualization
      - CSVLogger: saves training history to CSV
    """

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('runs', f'{run_name}_{timestamp}')
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'logs'), exist_ok=True)

    checkpoint_path = os.path.join(run_dir, 'checkpoints', 'best_model.keras')
    csv_path = os.path.join(run_dir, f'{run_name}_training_log.csv')
    tensorboard_log_dir = os.path.join(run_dir, 'logs')

    monitor_lower = monitor.lower()
    if any(k in monitor_lower for k in ['acc', 'auc', 'precision', 'recall']):
        mode = 'max'
    else:
        mode = 'min'

    checkpoint = callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=monitor,
        mode=mode,
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )

    early_stop = callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        mode=mode,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        mode=mode,
        verbose=1
    )

    tensorboard = callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1,
        write_graph=True,
        update_freq='epoch'
    )

    csv_logger = callbacks.CSVLogger(
        filename=csv_path,
        append=False,
        separator=','
    )

    print(f"\033[92m[CALLBACKS]\033[0m Initialized for run: {run_name}")
    print(f"  → Logs: {tensorboard_log_dir}")
    print(f"  → Checkpoints: {checkpoint_path}")

    return [checkpoint, reduce_lr, early_stop, tensorboard, csv_logger]