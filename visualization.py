# visualization.py

import pandas as pd
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display
from typing import Callable
from extract_features import load_and_process_sample
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd


def plot_signal(signal, label):
    """Plot X, Y, Z components of a signal"""
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(signal[:, 0], label='X', alpha=0.7)
    ax.plot(signal[:, 1], label='Y', alpha=0.7)
    ax.plot(signal[:, 2], label='Z', alpha=0.7)
    ax.set_title(f'Signal - {label}')
    ax.set_ylabel('Acceleration')
    ax.set_xlabel('Time')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def signal_viewer(
        data_dir: Path,
        labels_csv: Path,
):
    """
    Launch an interactive viewer for signal samples.
    
    Args:
        data_dir (Path): Path to directory containing .npz files.
        labels_csv (Path): Path to the CSV file with sample_id and label.
        load_and_process_sample (Callable): Function to load and process a sample.
        plot_signal (Callable): Function to plot the signal and label.
    """
    all_files = sorted([f.stem for f in data_dir.glob('*.npz')])
    df_labels = pd.read_csv(labels_csv)

    sample_selector = widgets.Dropdown(
        options=all_files,
        description='Sample ID:',
        layout=widgets.Layout(width='50%')
    )

    load_button = widgets.Button(description='Load & Plot', button_style='primary')
    output = widgets.Output()

    def on_load_clicked(b):
        with output:
            output.clear_output()
            sample_id = sample_selector.value
            signal, metadata = load_and_process_sample(f'{data_dir.name}/{sample_id}')
            label = df_labels.loc[df_labels['sample_id'] == sample_id, 'label'].values[0]
            plot_signal(signal, label)

    load_button.on_click(on_load_clicked)
    display(widgets.VBox([sample_selector, load_button, output]))
