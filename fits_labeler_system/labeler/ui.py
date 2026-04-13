from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Callable

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from .csv_logger import CsvDecisionLogger
from .loaders import BaseFitsLoader
from .scaling import BandScaler


@dataclass
class BandControl:
    low: tk.DoubleVar
    high: tk.DoubleVar
    a_log10: tk.DoubleVar


@dataclass
class SliderControl:
    var: tk.DoubleVar
    min_v: float
    max_v: float
    fine_step: float
    on_change_cb: Callable[[str], None]
    toggle_button: tk.Label


class FitsLabelerApp:
    """Tkinter UI for binary decisions over 4-band FITS samples."""

    TOGGLE_OFF_BG = "#b56565"
    TOGGLE_OFF_ACTIVE_BG = "#9f5555"
    TOGGLE_ON_BG = "#00cc44"
    TOGGLE_ON_ACTIVE_BG = "#00b83d"

    def __init__(
        self,
        loader: BaseFitsLoader,
        logger: CsvDecisionLogger,
        true_label_text: str = "True",
        false_label_text: str = "False",
        skip_previously_labeled: bool = False,
        title: str = "FITS Labeler",
    ) -> None:
        self.loader = loader
        self.logger = logger
        self.true_label_text = true_label_text
        self.false_label_text = false_label_text
        self.skip_previously_labeled = skip_previously_labeled

        all_ids = self.loader.discover()
        if self.skip_previously_labeled:
            done = logger.labeled_ids()
            self.sample_ids = [s for s in all_ids if s not in done]
        else:
            self.sample_ids = all_ids

        self.idx = 0
        self.current_sample = None
        self.band_controls: dict[str, BandControl] = {}
        self.images: dict[str, np.ndarray] = {}
        self.artists = {}
        self.slider_controls: dict[str, SliderControl] = {}
        self.active_slider_id: str | None = None

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("1400x900")

        self.progress_var = tk.StringVar(value="")
        self._build_layout()
        self._bind_shortcuts()
        self._load_current_sample()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=3)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)

        right = ttk.Frame(self.root, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(8, 8), dpi=100)
        axes = self.figure.subplots(2, 2)
        self.axes_by_band = {}
        for ax in axes.flat:
            ax.axis("off")

        self.canvas = FigureCanvasTkAgg(self.figure, master=left)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        ttk.Label(right, textvariable=self.progress_var, font=("Helvetica", 12, "bold")).grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )

        self.controls_frame = ttk.Frame(right)
        self.controls_frame.grid(row=1, column=0, sticky="nsew")
        self.controls_frame.columnconfigure(0, weight=1)

        btn_frame = ttk.Frame(right)
        btn_frame.grid(row=2, column=0, sticky="ew", pady=(10, 0))
        btn_frame.columnconfigure((0, 1), weight=1)

        ttk.Button(btn_frame, text=self.false_label_text, command=self._label_false).grid(
            row=0, column=0, sticky="ew", padx=(0, 5)
        )
        ttk.Button(btn_frame, text=self.true_label_text, command=self._label_true).grid(
            row=0, column=1, sticky="ew", padx=(5, 0)
        )

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Left>", self._on_arrow_key)
        self.root.bind("<Right>", self._on_arrow_key)
        self.root.bind("<Up>", self._on_arrow_key)
        self.root.bind("<Down>", self._on_arrow_key)

    def _load_current_sample(self) -> None:
        if not self.sample_ids:
            messagebox.showinfo("Finished", "No unlabeled samples found.")
            self.root.destroy()
            return

        if self.idx >= len(self.sample_ids):
            messagebox.showinfo("Finished", "All samples have been labeled.")
            self.root.destroy()
            return

        sample_id = self.sample_ids[self.idx]
        self.current_sample = self.loader.load(sample_id)
        self.images = self.current_sample.bands

        self._refresh_progress_text()
        self._reset_controls_if_needed()
        self._redraw_all_bands()

    def _reset_controls_if_needed(self) -> None:
        bands = list(self.images.keys())

        if not self.axes_by_band:
            for ax, band in zip(self.figure.axes, bands):
                self.axes_by_band[band] = ax

        if self.band_controls and set(self.band_controls.keys()) == set(bands):
            return

        for widget in self.controls_frame.winfo_children():
            widget.destroy()

        self.band_controls.clear()
        self.slider_controls.clear()
        self.active_slider_id = None

        for row, band in enumerate(bands):
            frame = ttk.LabelFrame(self.controls_frame, text=f"{band} scaling", padding=6)
            frame.grid(row=row, column=0, sticky="ew", pady=4)
            frame.columnconfigure(1, weight=1)

            low = tk.DoubleVar(value=40.0)
            high = tk.DoubleVar(value=99.8)
            a_log10 = tk.DoubleVar(value=-1.0)
            self.band_controls[band] = BandControl(low=low, high=high, a_log10=a_log10)

            self._add_slider(
                frame,
                "Low %",
                low,
                0.0,
                100.0,
                band,
                slider_name="low",
                fine_step=0.05,
                on_change_cb=lambda _, b=band: self._on_low_changed(b),
            )
            self._add_slider(
                frame,
                "High %",
                high,
                0.0,
                100.0,
                band,
                slider_name="high",
                fine_step=0.05,
                on_change_cb=lambda _, b=band: self._on_high_changed(b),
            )
            self._add_slider(
                frame,
                "log10(Asinh a)",
                a_log10,
                -3.0,
                3.0,
                band,
                slider_name="a_log10",
                resolution=0.01,
                fine_step=0.1,
                on_change_cb=lambda _, b=band: self._redraw_band(b),
            )

    def _add_slider(
        self,
        parent: ttk.LabelFrame,
        text: str,
        var: tk.DoubleVar,
        min_v: float,
        max_v: float,
        band: str,
        slider_name: str,
        fine_step: float,
        resolution: float = 0.1,
        on_change_cb=None,
    ) -> None:
        row = parent.grid_size()[1]
        ttk.Label(parent, text=text).grid(row=row, column=0, sticky="w", padx=(0, 8))

        callback = on_change_cb

        if callback is None:
            def default_callback(value: str) -> None:
                _ = value
                self._redraw_band(band)
            callback = default_callback

        scale = tk.Scale(
            parent,
            variable=var,
            from_=min_v,
            to=max_v,
            orient=tk.HORIZONTAL,
            resolution=resolution,
            length=220,
            command=callback,
        )
        scale.grid(row=row, column=1, sticky="ew")

        slider_id = f"{band}:{slider_name}"
        toggle_button = tk.Label(
            parent,
            text="Keys Off",
            width=8,
            relief=tk.RAISED,
            bd=1,
            bg=self.TOGGLE_OFF_BG,
            fg="white",
            padx=6,
            pady=3,
            cursor="hand2",
            highlightthickness=0,
            takefocus=0,
        )
        toggle_button.bind("<Button-1>", lambda _, sid=slider_id: self._toggle_slider_control(sid))
        toggle_button.grid(row=row, column=2, sticky="e", padx=(8, 0))

        self.slider_controls[slider_id] = SliderControl(
            var=var,
            min_v=min_v,
            max_v=max_v,
            fine_step=fine_step,
            on_change_cb=callback,
            toggle_button=toggle_button,
        )

    def _toggle_slider_control(self, slider_id: str) -> None:
        if self.active_slider_id == slider_id:
            self.active_slider_id = None
            self._refresh_slider_toggle_states()
            return

        self.active_slider_id = slider_id
        self._refresh_slider_toggle_states()

    def _refresh_slider_toggle_states(self) -> None:
        for sid, control in self.slider_controls.items():
            if sid == self.active_slider_id:
                control.toggle_button.config(
                    text="Keys On",
                    relief=tk.SUNKEN,
                    bg=self.TOGGLE_ON_BG,
                    fg="white",
                )
            else:
                control.toggle_button.config(
                    text="Keys Off",
                    relief=tk.RAISED,
                    bg=self.TOGGLE_OFF_BG,
                    fg="white",
                )

    def _on_arrow_key(self, event: tk.Event) -> None:
        key = event.keysym

        if self.active_slider_id is None:
            if key == "Left":
                self._label_false()
            elif key == "Right":
                self._label_true()
            return

        control = self.slider_controls.get(self.active_slider_id)
        if control is None:
            self.active_slider_id = None
            return

        if key in ("Left", "Down"):
            delta = -control.fine_step
        elif key in ("Right", "Up"):
            delta = control.fine_step
        else:
            return

        current = control.var.get()
        updated = max(control.min_v, min(control.max_v, current + delta))
        control.var.set(updated)
        control.on_change_cb("keyboard")

    def _on_low_changed(self, band: str) -> None:
        c = self.band_controls[band]
        low = c.low.get()
        high = c.high.get()
        if low > high:
            c.low.set(high)
        self._redraw_band(band)

    def _on_high_changed(self, band: str) -> None:
        c = self.band_controls[band]
        low = c.low.get()
        high = c.high.get()
        if high < low:
            c.high.set(low)
        self._redraw_band(band)

    def _scaler_for_band(self, band: str) -> BandScaler:
        c = self.band_controls[band]
        low = min(c.low.get(), c.high.get() - 0.1)
        high = max(c.high.get(), low + 0.1)
        asinh_a = 10 ** c.a_log10.get()
        return BandScaler(low_pct=low, high_pct=high, asinh_a=asinh_a)

    def _redraw_all_bands(self) -> None:
        for band in self.images:
            self._redraw_band(band)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _redraw_band(self, band: str) -> None:
        if band not in self.images:
            return

        ax = self.axes_by_band[band]
        scaler = self._scaler_for_band(band)
        scaled = scaler.apply(self.images[band])

        if band in self.artists:
            self.artists[band].set_data(scaled)
        else:
            self.artists[band] = ax.imshow(scaled, cmap="gray", origin="lower")
            ax.set_title(band)
            ax.axis("off")

        self.canvas.draw_idle()

    def _refresh_progress_text(self) -> None:
        total = len(self.sample_ids)
        index = self.idx + 1
        source_name = self.current_sample.source_path.name if self.current_sample else ""
        self.progress_var.set(f"Sample {index}/{total}: {source_name}")

    def _label_true(self) -> None:
        self._record_and_advance(self.true_label_text, True)

    def _label_false(self) -> None:
        self._record_and_advance(self.false_label_text, False)

    def _record_and_advance(self, response_text: str, response_bool: bool) -> None:
        if self.current_sample is None:
            return

        self.logger.append(
            sample_id=self.current_sample.sample_id,
            source_path=Path(self.current_sample.source_path),
            response_text=response_text,
            response_bool=response_bool,
            update_if_exists=not self.skip_previously_labeled,
        )
        self.idx += 1
        self._load_current_sample()

    def run(self) -> None:
        self.root.mainloop()
