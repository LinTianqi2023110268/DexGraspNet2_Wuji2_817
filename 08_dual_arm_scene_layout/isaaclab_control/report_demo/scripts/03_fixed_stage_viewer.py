#!/usr/bin/env python3
"""Show one report stage in one fixed, consistently sized Tk window.

The controller terminates this process before opening the next stage, so report
figures never overlap each other.  Multiple paths are tiled inside this one
window rather than creating multiple top-level windows.
"""

from __future__ import annotations

import argparse
import tkinter as tk
from pathlib import Path

from PIL import Image, ImageTk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--title", default="DGN2 Wuji2 Report Evidence")
    parser.add_argument("--geometry", default="1040x840+10+20")
    return parser.parse_args()


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    result = image.copy()
    result.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), "#202020")
    offset = ((width - result.width) // 2, (height - result.height) // 2)
    canvas.paste(result, offset)
    return canvas


def main() -> None:
    args = parse_args()
    for path in args.images:
        if not path.is_file():
            raise FileNotFoundError(path)

    root = tk.Tk()
    root.title(args.title)
    root.geometry(args.geometry)
    root.minsize(800, 620)
    root.configure(bg="#202020")

    title = tk.Label(
        root,
        text=args.title,
        font=("Sans", 15, "bold"),
        fg="white",
        bg="#202020",
        pady=8,
    )
    title.pack(fill="x")
    body = tk.Frame(root, bg="#202020")
    body.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    count = len(args.images)
    columns = 1 if count == 1 else 2
    rows = (count + columns - 1) // columns
    for row in range(rows):
        body.grid_rowconfigure(row, weight=1)
    for column in range(columns):
        body.grid_columnconfigure(column, weight=1)

    # Keep references alive for the lifetime of the Tk window.
    root._report_photos = []  # type: ignore[attr-defined]

    def redraw(_event=None) -> None:
        for child in body.winfo_children():
            child.destroy()
        root._report_photos.clear()  # type: ignore[attr-defined]
        cell_width = max((body.winfo_width() - 10 * columns) // columns, 200)
        cell_height = max((body.winfo_height() - 10 * rows) // rows, 200)
        for index, path in enumerate(args.images):
            image = Image.open(path).convert("RGB")
            photo = ImageTk.PhotoImage(fit_image(image, cell_width, cell_height))
            root._report_photos.append(photo)  # type: ignore[attr-defined]
            label = tk.Label(body, image=photo, bg="#202020", bd=0)
            label.grid(
                row=index // columns,
                column=index % columns,
                sticky="nsew",
                padx=5,
                pady=5,
            )

    # Windows use a fixed report layout, so one render is sufficient and avoids
    # Configure-event feedback while Tk is laying out the image labels.
    root.after(80, redraw)
    root.mainloop()


if __name__ == "__main__":
    main()
