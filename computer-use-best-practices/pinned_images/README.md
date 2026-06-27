# pinned_images

Drop reference images here (`.png`, `.jpg`, `.jpeg`, `.gif`, `.webp`), then run:

```bash
python -m computer_use "your task" --image-dir pinned_images
```

Every image in this folder is **pinned** in the model's context for the whole
run — never pruned — so its guidance stays available even late in a long task.
This is the folder-based shortcut for `--image a.png --image b.png` (and it
combines with `--image`).

Dropped images are gitignored; this folder and README stay tracked.
