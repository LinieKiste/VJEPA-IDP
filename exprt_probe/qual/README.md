# eXprt qualitative action-labelling — annotate, then probe

Goal: see how off-the-shelf **V-JEPA 2 + EK100** labels a few hand-picked eXprt tea actions
(the same zero-shot pipeline used on EgoPER tea). You annotate start/stop for interesting
actions; I run the probe over those segments and print/visualise its predictions.

## 1. Watch the videos
In `exprt_probe/qual/watch/`:
- `Normal.mp4`  — standard tea prep (pour water, add tea bag, stir, milk/sugar)
- `Spueli.mp4`  — anomaly: washing-up liquid added
- `glass_and_fork.mp4` — anomaly: tea in a glass, stirred with a fork
- `2tb_2stir.mp4` — anomaly: two tea bags + two separate stirs

Playback is real-time at **20 Hz** (the on-disk frame rate; originally 23.96 Hz, downsampled),
so **your player's clock = the timestamp I need**. Read start/stop seconds straight off the player.

## 2. Fill in `annotations.csv`
Columns: `video, start_s, stop_s, action, notes`
- `video` = one of `Normal | Spueli | glass_and_fork | 2tb_2stir`
- `start_s`, `stop_s` = seconds (decimals fine, e.g. `45.5`)
- `action` = your short label ("pour water", "add soap", "stir with fork", …)
- A few interesting actions per video is plenty (5–8 total is great). Add rows as needed.
- Delete the EXAMPLE row.

## 3. Tell me you're done
I run `exprt_probe/ek100_label.py`, which for each segment samples a clip, runs the EK100 head,
and prints its top verb / noun / action prediction next to your label — plus a montage PNG in
`exprt_probe/qual/figures/`. (Caveat, as on tea: EPIC vocabulary has no tea-specific nouns, so
verbs are the fair signal; predictions are anticipatory ~1 s.)
