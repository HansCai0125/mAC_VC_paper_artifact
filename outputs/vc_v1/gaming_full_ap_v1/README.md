# Full diagnostic mAP anti-gaming supplement

This output fills every missing AP50, AP75, mAP50:95, and AR50:95 value in
the completed formal anti-gaming grid. It reuses the frozen post-NMS
prediction caches and does not train a model or run model inference.

The mAP values are diagnostic measurements at the fixed confidence threshold
`theta=0.25` after each controlled box manipulation. They support
continuous anti-gaming response curves, but they are distinct from official
dataset AP computed from the detector's complete confidence ranking.

- Source formal protocol: `89ca7213897d7c99bfd5550a1b40f4b40d36e9ad8c9a86bfbabad7fee35504f7`
- Completed source runs: 18
- Per-run attack rows: 540
- Aggregate rows: 180
- Output protocol: `3ae6341a53580779ce3e4de4c351147e7800e055f0da9eada768142ddd8aa167`
