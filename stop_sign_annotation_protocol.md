# Stop-Sign Behavior — Annotation Protocol (Dashcam-Only)

**Purpose**  
Provide a clear, lightweight guide so remote annotators can consistently label **driver compliance near stop signs** using *video only*. Output should be directly useful for training/QA and match the repo’s Part‑2 deliverables.

---

## 0) Scope, assumptions, and simplifications
- **Input:** Forward-facing dashcam clips under `videos/` (no IMU/metadata). Use only the video; do not use speed, GPS, or any external sensors.  
- **Annotator setting:** Remote; limited context; only what’s visible in the clip.  
- **Task focus:** The **first** stop-sign interaction in the clip unless otherwise instructed.  
- **Timing aid:** Use player timestamps or frames (e.g., 30 fps → 45 frames ≈ **1.5 s**).  
- **Simplifications:**  
  - No speedometer or GPS—judge motion from parallax/optical flow and fixed references (stop line, crosswalk, pole, cracks).  
  - If the stop line isn’t visible, use the sign pole/crosswalk/edge of intersection as a proxy.  
  - When uncertain between **Rolling_Stop** and **No_Stop**, choose **Rolling_Stop**. Use **Unclear** sparingly.

---

## 1) Primary label (pick exactly one)
1. **Full_Stop** — Wheels fully stop for **≥ 1.5 s** **before or at the stop line**; proceeds when safe.  
2. **Rolling_Stop** — Clear slowing, but **no full stop** (**pause < 0.5 s**) before the line.  
3. **No_Stop** — No meaningful deceleration before crossing the line.  
4. **Creeping_Stop** — Multiple short inching pauses (creep‑pause‑creep) with **cumulative pause ≥ 1.5 s** before entering cross traffic.  
5. **Unclear** — Sign/line not visible or motion cannot be judged reliably.

> If multiple stop signs appear in quick succession, annotate the **first encountered** unless the instructions for a specific batch say otherwise.

---

## 2) Secondary tags (zero or more)
Use tags to capture context that affects visibility/behavior and helps model training/triage:
- **Sign_Visibility:** {Clear, Partial, Poor}  
- **Obstruction:** {None, Vehicle, Pedestrian, Object, Weather, Glare}  
- **Traffic_Crossing:** {None, Vehicle, Pedestrian, Bicycle}  
- **Lighting:** {Day, Dusk, Night}  
- **Camera_Motion:** {Stable, Shake}  
- **Weather:** {Clear, Rain, Snow, Fog}

---

## 3) Decision rubric (how to judge)
- **Stopping is about wheel motion**: rely on parallax vs. fixed ground cues (stop line, crack, manhole, crosswalk). A stop means the reference point **holds still** for the pause duration.  
- **Where it happens matters**: stopping **after** crossing the line generally **does not** count as *Full_Stop* → prefer *Rolling_Stop* unless clearly yielding before entering.  
- **Creeping vs. Full_Stop**: a single continuous pause ≥ 1.5 s → *Full_Stop*; multiple micro‑pauses accumulating ≥ 1.5 s → *Creeping_Stop*.  
- **If line not visible**: approximate using crosswalk or sign pole; still require the same pause durations.  
- **Edge occlusions**: if a clean full stop is visible earlier then view becomes blocked, *Full_Stop* is valid.

---

## 4) Quality controls (lightweight)
- **Dual‑pass audit (~10%)**: resolve discrepancies by consensus; capture reasons.  
- **Gold clips (~20)** across classes; quick weekly calibrations for annotators.  
- **Unclear rate**: if an annotator’s *Unclear* > 10%, sample their work for coaching.  
- **Agreement**: track inter‑annotator agreement (e.g., Cohen’s κ) on the audit set.

---

## 5) Output format
CSV, one row per clip:
```
clip_id,primary_label,sign_visibility,obstruction,traffic_crossing,lighting,camera_motion,weather,notes
abc123,Rolling_Stop,Clear,None,None,Day,Stable,Clear,"Slows, no full stop before line"
```

---

## 6) How this labeling supports training an effective model
- **Primary labels** map directly to driver‑compliance classes for a stop‑behavior model; *Full_Stop* (positive compliance) vs. *Rolling/No_Stop* (non‑compliance) support both multi‑class and binary objectives.  
- **Timing thresholds** (≥ 1.5 s, < 0.5 s) make supervision less ambiguous and more reproducible across annotators.  
- **Secondary tags** surface confounders (visibility, obstruction, lighting), enabling:  
  1) robust evaluation by strata,  
  2) targeted data curation/collection, and  
  3) optional multi‑task or conditioning signals.  
- **QC practices** (golds/audit) reduce label drift and maintain consistency as the dataset scales.

---

## 7) Quick checklist for annotators
1. Identify the first stop sign / approach elements (sign, stop line, crosswalk).  
2. Watch the approach and decide: **Full_Stop / Rolling_Stop / No_Stop / Creeping_Stop / Unclear**.  
3. Add relevant **secondary tags**.  
4. Write a short **note** if anything unusual impacted your decision.

