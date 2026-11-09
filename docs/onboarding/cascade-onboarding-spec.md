# Cascade onboarding — build spec

> **As built (2026-11-09):** the layout, motion and tone below were kept, but the steps were cut to
> what Cascade really does. There are 3 steps: pick a trace (the two OPT-125m samples, or profile your
> own on Modal), pick Accel-Sim GPU configs with replicas (1 to 64), then watch the run fork a Sail
> sandbox per simulation and compare to the real run, using the samples' recorded numbers. The trace
> connectors, fleet, routers, engine settings and replay numbers in this spec are not in Cascade and
> were left out. Colors and type follow the app's tokens instead of the ones listed here.

Build the first-run onboarding for Cascade as a 4-step flow modeled on the attached reference frames (`onboarding-ref-*.png`). The reference is a product called Detent; copy its **layout, motion and tone**, not its content or colors. Everything below is the Cascade version.

## The pattern (from the reference)

Full-viewport, two columns.

**Left column (≈46% width), the form.** Top bar: small logo + product name at left; a 4-segment progress bar centered (filled segments in the accent color, unfilled in a light grey), "Step N of 4" to its right. Content is vertically centered. A headline in two tones: the first sentence in near-black, the rest in mid-grey, all in one large light-weight sans (≈36px). Then the step's controls. Then a single dark pill-shaped primary button with a `→` glyph, and a quiet text "Back" / "Skip for now" link beside it.

**Right column (≈54%), the live preview.** A card with a soft radial gradient background (reference uses lavender→peach→blue; we use ours below). Small label top-left: "Your workspace, so far". Inside it, a browser-window mock: three dots, a URL pill in monospace, a sidebar on the left, a main area titled "Activity". The preview **updates in real time as the user acts on the left**: each action appends a row to the Activity timeline (icon + "You did X · detail") and adds an item to the sidebar. Pending items render faded (≈40% opacity) and fade in to full when confirmed. The bottom-left of the sidebar shows avatars + "Just you" / "4 people".

**Motion.** Everything is 200–300 ms ease-out. Connect a tool → the card's subtitle flips to "Connected · <account>", a green check appears, the new row fades in on the right. Step changes slide the left content out/in horizontally; the right card persists and only its contents change. Step 4 shows a live checklist: completed items with green checks, the active item with a spinner and a sub-line that updates ("Checking the refund policy in Notion · step 2 of 4"), queued items as grey dots. The button reads "Working…" and is disabled until done, then becomes "Open Cascade →".

**Tone.** Short sentences. Headline pattern: `<Imperative.> <One sentence of why.>` Body copy ≤ 2 lines. No exclamation marks.

## Cascade's four steps

### Step 1 · Connect your trace source
Headline: **Connect a trace source.** Cascade replays what you really ran. Point it at a run, or load the sample.
Controls: a 2×3 grid of source cards, each with an icon, name, and status line ("Not connected" → "Connected · <detail>"), a `+` or green check at the right:
- Modal (`Connected · cascade-demo`)
- vLLM endpoint (`Connected · localhost:8000`)
- AIPerf log (`Connected · aiperf_a100x4_c10.json`)
- OpenAI-compatible URL
- Sample trace (`Loaded · 1,200 requests · 94.2 s`)
- Upload JSON/Parquet
Right preview, sidebar section "Traces": each connected source appears as an item. Activity rows: "You created **cascade-demo**", "You connected **Modal** · cascade-demo", "You loaded **sample trace** · 1,200 requests".

### Step 2 · Pick your fleet
Headline: **Pick a fleet.** Start from the GPUs you have, then add the ones you don't.
Controls: a list of GPU rows (icon, name, memory, "Measured" or "Estimated" tag), each with `+ Add` → a count stepper. Rows: A100 80 GB (Measured), H100 SXM (Measured), L40S 48 GB (Estimated), H200 (Estimated), TPU v6e (Estimated, hatched). A line under the list: "Estimated profiles are marked and shown hatched in every chart."
Primary button label changes with selection: "Continue with 4 GPUs →".
Right preview, sidebar section "Fleet": one line per GPU type with a count. Activity rows: "You added **2× H100** · measured", "You added **2× L40S** · estimated". Sidebar footer: "4 GPUs".

### Step 3 · Choose a policy
Headline: **Choose a router.** This decides which GPU each request lands on. You can change it any time.
Controls: three selectable cards, each with a title, one line of description, and a small inline icon strip showing which fleet types it suits:
- Round-robin · "Simple and fair. Ignores load."
- Least-loaded · "Sends each request to the shortest queue."
- KV-aware · "Prefers the GPU that already holds the prompt's cache."
Below, a collapsed "Engine settings" disclosure: max requests 64, batch size 32, tensor parallel 1.
Right preview, sidebar section "Policies": the chosen router. Activity row: "You chose **least-loaded** · engine defaults".

### Step 4 · Run the first replay
Headline: **Try it out.** Cascade is replaying your trace on the fleet you built.
Controls: the live checklist:
- ✓ Trace loaded · 1,200 requests
- ✓ Fleet built · 2× H100, 2× L40S
- ✓ Router set · least-loaded
- ◌ Replaying · sub-line cycles through "Replaying request 412 of 1,200 · 00:41", "Detecting hotspots", "Comparing to the real run"
- then ✓ Replay done · "p99 TTFT 1,105 ms · 2 hotspots · twin error ±6.8%"
Button: "Working…" (disabled) → "Open Cascade →".
Right preview: the Activity timeline gets a final row "**Replay** finished · 2 hotspots on gpu2 and gpu1", then an inset card appears under it showing a 4-lane mini heatmap (one lane per GPU, 24 cells each) with two cells outlined in orange, and a line "twin error ±6.8% vs real run" in green. Sidebar footer: "1 replay".

### Done state
Headline: **You're set up.** Cascade ran its first replay. Change the fleet and run it again whenever you like.
Checklist all green. A green banner: "Replayed 1,200 requests · p99 −18% vs 4× A100". Button "Open Cascade →".

## Cascade visual tokens (replace the reference's)

- Background: Paper `#F4F3EE`. Text: Ink `#0B0F14`; secondary `#6B7580`.
- Accent for progress bar, checks, "Connected" and measured values: Trace green `#2EE59D` (use `#1E7F86` for small text on Paper, the bright green fails contrast on light backgrounds).
- Simulated / estimated values: Twin violet `#8B7CFF`. Hotspots and warnings: Hotspot orange `#FF6B35`.
- Primary button: Ink background, `#E8EAED` text, 999px radius.
- Preview card gradient: `radial-gradient(60% 80% at 20% 90%, #FFB36B33, transparent), radial-gradient(50% 70% at 85% 10%, #8B7CFF40, transparent), #F4F3EE`. Card and browser mock radius 14px, hairline border `#DAD8D0`.
- Type: Geist for UI, Geist Mono for the URL pill, counts, GPU names, request ids and every number. Headline 36px/1.15, weight 400; first sentence weight 500 in Ink.
- Icons: inline stroke SVG, 1.5px, no emoji.

## Implementation notes

- One route, `/onboarding`, state in a single store: `{ step, sources[], fleet[], router, engine, replay: { status, progress, result } }`. The right column is a pure render of that store, which is how it stays in sync.
- Progress bar segments, checks and timeline rows all animate with the same 240 ms ease-out; respect `prefers-reduced-motion`.
- Step 4's "replay" is a scripted timer against the sample trace for now (2.5 s total); wire it to the real twin later behind the same `replay.status` field.
- The 4-lane mini heatmap in the preview is the same component as the dashboard's lanes, at small size. Build it once.
- Keyboard: Enter advances, Esc goes back, every card is a real `<button>`.
- Works down to 1024px wide by stacking the preview under the form; below 768px hide the preview and show a one-line summary chip instead.
