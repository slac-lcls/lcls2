# Interactive psana event workflow

Open **[index.html](index.html)** in a browser. The file includes its model, CSS and JavaScript; it works from `file://`, without a server, repository checkout, web fonts, CDN or network access. Share that one file. The page's **Download HTML** button saves another self-contained copy. Source and issue links require internet.

Status: **implemented-reference**. Reviewed source: **`410ebba1ce4d8b2a9d8d7fab56f6dc1fa26a6ef8`** on **2026-10-15**, branch **`features/psana2-gpu`**. Source paths are recorded in `workflow-model.json` under `baseline.source_paths`.

The reconstruction stack, including PR #151, is merged into that branch. Its `psana/` source is byte-identical to the previously inspected `e2cdad6e18fd9263d7adfeed0ce58d5ad074e070`. The call paths were reviewed on October 13 and carried forward after that comparison. This is a reference to the inspected implementation, not a claim about master or another checkout. Proposals and limitations remain explicitly labeled.

## Read the map

- **Overview** follows setup, SMD reading, event building and BD processing to public Event delivery.
- **CPU path** explains SMD offsets, bigdata reads, datagrams and ordered transitions.
- **GPU path** expands host orchestration, descriptors, input ownership, slot execution, requested constants, external tasks and host publication.
- **Show serial roles** explains how the same responsibilities are carried out locally. It does not create fictional SMD0/EB/BD processes in serial mode.
- Click a diagram component or its keyboard-accessible list link to inspect purpose, inputs/outputs, class/script relationships, ownership and commit-pinned sources.
- Each component's **Class and script relationships** section includes call-path trees. Setup and processing are separate, with MPI/serial alternatives, iterator handoffs, conditional branches and CUDA launches labeled. Click a method to open its definition at the inspected commit; collapse a phase using its heading. The device-input panel traces reader initialization, lazy file/buffer creation, bulk/per-datagram reads, completion and parsing from the public event iterator.
- **Follow event T** advances a conceptual event through its representations. CPU view omits GPU-only stages. The tour is explanatory; its steps are not measured timings or a literal serial call trace.
- Connection colors and line styles distinguish iterator/call handoff, MPI/local data transfer, association, I/O submission, CUDA dependencies and ownership. Connection numbers map to **Show connections**. Routes have no timing meaning.
- The transition example and buffer-lifetime reference are separate from the event-flow diagram.

Component routes are stable, for example `index.html#/gpu/input`. The serial toggle is retained in the route as `?serial=1`. On narrow screens the map scrolls horizontally while text wraps. A no-JavaScript component reference is embedded as a fallback.

## What is current and what is proposed

Current source is authoritative. The model explicitly distinguishes SMD chunks, EB communication batches, complete-event execution subbatches and input read groups. GPU BD orchestration is a CPU process. GPUBAT1 contains L1 event/read metadata; transitions remain in the CPU/SMD packet. The EB-local `step_batch` maintains history and is not an additional transmitted packet.

The viewer calls out these OPEN follow-ups, checked on 2026-10-15:

- [#153](https://github.com/slac-lcls/lcls2/issues/153): ordered GPU transition handling and working step iteration. The current GPU transition scan precedes this packet's L1 submission; an earlier drain cannot protect L1 events not submitted yet. CPU behavior and the proposed GPU contract are shown separately.
- [#168](https://github.com/slac-lcls/lcls2/issues/168): restore CUDA-IPC sharing of requested constants. Current device constant copies are per BD. CPU host calibration sharing is a separate mechanism.
- [#170](https://github.com/slac-lcls/lcls2/issues/170): missing GPU input behavior and CPU compatibility require an explicit policy.

This is an architecture introduction, not a complete bug inventory or a support guarantee. Offline experiment/run processing is the initial scope. Direct single-file, shared-memory, DRP, live processing, integrating-detector specifics, every detector layout, and downstream small-data/SRV internals remain future documentation work. The source has been inspected but no psana/GPU/MPI/data runtime tests were run for this documentation task. No data-filesystem access is necessary to maintain this prototype.

## Maintain and generate

All workflow prose, source anchors, relationships, tour steps and map positions live in **`workflow-model.json`**. The original renderer is in `viewer.js`, layout in `viewer.css`, and the page shell in `template.html`. The model is intentionally a smaller schema than Cong's whole-DAQ viewer; it is not interchangeable with `daq-model.json`.

Schema version 2 adds `call_paths` to every component. A phase has `title`, `when` and `roots`; each call has `label`, `note`, optional `children`, and a `source` anchor. Non-repository user/library calls and explanatory handoffs instead declare `kind`. Children represent calls or explicitly labeled iterator/launch handoffs; siblings follow source order unless marked as alternatives. These are manually reviewed paths for the box's responsibility, not exhaustive runtime traces of every helper or error branch. The same trees appear in the no-JavaScript reference.

From the repository root, using Python 3.8 or newer:

```bash
python3 psana/docs/architecture/event-flow/build.py
python3 psana/docs/architecture/event-flow/build.py --check
```

No Python packages are needed for generation. The pinned source commit must be available in the local Git repository. `build.py` validates unique node/edge IDs, required content, view membership, routes, tour coverage, existing source files and source-anchor needles. It derives GitHub line links from `git show <baseline>:<path>`; it does not silently repoint them at the current working tree. `--check` also requires deterministic generated HTML to match the checked-in file and disallows remote runtime assets.

Source anchor `needle` is a literal source line fragment. If the same method name appears more than once, the first match is intentional only when targeting a base-class definition (for example `Run.events`). Use a unique fragment for other cases. Source links are evidence entry points; the validator checks the anchor exists, not that every English claim follows from it. Changes need human/source review.

Call-tree method anchors also use `scope`, the exact containing class declaration, so methods such as `__init__` resolve to the correct class. The builder rejects missing or ambiguous anchors within that scope. When updating a path, trace the actual callers as well as the definitions: an existing method alone does not prove a call relationship.

To add a component: give it a stable ID, role and serial explanation, semantic input/output lists, relationship steps, ownership text and source references; place it in each relevant view; add typed edges and tour coverage. Keep reader-facing descriptions in the model rather than adding psana facts to the viewer. Explicit route bend points avoid connections crossing component boxes.

## Browser validation

Optional browser checks need Playwright and its Chromium browser in a separate environment:

```bash
python3 -m venv /tmp/psana-workflow-browser-venv
/tmp/psana-workflow-browser-venv/bin/pip install playwright
PLAYWRIGHT_BROWSERS_PATH=/tmp/psana-workflow-browsers /tmp/psana-workflow-browser-venv/bin/playwright install chromium
PLAYWRIGHT_BROWSERS_PATH=/tmp/psana-workflow-browsers /tmp/psana-workflow-browser-venv/bin/python psana/docs/architecture/event-flow/browser_check.py
```

The checker opens the generated file with browser networking disabled, exercises all component routes and views, source permalinks, serial toggle/reload, the complete event tour, keyboard activation, connection table, downloadable HTML and mobile layout, and checks for JavaScript errors and external requests. It also opens the downloaded file with scripting disabled to verify the static component reference. It stores screenshots and `result.json` under `/tmp/psana-workflow-browser-check` by default; override with `--artifacts <directory>`.

No GPU, experimental data or psana installation is used. Browser installation is development tooling, not a runtime requirement for readers. Checks were performed with Chromium 153; Safari/Firefox and screen-reader testing remain unperformed.

## Branch and publication

These docs target `features/psana2-gpu`. The generated `index.html` is committed so reviewers can download and open it before a hosted site exists. Its Download HTML button saves a portable `psana-event-flow.html` copy. The documentation-home link uses the feature branch's GitHub page and therefore works from a downloaded copy when online.

Keep the model and renderer as the editing sources; regenerate the HTML after changes. Review changed source paths and callers before advancing the inspected commit. Do not replace that revision automatically with HEAD or label proposals as implemented because an issue closed.

Publishing with GitHub Pages is a separate follow-up. A future site build can copy this standalone viewer unchanged alongside rendered Markdown guides. The documentation check workflow does not deploy a site.

## Reference and attribution

Cong Wang's [interactive LCLS-II DAQ documentation](https://carbonscott.github.io/lcls2/dev/design/) supplied the approach: a single content model, typed relationships, clickable component details and a “follow one event” tour. Inspected its [model, schema, editing guide and viewer](https://github.com/carbonscott/lcls2/tree/e49563c40bbbbbf0ef96ed3520079c42388a14a2/docs/design) on the `better-docs` branch. Its viewer loads a separate JSON file and web fonts.

This prototype uses an original, smaller viewer with embedded JSON/assets for portable downloads. It does not copy Cong's JavaScript, CSS or DAQ prose. It omits the whole-DAQ viewer's nested submaps, relationship matrices and sequence-chart rendering for this first twelve-component psana map. The model/renderer separation is retained so those features can be added later.

Existing source documentation used as supporting material:

- [GPU design](../../../psana/gpu/docs/design.md)
- [External user kernels](../../../psana/gpu/docs/user_kernels.md)
- [GPU limitations](../../../psana/gpu/docs/limitations.md)
- [GPU README](../../../psana/gpu/README.md)

Repository redistribution terms are in [LICENSE.md](../../../../LICENSE.md), and are included in the standalone HTML.
