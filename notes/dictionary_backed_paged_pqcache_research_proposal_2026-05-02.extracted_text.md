# dictionary_backed_paged_pqcache_research_proposal_2026-05-02.pptx

## Slide 1
- REVISED RESEARCH PROPOSAL
- Dictionary-Backed Paged PQCache
- Binary recall plus page-local PQ, centroid interning, variance-bounded merge, sparse V pages, and completion fallback.
- Novelty
- Use page-local PQ when pages seal, then merge reusable centroids into a shared dictionary.
- Claim
- Tame decode drift without paying linear codebook metadata per page.
- Plan
- Prove on CPU, expose GPU pain, then map surviving primitives to hardware.
- May 2, 2026
- Updated to include page-wise PQ and centroid interning as the proposed novelty.
- 1

## Slide 2
- ONE-LINE THESIS
- The algorithm should fit locally, but score globally.
- Global PQ controls metadata but drifts; page-local PQ controls error but grows metadata. Centroid interning is the bridge.
- This is the revised core proposal.
- 2
- Global PQ
- small metadata, but centroid variance can grow as decode distribution drifts
- Naive page PQ
- good local fit, but every sealed page carries its own centroid table
- Dictionary-backed page PQ
- train locally, merge reusable centroids, keep private/outlier centroids only when needed
- active exact page -> page PQ on seal -> centroid dictionary merge -> compact page aliases
- Research question: can merge guarantees preserve attention ranking while making codebook metadata grow sublinearly with sequence length?

## Slide 3
- FAILURE MODE
- Long decode makes global centroids stale, but online splitting is the wrong hot-path fix.
- The risk is not just reconstruction error. It is top-k membership flipping because centroid residuals widen over generated tokens.
- The proposed fix moves adaptation to page-seal time.
- 3
- Issue
- Why it matters
- Avoid
- Centroid variance grows
- members assigned to one centroid become less coherent as generated tokens drift
- blindly trust old PQ scores
- Top-k recall degrades
- small logit errors change candidate membership when many keys are near-tied
- optimize only MSE
- Online split is messy
- old token codes need versioning; selector hardware becomes dynamic
- split in decode critical path
- Page-seal adaptation
- train after page contents are known, then compress with measured error stats
- none; this is the proposed control point

## Slide 4
- ALGORITHM LIFECYCLE
- Exact active pages become dictionary-backed PQ pages only after they seal.
- This keeps recent tokens accurate and moves clustering/merging off the latency-critical decode path.
- Page size is a tunable parameter, e.g. 128-512 tokens.
- 4
- Active page
- int8/fp16 K while filling
- Seal
- page reaches fixed size
- Train local PQ
- fit page distribution
- Intern
- merge/reuse centroids in dictionary
- Encode
- aliases: local id -> global id
- Decode later
- binary recall + dictionary PQ
- Critical-path decode keeps using the current exact page plus previously sealed dictionary pages.

## Slide 5
- CENTROID INTERNING
- Centroid interning gives page-local fit without page-local metadata growth.
- Each subspace gets a shared dictionary; a page stores only compact aliases into the dictionary plus private centroids when needed.
- Interning is per PQ subspace, not one global table for the whole K vector.
- 5
- Local page codebook
- centroids trained on the sealed page for each PQ subspace
- Shared dictionary
- reused centroids with count, SSE, radius, and logit-error metadata
- Page alias table
- small local code maps to global centroid id or private centroid id
- token_code[j] = local_id; page_alias[j][local_id] -> dictionary_centroid_id

## Slide 6
- MERGE GUARANTEE
- A centroid merge should be allowed only if variance and attention-logit error stay bounded.
- Centroid distance alone is not enough; merging must protect both reconstruction and candidate ranking.
- CPU experiments should test simple bounds before learned policies.
- 6
- Variance test
- SSE_merged / n_merged <= tau for this layer/head/subspace
- Radius test
- ||mu_new - mu_old|| + r_new <= allowed_radius
- Logit test
- ||q|| * ||mu_new - mu_old|| <= epsilon for calibration queries
- SSE_merge = SSE_old + SSE_new + n_old*n_new/(n_old+n_new) * ||mu_old - mu_new||^2
- If the merge fails: allocate a private centroid, route high-error tokens to an outlier tier, or increase candidate quota for that page.

## Slide 7
- MEMORY FORMAT
- The page stores aliases, not a full repeated codebook.
- The dictionary can grow sublinearly when centroid structure repeats across pages; hard pages still keep private entries.
- This is the main metadata-scaling argument.
- 7
- Object
- Stored fields
- Growth
- Purpose
- Dictionary
- centroid vector, count, SSE, radius
- sublinear if centroids repeat
- global scoring table
- Page alias
- local id -> global/private id
- linear but tiny
- small token codes
- Token code
- local id per subspace
- linear
- compressed K representation
- Outlier tier
- int8/full K or private centroid
- sparse
- protect rare keys
- Binary sidecar
- b_i hash/sign code
- linear, tiny
- candidate recall

## Slide 8
- SCORE PATH
- Dictionary-backed PQ lets pages fit locally while queries score against shared centroids.
- The alias indirection is the cost; codebook reuse and comparable LUT scoring are the payoff.
- On GPU this indirection may be painful; on custom hardware it is a natural SRAM lookup.
- 8
- q_j
- subspace query
- LUT
- q_j dot dictionary_j
- token local id
- small per-page code
- alias lookup
- local id -> centroid id
- score add
- LUT[centroid id]
- top-K
- reranked candidates
- If a page uses private centroids, the page alias points to a private dictionary bank with the same scoring interface.

## Slide 9
- SELECTOR INTERACTION
- Binary recall should be uncertainty-aware when PQ pages have different error bounds.
- High-variance pages should not silently lose important tokens because their approximate scores are less trusted.
- This is where the merge statistics feed the candidate engine.
- 9
- Binary recall
- cheap full-cache candidate generation using b_i
- Error-aware admission
- boost quota or use score upper bound for high-radius pages
- PQ rerank
- dictionary LUT + norm + page bias; optional exact-ish K for uncertain rows
- score_upper_i = score_hat_i + ||q|| * radius(page_or_centroid_i)
- Candidate quota is a quality control knob, not just a latency knob.

## Slide 10
- V AND FALLBACK
- K compression is the search problem; V compression is the bandwidth problem.
- The new centroid scheme should compose with quantized V pages and a completion fallback for diffuse attention.
- Do not let K-side novelty hide the V-fetch cost.
- 10
- K side
- binary sidecar + dictionary-backed PQ + optional outlier K
- V side
- selected int4/int8 V pages with scales and token maps
- Fallback
- page/segment summary adds missing softmax mass
- output = (selected_num + completion_num) / (selected_den + completion_den)

## Slide 11
- CPU PLAN
- CPU implementation should prove whether centroid interning preserves attention ranking.
- The first prototype should be exact, inspectable, and instrumented; speed is secondary.
- Metrics should be per layer, KV head, page age, and task type.
- 11
- Build
- Measurement
- Gate
- Trace collector
- Q/K/V, exact logits, exact top-k, output vectors
- reproducible baseline
- Page PQ encoder
- local codebooks, private centroids, alias tables
- fit quality by page
- Interning policy
- merge rate, dictionary growth, variance/logit bounds
- sublinear metadata without recall cliff
- Selector simulator
- binary recall, PQ rerank, error-aware quota
- top-k recall and output error
- V/fallback simulator
- quantized V bytes and completion recovery
- quality per fetched byte

## Slide 12
- ABLATION GRID
- The ablation grid should decide if this is novelty or just complexity.
- The critical proof is a three-way comparison: global PQ, naive page PQ, and dictionary-backed page PQ.
- Report both mean and tail behavior; rare retrieval failures matter.
- 12
- Ablation
- Knob
- Quality metric
- Memory metric
- Decision
- Page size
- 128 / 256 / 512
- active-page error, seal frequency
- alias and page overhead
- choose latency/fit balance
- Merge threshold
- tau, radius, epsilon
- top-k recall tail
- dictionary growth
- metadata-quality curve
- Private centroids
- none / limited / unlimited
- outlier recovery
- private bank bytes
- cap hard pages
- Alias width
- 4 / 5 / 6 / 8 bits
- local centroid pressure
- token code bytes
- min useful alias
- Quota policy
- fixed / score upper / entropy
- miss rate
- candidate count
- quality guardrail

## Slide 13
- GPU PLAN
- GPU benchmarking should test whether alias indirection erases the byte savings.
- The GPU prototype is where the proposal becomes systems-real: bit packing, dictionary LUTs, page aliases, top-k, and sparse gathers all matter.
- A negative GPU result can still strengthen the hardware argument.
- 13
- Kernel 1
- binary scan and top-M candidate generation
- Kernel 2
- alias lookup plus dictionary PQ LUT rerank
- Kernel 3
- sparse V page fetch, dequant, selected attention
- benchmark = quality + tokens/sec + HBM bytes + alias stalls + gather efficiency
- Compare against dense/GQA, KV quantization, global PQCache-like selection, and naive page-local PQ.

## Slide 14
- HARDWARE SPLIT
- The chip story separates the seal engine from the decode engine.
- Page-local training/merging can run off the hot path; decode needs predictable SRAM lookups, popcount, top-k, LUT scoring, and page fetch.
- This separation keeps adaptive PQ out of the per-token critical path.
- 14
- Seal engine
- local clustering, merge tests, alias table build, outlier routing
- Decode engine
- binary scan, score upper, dictionary LUT, top-k, V page scheduler
- sealed page metadata
- On-chip state
- dictionary SRAM, page alias SRAM/cache, top-k buffers, V reorder buffer

## Slide 15
- HARDWARE DATAPATH
- Custom hardware should make the dictionary indirection regular.
- This is the part GPUs may dislike and an ASIC can directly support with small SRAM banks and fixed reduction trees.
- The datapath still supports PQCache-like global mode by bypassing page aliases.
- 15
- b_i SRAM
- binary sidecar
- Popcount
- coarse recall
- Alias SRAM
- local -> centroid
- Dict SRAM
- centroid LUT
- Top-K
- rerank
- V pages
- fetch/dequant
- Fallback
- summary
- Hardware primitive list: bit-sliced scan, variance-aware admission, alias lookup, dictionary LUT reduction, page-aware sparse V fetch.

## Slide 16
- EVALUATION GATES
- The project should advance only if metadata growth and attention error both stay controlled.
- This slide is the advisor-facing test plan for whether the novelty is real.
- Hard gates can be tuned after a first model/trace target is chosen.
- 16
- Gate
- Pass signal
- Fail signal
- Action
- Dictionary growth
- centroids grow sublinear with pages
- near page-local codebook cost
- tighten merge or abandon
- Attention fidelity
- top-k recall/output error matches page-local PQ
- merge causes tail misses
- use private centroids/outliers
- GPU reality
- byte savings survive alias/top-k/gather overhead
- latency worse than KV quant
- hardware-only argument
- Hardware case
- bottlenecks map to compact SRAM/datapath
- needs broad vector DB engine
- narrow scope
- Quality fallback
- completion recovers diffuse-head failures
- fallback hides selector weakness
- revisit selector

## Slide 17
- ADVISOR PITCH
- The novelty is a compression-index co-design, not another sparse-attention heuristic.
- The proposal can be explained as a way to get page-local adaptation with global-style metadata reuse.
- This is the short version for a meeting.
- 17
- Problem
- Global PQ can drift during long decode; naive page PQ has linear codebook overhead.
- Idea
- Keep the active page exact, train page PQ on seal, then intern/merge centroids into a shared dictionary.
- Guarantee
- Merge only under variance and attention-logit bounds; otherwise use private centroids or outlier storage.
- Payoff
- Compressed K becomes searchable, locally adaptive, and metadata-efficient enough to motivate hardware.

## Slide 18
- MILESTONES
- The implementation path should disprove weak variants before GPU or hardware work expands.
- Start with the algorithmic claim: dictionary-backed page PQ should match page-local quality at much lower metadata cost.
- Timeline is a research sequence, not a calendar promise.
- 18
- 1
- CPU trace + dense baseline
- exact Q/K/V, logits, top-k, output error
- 2
- Page PQ + interning
- merge stats, dictionary growth, private centroid use
- 3
- Selector + fallback
- binary recall, error-aware quota, V bytes, completion
- 4
- GPU benchmark
- alias/LUT/top-k/gather bottleneck profile
- 5
- Hardware model
- SRAM, traffic, latency, energy per generated token

## Slide 19
- SOURCES
- Primary references for the revised proposal.
- The centroid-intering mechanism is the proposed new idea; these sources motivate the components around it.
- URLs are plain text for easy copy/paste.
- 19
- PQCache
- https://arxiv.org/abs/2407.12820
- Self-Indexing KVCache
- https://arxiv.org/abs/2603.14224
- DASH-KV
- https://arxiv.org/abs/2604.19351
- Top-K + Completion
- https://arxiv.org/abs/2604.05438
- ShadowKV
- https://arxiv.org/abs/2410.21465
- ParisKV
- https://arxiv.org/abs/2602.07721
- Proposal-specific terms to use with advisor: paged PQ, centroid interning, dictionary-backed PQ, variance/logit-bounded merge, page alias table.
