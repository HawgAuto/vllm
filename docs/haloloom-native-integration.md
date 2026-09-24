# HaloLoom native serving integration

This branch retains the upstream engine at `21ee743f474a35a056e8d5ade2c9aa744e726e6e` and its native Qwen conditional-model/MTP implementation. It carries the existing Quark checkpoint/SSD-PLE integration plus:

- Source-distributed mixed-Quark and dense-MTP consumer modules. These used to exist only as installed runtime additions; the wheel now declares them explicitly.
- The exact bounded BF16 SSD-PLE capture guard used by the supported serving path.
- Opt-in TraceLens execution metadata for steady profiling and graph capture, including observed tensor geometry and distinct rank/capture descriptors. Missing or ambiguous provenance does not admit exact-kernel replay.
- Separate regression tests for the TraceLens execution contract. The original profiler test file is unchanged.

The mixed-precision adapter is deliberately narrow: explicit hash-bound policies, supported physical wire layouts and the declared expert geometry are required. It is not an arbitrary-model mixed-precision format converter and never silently chooses a BF16 expansion or another backend. Model weights, policy artifacts, private captures and campaign scripts are not shipped here.

The vLLM native binary provenance and Python integration provenance must both be recorded when building a composite wheel. Python-only integration does not justify relabelling native extensions as freshly rebuilt. The HaloLoom release recipe records the compiled donor wheel, source revision and every replaced Python member separately.

Source-level CPU tests are not a GPU performance or model-quality claim. A rebuilt image needs its own installed-file inventory and bounded real runtime validation before a public release pointer is advanced.
