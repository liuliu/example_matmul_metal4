# Repo Notes

## Device Conv2D Tensor Op

- On the M5 iPad, the `convolution2d` tensor-op path is not correct when writing directly to a regular destination tensor, even though the same direct path passed on macOS.
- The first fully passing device configuration is `cooperative-half-offset-simdgroups`.
- The working shader pattern is:
  - keep `set_offsets(...)` for valid-region extraction
  - get a cooperative half destination tensor
  - zero-initialize the cooperative destination
  - run `conv2d_op.run(activation, weights, cOutput)`
  - store with `cOutput.store(output)`
- The device debug sweep is implemented in `ConvolutionHarness.debugTensorOpVariants()`. Use it before changing the Conv2D tensor-op path again.
- A good device sanity check is that the app report shows:
  - `tensorOpVariantDebug.firstPassingVariantName == "cooperative-half-offset-simdgroups"`
  - `validation.allPassed == true`
- Current correctness-first performance baseline on the M5 iPad for `N=1, H=W=128, C=32, O=32, KH=KW=3` is about `49.4 GFLOP/s` for the corrected tensor-op path.
- For tiled Conv2D on device, the edge tiles should follow the `matmul.swift` pattern:
  - keep a static convolution descriptor for the full tile
  - use static `slice<...>` for interior tiles
  - use dynamic `slice(...)` only on edge tiles
  - do not switch the convolution descriptor itself to dynamic output sizes for edge handling
- On the M5 iPad, the single-dispatch tiled Conv2D path with `tile=8x8`, `execution_simdgroups<4>`, `N=1`, `H=W=256`, `C=O=512`, `KH=KW=3` runs at about `9.71 TFLOP/s` GPU time. The corresponding `<1>` run is about `6.85 TFLOP/s`.

## Device Conv3D Tensor Op

- The current Conv3D tensor-op path is implemented by reusing the 2D tensor op over spatial slices and launching once per `(outputDepth, kernelDepth)` pair.
- `convolution2d::set_offsets(...)` behaves like a sampling translation. The local probe in `Sources/convolution/convolution.swift` showed that for a `3x3` kernel, increasing the offset by `+1` shifts the sampled activation by `+1` in that axis. For a full-output same-padding `3x3` Conv2D, `offset(0, 0)` matched the CPU zero-padding reference.
- For `KD=3`, the host launches 3 kernels per output-depth slice:
  - the first launch uses `convolution2d_descriptor::mode::multiply`
  - later launches use `convolution2d_descriptor::mode::multiply_accumulate`
- On device, accumulation should use the cooperative destination path too:
  - for the first launch, zero-initialize the cooperative destination
  - for later launches, call `cOutput.load(output)` before `run(...)`
  - then store back with `cOutput.store(output)`
- The first device validation sweep for this Conv3D path passed with zero mismatches on 4 cases, including `strideZ=2` and full `dilationZ/Y/X=2`.
- On the M5 iPad, the first large Conv3D tensor-op profile with `execution_simdgroups<4>`, `tile=8x8`, `N=1`, `D=3`, `H=W=256`, `C=O=512`, `KD=KH=KW=3` reached about `9.11 TFLOP/s` GPU time.
- For the first `padLeft=padRight=1`, `padTop=padBottom=0` Conv3D experiment on device, a pure full-width tensor-op dispatch was almost correct but still failed on the left edge. The working device pattern is:
  - run the full padded tensor-op dispatch with `set_offsets(...)`
  - overwrite the leftmost column with a specialized tensor-op correction pass
  - do not use `cOutput.store(output)` for that correction pass; instead, manually scatter each valid cooperative-tensor element back to `output_buf`
- With that manual-store correction, the iPad validation for left/right padding passed with zero mismatches on 3 cases, including `batch=2` and `strideZ=2`.
- A later attempt to remove the special correction launch and rely on a single padded dispatch with custom writeback did not validate on the M5 iPad. Two variants were tried:
  - full dispatch with manual flat scatter replacing `cOutput.store(output)`
  - full dispatch with padding encoded into `set_offsets(...)` plus manual flat scatter
- Neither variant reproduced `cOutput.store(output)` semantics on device. Also, `metal::tensor` does not expose `map_iterator`, so there is no direct tensor-space equivalent of the cooperative-tensor iterator remap trick used between cooperative tensors in `attention.swift`.
- The current weight-permutation benchmark assumes source weights in `OIDHW` and permutes them to `DHWIO`, which is the layout consumed by the Conv3D tensor-op path.
- On the M5 iPad, the `OIDHW -> DHWIO` permutation kernel validated with zero mismatches.
- For `N=1`, `D=3`, `H=W=256`, `C=O=512`, `KD=KH=KW=3`, `tile=8x8`, `execution_simdgroups<4>`:
  - direct Conv3D tensor-op ran at about `9.18 TFLOP/s` GPU time
  - permutation alone ran at about `3.42 ms` GPU time, or `8.28 GB/s`
  - `permute + Conv3D` ran at about `9.07 TFLOP/s` GPU time
- On that shape, adding the permutation as measured GPU overhead reduced effective Conv3D throughput by about `1.1%`.

## Local Readable Padding Probes

- For debugging padding, reduce the problem to a readable 2D tensor-op kernel first. The standalone local probe is in `Sources/convolution2d_padding/convolution2d_padding.swift`.
- The local 2D result is:
  - `set_offsets(...)` is sufficient to express zero padding
  - the tiled edge bug was not `set_offsets(...)` itself
  - the fix is to clamp the activation slice origin on edge tiles, then add the clamp delta back into `set_offsets(...)`
  - use explicit cooperative-tensor scatter for writeback
- With that pattern, the readable local 2D probe passed all 3 left/right-padding cases for `tile=4x4` with zero mismatches on the M3 Pro.
- The same pattern now also validates in the readable local Conv3D probe at `Sources/convolution3d_padding/convolution3d_padding.swift`:
  - clamp the spatial activation slice on edge tiles
  - compensate the clamp with adjusted `set_offsets(...)`
  - for accumulation, manually load from `output_buf` into the cooperative tensor and manually scatter back after `run(...)`
- After that change, the readable local Conv3D padding probe passed both the `8x8` oversized-tile run and the `4x4` tiled run with zero mismatches on all 3 validation cases.

## `set_offsets(...)` Padding Handoff

- `convolution2d::set_offsets(...)` can implement zero padding. The local probe in `Sources/convolution/convolution.swift` showed that increasing an offset by `+1` shifts sampling by `+1` in that axis.
- For a spatial tensor-op tile, the base padding offset is:
  - `baseOffsetX = ((kernelWidth - 1) * dilationX / 2) - paddingLeft`
  - `baseOffsetY = ((kernelHeight - 1) * dilationY / 2) - paddingTop`
- That base offset is enough when the activation slice origin is the natural unclamped one.
- The tiled edge bug came from the activation slice origin, not from `set_offsets(...)` itself.
- The working edge rule is:
  - `unclampedInputOriginX = outputOriginX * strideX`
  - `unclampedInputOriginY = outputOriginY * strideY`
  - `clampedInputOriginX = min(unclampedInputOriginX, max(0, inputWidth - inputTileWidth))`
  - `clampedInputOriginY = min(unclampedInputOriginY, max(0, inputHeight - inputTileHeight))`
  - `adjustedOffsetX = baseOffsetX + (unclampedInputOriginX - clampedInputOriginX)`
  - `adjustedOffsetY = baseOffsetY + (unclampedInputOriginY - clampedInputOriginY)`
- In words: clamp the activation slice so edge tiles still see the needed halo, then add the clamp delta back into `set_offsets(...)` so sampling still lands on the correct global input positions.
- For Conv3D implemented as repeated spatial Conv2D launches:
  - use the same clamped-slice plus adjusted-offset rule on every `(outputDepth, kernelDepth)` launch
  - for the first depth slice, zero-init the cooperative destination
  - for later depth slices, manually load from `output_buf` into the cooperative tensor and manually scatter back after `run(...)`
- The single-dispatch readable version validated on both machines:
  - local M3 Pro: `Sources/convolution2d_padding/convolution2d_padding.swift` and `Sources/convolution3d_padding/convolution3d_padding.swift` both passed all tiled validation cases
  - iPad M5: the shared device path in `Sources/ConvolutionShared/ConvolutionShared.swift` passed all 3 Conv3D left/right padding validation cases with `maxAbsoluteError = 0.0` and `mismatches = 0`
