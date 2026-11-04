# Extended non-negative reals `ℝ⨂` and the QLL tensor `⨂`

Adds a first-class type for the **multiplicative extended non-negative reals**

```
ℝ⨂ := [0, ∞]
```

and its first Quantitative Linear Logic operator

```
Tensor : [0, ∞] × [0, ∞] → [0, ∞]      written  ⨂
```

`⨂` agrees with ordinary multiplication except on the one case IEEE-754 leaves
indeterminate: QLL **defines** `0 ⨂ ∞ = ∞ ⨂ 0 = 0`, where `0.0 * INFINITY` is NaN.

```
0 ⨂ ∞ = 0        ∞ ⨂ 0 = 0        a ⨂ ∞ = ∞  (a > 0)
∞ ⨂ a = ∞ (a>0)  ∞ ⨂ ∞ = ∞        otherwise  a ⨂ b = a · b
```

`ℝ⨁ := [-∞, ∞]` is deliberately untouched: `ℝ⨁` still lexes to `TINT32` and `⨁`
to `SUM`, exactly as before.

## Starting point

Commit `4cbf5b7` had added `RealAdd`/`RealMul` to `base_type` and `Tensor` to
`binop`, plus branches in every backend's `match`. That made the OCaml compile
but did **not** create a path from source to those constructors:

- `lexer.mll` mapped `ℝ⨂ → TUINT32` and `⨂ → MUL`, so both aliased to existing
  integer/multiply tokens;
- `parser.mly` had no `TENSOR`/`TREALMUL` token, no `base_type` production and
  no `const` constructor.

`Ast.RealMul` and `Ast.Tensor` were therefore unreachable. Two latent crashes
also sat on the path: `typeof_expr` (`tcenv.ml`) and `infer_binop_label`
(`infer.ml`) both ended in catch-all `failwith`s that `Tensor` fell into, and
`typeof_expr` is called unconditionally by the default-on TAC pass.

## Representation

`ℝ⨂` is a **non-negative IEEE-754 float including the native `+∞`**. No
`(value, is_infinity)` pair is needed. Each backend uses its natural width.

| Backend                                 | `ℝ⨂` representation                              | `⨂` compiles to                                                   |
| --------------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------- |
| **CPPFLOAT**                      | `double` (binary64), `∞` = `(double)INFINITY` | `qll_tensor` (guarded)                                             |
| **SECFLOAT**                      | secret:`FPArray`; public: `float`                | secret:`__fp_op->mul` (**no guard**); public: `qll_tensor` |
| **EMP**                           | secret:`Float`; public: `float`                  | secret + public:`qll_tensor` (guarded)                             |
| ABY, CPPRING, SCI, PORTHOS, FSS, OBLIVC | *unsupported*                                      | explicit`failwith` pointing at CPPFLOAT                            |

### Why SECFLOAT needs no guard

SecFloat's `FPArray` is **not** a packed IEEE word. It is a decomposed
`(sign, zero-bit, mantissa, exponent)` secret sharing
(`SCI/src/FloatingPoint/floating-point.h`) which

- encodes `∞` as `e = e_max()+1, m = 0`, and has **no NaN state at all**;
- round-trips a native `INFINITY` through `FPOp::input` / `get_native_type`;
- sets `ret_z = OR(x_z, y_z)` in `FPOp::mul`
  (`floating-point.cpp:959`), so a zero operand forces a zero result whatever
  the other operand is.

That last line *is* the QLL rule. `⨂` on secret `ℝ⨂` therefore costs **zero
extra gates** over ordinary multiplication.

### Why EMP does need one

emp's `Float` is a packed 32-bit IEEE word (`std::array<Bit,32>`). Its Bristol
multiplier returns `inf` for `0 * inf` (measured against emp-tool, not IEEE's
NaN — wrong for QLL either way), so `⨂` gets an explicit zero guard. The zero
test scans exponent+mantissa (bits 0..30): **31 OR gates** instead of two
507-gate `Float::equal` circuits, and skipping the sign bit makes `-0.0` count
as zero too.

### Closure

The only NaN-producing IEEE multiply is `0 × ∞`, which every guard intercepts,
so `⨂` is **closed on `[0, ∞]`** — NaN can never enter an `ℝ⨂` value. Overflow
saturates to `+∞` (the top element); underflow collapses to `0`.

## Surface syntax

Literals follow the existing suffix convention (`2u`, `2L`, `2uL`): a literal's
base type is fixed by its lexical form, since `typeof_const` is a total function
from constructor to base type.

```
ℝ⨂_pl a = 2.0m ;      (* suffix m; `a` stays reserved for ℝ⨁ later *)
ℝ⨂_fl b = ∞ ;
ℝ⨂_fl c = a ⨂ b ;
```

**Labels.** `ℝ⨂` mirrors `float`: `pl` (public) and `fl` (Baba) are accepted,
`al`/`bl` are type errors, and an unlabelled `ℝ⨂` defaults to `Secret Baba`.

**Not permitted:** ordinary `+ - * /` on `ℝ⨂` (they would emit raw C++ `*` and
reintroduce `0 * ∞ = NaN`); mixing `ℝ⨂` with `float`; negative `ℝ⨂` literals
(there is no negation production for `REALMUL`/`INFTY`).

**Permitted:** comparisons `> < >= <= ==`, since IEEE ordering on `[0, ∞]` is
exactly the extended-real ordering.

## Changes by file

**Front end**

- `ast.ml` — `RealMulC of float` constant; `typeof_const`; `expr_to_string`;
  `is_realmul_bt`, `is_baba_bt` helpers.
- `lexer.mll` — `ℝ⨂ → TREALMUL` (was `TUINT32`), `⨂ → TENSOR` (was `MUL`),
  new `∞ → INFTY` and `flt "m" → REALMUL`, plus `cvt_realmul_literal`.
- `parser.mly` — `TREALMUL`/`TENSOR`/`INFTY`/`REALMUL` tokens; `base_type` and
  `const` productions; `TENSOR` at the `MUL DIV MOD` precedence level.

**Type checking**

- `tcenv.ml` — `Tensor` joined into `typeof_expr`'s arithmetic group (this is
  what unblocks the TAC pass).
- `infer.ml` — `Tensor` label rule (`Secret Baba`); `ℝ⨂` default label;
  Baba-aware comparisons via `is_baba_bt`.
- `tc.ml` — `Tensor : ℝ⨂ × ℝ⨂ → ℝ⨂` rule; `check_expected_realmul_typ` and
  `check_expected_ordered_typ`; `ℝ⨂` excluded from `check_expected_numeric_typ`;
  well-formedness restricted to `pl`/`fl`.

In `tcenv.ml`, `infer.ml` and `tc.ml` the trailing `| _ -> failwith` on the
now-exhaustive `binop` matches was **removed**, so warning 8 (an error under the
Makefile's `-w -11@8`) catches future binop additions at compile time instead of
deferring to a runtime crash.

**Back ends**

- `codegenlib.ml` — `o_realmul` (binary64) and `o_realmul_f` (binary32)
  constant emitters; `Float.to_string infinity` is `"inf"`, not a C++ literal.
- `codegencppfloat.ml` — `double`; `qll_tensor` + `qll_read_realmul` prelude;
  `Tensor` call site; removed the phantom `RealMulArray`/`make_vector_realmul`
  secret paths (those types do not exist in `secfloat.h`).
- `codegensecfloat.ml` — `ℝ⨂` aliased onto `FPArray` / `__fp_op` /
  `__public_float_to_baba` / `make_vector_float` / `get_native_type<float>`;
  `Tensor → "mul"`; public `qll_tensor` + `qll_read_realmul` prelude.
- `codegenemp.ml` — `ℝ⨂` aliased onto emp `Float`; guarded `qll_tensor`
  (`float` and `Float` overloads) + `qll_is_zero` + `qll_read_realmul` prelude.
- `codegencppring.ml`, `codegenfss.ml`, `codegenoblivc.ml`, `codegenporthos.ml`,
  `codegensci.ml` — explicit, backend-named rejections for `ℝ⨂` types,
  constants and `Tensor`, replacing generic `"impossible branch"` failures and
  invented type names.

**Tests**

- `test_suite_float/tensor_realmul.ezpc` — public `ℝ⨂` (CPPFLOAT).
- `test_suite_float/tensor_realmul_secret.ezpc` — secret `ℝ⨂` (SECFLOAT, EMP).
- `test_suite/dot_product.ezpc` — **restored** to its pre-`4cbf5b7` `int32`
  form. Its `ℝ⨂`/`⨁` rewrite only ever compiled because the unicode aliased to
  integer tokens; it no longer typechecks and did not match its precompiled
  reference.

## Incidental fixes

1. **ABY `PutMULGate` regression.** `codegen.ml` emitted `acirc->(a, b)` for
   every secret multiply (`| Mul -> aux "" false`). Restored. This is the only
   codegen change this work makes to the pre-existing test suite.
2. **Silent `∞` loss on input.** `istream::operator>>` for float/double is not
   portable for infinity: libstdc++ accepts `"inf"`, libc++ rejects it and
   (per C++11) stores `0`, turning an infinite input into a zero one with no
   error. All three supporting backends now emit `qll_read_realmul`, which
   accepts `inf` / `INF` / `infinity` / `∞`.

## Known limitations

- **`output(ALL, x > y)` does not typecheck under `fl`** — `typeof_expr` reports
  `baba bool` while `tc_expr` reports `boolean bool`. This is **pre-existing and
  not specific to `ℝ⨂`**: it fails identically for `float_fl`. Bind the
  comparison to a `bool_bl` first, as the existing float tests do.
- **Negative input is not validated.** Literals cannot be negative, but a value
  read from stdin is not range-checked against `[0, ∞]`.
- **Ring backends.** `Z_2^l` has no float representation, so `ℝ⨂` is rejected
  there. The intended encoding is fixed-point with a reserved top sentinel
  (`x ↦ round(x·2^s)`, `∞ ↦ TOP = 2^l−1`, multiply truncating and saturating
  upward to `TOP`), which mirrors the IEEE overflow-to-`∞` behaviour. Not built.
- **`ℝ⨁`** is untouched; no operators beyond `⨂` are implemented.

## Verification

```bash
cd EzPC/EzPC && make

# public ℝ⨂, end to end
./ezpc.sh test_suite_float/tensor_realmul.ezpc --codegen CPPFLOAT --bitlen 32 --o_prefix /tmp/t
g++ -std=c++17 -o /tmp/t /tmp/t0.cpp && /tmp/t     # 0 0 inf inf inf 4 0 1, no NaN

# secret ℝ⨂ codegen
./ezpc.sh test_suite_float/tensor_realmul_secret.ezpc --codegen SECFLOAT --bitlen 32 --o_prefix /tmp/sf
./ezpc.sh test_suite_float/tensor_realmul_secret.ezpc --codegen EMP      --bitlen 32 --o_prefix /tmp/emp

make runtest
```

What was actually checked in this environment:

- **CPPFLOAT** — compiled and run; all six semantic cases plus left
  associativity, arrays, `input`, and `m`-suffix disambiguation.
- **EMP** — the compiler's *emitted* `qll_tensor`/`qll_is_zero` were extracted
  verbatim and run against real emp-tool circuits under `PlainProt`: all 10 QLL
  cases pass, no NaN. The network/`sh2pc` layer was not exercised
  (`emp-sh2pc` is not installed) but does not affect circuit semantics.
- **SECFLOAT** — generated code inspected and its emitted *public* helpers
  compiled and run. The secret path rests on reading `FPOp::mul`; SCI is not
  built here, so it was not executed.
- **Regression** — compared against a scratch worktree of the pre-change tree:
  across all 11 existing test programs the only differences are the
  `PutMULGate` restoration and `dot_product` (whose source was restored).
  The residual diff against `test_suite/precompiled_output/` is pre-existing
  line-number noise in comments (the local `cpp` emits 4 extra lines for the
  library include) and reproduces identically on the unmodified tree.
