#import "@preview/lovelace:0.3.1": *


#set document(title: "Specification QLL in MPC")

#let sint = math.sans[int]

// Bit diagram of a SecFloat float (z, s, e, m): bit indices, the encoded bits
// coloured per field with the hex word, and the value decomposition below.
// e is encoded in two's complement on p + 2 bits, m unsigned on q + 1 bits.
#let float-bits(z, s, e, m, p: 8, q: 23) = {
  let pe = p + 2
  let qm = q + 1
  let n = 2 + pe + qm
  assert(z in (0, 1) and s in (0, 1), message: "z and s must be bits")
  assert(
    -calc.pow(2, p - 1) + 1 <= e and e <= calc.pow(2, p - 1),
    message: "exponent out of range",
  )
  assert(
    m == 0 or (calc.pow(2, q) <= m and m < calc.pow(2, q + 1)),
    message: "mantissa not normalised",
  )

  let colors = (
    z: rgb("#f3e3a0"),
    s: rgb("#c5d3f2"),
    e: rgb("#c6ebc2"),
    m: rgb("#f4c4c4"),
  )
  let field(i) = if i == 0 { "z" } else if i == 1 { "s" } else if i < 2 + pe { "e" } else { "m" }
  let bits(x, w) = range(w).rev().map(i => calc.rem(calc.quo(x, calc.pow(2, i)), 2))
  let e-enc = if e < 0 { e + calc.pow(2, pe) } else { e }
  let all = (z, s) + bits(e-enc, pe) + bits(m, qm)

  let word = all.fold(0, (acc, b) => acc * 2 + b)
  let hex = upper(str(word, base: 16))
  let hex = "0" * (calc.quo(n + 3, 4) - hex.len()) + hex

  // Round to 8 significant digits.
  let fmt(x) = {
    if x == 0 { return "0" }
    let d = 7 - calc.floor(calc.log(calc.abs(x), base: 10))
    str(calc.round(x, digits: calc.max(d, 0)))
  }

  let index-row = range(n).map(i => {
    let pos = n - i
    grid.cell(
      fill: if calc.odd(calc.quo(pos - 1, 4)) { luma(225) } else { none },
      text(size: 6pt, str(pos)),
    )
  })
  let bit-row = all
    .enumerate()
    .map(((i, b)) => grid.cell(
      fill: colors.at(field(i)),
      stroke: 0.5pt,
      text(size: 8pt, str(b)),
    ))
  let label-row = (
    grid.cell(text(size: 7pt)[$z$]),
    grid.cell(text(size: 7pt)[$s$]),
    grid.cell(colspan: pe, text(size: 7pt)[$e$ (#pe bits)]),
    grid.cell(colspan: qm, text(size: 7pt)[$m$ (#qm bits)]),
  )

  let mant = m / calc.pow(2, q)
  let value = (1 - z) * (1 - 2 * s) * calc.pow(2.0, e) * mant
  let sign = if s == 1 [$-$] else [$+$]
  let result = if z == 1 [$#sign 0$] else if e == calc.pow(2, p - 1) [$#sign oo$] else [#fmt(value)]
  let tag(c, body) = box(fill: c, inset: (x: 3pt, y: 2pt), outset: (y: 1pt), body)

  block(breakable: false, width: 100%, {
    grid(
      columns: (1fr, auto),
      column-gutter: 8pt,
      align: horizon,
      grid(
        columns: (1fr,) * n,
        align: center + horizon,
        inset: (y: 2.5pt),
        ..index-row,
        ..bit-row,
        ..label-row,
      ),
      [$=$ #raw("0x" + hex)],
    )
    v(2pt)
    tag(colors.z)[$#(1 - z)$]
    [ $times$ ]
    tag(colors.s)[$#sign 1$]
    [ $times$ ]
    tag(colors.e)[$2^(#e)$]
    [ $times$ ]
    tag(colors.m)[#fmt(mant)]
    [ $=$ #result]
  })
}

#align(center)[
  #text(size: 20pt, weight: "bold")[Specification QLL in MPC]
]



= Float structure

Following SecFloat @secfloat[Section V], a floating-point number is
parameterised by $p, q in ZZ^+$ ($p = 8$, $q = 23$ for IEEE single precision
@IEEE-754) and is a tuple $alpha = (z, s, e, m)$:

#figure(
  table(
    columns: 3,
    align: (center, left, left),
    table.header([*Field*], [*Domain*], [*Meaning*]),
    [$z$], [${0, 1}$], [Zero bit, set iff $alpha = 0$.],
    [$s$], [${0, 1}$], [Sign bit, set only if $alpha <= 0$.],
    [$e$], [${0, 1}^(p+2)$, values in $[-2^(p-1) + 1, 2^(p-1)]$], [*Unbiased* signed exponent.],
    [$m$],
    [${0, 1}^(q+1)$, values in $[2^q, 2^(q+1) - 1] union {0}$],
    [Normalised unsigned fixed-point mantissa with scale $q$.],
  ),
  caption: [Components of a float $alpha = (z, s, e, m)$.],
)

The tuple represents the real number
$ alpha = (1 - z) dot (1 - 2s) dot 2^(sint_(p+2) (e)) dot m / 2^q $
where $sint_(p+2) (e)$ is the two's-complement value of $e$.

A nonzero real $x$ is encoded as
$
  e = floor(log_2 abs(x)) quad m = round(abs(x) / 2^e dot 2^q)
$
(if $m$ overflows to $2^(q+1)$, halve it and increment $e$).

For a float 32 value $r$ $"ULP"(r) = 2^(e-23)$ where $e$ is the binary exponent of $r$ i.e. the integer such that $2^e <= |r| <= 2^(e+1)$.
So the value $2^(e-23)$ the least significant mantissa bit for a normalized float32 with exponent $e$.

$
  "ulp(r)" = 2^(e-23)
$

== Examples

// EzPC/SCI/src/FloatingPoint/floating-point.cpp:250-320
$pi approx 3.1415927$: $e = floor(log_2 pi) = 1$ and $m = round(pi / 2 dot 2^23) = 13176795$, so
$ alpha = (0, 0, 1, 13176795) $

#float-bits(0, 0, 1, 13176795)

=== Special values $plus.minus oo$

$ (-1)^s oo = (0, s, 2^(p-1), 2^q) $

#float-bits(0, 0, 128, calc.pow(2, 23))
#float-bits(0, 1, 128, calc.pow(2, 23))

=== Special values $plus.minus 0$

$ (-1)^s 0 = (1, s, -2^(p-1) + 1, 0) $

#float-bits(1, 0, -127, 0)
#float-bits(1, 1, -127, 0)

`NaN`s and denormals are not supported.

= QLL Operators

== Extended Primitive Operations


#figure(
  box(stroke: 1pt, inset: 8pt, width: 100%)[
    #align(center)[#smallcaps[*Functionality* $cal(F)^23_"IntParity" (beta)$]]
    #v(0.6em)
    #align(left)[
      #pseudocode-list(
        indentation: 2em,
        stroke: none,
      )[
        + $e = 1{β.e < 0} ? 0 : min(β.e, q+1)$
        + $S = L_("pow"2) e$
        + $s = β.m *_(q+1) S$
        + $"isInt" = (¬c ∧ 1{s mod 2^23 = 0}) || β.z$
        + $"isOdd" = "isInt" ∧ 1{s ≥ 2^23}$
        + *Return* isInt, isOdd
      ]
    ]
  ],
  caption: [Integer parity test for $beta$, used by $cal(F)^(8,23)_"FPpow"$.\ The function $L_("pow"2)$ is computed through the LUT operator.],
)

#grid(
  columns: (1fr, 1fr),
  column-gutter: 16pt,
  align: top,
  block(inset: 8pt, width: 100%)[
    _Option 1:_
    $
        a & = 2^log_2(a) \
      a^n & = (2^log_2(a))^n \
          & = 2^(n dot log_2(a))
    $
  ],
  block(inset: 8pt, width: 100%)[
    _Option 2:_
    $
                 a & = e^ln(a) \
               a^n & = (e^ln(a))^n \
                   & = e^(n dot ln(a)) \
      "with" ln(x) & = log_2 x dot ln 2
    $
  ],
)

#figure(
  box(stroke: 1pt, inset: 8pt, width: 100%)[
    #align(center)[#smallcaps[*Functionality* $cal(F)^(8,23)_"FPpow" (alpha, beta)$]]
    #v(0.6em)
    #align(left)[
      #pseudocode-list(
        indentation: 2em,
        stroke: none,
      )[
        + (isInt, isOdd) = $cal(F)_"IntParity"(beta)$
        + s = α.s & isOdd
        + *if* 1{β.z = 1} *then* *Return* $"Float"_(p,q)(1)$
        + *else if* 1{α.z = 1} *then*
          + *if* 1{β.s = 0} *then* *Return* (1, s, 1 − 2#super[p−1], 0) *else* *Return* (0, s, 2#super[p−1], 2#super[q])
        + *else*
          + $alpha' = (0, 0, α.e, α.m)$
          + δ = β $⊡_(p,q)$ $cal(F)^(8,23)_"FPlog2" (alpha')$
          + γ = $cal(F)^(8,23)_"FPexp2"(δ)$
          + *Return* (γ.z, s, γ.e, γ.m)
      ]
    ]
  ],
  caption: [Floating-Point power: $alpha^beta$],
)

_Error propagation:_

We want to compute the error propagation of the expression $hat(y) = "FPexp2"("FPmul"(b, "FPlog2"(a)))$ and $y = 2^t = a^b$ the expected real result.
We recall that $log_2(a) < 1 "ULP"$, $b * - < 0.5 "ULP"$ and finally $2^- < 1 "ULP"$.
The ULP for the FPLog2 function can be computed as follows $"ulp"(log_2(x)) = 2^(floor(log_2 |log_2(x)|) - 23)$. Let $ell = log_2(a)$.
The `FPlog2` functionality has an error of $<1$ ULP, so if $epsilon_log$ denotes the error of `FPlog2`,
$
  abs(epsilon_log) < 2^(E_ell - 23) quad "with" quad E_ell = floor(log_2(abs(ell)))
$
After multiplication by $b$, this error becomes $b epsilon_log$, and the floating-point multiplication adds its own rounding error $epsilon_times$, with
$
  epsilon = b epsilon_log + epsilon_times quad "where" epsilon_times < 0.5
$
Now let $t = b ell$, since `FPexp2` receives $t + epsilon$ instead of $t$, the value before the final floating-point rounding is
$
  2^(t + epsilon) = 2^t 2^epsilon = y 2^epsilon.
$
Hence, the absolute difference $y 2^epsilon - y = y(2^epsilon -1) approx ln(2)|epsilon|$.
Thus, including the final rounding error $epsilon_exp$ introduced by `FPexp2`:
$
  abs(hat(y) - y) & <= y abs(2^(b epsilon_log + epsilon_times) - 1) + abs(epsilon_exp) \
                  & < y abs(2^epsilon - 1) + "ULP"(2^(t + (b epsilon_log + epsilon_times))) \
                  & approx y ln(2) abs(epsilon) + "ULP"(2^(t + (b epsilon_log + epsilon_times)))
$

== Logical disjunction $or^rho$

$
  ⟦ a or^rho b ⟧ = a plus.o^(p) b & = (a^(p) + b^(p))^(frac(1, p)) \
                                  & = M (1 + (m/M)^rho)^(1/p)
$
where
$
  M = max(a, b) quad m = min(a, b)
$

#figure(
  box(stroke: 1pt, inset: 8pt, width: 100%)[
    #align(center)[#smallcaps[*Functionality* $cal(F)^(8,23)_"FPPMEAN" (alpha, beta, rho)$]]
    #v(0.6em)
    #align(left)[
      #pseudocode-list(
        indentation: 2em,
        stroke: none,
      )[
        + m = min(α1, α2)
        + M = max(α1, α2)
        + *if* 1{M.z = 1} *then*
          + *Return* $"Float"_(8,32)(0)$
        + *else*
          + r = m $⊘_(8,23)$ M
          + β = $cal(F)^(8,23)_"FPpow" (r, rho)$
          + γ = $"Float"_(8,23)(1) ⊞_(8,23) β$
          + δ = $"Float"_(8,23)(1) ⊘_(8,23) rho$
          + η = $cal(F)^(8,23)_"FPpow" (γ, δ)$
          + *Return* M $⊗_(8,23)$ η
      ]
    ]
  ],
  caption: [Floating-Point $rho$-mean equivalent to $or^rho$],
)

_Correctness_: TODO

== Logical conjunction $and^rho$

$
  ⟦ a and^p b ⟧ = a plus.o^(-rho) b & = (a^(-rho) + b^(-rho))^(-frac(1, p)) \
                                    & =m (1 + (m/N)^rho)^(-1/rho)
$

#figure(
  box(stroke: 1pt, inset: 8pt, width: 100%)[
    #align(center)[#smallcaps[*Functionality* $cal(F)^(8,23)_"FPHPMEAN" (alpha, beta, rho)$]]
    #v(0.6em)
    #align(left)[
      #pseudocode-list(
        indentation: 2em,
        stroke: none,
      )[
        + m = min(α1, α2)
        + M = max(α1, α2)
        + *if* 1{m.z = 1} *then*
          + *Return* $"Float"_(8,23)(0)$
        + *else*
          + r = m $⊘_(8,23)$ M
          + β = $cal(F)^(8,23)_"FPpow"(r, rho)$
          + γ = $"Float"_(8,23)(1) ⊞_(8,23) β$
          + δ = $"Float"_(8,23)(-1) ⊘_(8,23) rho$
          + η = $cal(F)^(8,23)_"FPpow"(γ, δ)$
          + *Return* m $⊗_(8,23)$ η
      ]
    ]
  ],
  caption: [Floating-Point Harmonic $rho$-mean equivalent to $and^rho$],
)

_Correctness_: TODO

#bibliography("refs.bib")
