#import "@preview/lovelace:0.3.1": *
#import "@preview/fletcher:0.5.8": diagram, node, edge
#import "@preview/cetz:0.4.2"

#set document(title: "Implementation of QLL in MPC")
// #set par.line(numbering: "1")
#set page(numbering: "1") // "1" instead for plain numbers
#set heading(numbering: "1.1")
#set quote(block: true)

#let sint = math.sans[int]

// Code listings: line numbers in a grey gutter, light background, numbered "Listing N"
#show figure.where(kind: raw): set block(breakable: true)
#let listing(code, caption: none) = figure(
  kind: raw, supplement: [Listing], caption: caption,
  {
    show raw.line: it => {
      box(width: 1.6em, align(right, text(fill: luma(150), str(it.number))))
      h(0.8em)
      it.body
    }
    set text(size: 8pt)
    set par(leading: 0.5em)
    block(fill: luma(247), inset: 8pt, radius: 3pt, width: 100%, align(left, code))
  },
)
#let output(code) = block(
  fill: luma(30), inset: 8pt, radius: 3pt, width: 100%,
  text(size: 8pt, fill: luma(235), code),
)

// highlighting colors
#let note-colors = (
  review:   rgb("#c5d3f2"),   // blue
  question: rgb("#f4e3a1"),   // yellow
  todo:     rgb("#f4c4c4"),   // red
  comment:  rgb("#f8d1a8"),   // orange
)
#let mark(kind, label, body) = highlight(
  fill: note-colors.at(kind), radius: 2pt, extent: 1pt,
)[*#label:* #body]

#let review(body)   = mark("review",   "Review", body)
#let question(body) = mark("question", "Question",      body)
#let todo(body)     = mark("todo",     "To-do",   body)
#let comment(body)  = mark("comment",  "Comment", body)

#let Q = question
#let R = review
#let C = comment
#let T = todo

#let legend = {
  let item(kind, label, desc) = box(inset: (right: 1em))[
    #box(fill: note-colors.at(kind), radius: 2pt, inset: (x: 3pt, y: 1pt))[*#label*] #desc
  ]
  block(inset: 6pt, stroke: 0.5pt + luma(180), radius: 3pt)[
    #item("review",   "Review", [\#R])
    #item("question", "Question", [\#Q])
    #item("todo",     "To-do", [\#T])
    #item("comment",  "Comment", [\#C])
  ]
}

#let fixed-bits(t) = {
  let (l, f, x) = t
  assert(type(l) == int and type(f) == int and 0 <= f and f <= l and l >= 1,
    message: "need integers 0 <= f <= l")
  let alpha = int(calc.round(x * calc.pow(2.0, f)))
  let lo = -calc.pow(2, l - 1)
  let hi = calc.pow(2, l - 1) - 1
  assert(lo <= alpha and alpha <= hi,
    message: "x * 2^f = " + str(alpha) + " outside [" + str(lo) + ", " + str(hi) + "]")

  let colors = (s: rgb("#c5d3f2"), i: rgb("#c6ebc2"), f: rgb("#f4c4c4"))
  let k = l - f // integer bits, sign bit included
  let field(j) = if j == 0 { "s" } else if j < k { "i" } else { "f" }
  let enc = if alpha < 0 { alpha + calc.pow(2, l) } else { alpha }
  let all = range(l).rev().map(j => calc.rem(calc.quo(enc, calc.pow(2, j)), 2))
  let hex = upper(str(enc, base: 16))
  let hex = "0" * (calc.quo(l + 3, 4) - hex.len()) + hex

  let fmt(v) = {
    if v == 0 { return "0" }
    let d = 7 - calc.floor(calc.log(calc.abs(v), base: 10))
    str(calc.round(v, digits: calc.max(d, 0)))
  }

  let index-row = range(l).map(j => {
    let pos = l - j - 1 // bit position, weight 2^(pos - f)
    grid.cell(
      fill: if calc.odd(calc.quo(pos, 4)) { luma(225) } else { none },
      text(size: 6pt, str(pos)),
    )
  })
  let bit-row = all.enumerate().map(((j, b)) => grid.cell(
    fill: colors.at(field(j)),
    stroke: 0.5pt,
    text(size: 8pt, str(b)),
  ))
  let label-row = ()
  // The sign bit belongs to the integer part, or to the fraction when f = l.
  let ni = calc.max(k - 1, 0)
  let nf = if k == 0 { f - 1 } else { f }
  label-row.push(grid.cell(text(size: 7pt)[$s$]))
  if ni > 0 { label-row.push(grid.cell(colspan: ni, text(size: 7pt)[int (#ni bits)])) }
  if nf > 0 { label-row.push(grid.cell(colspan: nf, text(size: 7pt)[frac (#nf bits)])) }

  let ip = calc.div-euclid(alpha, calc.pow(2, f))
  let fp = calc.rem-euclid(alpha, calc.pow(2, f))
  let xhat = alpha / calc.pow(2, f)
  let tag(c, body) = box(fill: c, inset: (x: 3pt, y: 2pt), outset: (y: 1pt), body)

  block(breakable: false, width: 100%, {
    grid(
      columns: (1fr, auto),
      column-gutter: 8pt,
      align: horizon,
      grid(
        columns: (1fr,) * l,
        align: center + horizon,
        inset: (y: 2.5pt),
        ..index-row,
        ..bit-row,
        ..label-row,
      ),
      [$=$ #raw("0x" + hex)],
    )
    v(2pt)
    [$alpha = round(#fmt(x) dot 2^#f) = #alpha$, #h(0.6em) $alpha / 2^#f = $ ]
    tag(colors.i)[$#ip$]
    if f > 0 {
      [ $+$ ]
      tag(colors.f)[$#fp / 2^#f$]
    }
    [ $= #fmt(xhat)$]
    if xhat != x [ #h(0.6em) (error $#fmt(xhat - x)$)]
  })
}

#align(center)[
  #text(size: 20pt, weight: "bold")[QLL in MPC]
]
// Headings tagged <todo>, <review>, <question> or <comment> get that colour,
// both in the text and in the table of contents.
#let tag-of(h) = if h.has("label") { str(h.label) } else { none }

#show heading: it => {
  let k = tag-of(it)
  if k != none and k in note-colors {
    highlight(fill: note-colors.at(k), radius: 2pt, extent: 1pt, it)
  } else { it }
}

#show outline.entry: it => {
  let k = tag-of(it.element)
  if k != none and k in note-colors {
    highlight(fill: note-colors.at(k), radius: 2pt, extent: 1pt, it)
  } else { it }
}

#show outline.entry: set block(spacing: 0.45em)
#show outline.entry.where(level: 1): set text(weight: "bold")
#outline(
  title: underline(text(size: 14pt)[Contents]),
  depth: 1, // raise to show more headers
  indent: 1.2em,
)
// Backward/forward error diagrams after Higham, Figs. 1.1 and 1.2
#let error-diagram(mixed: false) = cetz.canvas(length: 1.5cm, {
  import cetz.draw: *
  set-style(stroke: 0.6pt, mark: (fill: black, scale: 1.8))
  let dot(p) = circle(p, radius: 0.05, fill: black, stroke: none)
  let both = (start: "stealth", end: "stealth")

  let X  = (0, 0)
  let Y  = (if mixed { 3.8 } else { 3.75 }, 0)
  let XD = (0.75, -0.77)
  let YH = (4.55, -1.52)
  let FD = (3.8, -2.27)           // f(x + Δx), mixed case only

  content((0, 0.7), text(weight: "bold")[Input space])
  content((Y.at(0) + 0.4, 0.7), text(weight: "bold")[Output space])

  // exact (solid) and computed (dashed)
  line(X, Y)
  line(XD, if mixed { FD } else { YH })
  line(X, YH, stroke: (dash: (9pt, 5pt), thickness: 0.6pt))

  // error arrows
  line((X, 10%, XD), (XD, 10%, X), mark: both)
  line((Y, 8%, YH), (YH, 8%, Y), mark: both)
  if mixed {
    line((FD, 12%, YH), (YH, 12%, FD), mark: both)
    content((4.45, -2.0), $italic(Delta) y$, anchor: "west")
  }

  // labels
  line((-0.55, -0.35), (-0.08, -0.35))
  content((-0.62, -0.35), [backward error], anchor: "east")
  line((4.3, -0.77), (5.0, -0.77))
  content((5.08, -0.77), [forward error], anchor: "west")

  for p in (X, Y, XD, YH) { dot(p) }
  content((X.at(0) - 0.12, 0), $x$, anchor: "east")
  content((Y.at(0) + 0.12, 0), $y = f(x)$, anchor: "west")
  content((XD.at(0) - 0.08, XD.at(1)), $x + italic(Delta) x$, anchor: "east")
  if mixed {
    dot(FD)
    content((YH.at(0) + 0.12, YH.at(1)), $hat(y)$, anchor: "west")
    content((FD.at(0) + 0.1, FD.at(1) - 0.12), $f(x + italic(Delta) x)$, anchor: "north-west")
  } else {
    content((YH.at(0) + 0.12, YH.at(1)), $hat(y) = f(x + italic(Delta) x)$, anchor: "west")
  }
})

////////////////////////////////////////////////////////////////////////////////////
///           Figure code above, text below
//////////////////////////////////////////////////////////////////////////////////// 
= Overview
Purpose of this document is to introduce concepts that are used and to track progress. Project's aim is to implement QLL in MPC for a NIDS use case. Currently the project is broken up into 4 parts. 1) Implementation and Evaluation 2) Accuracy 3) Correctness 4) Privacy. Building to this statement, "One training step in MPC is a private, exact implementation of the fixed-point training step, and each operation in that step is within $epsilon$ of real arithmetic, provided the values stay within stated ranges."

#legend
//// Accuracy
= Accuracy: Fixed-point vs. reals
== Fixed point structure

A fixed-point number has a total bitlength $l$, of which $f$ are fractional bits (the _scale_). A real number $x$ is encoded as the integer 
$ alpha = round(x dot 2^f) in [-2^(l-1), 2^(l-1) - 1] $
stored in two's complement on $l$ bits, and decodes to
$ hat(x) = floor(alpha / 2^f) + (alpha mod 2^f) / 2^f, $
the real valued number integer part is represeted by the top $l-f$ bits and fractional part represented by the lower $f$ bits. Rounding error is $(alpha mod 2^f) / 2^f$.

The integer can take on any value within this range (values beyond this range wrap around):

#align(center + horizon)[$ [-2^(l-1), 2^(l-1)-1]$]

To convert the integer to a real-valued number dividing by $2^f$, our range becomes:

#align(center + horizon)[$ [-2^(l-1-f), 2^(l-1-f)-2^(-f)]$]

For example, for $l=8,f=4$ that range becomes:

#align(center + horizon)[$[-8, 7.9375]$]
=== Examples

#fixed-bits((8, 4, 1.25))
#fixed-bits((8, 4, -1.25))
#fixed-bits((8, 4, 3.1415927))
#fixed-bits((32, 16, 3.1415927))

== Rounding and Truncation
Truncation right-shifts until there are $2^f$ fractional bits. A product $a * b$ results in $2^(2f)$ fractional bits and truncation bits shifts until there are $2^f$ fractional bits.

Rounding decides how to handle those dropped bits. Rounding can be towards 0, away from 0, towards $plus.minus infinity$, round to nearest with ties broken by a subsequent rounding direction, or stochastic rounding. Rounding down is a right shift which is the same as truncation.

=== Stochastic Rounding
Rounds up with a probability equal to the truncated fraction.
Multiplying two real valued numbers result in twice the fractional bits:
#align(center + horizon)[$(v*2^f)(w*2^f) = (v*w)*2^(2f)$]
Truncate $2^f$ bits and round up with probability $Phi = (p mod 2^f)/2^f$ where $p = v*w$:

$ "result" = cases(
  ceil(x) & "with probability" Phi,
  floor(x) & "with probability" 1 - Phi,
) $

== Types of Error
#quote["There are three main sources of errors in numerical computation: rounding, data
uncertainty, and truncation"@higham2002.]

#T[Rounding and truncation are explained above.] Error due to uncertainty can come from measurement errors or error that is introduced from a previous computation. 
=== Absolute Error
Given exact value $y$ and computed value $hat(y)$ the absolute error is:
#align(center + horizon)[$epsilon = abs(y - hat(y))$]
=== Relative Error
Given exact value $y$ and computed value $hat(y)$ the realtive error is:
#align(center + horizon)[$epsilon = (abs(y - hat(y)))/hat(y)$]
=== (Mixed) Forward and Backward Error
#C[Something that I came across while skimming @higham2002 was forward and backward error.] Given an exact function $f$, a computed version $hat(f)$, and an exact value $x$. 

Forward error represents how far the true output is from the computed output:
#align(center + horizon)[$abs(f(x) - hat(f)(x)) <= epsilon$]

Backward error represents if there exists a pertubed input that matches the computed output:
#align(center + horizon)[$hat(f)(x) = f(x + Delta x)$ with $abs(Delta x) <= epsilon$]

#figure(
  error-diagram(),
  caption: [Backward and forward errors for $y = f(x)$. Solid line = exact; dashed line = computed $hat(f)$. After @higham2002[Fig. 1.1].],
) <fig:back-fwd>

Mixed forward and backward error asks if there is a pertubed input for a pertubed output:
#align(center + horizon)[$f(x) + Delta y = f(x + Delta x)$ with $abs(Delta x) <= epsilon_1$ and $abs(Delta y) <= epsilon$]

#figure(
  error-diagram(mixed: true),
  caption: [Mixed forward--backward error for $y = f(x)$. Solid line = exact; dashed line = computed $hat(f)$. After @higham2002[Fig. 1.2].],
) <fig:mixed>

=== Accumulation of Rounding Errors 
#C[Explain, that accumulation could be due to a few rounding errors.]

#C[@higham2002, a good approximation is more important than accurate rounding. review polynomial approx.]

== Gappa
#quote(attribution:[Gappa Docs])[This tool manipulates logical formulas stating the enclosures of expressions in some intervals.]

Given definitions, a logical formula, and hints Gappa proves bounds on computer arithmetic that involves rounding. Goals in Gappa can be written in the following forms:

#figure(
  table(
    columns: (auto, 1fr, 1fr),
    align: left,
    stroke: none,
    inset: (x: 5pt, y: 3pt),
    table.header([*Goal form*], [*Means*], [*Output when proved*]),
    table.hline(stroke: 0.6pt),
    [`e in ?`], ["find a bound"], [`Results: e in [a, b]`],
    [`e in [a, b]`], [check this given interval], [silent (no output means proved)],
    [`|e| <= b`], [absolute-value bound], [silent],
    [`e <= b`, `e >= a`], [one-sided bound], [silent],
    [`e1 -/ e2 in ?`], [relative error $(e_1 - e_2) \/ e_2$],
      [interval],
    [`@FIX(e, k)`], [$e$ is a multiple of $2^k$ (fits on the fixed-point grid)], [silent],
    [`@FLT(e, p)`], [$e$ fits in a $p$-bit significand], [silent],
    [`e <> 0`], [$e$ is nonzero], [silent, or `NZR(e)` failure],
    [`e1 = e2`], [exact equality], [silent, or `EQL(e1, e2)` failure],
    [`G1 /\ G2`], [both], [],
    [`G1 \/ G2`], [either], [],
    [`not G`], [negation], [],
    [`H -> (H2 -> G)`], [a nested hypothesis], [],
    table.hline(stroke: 0.6pt),
  ),
  caption: [Ways to write Gappa goals.],
) <tab:gappa-goals>

Gappa only supports one logical formula per file and doesn't support stochastic rounding so I compute the error from rounding up as well as the error from rounding down. Stocastic error picks one of the two so the error lies within the union.

Gappa's hints help give gappa identities to substitue for simpiler error calculations, where/how to split a range, or to tell Gappa that two values are close/approximations.

A Gappa file is usually structured in four parts: 
+ *Number format and rounding*: (lines 1-2) `fixed<-20,dn>` rounds to the grid $2^(-20)$, in direction down.
+ *The computation*: written once exactly (lines 4) and the rounded versions (lines 5--6)
+ *Logical Formula*: (lines 8-18) The hypothesis bounds both inputs. The goal checks that the exact output stays within a range and then asks for the rounding error in $2^(-20)$ units (ULP) as well as the range of the rounded results. 

=== Example Gappa file FixMul.g

#listing(caption: [`FixMul.g`: rounding a fixed-point product to $f = 20$ fractional bits.])[
```gappa
@rnd_dn = fixed<-20, dn>;
@rnd_up = fixed<-20, up>;

p = x * y;
z_dn = rnd_dn(p);
z_up = rnd_up(p);

{
    # hypo
    x in [-1b11, 1b11] /\
    y in [-1b11, 1b11] ->
    # goal
    p in [-1b22, 1b22] /\ # 2^(l - 2 - 2f) i.e. @ (64 - 2 - 2*20)
    (z_dn - x * y) * 1b20 in ? /\
    (z_up - x * y) * 1b20 in ? /\
    z_dn in ? /\
    z_up in ?
}
```
] <lst:fixmul>

#output[
```
$ gappa FixMul.g
Results:
  z_dn in [-1b22 {-4.1943e+06, -2^(22)}, 1b22 {4.1943e+06, 2^(22)}]
  z_up in [-1b22 {-4.1943e+06, -2^(22)}, 1b22 {4.1943e+06, 2^(22)}]
  (z_dn - p) * 1b20 in [-1, 0]
  (z_up - p) * 1b20 in [0, 1]
```
]

From the results we can see that rounding cost 1 ULP. And a stocastic rounding error lies in $[-2^(-20), 2^(-20)]$
#R[Trio multiplication in gappa implemented in Trio_mul.g, run with gappa Trio_Mul.g]

=== How to capture error in / error out in functions
#Q[Should I write forms of these gappa files/operations that incorporate incoming error from another function because I'd like to be able to compose these functions that mirror NN training.]

== Numerical stability 
#Q[How to define numerical stability in our case, what is leading cause of errors? If its rounding then we could use backward stability.]

//// Correctness
= Correctness: MPC vs fixed-point
== Secure computation in Rocq <todo>
Does $Pi_"mult"$(share x share y) = mult(x y)

//// Privacy
= Privacy
== Secure protocol with SSProve <todo>
Is Trio-MPC protocol secure in additive secret sharing for 3PC, 1-dealer, SH, HM setting.

SSProve has a SecretSharing.v file that computes 2-out-of-2 SS with XOR and matches "Joy of Cryptography" implementation. Can we extend it to our case? Starting with $Pi_"mult"$.

#T[Specify real view -> specify protocol view -> specify each parties share, simulate protocol messages given a party's input and output.]

//// Implementation details and evaluating finetuning runs
= Implementation and Evaluation
== Trio in MPC <todo>
== Code structure
#T[Fill out details]
#let file(pos, body, ..args) = node(pos, raw(body), stroke: 0.5pt, corner-radius: 2pt, ..args)
#let manual = (fill: luma(235), stroke: (dash: "dashed", thickness: 0.5pt))
#let io = (fill: luma(235))

#figure(
  text(size: 6.5pt, diagram(
    spacing: (6pt, 10pt),
    edge-stroke: 0.6pt,
    // acasXu.sh: generate, compile, run
    file((0, 1), "acasXu.sh", fill: rgb("#c5d3f2")),
    file((0.7, 0), "nn.py"),
    file((1.3, 0), "loss.py"),
    file((1, 1), "torch_to_hpmpc.py"),
    file((2, 1), "acasXu_generated.hpp"),
    file((3, 1), "acasXu.hpp"),
    file((4, 1), "main.cpp\n× 3 parties", fill: rgb("#c6ebc2")),
    edge((0, 1), (1, 1), "-|>"),
    edge((0.7, 0), (1, 1), "-|>"),
    edge((1.3, 0), (1, 1), "-|>"),
    edge((1, 1), (2, 1), "-|>"),
    edge((2, 1), (3, 1), "-|>"),
    edge((3, 1), (4, 1), "-|>"),

    // inputs, made by hand before acasXu.sh
    file((1, 2), "acasXu_2_9.onnx", ..io),
    file((2, 2), "onnx_converter.py\nto_hpmpc", ..manual),
    file((3, 2), "weights.inp (P1)", ..io),
    file((2, 3), "acasXu_data.py", ..manual),
    file((3, 3), "property.inp (P1)\ndata.inp (P2)", ..io),
    edge((1, 2), (2, 2), "-|>"),
    edge((2, 2), (3, 2), "-|>"),
    edge((2, 3), (3, 3), "-|>"),
    edge((3, 2), (4, 1), "-|>"),
    edge((3, 3), (4, 1), "-|>"),

    // output, converted by hand after
    file((5, 1), "finetuned.inp (P1)", ..io),
    file((5, 2), "onnx_converter.py\nto_onnx", ..manual),
    file((5, 3), "acasXu_2_9_1_mpc.onnx", ..io),
    edge((4, 1), (5, 1), "-|>"),
    edge((5, 1), (5, 2), "-|>"),
    edge((5, 2), (5, 3), "-|>"),
  )),
  caption: [ACAS Xu fine-tuning pipeline; dashed steps are run by hand before and after.],
) <fig:pipeline>

== Fine-tuning results <review>
Following the cleartext experiments of finetuning a pretrained AcasXu model with a QLL property and successfully verifying that the property holds with Vehicle, I applied a similar technique under an MPC setting. 

Below are three preliminary results, the first run trains while finding datapoints that violate the QLL property and retraining on those points.

The second and third runs use the same dataset and performs traditional ML training without searching for adversarial inputs.

#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    stroke: none,
    table.header([*Run*], [*Optimizer steps*], [*Time*], [*Rounds*]),
    table.hline(stroke: 0.6pt),
    [Cleartext, adversarial inputs], [24,600], [6,000.78 s], [--],
    [Cleartext, dataset fine-tuning], [1,200],  [1.34 s],     [--],
    [MPC, dataset fine-tuning],       [1,200],  [199.91 s],   [4,502,375],
    table.hline(stroke: 0.6pt),
  ),
  caption: [Training time for 30 epochs of fine-tuning ACAS Xu 2.9 (property 1).],
) <tab:runs>

== Miscellaneous
Random notes and diagrams:
#figure(
  diagram(
    spacing: (3cm, 1.5cm),
    node((0, 0), $x$),
    node((1, 0), $y$),
    node((0, 1), $⟦x⟧$),
    node((1, 1), $⟦y⟧$),
    edge((0, 0), (1, 0), $cal(F)$, "->"),
    edge((0, 0), (0, 1), $"share"$, "->", label-side: right),
    edge((0, 1), (1, 1), $Pi$, "->", label-side: right),
    edge((1, 1), (1, 0), $"reconstruct"$, "->", label-side: right),
  ),
  caption: [MPC protocol: $cal(F) = "reconstruct" compose Pi compose "share"$.],
)

*Disclaimer:* GenAI used in this document for formatting, learning typst, and making diagrams. 

#bibliography("refs.bib")