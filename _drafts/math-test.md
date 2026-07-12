---
title: Math Rendering Test
description: Draft page exercising KaTeX rendering paths.
math: true
---

Euler's identity, $$e^{i\pi} + 1 = 0$$, relates five constants in one line.

$$
\int_{-\infty}^{\infty} e^{-x^2}\,dx = \sqrt{\pi}
$$

<details>
<summary>Derivation notes</summary>

The Gaussian integral \( \int_{-\infty}^{\infty} e^{-x^2}\,dx \) is computed by squaring it and switching to polar coordinates:

\[
\left( \int_{-\infty}^{\infty} e^{-x^2}\,dx \right)^2 = \int_0^{2\pi} \int_0^\infty e^{-r^2} r \, dr \, d\theta = \pi
\]

</details>
