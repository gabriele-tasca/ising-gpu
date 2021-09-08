# ising-gpu
CUDA C simulation of the two-dimensional Ising model.

The program stores 32 separate spin variables in the 32 bits of an unsigned integer. Each of the 32 spins make up 32 separate Ising grids that can be simulated in parallel by applying bitwise operations, allowing for maximum efficiency in both memory usage and speed.

The program was used to compute the magnetization curve and find the critical temperature, as well as to study the spin-spin correlation function at different temperatures.

1

# ising-gpu

2

CUDA C simulation of the two-dimensional Ising model.

3

​

4

The program stores 32 separate spin variables in the 32 bits of an unsigned integer. Each of the 32 spins make up 32 separate Ising grids that can be simulated in parallel by applying bitwise operations, allowing for maximum efficiency in both memory usage and speed.

5

​

6

The program was used to compute the magnetization curve and find the critical temperature, as well as to study the spin-spin correlation function at different temperatures.

1

# ising-gpu

2

CUDA C simulation of the two-dimensional Ising model.

3

​

4

The program stores 32 separate spin variables in the 32 bits of an unsigned integer. Each of the 32 spins make up 32 separate Ising grids that can be simulated in parallel by applying bitwise operations, allowing for maximum efficiency in both memory usage and speed.

5

​

6

The program was used to compute the magnetization curve and find the critical temperature, as well as to study the spin-spin correlation function at different temperatures.

![Critical Ising](https://github.com/gabriele-tasca/ising-gpu/blob/master/critical-ising.png)

A snapshot of the state of the system near the critical temperature. The characteristic fractal/scale-free patterns are visible.


![Critical Ising](https://github.com/gabriele-tasca/ising-gpu/blob/master/m_plot.svg)

Computed magnetization curve.


![Critical Ising](https://github.com/gabriele-tasca/ising-gpu/blob/master/3corrs.svg)

Spin-spin correlation function at 3 temperatures.
