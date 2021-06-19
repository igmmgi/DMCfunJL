# DiffusionModelConflict
Julia implementation of the diffusion process model (Diffusion Model for
Conflict Tasks, DMC) presented in Automatic and controlled stimulus
processing in conflict tasks: Superimposed diffusion processes and delta
functions
(https://www.sciencedirect.com/science/article/pii/S0010028515000195).

NB. See also R/Cpp package DMCfun for further functionality inlcuding fitting
procedures.

## Installation
``` julia
] # julia pkg manager
add https://github.com/igmmgi/DiffusionModelConflict.git # install from  GitHub
test DiffusionModelConflict # optional
```

## Basic Example
``` julia
using DiffusionModelConflict

# Example 1
res = dmc_sim(Prms(fullData = true));
dmc_summary(res)
dmc_plot_full(res)
```

![alt text](/figures/figure1.png)

``` julia
# Example 2
res = dmc_sim(Prms(tau = 120));
dmc_summary(res)
dmc_plot(res)
```

![alt text](/figures/figure2.png)
