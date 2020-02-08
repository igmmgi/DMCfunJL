module DMCfunJL

using 
DataFrames,
Distributions,
KernelDensity,
Parameters,
Plots,
StatsBase

export
Prms,
Simulation,
DMC,
dmc_sim,
dmc_trials_full,
dmc_trials,
dmc_calculate_delta,
dmc_calculate_caf,
dmc_summary,
dmc_plot,
dmc_plot_full,
dmc_plot_activation,
dmc_plot_trials,
dmc_plot_cdf,
dmc_plot_pdf,
dmc_plot_caf,
dmc_plot_delta

@with_kw struct Prms
  amp = 20.0
  tau = 30.0
  aaShape = 2.0
  mu = 0.5 
  sigm = 4 
  bnds = 75.0
  resMean = 300.0
  resSD = 30.0
  nTrl = 100000 
  tmax = 1000 
  varDR = false 
  drLim = [0.1 0.7]
  drShape = 3.0
  varSP = false 
  spShape = 3.0
  fullData = false 
  nTrlData = 5
  stepDelta = 5
  stepCAF = 20
end

struct Simulation
  drift::Array{Float64}
  activation::Array{Float64}
  trials::Array{Float64}
  rts::Array{Float64}
  rtCorr::Float64
  sdRtCorr::Float64
  perErr::Float64
  rtErr::Float64
  sdRtErr::Float64
  delta::Array{Float64}
  caf::Array{Float64}
end

struct DMC
  prms::Prms
  comp::Simulation
  incomp::Simulation
end

function dmc_sim(prms::Prms)

  @unpack amp, tau, aaShape, mu, sigm, bnds, resMean, resSD, nTrl, tmax, varDR, drLim, drShape, varSP, spShape, fullData, nTrlData, stepDelta, stepCAF = prms 

  tim            = 1:tmax
  drift          = amp * exp.(-(tim) / tau) .* ((exp.(1) .* (tim) ./ (aaShape - 1) ./ tau).^(aaShape - 1))
  drift_rate     = fill(mu, nTrl)
  starting_point = fill(0.0, nTrl)

  res = Simulation[] 
  for comp  in [1, -1]    # comp/incomp

    muVec = comp * ( drift .* ((aaShape - 1) ./ tim .- 1 ./ tau)) 

    # variable starting points and drift rates
    if varDR
      drift_rate = rand(Beta(drShape), nTrl) .* (drLim[2] - drLim[1]) .+ drLim[1]
    end
    if varSP
      starting_point = rand(Beta(spShape), nTrl) .* (bnds * 2) .- bnds
    end

    if fullData
        activation, trials, rts, errs = dmc_trials_full(nTrl, nTrlData, tmax, bnds, resMean, resSD, muVec, starting_point, drift_rate, sigm)
    else
        activation, trials, rts, errs = dmc_trials(nTrl, nTrlData, tmax, bnds, resMean, resSD, muVec, starting_point, drift_rate, sigm)
    end

    push!(res, Simulation(drift, 
                          activation,
                          trials,
                          rts,
                          round(mean(rts[.!errs])), 
                          round(std(rts[.!errs])),
                          round(sum(errs) / nTrl * 100, digits = 2),
                          round(mean(rts[errs])),
                          round(std(rts[errs])),
                          dmc_calculate_delta(rts, stepDelta),
                          dmc_calculate_caf(rts, errs, stepCAF)))

  end

  return(DMC(prms, res[1], res[2]))

end

function dmc_trials_full(nTrl, nTrlData, tmax, bnds, resMean, resSD, muVec, starting_point, drift_rate, sigm)
    
  rts = fill(convert(Float64, tmax), nTrl)
  errs = fill(true, nTrl)
  activation = zeros(tmax) 
  trials = zeros(tmax, nTrlData)
  trial_activation = fill(0.0, tmax) 

  for t in 1:nTrl
      criterion = false
      trial_activation[1] = starting_point[t] + muVec[1] + drift_rate[t] + (sigm * rand(Normal()))
      @inbounds for i = 2:tmax
          trial_activation[i] = trial_activation[i - 1] + muVec[i] + drift_rate[t] + (sigm * rand(Normal()))
          if !criterion && (abs(trial_activation[i]) >= bnds)
              rts[t] = i + rand(Normal(resMean, resSD))
              errs[t] = trial_activation[i] > 0 ? false : true
              criterion = true
          end
      end
      if t <= nTrlData
          trials[:, t] = trial_activation
      end
      activation .+= trial_activation
  end
  activation ./= nTrl

  return(activation, trials, rts, errs)

end

function dmc_trials(nTrl, nTrlData, tmax, bnds, resMean, resSD, muVec, starting_point, drift_rate, sigm)
 
  rts = fill(convert(Float64, tmax), nTrl)
  errs = fill(true, nTrl)
  activation = zeros(tmax) 
  trials = zeros(tmax, nTrlData)
 
  for t = 1:nTrl
      trial_activation = starting_point[t]
      @inbounds for i = 1:tmax
          trial_activation += muVec[i] .+ drift_rate[t] .+ (sigm .* rand(Normal()))
          if (abs(trial_activation) >= bnds)
              rts[t] = i + rand(Normal(resMean, resSD))
              errs[t] = trial_activation > 0 ? false : true
              break
          end
      end
  end

  return(activation, trials, rts, errs)

end

function dmc_calculate_delta(rts, stepDelta)
  return quantile(rts, (stepDelta:stepDelta:100 - stepDelta) / 100, sorted = false)
end

function dmc_calculate_caf(rts, errs, stepCAF)
  nBins = div(100, stepCAF)
  edges = quantile(rts, 0:(1 / nBins):1)
  caf = fill(0.0, nBins)
  for bin = 1:length(edges) - 1
    caf[bin] = proportions(errs[(rts .> edges[bin]) .& (rts .< edges[bin + 1])])[1]
  end
  return caf
end

function dmc_summary(res::DMC)
  return(DataFrame(Comp     = ["comp", "incomp"],
                   rtCorr   = [res.comp.rtCorr,   res.incomp.rtCorr],
                   sdRtCorr = [res.comp.sdRtCorr, res.incomp.sdRtCorr],
                   perErr   = [res.comp.perErr,   res.incomp.perErr],
                   rtErr    = [res.comp.rtErr,    res.incomp.rtErr],
                   sdRtErr  = [res.comp.sdRtErr,  res.incomp.sdRtErr]))
end

function dmc_plot_activation(res::DMC)
  p1 = plot(res.comp.activation, color = "green", xlabel = "Time (ms)", ylabel = "E[X(t)]", legend = :none, framestyle = :box, linewidth = :2)
  plot!(res.incomp.activation, color = "red", linewidth = :2)
  plot!(res.comp.drift, color = "black", linewidth = :2)
  plot!(-res.incomp.drift, color = "black", linestyle = :dot, linewidth = :2)
  xlims!((0, res.prms.tmax))
  ylims!((-res.prms.bnds - 20, res.prms.bnds + 20))
  return p1
end

function dmc_plot_trials(res::DMC)
  p2 = plot(xlabel = "Time (ms)", ylabel = "X(t)", legend = :none, framestyle = :box)
  xlims!((0, res.prms.tmax))
  ylims!((-res.prms.bnds - 20, res.prms.bnds + 20))
  hline!([-res.prms.bnds, res.prms.bnds], color = "black")
  for plt = 1:res.prms.nTrlData
    idx = findfirst(abs.(res.comp.trials[:, plt]) .>= res.prms.bnds)
    plot!(res.comp.trials[1:idx, plt], color = "green")
    idx = findfirst(abs.(res.incomp.trials[:, plt]) .>= res.prms.bnds)
    plot!(res.incomp.trials[1:idx, plt], color = "red")
  end
  return p2
end

function dmc_plot_cdf(res::DMC)
  kde_comp = kde(res.comp.rts)
  p3 = plot(kde_comp.x, kde_comp.density, color = "green", xlabel = "Time (ms)", ylabel = "CDF", legend = :none, framestyle = :box, linewidth = :2)
  kde_incomp = kde(res.incomp.rts)
  plot!(kde_incomp.x, kde_incomp.density, color = "red", linewidth = :2)
  return p3
end

function dmc_plot_pdf(res::DMC)
  ecdf_comp = ecdf(res.comp.rts)
  p4 = plot(1:1000, ecdf_comp(1:1000), color = "green", xlabel = "Time (ms)", ylabel = "PDF", legend = :none, framestyle = :box, linewidth = :2)
  ecdf_incomp = ecdf(res.incomp.rts)
  plot!(1:1000, ecdf_incomp(1:1000), color = "red", linewidth = :2)
  return p4
end

function dmc_plot_caf(res::DMC)
  p5 = plot(res.comp.caf, 
            color = "green", xlabel = "RT Bin", ylabel = "CAF", framestyle = :box, label = "Compatible", 
            legend = :bottomright, markershape = :circle, markersize = :2, legendfontsize = :8)
  ylims!((0, 1.1))
  plot!(res.incomp.caf, color = "red", label = "Incompatible", markershape = :circle, markersize = :2)
  return p5
end

function dmc_plot_delta(res::DMC)
  p6 = plot([res.incomp.delta .+ res.comp.delta] ./ 2, [res.incomp.delta .- res.comp.delta], 
            legend = :none, framestyle = :box, color = "black", xlabel = "Time (ms)", ylabel = "Delta (ms)", 
            markershape = :circle, markersize = :2)
  xlims!((0, 1000))
  ylims!((-100, 100))
  return p6
end

function dmc_plot_full(res::DMC)

  p1 = dmc_plot_activation(res)
  p2 = dmc_plot_trials(res)
  p3 = dmc_plot_cdf(res) 
  p4 = dmc_plot_pdf(res) 
  p5 = dmc_plot_caf(res) 
  p6 = dmc_plot_delta(res) 

  return plot(p1, p2, p3, p4, p5, p6, layout = (3, 2), size = (800, 600))

end

function dmc_plot(res::DMC)

  p3 = dmc_plot_cdf(res) 
  p4 = dmc_plot_pdf(res) 
  p5 = dmc_plot_caf(res) 
  p6 = dmc_plot_delta(res) 

  return plot(p3, p4, p5, p6, layout = (2, 2), size = (800, 600))

end

end

