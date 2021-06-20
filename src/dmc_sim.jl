export Prms, DmcSim, dmc_sim, plot

@with_kw struct Prms
    amp = 20.0
    tau = 30.0
    aaShape = 2.0
    drc = 0.5
    sigm = 4
    bnds = 75.0
    resDist = 1
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
    nDelta = 19
    nCAF = 5
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

struct DmcSim
    prms::Prms
    comp::Simulation
    incomp::Simulation
    summary::DataFrame
end

function dmc_sim(p::Prms)

    @unpack amp,
    tau,
    aaShape,
    drc,
    sigm,
    bnds,
    resDist,
    resMean,
    resSD,
    nTrl,
    tmax,
    varDR,
    drLim,
    drShape,
    varSP,
    spShape,
    fullData,
    nTrlData,
    nDelta,
    nCAF = p

    tim = 1:p.tmax
    drift =
        @. amp * exp(-(tim) / tau) * ((exp(1) * (tim) / (aaShape - 1) / tau)^(aaShape - 1))
    drift_rate = fill(drc, nTrl)
    starting_point = fill(0.0, nTrl)

    res = Simulation[]
    for comp in [1, -1]    # comp/incomp

        drcVec = @. comp * (drift * ((aaShape - 1) / tim - 1 / tau))

        # residual rt distribution (normal 1 vs. uniform 2)
        if resDist == 1
            residual_rt = rand(Normal(resMean, resSD), nTrl)
        elseif resDist == 2
            range = sqrt((resSD * resSD / (1.0 / 12.0)) / 2.0)
            residual_rt = rand(Uniform(resMean - range, resMean + range), nTrl)
        end

        # variable starting points and drift rates
        if varDR
            drift_rate = rand(Beta(drShape), nTrl) .* (drLim[2] - drLim[1]) .+ drLim[1]
        end
        if varSP
            starting_point = rand(Beta(spShape), nTrl) .* (bnds * 2) .- bnds
        end

        if fullData
            activation, trials, rts, errs = _dmc_trials_full(
                nTrl,
                nTrlData,
                tmax,
                bnds,
                residual_rt,
                drcVec,
                starting_point,
                drift_rate,
                sigm,
            )
        else
            activation, trials, rts, errs = _dmc_trials(
                nTrl,
                nTrlData,
                tmax,
                bnds,
                residual_rt,
                drcVec,
                starting_point,
                drift_rate,
                sigm,
            )
        end

        push!(
            res,
            Simulation(
                drift,
                activation,
                trials,
                rts,
                round(mean(rts[.!errs])),
                round(std(rts[.!errs])),
                round(sum(errs) / nTrl * 100, digits = 2),
                round(mean(rts[errs])),
                round(std(rts[errs])),
                _dmc_calculate_delta(rts, nDelta),
                _dmc_calculate_caf(rts, errs, nCAF),
            ),
        )

    end

    return (DmcSim(p, res[1], res[2], _dmc_summary(res)))

end

function _dmc_trials_full(
    nTrl,
    nTrlData,
    tmax,
    bnds,
    residual_rt,
    drcVec,
    starting_point,
    drift_rate,
    sigm,
)

    rts = fill(convert(Float64, tmax), nTrl)
    errs = fill(true, nTrl)
    activation = zeros(tmax)
    trials = zeros(tmax, nTrlData)

    # TO DO: threads with random?
    for t = 1:nTrl
        criterion = false
        trial_activation = starting_point[t]
        @inbounds for i = 1:tmax
            trial_activation += drcVec[i] + drift_rate[t] + (sigm * rand(Normal()))
            if !criterion && (abs(trial_activation) >= bnds)
                rts[t] = i + residual_rt[t]
                errs[t] = trial_activation < 0
                criterion = true
            end
            if t <= nTrlData
                trials[i, t] = trial_activation
            end
            activation[i] += trial_activation
        end
    end
    activation ./= nTrl

    return (activation, trials, rts, errs)

end

function _dmc_trials(
    nTrl,
    nTrlData,
    tmax,
    bnds,
    residual_rt,
    drcVec,
    starting_point,
    drift_rate,
    sigm,
)

    rts = fill(convert(Float64, tmax), nTrl)
    errs = fill(true, nTrl)
    activation = zeros(tmax)
    trials = zeros(tmax, nTrlData)

    # TO DO: threads with random?
    for t = 1:nTrl
        trial_activation = starting_point[t]
        @inbounds for i = 1:tmax
            trial_activation += @. drcVec[i] + drift_rate[t] + (sigm * rand(Normal()))
            if (abs(trial_activation) >= bnds)
                rts[t] = i .+ residual_rt[t]
                errs[t] = trial_activation < 0
                break
            end
        end
    end

    return (activation, trials, rts, errs)

end

function _dmc_calculate_delta(rts, nDelta)
    return quantile(rts, range(0, stop = 1, length = nDelta + 2)[2:end-1], sorted = false)
end

function _dmc_calculate_caf(rts, errs, nCAF)
    edges = quantile(rts, range(0, stop = 1, length = nCAF + 1))
    caf = fill(0.0, nCAF)
    for bin = 1:nCAF
        caf[bin] = proportions(errs[(rts.>edges[bin]).&(rts.<edges[bin+1])])[1]
    end
    return caf
end

function _dmc_summary(res)
    return (DataFrame(
        Comp = ["comp", "incomp"],
        rtCorr = [res[1].rtCorr, res[2].rtCorr],
        sdRtCorr = [res[1].sdRtCorr, res[2].sdRtCorr],
        perErr = [res[1].perErr, res[2].perErr],
        rtErr = [res[1].rtErr, res[2].rtErr],
        sdRtErr = [res[1].sdRtErr, res[2].sdRtErr],
    ))
end


# Plotting
# TO DO: plot recepies/themes?
# TO DO: why is this so slow on first run? kwargs or just standard time-to-first plot?
# TO DO: margins seem a bit tight on right of plot
# TO DO: filter potential kwargs for each individual plot
function dmc_plot(res::DmcSim; figType = "summary", kwargs...)
    fig = Figure()
    if figType == "summary"
        if res.prms.fullData
            fig = _dmc_plot_activation(res, fig = fig, figrow = 1, figcol = 1; kwargs...)[1]
            fig = _dmc_plot_trials(res, fig = fig, figrow = 1, figcol = 2; kwargs...)[1]
            fig = _dmc_plot_pdf(res, fig = fig, figrow = 2, figcol = 1; kwargs...)[1]
            fig = _dmc_plot_cdf(res, fig = fig, figrow = 2, figcol = 2; kwargs...)[1]
            fig = _dmc_plot_caf(res, fig = fig, figrow = 3, figcol = 1; kwargs...)[1]
            fig = _dmc_plot_delta(res, fig = fig, figrow = 3, figcol = 2, kwargs...)[1]
        else
            fig = _dmc_plot_pdf(res, fig = fig, figrow = 1, figcol = 1; kwargs...)[1]
            fig = _dmc_plot_cdf(res, fig = fig, figrow = 1, figcol = 2; kwargs...)[1]
            fig = _dmc_plot_caf(res, fig = fig, figrow = 2, figcol = 1; kwargs...)[1]
            fig = _dmc_plot_delta(res, fig = fig, figrow = 2, figcol = 2, kwargs...)[1]
        end
    elseif figType == "activation"
        fig = _dmc_plot_activation(res, fig = fig, figrow = 1, figcol = 1; kwargs...)[1]
    elseif figType == "trials"
        fig = _dmc_plot_trials(res, fig = fig, figrow = 1, figcol = 1; kwargs...)[1]
    elseif figType == "pdf"
        fig = _dmc_plot_pdf(res, fig = fig, figrow = 1, figcol = 1; kwargs...)[1]
    elseif figType == "cdf"
        fig = _dmc_plot_cdf(res, fig = fig, figrow = 1, figcol = 1; kwargs...)[1]
    elseif figType == "caf"
        fig = _dmc_plot_caf(res, fig = fig, figrow = 1, figcol = 1; kwargs...)[1]
    elseif figType == "delta"
        fig = _dmc_plot_delta(res, fig = fig, figrow = 1, figcol = 1; kwargs...)[1]
    end
    return fig
end

function _dmc_plot_activation(
    res::DmcSim;
    fig = nothing,
    figrow = 1,
    figcol = 1,
    showlegend = true,
    xlim = (),
    ylim = (),
    xlabel = "Time (ms)",
    ylabel = "E[X(t)]",
    labels = ("Compatible", "Incompatible"),
)
    fig = isnothing(fig) ? Figure() : fig
    ax = fig[figrow, figcol] = Axis(fig, xlabel = xlabel, ylabel = ylabel)
    l1 = lines!(ax, 1:res.prms.tmax, res.comp.activation, color = :green, label = labels[1])
    l2 = lines!(ax, 1:res.prms.tmax, res.incomp.activation, color = :red, label = labels[2])
    lines!(ax, res.comp.drift, color = "black", linewidth = :2)
    lines!(ax, -res.incomp.drift, color = "black", linestyle = :dot, linewidth = :2)
    Makie.hlines!(ax, [-res.prms.bnds, res.prms.bnds], color = :grey)
    xlim = isempty(xlim) ? (0, res.prms.tmax) : xlim
    xlims!(ax, xlim)
    ylim = isempty(ylim) ? (-res.prms.bnds - 20, res.prms.bnds + 20) : ylim
    ylims!(ax, ylim)
    if showlegend
        axislegend(
            ax,
            [l1, l2],
            [labels[1], labels[2]],
            position = :rb,
            orientation = :vertical,
            framevisible = false,
            labelsize = 16,
        )
    end
    return fig, ax
end

function _dmc_plot_trials(
    res::DmcSim;
    fig = nothing,
    figrow = 1,
    figcol = 1,
    showlegend = true,
    xlim = (),
    ylim = (),
    xlabel = "Time (ms)",
    ylabel = "X(t)",
    labels = ["Compatible", "Incompatible"],
)
    fig = isnothing(fig) ? Figure() : fig
    ax = fig[figrow, figcol] = Axis(fig, xlabel = xlabel, ylabel = ylabel)
    l1 = l2 = 0
    for plt = 1:res.prms.nTrlData
        idx = findfirst(abs.(res.comp.trials[:, plt]) .>= res.prms.bnds)
        l1 = lines!(res.comp.trials[1:idx, plt], color = :green, label = labels[1])
        idx = findfirst(abs.(res.incomp.trials[:, plt]) .>= res.prms.bnds)
        l2 = lines!(res.incomp.trials[1:idx, plt], color = :red, label = labels[2])
    end
    Makie.hlines!(ax, [-res.prms.bnds, res.prms.bnds], color = :grey)
    xlim = isempty(xlim) ? (0, res.prms.tmax) : xlim
    xlims!(ax, xlim)
    ylim = isempty(ylim) ? (-res.prms.bnds - 20, res.prms.bnds + 20) : ylim
    ylims!(ax, ylim)
    if showlegend
        axislegend(
            ax,
            [l1, l2],
            [labels[1], labels[2]],
            position = :rb,
            orientation = :vertical,
            framevisible = false,
            labelsize = 16,
        )
    end
    return fig, ax
end

function _dmc_plot_pdf(
    res::DmcSim;
    fig = nothing,
    figrow = 1,
    figcol = 1,
    showlegend = true,
    xlim = (),
    ylim = (),
    xlabel = "Time (ms)",
    ylabel = "PDF",
    labels = ("Compatible", "Incompatible"),
)
    fig = isnothing(fig) ? Figure() : fig
    ax = fig[figrow, figcol] = Axis(fig, xlabel = xlabel, ylabel = ylabel)
    l1 =
        lines!(ax, Makie.KernelDensity.kde(res.comp.rts), color = :green, label = labels[1])
    l2 =
        lines!(ax, Makie.KernelDensity.kde(res.incomp.rts), color = :red, label = labels[2])
    xlim = isempty(xlim) ? (0, res.prms.tmax) : xlim
    xlims!(ax, xlim)
    ylim = isempty(ylim) ? (0, 0.01) : ylim
    ylims!(ax, ylim)
    if showlegend
        axislegend(
            ax,
            [l1, l2],
            [labels[1], labels[2]],
            position = :rt,
            orientation = :vertical,
            framevisible = false,
            labelsize = 16,
        )
    end
    return fig, ax
end

function _dmc_plot_cdf(
    res::DmcSim;
    fig = nothing,
    figrow = 1,
    figcol = 1,
    showlegend = true,
    xlim = (),
    ylim = (),
    xlabel = "Time (ms)",
    ylabel = "CDF",
    labels = ("Compatible", "Incompatbile"),
)
    fig = isnothing(fig) ? Figure() : fig
    ax = fig[figrow, figcol] = Axis(fig, xlabel = xlabel, ylabel = ylabel)
    l1 = lines!(ax, ecdf(res.comp.rts)(1:res.prms.tmax), color = :green, label = labels[1])
    l2 = lines!(ax, ecdf(res.incomp.rts)(1:res.prms.tmax), color = :red, label = labels[2])
    xlim = isempty(xlim) ? (0, res.prms.tmax) : xlim
    xlims!(ax, xlim)
    ylim = isempty(ylim) ? (0, 1.05) : ylim
    ylims!(ax, ylim)
    if showlegend
        axislegend(
            ax,
            [l1, l2],
            [labels[1], labels[2]],
            position = :rb,
            orientation = :vertical,
            framevisible = false,
            labelsize = 16,
        )
    end
    return fig, ax
end

function _dmc_plot_caf(
    res::DmcSim;
    fig = nothing,
    figrow = 1,
    figcol = 1,
    showlegend = true,
    ylim = (),
    xlabel = "RT Bin",
    ylabel = "CAF",
    labels = ("Compatible", "Incompatible"),
)
    fig = isnothing(fig) ? Figure() : fig
    ax = fig[figrow, figcol] = Axis(fig, xlabel = xlabel, ylabel = ylabel)
    l1 = lines!(ax, res.comp.caf, color = :green)
    l2 = lines!(ax, res.incomp.caf, color = :red)
    scatter!(ax, res.comp.caf, color = :green, marker = :circle)
    scatter!(ax, res.incomp.caf, color = :red, marker = :circle)
    ylim = isempty(ylim) ? (0, 1.05) : ylim
    ylims!(ax, ylim)
    if showlegend
        axislegend(
            ax,
            [l1, l2],
            [labels[1], labels[2]],
            position = :rb,
            orientation = :vertical,
            framevisible = false,
            labelsize = 16,
        )
    end
    return fig, ax
end

function _dmc_plot_delta(
    res::DmcSim;
    fig = nothing,
    figrow = 1,
    figcol = 1,
    xlim = (),
    ylim = (),
    xlabel = "Time (ms)",
    ylabel = "Delta (ms)",
)
    fig = isnothing(fig) ? Figure() : fig
    ax = fig[figrow, figcol] = Axis(fig, xlabel = xlabel, ylabel = ylabel)
    lines!(
        ax,
        (res.incomp.delta .+ res.comp.delta) ./ 2,
        (res.incomp.delta .- res.comp.delta),
        color = :black,
    )
    scatter!(
        ax,
        (res.incomp.delta .+ res.comp.delta) ./ 2,
        (res.incomp.delta .- res.comp.delta),
        color = :black,
        marker = :circle,
    )
    xlim = isempty(xlim) ? (0, res.prms.tmax) : xlim
    xlims!(ax, xlim)
    ylim = isempty(ylim) ? (-50, 100) : ylim
    ylims!(ax, ylim)
    return fig, ax
end
