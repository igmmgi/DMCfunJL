__precompile__()
module DiffusionModelConflict

using CSV,
    DataFrames, Distributions, Parameters, Makie, Glob, GLMakie, Statistics, StatsBase

include("dmc_sim.jl")
include("dmc_data.jl")

end
