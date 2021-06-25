__precompile__()
module DiffusionModelConflict

using CSV,
    DataFrames, Distributions, Parameters, Makie, Glob, GLMakie, Random, Statistics, StatsBase

include("dmc_sim.jl")
include("dmc_data.jl")

end
