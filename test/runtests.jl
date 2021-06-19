using DiffusionModelConflict
using Test

@testset "test_dmc_sim1" begin

    # Simulation 1 (Figure 3)
    # amp = 20, tau = 30, mu = 0.5, sigm = 4, bnds = 75, resMean = 300, resSD = 30

    res = dmc_sim(Prms())

    @test isapprox(res.comp.rtCorr, 440, atol = 3)
    @test isapprox(res.comp.sdRtCorr, 106, atol = 3)
    @test isapprox(res.comp.perErr, 0.7, atol = 0.5)
    @test isapprox(res.incomp.rtCorr, 458, atol = 3)
    @test isapprox(res.incomp.sdRtCorr, 95, atol = 3)
    @test isapprox(res.incomp.perErr, 1.4, atol = 0.5)

end

@testset "test_dmc_sim2" begin

    # Simulation 2 (Figure 4)
    # amp = 20, tau = 150, mu = 0.5, sigm = 4, bnds = 75, resMean = 300, resSD = 30

    res = dmc_sim(Prms(tau = 150))

    @test isapprox(res.comp.rtCorr, 422, atol = 3)
    @test isapprox(res.comp.sdRtCorr, 90, atol = 3)
    @test isapprox(res.comp.perErr, 0.3, atol = 0.5)
    @test isapprox(res.incomp.rtCorr, 483, atol = 3)
    @test isapprox(res.incomp.sdRtCorr, 103, atol = 3)
    @test isapprox(res.incomp.perErr, 2.2, atol = 0.5)

end

@testset "test_dmc_sim3" begin

    # Simulation 3 (Figure 5)
    # amp = 20, tau = 90, mu = 0.5, sigm = 4, bnds = 75, resMean = 300, resSD = 30

    res = dmc_sim(Prms(tau = 90))

    @test isapprox(res.comp.rtCorr, 420, atol = 3)
    @test isapprox(res.comp.sdRtCorr, 96, atol = 3)
    @test isapprox(res.comp.perErr, 0.3, atol = 0.5)
    @test isapprox(res.incomp.rtCorr, 477, atol = 3)
    @test isapprox(res.incomp.sdRtCorr, 96, atol = 3)
    @test isapprox(res.incomp.perErr, 2.4, atol = 0.5)

end

@testset "test_dmc_sim4" begin

    # Simulation 4 (Figure 6)
    # amp = 20, tau = 30, mu = 0.5, sigm = 4, bnds = 75, resMean = 300, resSD = 30
    # varSP = true

    res = dmc_sim(Prms(varSP = true))

    @test isapprox(res.comp.rtCorr, 436, atol = 3)
    @test isapprox(res.comp.sdRtCorr, 116, atol = 3)
    @test isapprox(res.comp.perErr, 1.7, atol = 0.5)
    @test isapprox(res.incomp.rtCorr, 452, atol = 3)
    @test isapprox(res.incomp.sdRtCorr, 101, atol = 3)
    @test isapprox(res.incomp.perErr, 6.9, atol = 0.5)

end
