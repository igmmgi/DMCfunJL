export DmcOb,
    DataRaw, DataOb, dmc_observed_data, flankerDataRaw, flankerData, simonDataRaw, simonData

struct DmcOb
    subject::DataFrame
    agg::DataFrame
end

struct DataRaw
    data::DataFrame
end
struct DataOb
    subject::DataFrame
    agg::DataFrame
end

nanmean(x) = mean(filter(!isnan, x))
nanstd(x) = std(filter(!isnan, x))

function _read_files(files::String, toskip)
    fn = glob(files)
    return (reduce(vcat, @. DataFrame(CSV.File(fn, skipto = toskip))))
end

function _read_files(files::Vector, toskip)
    fn = glob(files[1], files[2])
    return (reduce(vcat, @. DataFrame(CSV.File(fn, skipto = toskip))))
end

function dmc_observed_data(
    dat;
    nCAF = 5,
    nDelta = 19,
    outlier = [200, 1200],
    columns = ["Subject", "Comp", "RT", "Error"],
    compCoding = ["comp", "incomp"],
    errorCoding = [0, 1],
    toskip = -1,
)

    # read from external files
    dat = typeof(dat) != DataFrame ? _read_files(dat, toskip) : dat

    # select columns and check appropriate columns
    dat = dat[:, columns]
    if ncol(dat) != 4
        error("dat does not contain required/requested columns!")
    end
    if names(dat) != columns
        rename!(dat, columns)
    end

    # coding of compatibility columns
    if compCoding != ["comp", "incomp"]
        dat[:, :Comp] = ifelse.(dat[:, :Comp] .== compCoding[1], "comp", "incomp")
    end

    # coding of error coumns
    if errorCoding != [0, 1]
        dat[:, :Error] = ifelse.(dat[:, :Error] .== errorCoding[1], 0, 1)
    end

    # define outliers
    rtMin, rtMax = outlier[1], outlier[2]
    dat[:, :outlier] =
        ifelse.((dat[:, :RT] .<= rtMin) .| (dat[:, :RT] .>= rtMax), true, false)

    # aggregate across trials
    datSubject = combine(groupby(dat, [:Subject, :Comp])) do d
        N = nrow(d)
        nCor = sum(d.Error .== 0)
        nErr = sum(d.Error .== 1)
        nOut = sum(d.outlier .== 1)
        rtCor = nanmean(d.RT[(d.Error.==0).&(d.outlier.==0)])
        rtErr = nanmean(d.RT[(d.Error.==1).&(d.outlier.==0)])
        perErr = (nErr ./ (nErr .+ nCor)) * 100
        return (
            N = N,
            nCor = nCor,
            nErr = nErr,
            nOut = nOut,
            rtCor = rtCor,
            rtErr = rtErr,
            perErr = perErr,
        )
    end

    # aggregate across trials
    datSubject = sort(
        combine(groupby(dat, [:Subject, :Comp])) do d
            N = nrow(d)
            nCor = sum(d.Error .== 0)
            nErr = sum(d.Error .== 1)
            nOut = sum(d.outlier .== 1)
            rtCor = nanmean(d.RT[(d.Error.==0).&(d.outlier.==0)])
            rtErr = nanmean(d.RT[(d.Error.==1).&(d.outlier.==0)])
            perErr = (nErr ./ (nErr .+ nCor)) * 100
            return (
                N = N,
                nCor = nCor,
                nErr = nErr,
                nOut = nOut,
                rtCor = rtCor,
                rtErr = rtErr,
                perErr = perErr,
            )
        end,
        [:Subject, :Comp],
    )

    # aggregate across trials
    datAgg = sort(
        combine(groupby(datSubject, [:Comp])) do d
            N = nrow(d)
            rtCor = nanmean(d.rtCor)
            sdRtCor = nanstd(d.rtCor)
            seRtCor = sdRtCor / sqrt(N)
            rtErr = nanmean(filter(!isnan, d.rtErr))
            sdRtErr = nanstd(filter(!isnan, d.rtErr))
            seRtErr = sdRtErr / sqrt(N)
            perErr = nanmean(d.perErr)
            sdPerErr = nanstd(d.perErr)
            sePerErr = sdPerErr / sqrt(N)
            return (
                N = N,
                rtCor = rtCor,
                sdRtCor = sdRtCor,
                seRtCor = seRtCor,
                rtErr = rtErr,
                sdRtErr = sdRtErr,
                seRtErr = seRtErr,
                perErr = perErr,
                sdPerErr = sdPerErr,
                sePerErr = sePerErr,
            )
        end,
        [:Comp],
    )

    return DmcOb(datSubject, datAgg)

end

# data files
function flankerDataRaw()
    return DataRaw(DataFrame(CSV.File(joinpath(@__DIR__, "../data/flankerData.csv"))))
end
function simonDataRaw()
    return DataRaw(DataFrame(CSV.File(joinpath(@__DIR__, "../data/simonData.csv"))))
end

function flankerData()
    dat = dmc_observed_data(flankerDataRaw().data)
    return DataOb(dat.subject, dat.agg)
end
function simonData()
    dat = dmc_observed_data(simonDataRaw().data)
    return DataOb(dat.subject, dat.agg)
end
