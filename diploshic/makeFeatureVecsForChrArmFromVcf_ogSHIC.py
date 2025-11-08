import os
import allel
import h5py
import numpy as np
import sys
import time
from joblib import Parallel, delayed
from diploshic.fvTools import *

if not len(sys.argv) in [13, 14, 15, 16, 17, 18]:
    sys.exit(
        "usage:\npython makeFeatureVecsForChrArmFromVcf_ogSHIC.py chrArmFileName chrArm chrLen targetPop winSize numSubWins maskFileName sampleToPopFileName ancestralArmFaFileName statFileName outFileName [segmentStart segmentEnd] [windowOffset] [threads]\n"
    )

# Handle different argument combinations
# First, check if last argument is threads (small positive integer)
threads = 1  # default
last_arg_is_threads = False
if len(sys.argv) >= 14:
    try:
        potential_threads = int(sys.argv[-1])
        # If it's a reasonable thread count (1-128), treat it as threads
        if 0 <= potential_threads <= 128:
            threads = potential_threads
            last_arg_is_threads = True
    except (ValueError, IndexError):
        pass

# Adjust argument count if last arg was threads
effective_argc = len(sys.argv) - 1 if last_arg_is_threads else len(sys.argv)

if effective_argc == 17:  # All optional args: segmentStart segmentEnd windowOffset
    (
        chrArmFileName,
        chrArm,
        chrLen,
        targetPop,
        winSize,
        numSubWins,
        maskFileName,
        unmaskedFracCutoff,
        sampleToPopFileName,
        ancestralArmFaFileName,
        statFileName,
        outfn,
        segmentStart,
        segmentEnd,
        windowOffset,
    ) = sys.argv[1:16] if last_arg_is_threads else sys.argv[1:]
    segmentStart, segmentEnd, windowOffset = int(segmentStart), int(segmentEnd), int(windowOffset)
elif effective_argc == 16:  # Could be segmentStart+segmentEnd+windowOffset
    end_idx = 16 if last_arg_is_threads else len(sys.argv)
    try:
        potential_segment_start = int(sys.argv[13])
        potential_segment_end = int(sys.argv[14])
        potential_window_offset = int(sys.argv[15])
        # If all three parse as integers, assume segmentStart, segmentEnd, windowOffset
        segmentStart, segmentEnd, windowOffset = potential_segment_start, potential_segment_end, potential_window_offset
        # Extract the base arguments
        (
            chrArmFileName,
            chrArm,
            chrLen,
            targetPop,
            winSize,
            numSubWins,
            maskFileName,
            unmaskedFracCutoff,
            sampleToPopFileName,
            ancestralArmFaFileName,
            statFileName,
            outfn,
        ) = sys.argv[1:13]
    except (ValueError, IndexError):
        # If parsing fails, treat as just windowOffset
        (
            chrArmFileName,
            chrArm,
            chrLen,
            targetPop,
            winSize,
            numSubWins,
            maskFileName,
            unmaskedFracCutoff,
            sampleToPopFileName,
            ancestralArmFaFileName,
            statFileName,
            outfn,
            windowOffset,
        ) = sys.argv[1:end_idx]
        segmentStart = None
        windowOffset = int(windowOffset)
elif effective_argc == 15:  # segmentStart and segmentEnd only
    end_idx = 15 if last_arg_is_threads else len(sys.argv)
    (
        chrArmFileName,
        chrArm,
        chrLen,
        targetPop,
        winSize,
        numSubWins,
        maskFileName,
        unmaskedFracCutoff,
        sampleToPopFileName,
        ancestralArmFaFileName,
        statFileName,
        outfn,
        segmentStart,
        segmentEnd,
    ) = sys.argv[1:end_idx]
    segmentStart, segmentEnd = int(segmentStart), int(segmentEnd)
    windowOffset = 0
else:  # len(sys.argv) == 13, no optional args (or 14 with threads)
    end_idx = 13 if last_arg_is_threads else len(sys.argv)
    (
        chrArmFileName,
        chrArm,
        chrLen,
        targetPop,
        winSize,
        numSubWins,
        maskFileName,
        unmaskedFracCutoff,
        sampleToPopFileName,
        ancestralArmFaFileName,
        statFileName,
        outfn,
    ) = sys.argv[1:end_idx]
    segmentStart = None
    windowOffset = 0

unmaskedFracCutoff = float(unmaskedFracCutoff)
chrLen, winSize, numSubWins = int(chrLen), int(winSize), int(numSubWins)
assert winSize % numSubWins == 0 and numSubWins > 1
subWinSize = int(winSize / numSubWins)

# Auto-detect number of CPUs if threads==0
if threads == 0:
    import os
    threads = os.cpu_count() or 1
    sys.stderr.write(f"Auto-detected {threads} CPUs for parallel processing.\n")


def getSubWinBounds(chrLen, subWinSize, windowOffset=0):
    # Start windows from windowOffset + 1 instead of 1
    firstSubWinStart = windowOffset + 1
    lastSubWinEnd = chrLen - ((chrLen - windowOffset) % subWinSize)
    lastSubWinStart = lastSubWinEnd - subWinSize + 1
    
    subWinBounds = []
    for subWinStart in range(firstSubWinStart, lastSubWinStart + 1, subWinSize):
        subWinEnd = subWinStart + subWinSize - 1
        if subWinEnd <= chrLen:  # Don't exceed chromosome length
            subWinBounds.append((subWinStart, subWinEnd))
    return subWinBounds


def getSnpIndicesInSubWins(subWinSize, lastSubWinEnd, snpLocs, windowOffset=0):
    subWinStart = windowOffset + 1  # Start from offset
    subWinEnd = subWinStart + subWinSize - 1
    snpIndicesInSubWins = [[]]
    
    for i in range(len(snpLocs)):
        while snpLocs[i] <= lastSubWinEnd and not (
            snpLocs[i] >= subWinStart and snpLocs[i] <= subWinEnd
        ):
            subWinStart += subWinSize
            subWinEnd += subWinSize
            snpIndicesInSubWins.append([])
        if snpLocs[i] <= lastSubWinEnd:
            snpIndicesInSubWins[-1].append(i)
    
    # Add empty windows for any remaining subwindows
    while subWinEnd < lastSubWinEnd:
        snpIndicesInSubWins.append([])
        subWinStart += subWinSize
        subWinEnd += subWinSize
    return snpIndicesInSubWins


chrArmFile = allel.read_vcf(chrArmFileName)
chroms = chrArmFile["variants/CHROM"]
positions = np.extract(chroms == chrArm, chrArmFile["variants/POS"])

if maskFileName.lower() in ["none", "false"]:
    sys.stderr.write(
        "Warning: a mask.fa file for the chr arm with all masked sites N'ed out is strongly recommended"
        + " (pass in the reference to remove Ns at the very least)!\n"
    )
    unmasked = [True] * chrLen
else:
    unmasked = readMaskDataForScan(maskFileName, chrArm)
    assert len(unmasked) == chrLen

if statFileName.lower() in ["none", "false"]:
    statFileName = None

samples = chrArmFile["samples"]
if not sampleToPopFileName.lower() in ["none", "false"]:
    sampleToPop = readSampleToPopFile(sampleToPopFileName)
    sampleIndicesToKeep = [
        i
        for i in range(len(samples))
        if sampleToPop.get(samples[i], "popNotFound!") == targetPop
    ]
else:
    sampleIndicesToKeep = [i for i in range(len(samples))]

rawgenos = np.take(
    chrArmFile["calldata/GT"],
    [i for i in range(len(chroms)) if chroms[i] == chrArm],
    axis=0,
)
genos = allel.GenotypeArray(rawgenos)
refAlleles = np.extract(chroms == chrArm, chrArmFile["variants/REF"])
altAlleles = np.extract(chroms == chrArm, chrArmFile["variants/ALT"])
if segmentStart != None:
    snpIndicesToKeep = [
        i
        for i in range(len(positions))
        if segmentStart <= positions[i] <= segmentEnd
    ]
    if len(snpIndicesToKeep) == 0:
        sys.exit(
            "Error: no SNPs in the given segment of the chr arm; exiting\n"
        )
    positions = np.array([positions[i] for i in snpIndicesToKeep])
    refAlleles = [refAlleles[i] for i in snpIndicesToKeep]
    altAlleles = [altAlleles[i] for i in snpIndicesToKeep]
    genos = allel.GenotypeArray(genos.subset(sel0=snpIndicesToKeep))
genos = allel.GenotypeArray(genos.subset(sel1=sampleIndicesToKeep))
alleleCounts = genos.count_alleles()

# remove all but mono/biallelic unmasked sites
isBiallelic = alleleCounts.is_biallelic()
for i in range(len(isBiallelic)):
    if not isBiallelic[i]:
        unmasked[positions[i] - 1] = False

# polarize
if not ancestralArmFaFileName.lower() in ["none", "false"]:
    sys.stderr.write("polarizing snps\n")
    ancArm = readFaArm(ancestralArmFaFileName, chrArm).upper()
    startTime = time.perf_counter()
    # NOTE: mapping specifies which alleles to swap counts for based on polarization; leaves unpolarized snps alone
    # NOTE: those snps need to be filtered later on (as done below)!
    # this will also remove sites that could not be polarized
    mapping, unmasked = polarizeSnps(
        unmasked, positions, refAlleles, altAlleles, ancArm
    )
    sys.stderr.write("took %s seconds\n" % (time.perf_counter() - startTime))
    statNames = [
        "pi",
        "thetaW",
        "tajD",
        "thetaH",
        "fayWuH",
        "maxFDA",
        "HapCount",
        "H1",
        "H12",
        "H2/H1",
        "ZnS",
        "Omega",
        "distVar",
        "distSkew",
        "distKurt",
    ]
else:
    statNames = [
        "pi",
        "thetaW",
        "tajD",
        "HapCount",
        "H1",
        "H12",
        "H2/H1",
        "ZnS",
        "Omega",
        "distVar",
        "distSkew",
        "distKurt",
    ]

snpIndicesToKeep = [
    i for i in range(len(positions)) if unmasked[positions[i] - 1]
]
genos = allel.GenotypeArray(genos.subset(sel0=snpIndicesToKeep))
positions = np.array([positions[i] for i in snpIndicesToKeep])
alleleCounts = allel.AlleleCountsArray(
    [[alleleCounts[i][0], max(alleleCounts[i][1:])] for i in snpIndicesToKeep]
)
if not ancestralArmFaFileName.lower() in ["none", "false"]:
    mapping = [mapping[i] for i in snpIndicesToKeep]
    alleleCounts = alleleCounts.map_alleles(mapping)
haps = genos.to_haplotypes()

subWinBounds = getSubWinBounds(chrLen, subWinSize, windowOffset)
precomputedStats = {}  # not using this

# Initialize phase timing
phase_times = {}
phase_start = time.perf_counter()

# Track data preparation phase (everything up to this point)
phase_times["data_preparation"] = time.perf_counter() - phase_start

header = "chrom classifiedWinStart classifiedWinEnd bigWinRange".split()
statHeader = "chrom start end".split()
for statName in statNames:
    statHeader.append(statName)
    for i in range(numSubWins):
        header.append("%s_win%d" % (statName, i))
statHeader = "\t".join(statHeader)
header = "\t".join(header)
outFile = open(outfn, "w")
outFile.write(header + "\n")
statVals = {}
for statName in statNames:
    statVals[statName] = []

startTime = time.perf_counter()
goodSubWins = []
lastSubWinEnd = chrLen - ((chrLen - windowOffset) % subWinSize)
snpIndicesInSubWins = getSnpIndicesInSubWins(
    subWinSize, lastSubWinEnd, positions, windowOffset
)
subWinIndex = 0
firstSubWinStart = windowOffset + 1
lastSubWinStart = lastSubWinEnd - subWinSize + 1
if statFileName:
    statFile = open(statFileName, "w")
    statFile.write(statHeader + "\n")

# Choose between parallel and sequential execution
if threads > 1:
    sys.stderr.write(f"Using parallel mode with {threads} threads.\n")

    # Build list of subwindows to process
    subwindows_to_process = []
    subWinIndex = 0
    for subWinStart in range(firstSubWinStart, lastSubWinStart + 1, subWinSize):
        subWinEnd = subWinStart + subWinSize - 1
        if subWinEnd > chrLen:
            break
        if segmentStart != None and not (subWinStart >= segmentStart and subWinEnd <= segmentEnd):
            subWinIndex += 1
            continue
        subwindows_to_process.append((subWinIndex, subWinStart, subWinEnd))
        subWinIndex += 1

    # Batch subwindows into chunks
    total_subwindows = len(subwindows_to_process)
    target_num_jobs = threads * 15  # ~15 chunks per worker
    chunk_size = max(1, total_subwindows // target_num_jobs)

    subwindow_chunks = []
    for i in range(0, total_subwindows, chunk_size):
        subwindow_chunks.append(subwindows_to_process[i:i+chunk_size])

    # Worker function
    def process_subwindow_batch(subwin_batch, haps, positions, alleleCounts, snpIndicesInSubWins, unmasked, statNames, unmaskedFracCutoff):
        import allel
        import numpy as np
        import scipy.stats
        from diploshic import shicstats as dps

        batch_results = []
        for subwin_tuple in subwin_batch:
            subWinIndex, subWinStart, subWinEnd = subwin_tuple
            snpIndices = snpIndicesInSubWins[subWinIndex]
            unmaskedFrac = unmasked[subWinStart - 1 : subWinEnd].count(True) / float(subWinEnd - subWinStart + 1)

            if len(snpIndices) > 0 and unmaskedFrac >= unmaskedFracCutoff:
                hapsInSubWin = allel.HaplotypeArray(haps.subset(sel0=snpIndices))
                snpLocsInSubWin = positions.take(snpIndices)
                alleleCountsInSubWin = alleleCounts.take(snpIndices, axis=0)

                stat_results = {}
                for statName in statNames:
                    if statName == "tajD":
                        stat_results[statName] = allel.stats.diversity.tajima_d(
                            alleleCountsInSubWin, pos=snpLocsInSubWin, start=subWinStart, stop=subWinEnd
                        )
                    elif statName == "pi":
                        stat_results[statName] = allel.stats.diversity.sequence_diversity(
                            snpLocsInSubWin, alleleCountsInSubWin, start=subWinStart, stop=subWinEnd, is_accessible=unmasked
                        )
                    elif statName == "thetaW":
                        stat_results[statName] = allel.stats.diversity.watterson_theta(
                            snpLocsInSubWin, alleleCountsInSubWin, start=subWinStart, stop=subWinEnd, is_accessible=unmasked
                        )
                    elif statName == "thetaH":
                        from diploshic.fvTools import thetah
                        stat_results[statName] = thetah(
                            snpLocsInSubWin, alleleCountsInSubWin, start=subWinStart, stop=subWinEnd, is_accessible=unmasked
                        )
                    elif statName == "fayWuH":
                        # Requires thetaH to be calculated first
                        stat_results[statName] = stat_results["thetaH"] - stat_results["pi"]
                    elif statName == "maxFDA":
                        from diploshic.fvTools import maxFDA
                        stat_results[statName] = maxFDA(
                            snpLocsInSubWin, alleleCountsInSubWin, start=subWinStart, stop=subWinEnd, is_accessible=unmasked
                        )
                    elif statName == "HapCount":
                        stat_results[statName] = len(hapsInSubWin.distinct())
                    elif statName == "H1":
                        h1, h12, h123, h21 = allel.stats.selection.garud_h(hapsInSubWin)
                        stat_results["H1"] = h1
                        if "H12" in statNames:
                            stat_results["H12"] = h12
                        if "H123" in statNames:
                            stat_results["H123"] = h123
                        if "H2/H1" in statNames:
                            stat_results["H2/H1"] = h21
                    elif statName == "ZnS":
                        r2Matrix = dps.computeR2Matrix(hapsInSubWin)
                        stat_results["ZnS"] = dps.ZnS(r2Matrix)[0]
                        if "Omega" in statNames:
                            stat_results["Omega"] = dps.omega(r2Matrix)[0]
                    elif statName == "distVar":
                        dists = dps.pairwiseDiffs(hapsInSubWin) / float(unmasked[subWinStart - 1 : subWinEnd].count(True))
                        stat_results["distVar"] = np.var(dists, ddof=1)
                        if "distSkew" in statNames:
                            stat_results["distSkew"] = scipy.stats.skew(dists)
                        if "distKurt" in statNames:
                            stat_results["distKurt"] = scipy.stats.kurtosis(dists)
                    elif statName in ["H12", "H123", "H2/H1", "Omega", "distSkew", "distKurt"]:
                        pass  # Already handled

                batch_results.append((subWinIndex, True, stat_results))
            else:
                # Monomorphic window
                stat_results = {}
                for statName in statNames:
                    if statName in ["tajD", "pi", "thetaW", "thetaH", "fayWuH", "maxFDA", "distVar", "distSkew", "distKurt", "ZnS", "Omega"]:
                        stat_results[statName] = 0.0
                    elif statName == "HapCount":
                        stat_results[statName] = 1
                    elif statName == "H1":
                        stat_results["H1"] = 1.0
                        if "H12" in statNames:
                            stat_results["H12"] = 1.0
                        if "H123" in statNames:
                            stat_results["H123"] = 1.0
                        if "H2/H1" in statNames:
                            stat_results["H2/H1"] = 0.0
                    elif statName in ["H12", "H123", "H2/H1"]:
                        pass  # Already handled

                batch_results.append((subWinIndex, False, stat_results))

        return batch_results

    # Process subwindow chunks in parallel
    loop_start = time.perf_counter()
    batch_results = Parallel(n_jobs=threads, backend='loky', verbose=0)(
        delayed(process_subwindow_batch)(chunk, haps, positions, alleleCounts, snpIndicesInSubWins, unmasked, statNames, unmaskedFracCutoff)
        for chunk in subwindow_chunks
    )

    # Flatten and sort results
    results = []
    for batch in batch_results:
        results.extend(batch)
    results.sort(key=lambda x: x[0])

    # Reconstruct statVals in sorted order and write feature vectors
    for subWinIndex, is_good, stat_results in results:
        goodSubWins.append(is_good)
        for statName in statNames:
            statVals[statName].append(stat_results[statName])

        # Write to stat file if requested
        if statFileName and is_good:
            subWinStart = firstSubWinStart + (subWinIndex * subWinSize)
            subWinEnd = subWinStart + subWinSize - 1
            statFile.write(
                "\t".join(
                    [chrArm, str(subWinStart), str(subWinEnd)]
                    + [str(stat_results[statName]) for statName in statNames]
                )
                + "\n"
            )

        # Write feature vector if we have enough good subwindows
        if goodSubWins[-numSubWins:].count(True) == numSubWins:
            outVec = []
            for statName in statNames:
                outVec += normalizeFeatureVec(statVals[statName][-numSubWins:])
            # Calculate window boundaries
            subWinStart = firstSubWinStart + (subWinIndex * subWinSize)
            subWinEnd = subWinStart + subWinSize - 1
            midSubWinEnd = int(subWinEnd - subWinSize * (numSubWins // 2))
            midSubWinStart = midSubWinEnd - subWinSize + 1
            outFile.write(
                "%s\t%d\t%d\t%d-%d\t"
                % (chrArm, midSubWinStart, midSubWinEnd, subWinEnd - winSize + 1, subWinEnd)
                + "\t".join([str(x) for x in outVec])
            )
            outFile.write("\n")

    phase_times["main_computation"] = time.perf_counter() - loop_start

else:
    # Sequential execution (threads == 1)
    loop_start = time.perf_counter()
    for subWinStart in range(firstSubWinStart, lastSubWinStart + 1, subWinSize):
        subWinEnd = subWinStart + subWinSize - 1
        if subWinEnd > chrLen:
            break
        unmaskedFrac = unmasked[subWinStart - 1 : subWinEnd].count(True) / float(subWinEnd - subWinStart + 1)
        if segmentStart == None or (subWinStart >= segmentStart and subWinEnd <= segmentEnd):
            sys.stderr.write(
                "%d-%d num unmasked snps: %d; unmasked frac: %f\n"
                % (subWinStart, subWinEnd, len(snpIndicesInSubWins[subWinIndex]), unmaskedFrac)
            )
        if len(snpIndicesInSubWins[subWinIndex]) > 0 and unmaskedFrac >= unmaskedFracCutoff:
            hapsInSubWin = allel.HaplotypeArray(haps.subset(sel0=snpIndicesInSubWins[subWinIndex]))
            statValStr = []
            for statName in statNames:
                calcAndAppendStatValForScan(
                    alleleCounts, positions, statName, subWinStart, subWinEnd,
                    statVals, subWinIndex, hapsInSubWin, unmasked, precomputedStats
                )
                statValStr.append("%s: %s" % (statName, statVals[statName][-1]))
            sys.stderr.write("\t".join(statValStr) + "\n")
            goodSubWins.append(True)
            if statFileName:
                statFile.write(
                    "\t".join(
                        [chrArm, str(subWinStart), str(subWinEnd)]
                        + [str(statVals[statName][-1]) for statName in statNames]
                    )
                    + "\n"
                )
        else:
            for statName in statNames:
                appendStatValsForMonomorphicForScan(statName, statVals, subWinIndex)
            goodSubWins.append(False)
        if goodSubWins[-numSubWins:].count(True) == numSubWins:
            outVec = []
            for statName in statNames:
                outVec += normalizeFeatureVec(statVals[statName][-numSubWins:])
            midSubWinEnd = int(subWinEnd - subWinSize * (numSubWins // 2))
            midSubWinStart = midSubWinEnd - subWinSize + 1
            outFile.write(
                "%s\t%d\t%d\t%d-%d\t"
                % (chrArm, midSubWinStart, midSubWinEnd, subWinEnd - winSize + 1, subWinEnd)
                + "\t".join([str(x) for x in outVec])
            )
            outFile.write("\n")
        subWinIndex += 1

    phase_times["main_computation"] = time.perf_counter() - loop_start

if statFileName:
    statFile.close()
outFile.close()

total_time = time.perf_counter() - startTime
sys.stderr.write("completed in %g seconds\n" % total_time)

# Print phase timing breakdown
sys.stderr.write("\n=== Phase Timing Breakdown ===\n")
for phase_name in ["data_preparation", "main_computation"]:
    if phase_name in phase_times:
        phase_time = phase_times[phase_name]
        pct = (phase_time / total_time) * 100 if total_time > 0 else 0
        sys.stderr.write(f"{phase_name:20s}: {phase_time:7.2f}s ({pct:5.1f}%)\n")
