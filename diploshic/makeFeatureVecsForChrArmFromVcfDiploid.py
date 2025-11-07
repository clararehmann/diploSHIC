import os
import allel
import h5py
import numpy as np
import sys
import time
import scipy.stats
from joblib import Parallel, delayed
from scipy.spatial.distance import squareform
from diploshic.fvTools import *

if not len(sys.argv) in [14, 16, 17, 18]:
    sys.exit(
        "usage:\npython makeFeatureVecsForChrArmFromVcfDiploid.py vcfFileName chrArm chrLen targetPop winSize numSubWins maskFileName unmaskedFracCutoff unmaskedGenoFracCutoff sampleToPopFileName statFileName outFileName threads [segmentStart segmentEnd] [windowOffset]\n"
    )

# Handle different argument combinations
if len(sys.argv) == 18:  # All optional args: threads segmentStart segmentEnd windowOffset
    (
        vcfFileName,
        chrArm,
        chrLen,
        targetPop,
        winSize,
        numSubWins,
        maskFileName,
        unmaskedFracCutoff,
        unmaskedGenoFracCutoff,
        sampleToPopFileName,
        statFileName,
        outfn,
        threads,
        segmentStart,
        segmentEnd,
        windowOffset,
    ) = sys.argv[1:]
    threads, segmentStart, segmentEnd, windowOffset = int(threads), int(segmentStart), int(segmentEnd), int(windowOffset)
elif len(sys.argv) == 17:  # threads + segmentStart+segmentEnd OR threads + windowOffset
    # Check if we have segmentStart and segmentEnd by seeing if the 14th arg looks like a reasonable coordinate
    try:
        potential_segment_start = int(sys.argv[14])
        potential_segment_end = int(sys.argv[15])
        potential_window_offset = int(sys.argv[16])
        # If all three parse as integers, assume threads, segmentStart, segmentEnd, windowOffset
        threads = int(sys.argv[13])
        segmentStart, segmentEnd, windowOffset = potential_segment_start, potential_segment_end, potential_window_offset
        # Extract the base arguments
        (
            vcfFileName,
            chrArm,
            chrLen,
            targetPop,
            winSize,
            numSubWins,
            maskFileName,
            unmaskedFracCutoff,
            unmaskedGenoFracCutoff,
            sampleToPopFileName,
            statFileName,
            outfn,
        ) = sys.argv[1:13]
    except (ValueError, IndexError):
        # If parsing fails, treat as threads + windowOffset
        (
            vcfFileName,
            chrArm,
            chrLen,
            targetPop,
            winSize,
            numSubWins,
            maskFileName,
            unmaskedFracCutoff,
            unmaskedGenoFracCutoff,
            sampleToPopFileName,
            statFileName,
            outfn,
            threads,
            windowOffset,
        ) = sys.argv[1:]
        segmentStart = None
        threads, windowOffset = int(threads), int(windowOffset)
elif len(sys.argv) == 16:  # threads + segmentStart + segmentEnd
    (
        vcfFileName,
        chrArm,
        chrLen,
        targetPop,
        winSize,
        numSubWins,
        maskFileName,
        unmaskedFracCutoff,
        unmaskedGenoFracCutoff,
        sampleToPopFileName,
        statFileName,
        outfn,
        threads,
        segmentStart,
        segmentEnd,
    ) = sys.argv[1:]
    threads, segmentStart, segmentEnd = int(threads), int(segmentStart), int(segmentEnd)
    windowOffset = 0
else:  # len(sys.argv) == 14, just threads, no other optional args
    (
        vcfFileName,
        chrArm,
        chrLen,
        targetPop,
        winSize,
        numSubWins,
        maskFileName,
        unmaskedFracCutoff,
        unmaskedGenoFracCutoff,
        sampleToPopFileName,
        statFileName,
        outfn,
        threads,
    ) = sys.argv[1:]
    threads = int(threads)
    segmentStart = None
    windowOffset = 0

unmaskedFracCutoff = float(unmaskedFracCutoff)
if unmaskedFracCutoff < 0.0 or unmaskedFracCutoff > 1.0:
    sys.exit(
        "unmaskedFracCutoff=%s but must be within [0, 1]. AAAAARRRRGHHHHHH!!!\n"
        % (unmaskedFracCutoff)
    )
unmaskedGenoFracCutoff = float(unmaskedGenoFracCutoff)
if unmaskedGenoFracCutoff < 0.0 or unmaskedGenoFracCutoff > 1.0:
    sys.exit(
        "unmaskedGenoFracCutoff=%s but must be within [0, 1]. AAAAARRRRGHHHHHH!!!\n"
        % (unmaskedGenoFracCutoff)
    )
chrLen, winSize, numSubWins = int(chrLen), int(winSize), int(numSubWins)
assert winSize % numSubWins == 0 and numSubWins > 1
subWinSize = int(winSize / numSubWins)


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


def readSampleToPopFile(sampleToPopFileName):
    table = {}
    with open(sampleToPopFileName) as sampleToPopFile:
        for line in sampleToPopFile:
            sample, pop = line.strip().split()
            table[sample] = pop
    return table


# Phase timing for performance analysis
phase_times = {}
phase_start = time.perf_counter()

vcfFile = allel.read_vcf(vcfFileName)
chroms = vcfFile["variants/CHROM"]
positions = np.extract(chroms == chrArm, vcfFile["variants/POS"])

phase_times["vcf_load"] = time.perf_counter() - phase_start

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

phase_start = time.perf_counter()
samples = vcfFile["samples"]
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
    vcfFile["calldata/GT"],
    [i for i in range(len(chroms)) if chroms[i] == chrArm],
    axis=0,
)
genos = allel.GenotypeArray(rawgenos).subset(sel1=sampleIndicesToKeep)
phase_times["data_prep"] = time.perf_counter() - phase_start

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
    positions = [positions[i] for i in snpIndicesToKeep]
    genos = allel.GenotypeArray(genos.subset(sel0=snpIndicesToKeep))

if isHaploidVcfGenoArray(genos):
    sys.stderr.write(
        "Detected haploid input. Converting into diploid individuals (combining haplotypes in order).\n"
    )
    genos = diploidizeGenotypeArray(genos)

phase_start = time.perf_counter()
alleleCounts = genos.count_alleles()

# remove all but mono/biallelic unmasked sites
isBiallelic = alleleCounts.is_biallelic()
for i in range(len(isBiallelic)):
    if not (
        isBiallelic[i]
        and calledGenoFracAtSite(genos[i]) >= unmaskedGenoFracCutoff
    ):
        unmasked[positions[i] - 1] = False
snpIndicesToKeep = [
    i for i in range(len(positions)) if unmasked[positions[i] - 1]
]
genos = allel.GenotypeArray(genos.subset(sel0=snpIndicesToKeep))
positions = [positions[i] for i in snpIndicesToKeep]
alleleCounts = allel.AlleleCountsArray(
    [[alleleCounts[i][0], max(alleleCounts[i][1:])] for i in snpIndicesToKeep]
)
phase_times["snp_filtering"] = time.perf_counter() - phase_start

statNames = [
    "pi",
    "thetaW",
    "tajD",
    "distVar",
    "distSkew",
    "distKurt",
    "nDiplos",
    "diplo_H1",
    "diplo_H12",
    "diplo_H2/H1",
    "diplo_ZnS",
    "diplo_Omega",
]

subWinBounds = getSubWinBounds(chrLen, subWinSize, windowOffset)

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

# Handle auto-detect for threads
if threads == 0:
    import os
    threads = os.cpu_count() or 1

# Convert positions and alleleCounts to numpy arrays if they're not already
# (they may have been converted to lists during filtering)
if not isinstance(positions, np.ndarray):
    positions = np.array(positions)
if not isinstance(alleleCounts, np.ndarray):
    alleleCounts = np.array(alleleCounts)

# Parallel execution for threads > 1
if threads > 1:
    sys.stderr.write(f"Using parallel mode with {threads} threads.\n")

    # Build lightweight list of subwindows to process (just indices and boundaries)
    subwindows_to_process = []
    subWinIndex = 0
    for subWinStart in range(firstSubWinStart, lastSubWinStart + 1, subWinSize):
        subWinEnd = subWinStart + subWinSize - 1
        if subWinEnd > chrLen:
            break

        # Only process windows within segment bounds if specified
        if segmentStart != None and not (subWinStart >= segmentStart and subWinEnd <= segmentEnd):
            subWinIndex += 1
            continue

        subwindows_to_process.append((subWinIndex, subWinStart, subWinEnd))
        subWinIndex += 1

    # Batch subwindows into chunks to reduce overhead while enabling load balancing
    # Use more chunks (threads * 10-20) to allow dynamic work distribution
    # This helps when some chunks are computationally heavier than others
    total_subwindows = len(subwindows_to_process)
    target_num_jobs = threads * 15  # ~15 chunks per worker for good load balancing
    chunk_size = max(1, total_subwindows // target_num_jobs)

    subwindow_chunks = []
    for i in range(0, total_subwindows, chunk_size):
        chunk = subwindows_to_process[i:i+chunk_size]
        subwindow_chunks.append(chunk)

    sys.stderr.write(f"Processing {total_subwindows} subwindows in {len(subwindow_chunks)} chunks (~{chunk_size} subwindows per chunk).\n")

    # Worker function - processes a BATCH of subwindows
    # Receives all data explicitly to enable memory mapping
    # Joblib will automatically memory-map large numpy arrays (genos, positions, alleleCounts)
    def process_subwindow_batch(subwin_batch, genos, positions, alleleCounts, snpIndicesInSubWins, unmasked, statNames, unmaskedFracCutoff):
        # Process each subwindow in the batch and return results for all
        batch_results = []

        for subwin_tuple in subwin_batch:
            subWinIndex, subWinStart, subWinEnd = subwin_tuple

            snpIndices = snpIndicesInSubWins[subWinIndex]
            unmaskedFrac = unmasked[subWinStart - 1 : subWinEnd].count(True) / float(
                subWinEnd - subWinStart + 1
            )

            # Build tuple: (subWinIndex, is_good, stat_values_dict)
            if len(snpIndices) > 0 and unmaskedFrac >= unmaskedFracCutoff:
                genosInSubWin = allel.GenotypeArray(genos.subset(sel0=snpIndices))
                genosNAlt = genosInSubWin.to_n_alt()

                # Get allele counts and positions for this subwindow
                snpLocsInSubWin = positions.take(snpIndices)
                alleleCountsInSubWin = alleleCounts.take(snpIndices, axis=0)

                # Compute statistics for this subwindow
                # This logic is extracted from calcAndAppendStatValForScanDiplo in fvTools.py
                stat_results = {}

                for statName in statNames:
                    if statName == "tajD":
                        stat_results[statName] = allel.stats.diversity.tajima_d(
                            alleleCountsInSubWin, pos=snpLocsInSubWin, start=subWinStart, stop=subWinEnd
                        )
                    elif statName == "pi":
                        stat_results[statName] = allel.stats.diversity.sequence_diversity(
                            snpLocsInSubWin,
                            alleleCountsInSubWin,
                            start=subWinStart,
                            stop=subWinEnd,
                            is_accessible=unmasked,
                        )
                    elif statName == "thetaW":
                        stat_results[statName] = allel.stats.diversity.watterson_theta(
                            snpLocsInSubWin,
                            alleleCountsInSubWin,
                            start=subWinStart,
                            stop=subWinEnd,
                            is_accessible=unmasked,
                        )
                    elif statName == "thetaH":
                        stat_results[statName] = thetah(
                            snpLocsInSubWin,
                            alleleCountsInSubWin,
                            start=subWinStart,
                            stop=subWinEnd,
                            is_accessible=unmasked,
                        )
                    elif statName == "nDiplos":
                        diplotypeCounts = dps.getHaplotypeFreqSpec(genosNAlt)
                        nDiplos = diplotypeCounts[genosNAlt.shape[1]]
                        stat_results["nDiplos"] = nDiplos
                        diplotypeCounts = diplotypeCounts[:-1]
                        dh1 = garudH1(diplotypeCounts)
                        dh2 = garudH2(diplotypeCounts)
                        dh12 = garudH12(diplotypeCounts)
                        if "diplo_H1" in statNames:
                            stat_results["diplo_H1"] = dh1
                        if "diplo_H12" in statNames:
                            stat_results["diplo_H12"] = dh12
                        if "diplo_H2/H1" in statNames:
                            stat_results["diplo_H2/H1"] = dh2 / dh1
                    elif statName == "diplo_ZnS":
                        if genosNAlt.shape[0] == 1:
                            stat_results["diplo_ZnS"] = 0.0
                            stat_results["diplo_Omega"] = 0.0
                        else:
                            r2Matrix = allel.stats.ld.rogers_huff_r(genosNAlt)
                            r2Matrix2 = squareform(r2Matrix ** 2)
                            stat_results["diplo_ZnS"] = np.nanmean(r2Matrix)
                            stat_results["diplo_Omega"] = dps.omega(r2Matrix2)[0]
                    elif statName == "distVar":
                        dists = dps.pairwiseDiffsDiplo(genosNAlt) / float(
                            unmasked[subWinStart - 1 : subWinEnd].count(True)
                        )
                        stat_results["distVar"] = np.var(dists, ddof=1)
                        stat_results["distSkew"] = scipy.stats.skew(dists)
                        stat_results["distKurt"] = scipy.stats.kurtosis(dists)
                    # Skip stats that are side-effects of other stats
                    elif statName in ["diplo_H12", "diplo_H2/H1", "diplo_Omega", "distSkew", "distKurt"]:
                        pass  # Already handled above

                batch_results.append((subWinIndex, True, stat_results))
            else:
                # Monomorphic window - return default values
                # This logic is extracted from appendStatValsForMonomorphicForScan in fvTools.py
                stat_results = {}
                for statName in statNames:
                    if statName == "tajD":
                        stat_results[statName] = 0.0
                    elif statName == "pi":
                        stat_results[statName] = 0.0
                    elif statName == "thetaW":
                        stat_results[statName] = 0.0
                    elif statName == "thetaH":
                        stat_results[statName] = 0.0
                    elif statName == "nDiplos":
                        stat_results[statName] = 1
                    elif statName == "diplo_H1":
                        stat_results["diplo_H1"] = 1.0
                        if "diplo_H12" in statNames:
                            stat_results["diplo_H12"] = 1.0
                        if "diplo_H2/H1" in statNames:
                            stat_results["diplo_H2/H1"] = 0.0
                    elif statName == "diplo_ZnS":
                        stat_results["diplo_ZnS"] = 0.0
                        stat_results["diplo_Omega"] = 0.0
                    elif statName == "distVar":
                        stat_results[statName] = 0.0
                        stat_results["distSkew"] = 0.0
                        stat_results["distKurt"] = 0.0
                    # Skip stats that are side-effects
                    elif statName in ["diplo_H12", "diplo_H2/H1", "diplo_Omega", "distSkew", "distKurt"]:
                        pass  # Already handled

                batch_results.append((subWinIndex, False, stat_results))

        return batch_results

    # Process subwindow chunks in parallel using joblib (multiprocessing with memory mapping)
    # backend='loky' uses multiprocessing with automatic memory mapping for large numpy arrays
    # verbose=0 suppresses progress output
    # Pass large arrays explicitly to enable automatic memory mapping
    phase_start = time.perf_counter()
    batch_results = Parallel(n_jobs=threads, backend='loky', verbose=0)(
        delayed(process_subwindow_batch)(chunk, genos, positions, alleleCounts, snpIndicesInSubWins, unmasked, statNames, unmaskedFracCutoff)
        for chunk in subwindow_chunks
    )
    phase_times["parallel_compute"] = time.perf_counter() - phase_start

    # Flatten batch results - each batch returns a list of (subWinIndex, is_good, stat_results)
    phase_start = time.perf_counter()
    results = []
    for batch in batch_results:
        results.extend(batch)

    # Sort by subWinIndex to ensure correct order
    results.sort(key=lambda x: x[0])
    phase_times["result_processing"] = time.perf_counter() - phase_start

    for subWinIndex, is_good, stat_results in results:
        goodSubWins.append(is_good)
        for statName in statNames:
            statVals[statName].append(stat_results[statName])

        # Write to stat file if requested
        if statFileName and is_good:
            # Need to reconstruct subWinStart and subWinEnd from subWinIndex
            subWinStart = firstSubWinStart + (subWinIndex * subWinSize)
            subWinEnd = subWinStart + subWinSize - 1
            statFile.write(
                "\t".join(
                    [chrArm, str(subWinStart), str(subWinEnd)]
                    + [str(stat_results[statName]) for statName in statNames]
                )
                + "\n"
            )

        # Check if we can write output (sliding window logic)
        if goodSubWins[-numSubWins:].count(True) == numSubWins:
            outVec = []
            for statName in statNames:
                outVec += normalizeFeatureVec(statVals[statName][-numSubWins:])

            # Calculate midpoint
            subWinEnd = firstSubWinStart + ((subWinIndex + 1) * subWinSize) - 1
            midSubWinEnd = int(subWinEnd - subWinSize * (numSubWins // 2))
            midSubWinStart = midSubWinEnd - subWinSize + 1

            outFile.write(
                "%s\t%d\t%d\t%d-%d\t"
                % (
                    chrArm,
                    midSubWinStart,
                    midSubWinEnd,
                    subWinEnd - winSize + 1,
                    subWinEnd,
                )
                + "\t".join([str(x) for x in outVec])
            )
            outFile.write("\n")

else:
    # Serial mode (threads == 1)
    for subWinStart in range(firstSubWinStart, lastSubWinStart + 1, subWinSize):
        subWinEnd = subWinStart + subWinSize - 1
        if subWinEnd > chrLen:  # Skip windows that exceed chromosome length
            break
        unmaskedFrac = unmasked[subWinStart - 1 : subWinEnd].count(True) / float(
            subWinEnd - subWinStart + 1
        )
        if (
            segmentStart == None
            or subWinStart >= segmentStart
            and subWinEnd <= segmentEnd
        ):
            sys.stderr.write(
                "%d-%d num unmasked snps: %d; unmasked frac: %f\n"
                % (
                    subWinStart,
                    subWinEnd,
                    len(snpIndicesInSubWins[subWinIndex]),
                    unmaskedFrac,
                )
            )
        if (
            len(snpIndicesInSubWins[subWinIndex]) > 0
            and unmaskedFrac >= unmaskedFracCutoff
        ):
            genosInSubWin = allel.GenotypeArray(
                genos.subset(sel0=snpIndicesInSubWins[subWinIndex])
            )
            statValStr = []
            for statName in statNames:
                calcAndAppendStatValForScanDiplo(
                    alleleCounts,
                    positions,
                    statName,
                    subWinStart,
                    subWinEnd,
                    statVals,
                    subWinIndex,
                    genosInSubWin,
                    unmasked,
                )
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
                appendStatValsForMonomorphicForScan(
                    statName, statVals, subWinIndex
                )
            goodSubWins.append(False)
        if goodSubWins[-numSubWins:].count(True) == numSubWins:
            outVec = []
            for statName in statNames:
                outVec += normalizeFeatureVec(statVals[statName][-numSubWins:])
            midSubWinEnd = int(subWinEnd - subWinSize * (numSubWins // 2))
            midSubWinStart = midSubWinEnd - subWinSize + 1
            outFile.write(
                "%s\t%d\t%d\t%d-%d\t"
                % (
                    chrArm,
                    midSubWinStart,
                    midSubWinEnd,
                    subWinEnd - winSize + 1,
                    subWinEnd,
                )
                + "\t".join([str(x) for x in outVec])
            )
            outFile.write("\n")
        subWinIndex += 1
if statFileName:
    statFile.close()
outFile.close()

total_time = time.perf_counter() - startTime
sys.stderr.write(
    "completed in %g seconds\n" % total_time
)

# Print phase timing breakdown if available
if phase_times:
    sys.stderr.write("\n=== Phase Timing Breakdown ===\n")
    for phase_name, phase_time in sorted(phase_times.items()):
        pct = (phase_time / total_time) * 100 if total_time > 0 else 0
        sys.stderr.write(f"{phase_name:20s}: {phase_time:7.2f}s ({pct:5.1f}%)\n")

    # Calculate serial fraction (Amdahl's law analysis)
    if "parallel_compute" in phase_times:
        serial_time = total_time - phase_times.get("parallel_compute", 0)
        parallel_frac = phase_times.get("parallel_compute", 0) / total_time if total_time > 0 else 0
        serial_frac = serial_time / total_time if total_time > 0 else 0
        sys.stderr.write(f"\n{'Serial fraction':20s}: {serial_frac:7.1%} ({serial_time:.2f}s)\n")
        sys.stderr.write(f"{'Parallel fraction':20s}: {parallel_frac:7.1%} ({phase_times.get('parallel_compute', 0):.2f}s)\n")
    sys.stderr.write("==============================\n")
