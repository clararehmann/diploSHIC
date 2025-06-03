import os
import allel
import h5py
import numpy as np
import sys
import time
from diploshic.fvTools import *

if not len(sys.argv) in [13, 15, 16, 17]:
    sys.exit(
        "usage:\npython makeFeatureVecsForChrArmFromVcf_ogSHIC.py chrArmFileName chrArm chrLen targetPop winSize numSubWins maskFileName sampleToPopFileName ancestralArmFaFileName statFileName outFileName [segmentStart segmentEnd] [windowOffset]\n"
    )

# Handle different argument combinations
if len(sys.argv) == 17:  # All optional args: segmentStart segmentEnd windowOffset
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
    ) = sys.argv[1:]
    segmentStart, segmentEnd, windowOffset = int(segmentStart), int(segmentEnd), int(windowOffset)
elif len(sys.argv) == 16:  # Could be segmentStart+segmentEnd OR just windowOffset
    # Check if we have segmentStart and segmentEnd by seeing if the 13th arg looks like a reasonable coordinate
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
        ) = sys.argv[1:]
        segmentStart = None
        windowOffset = int(windowOffset)
elif len(sys.argv) == 15:  # segmentStart and segmentEnd only
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
    ) = sys.argv[1:]
    segmentStart, segmentEnd = int(segmentStart), int(segmentEnd)
    windowOffset = 0
else:  # len(sys.argv) == 13, no optional args
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
    ) = sys.argv[1:]
    segmentStart = None
    windowOffset = 0

unmaskedFracCutoff = float(unmaskedFracCutoff)
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
    positions = [positions[i] for i in snpIndicesToKeep]
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
positions = [positions[i] for i in snpIndicesToKeep]
alleleCounts = allel.AlleleCountsArray(
    [[alleleCounts[i][0], max(alleleCounts[i][1:])] for i in snpIndicesToKeep]
)
if not ancestralArmFaFileName.lower() in ["none", "false"]:
    mapping = [mapping[i] for i in snpIndicesToKeep]
    alleleCounts = alleleCounts.map_alleles(mapping)
haps = genos.to_haplotypes()

subWinBounds = getSubWinBounds(chrLen, subWinSize, windowOffset)
precomputedStats = {}  # not using this

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
        hapsInSubWin = allel.HaplotypeArray(
            haps.subset(sel0=snpIndicesInSubWins[subWinIndex])
        )
        statValStr = []
        for statName in statNames:
            calcAndAppendStatValForScan(
                alleleCounts,
                positions,
                statName,
                subWinStart,
                subWinEnd,
                statVals,
                subWinIndex,
                hapsInSubWin,
                unmasked,
                precomputedStats,
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
sys.stderr.write(
    "completed in %g seconds\n" % (time.perf_counter() - startTime)
)
