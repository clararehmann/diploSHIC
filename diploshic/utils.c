#include <math.h>
#include <immintrin.h>  // AVX2 intrinsics
#ifdef __x86_64__
#include <cpuid.h>      // CPU feature detection
#endif

// CPU feature detection - check for AVX2 support
static int has_avx2_support() {
#ifdef __x86_64__
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid(7, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 5)) != 0;  // bit 5 is AVX2
    }
#endif
    return 0;
}

#ifdef __AVX2__
/*
 * SIMD-optimized r2 calculation using AVX2
 * Processes 8 samples simultaneously using 256-bit vectors
 * Expected speedup: 4-6x over scalar version
 */
double r2_avx2(int nSamps, int *haps, int i, int j){
	double pi  = 0.0;
	double pj  = 0.0;
	double pij = 0.0;
	double count = 0.0;

	int k;

	// Process 8 samples at a time with AVX2
	int vec_iterations = nSamps / 8;
	int remainder = nSamps % 8;

	// Accumulators for vectorized portion
	__m256i pi_vec  = _mm256_setzero_si256();
	__m256i pj_vec  = _mm256_setzero_si256();
	__m256i pij_vec = _mm256_setzero_si256();
	__m256i count_vec = _mm256_setzero_si256();

	// Constants for masking
	__m256i zero = _mm256_setzero_si256();
	__m256i one  = _mm256_set1_epi32(1);

	// Base pointers for the two SNPs
	int *hap_i_base = &haps[i * nSamps];
	int *hap_j_base = &haps[j * nSamps];

	// Vectorized loop: process 8 samples per iteration
	for(k = 0; k < vec_iterations * 8; k += 8){
		// Load 8 haplotype values for SNP i and SNP j
		__m256i hap_i_vec = _mm256_loadu_si256((__m256i*)&hap_i_base[k]);
		__m256i hap_j_vec = _mm256_loadu_si256((__m256i*)&hap_j_base[k]);

		// Create validity masks: valid if haplotype is 0 or 1
		__m256i valid_i_0 = _mm256_cmpeq_epi32(hap_i_vec, zero);
		__m256i valid_i_1 = _mm256_cmpeq_epi32(hap_i_vec, one);
		__m256i valid_i = _mm256_or_si256(valid_i_0, valid_i_1);

		__m256i valid_j_0 = _mm256_cmpeq_epi32(hap_j_vec, zero);
		__m256i valid_j_1 = _mm256_cmpeq_epi32(hap_j_vec, one);
		__m256i valid_j = _mm256_or_si256(valid_j_0, valid_j_1);

		// Both must be valid
		__m256i valid_both = _mm256_and_si256(valid_i, valid_j);

		// Create masks for counting
		__m256i is_one_i = _mm256_and_si256(valid_i_1, valid_both);
		__m256i is_one_j = _mm256_and_si256(valid_j_1, valid_both);

		// Both are 1: AND the masks
		__m256i both_one = _mm256_and_si256(is_one_i, is_one_j);

		// Accumulate counts (masks are -1 for true, 0 for false)
		// Subtract because mask is -1, effectively adding 1
		pi_vec  = _mm256_sub_epi32(pi_vec,  is_one_i);
		pj_vec  = _mm256_sub_epi32(pj_vec,  is_one_j);
		pij_vec = _mm256_sub_epi32(pij_vec, both_one);
		count_vec = _mm256_sub_epi32(count_vec, valid_both);
	}

	// Horizontal sum: add all 8 lanes together
	int pi_array[8], pj_array[8], pij_array[8], count_array[8];
	_mm256_storeu_si256((__m256i*)pi_array, pi_vec);
	_mm256_storeu_si256((__m256i*)pj_array, pj_vec);
	_mm256_storeu_si256((__m256i*)pij_array, pij_vec);
	_mm256_storeu_si256((__m256i*)count_array, count_vec);

	for(int lane = 0; lane < 8; lane++){
		pi  += pi_array[lane];
		pj  += pj_array[lane];
		pij += pij_array[lane];
		count += count_array[lane];
	}

	// Handle remainder samples with scalar code
	for(k = vec_iterations * 8; k < nSamps; k++){
		int hap_i = hap_i_base[k];
		int hap_j = hap_j_base[k];

		if((hap_i == 1 || hap_i == 0) && (hap_j == 1 || hap_j == 0)){
			if(hap_i == 1) pi++;
			if(hap_j == 1) pj++;
			if(hap_i == 1 && hap_j == 1) pij++;
			count += 1.0;
		}
	}

	// Same final computation as original (bit-exact)
	if (count == 0.0){
		return(-1.0);
	}
	else{
		pi  /= count;
		pj  /= count;
		pij /= count;

		double Dij = pij - (pi*pj);

		return (Dij*Dij) / ((pi*(1.0-pi)) * (pj*(1.0-pj)));
	}
}
#endif

// Scalar version of r2 (original implementation, used as fallback)
double r2(int nSamps, int *haps, int i, int j){
	double pi  = 0.0;
	double pj  = 0.0;
	double pij = 0.0;
	double count = 0.0;
	int k;

	// Optimize memory access by caching array values in local variables
	// This reduces redundant memory accesses from 3-4x to 1x per element
	for(k=0; k<nSamps; k++){
		int hap_i = haps[i*nSamps + k];
		int hap_j = haps[j*nSamps + k];

		// Only process if both haplotypes are valid (0 or 1)
		if((hap_i == 1 || hap_i == 0) && (hap_j == 1 || hap_j == 0)){
			if(hap_i == 1)
				pi++;

			if(hap_j == 1)
				pj++;

			if(hap_i == 1 && hap_j == 1)
				pij++;
			count += 1.0;
		}
	}
	if (count == 0.0){
		return(-1.0);
	}
	else{
		pi  /= count;
		pj  /= count;
		pij /= count;

		double Dij = pij - (pi*pj);

		return (Dij*Dij) / ((pi*(1.0-pi)) * (pj*(1.0-pj)));
	}
}

void computeR2Matrix(int nSamps, int nSnps, int *haps, double *r2Matrix){
	double r2Val;
	int i, j;

#ifdef __AVX2__
	// Use AVX2 if compiled with support and CPU has the capability
	static int use_avx2 = -1;  // -1 = not checked, 0 = no, 1 = yes
	if (use_avx2 == -1) {
		use_avx2 = has_avx2_support();
	}

	if (use_avx2) {
		for (i=0; i<nSnps-1; i++){
			for (j=i+1; j<nSnps; j++){
				r2Val = r2_avx2(nSamps, haps, i, j);
				r2Matrix[i*nSnps +j] = r2Val;
			}
		}
		return;
	}
#endif

	// Fallback to scalar version
	for (i=0; i<nSnps-1; i++){
		for (j=i+1; j<nSnps; j++){
			r2Val = r2(nSamps, haps, i, j);
			r2Matrix[i*nSnps +j] = r2Val;
		}
	}
}

void ZnS(int nSnps, double *r2Matrix, double *zns){
	double r2Sum = 0;
	int r2Denom = 0;
	int i, j;
	if (nSnps < 2)
		*zns = 0.0;
	else{
		for (i=0; i<nSnps-1; i++){
			for (j=i+1; j<nSnps; j++){
				if (r2Matrix[i*nSnps + j] >= 0.0){
					r2Sum += r2Matrix[i*nSnps + j];
					r2Denom += 1;
				}
			}
		}
		*zns = r2Sum/(float)r2Denom;
	}
}

double omegaAtSnp(int l, int nSnps, double *r2Matrix){
	double oSum=0.0, oSumL=0.0, oSumR = 0.0;
	double denom, numer;
	int lCount=0, rCount=0, crossCount=0;
	int i, j;

	for (i=0; i<nSnps-1; i++){
		for (j=i+1; j<nSnps; j++){
			if (r2Matrix[i*nSnps + j] >= 0.0){
				if (i < l && j >= l){
					oSum += r2Matrix[i*nSnps + j];
					crossCount += 1;
				}
				else if (i < l && j < l){
					oSumL += r2Matrix[i*nSnps + j];
					lCount += 1;
				}
				else if (i >= l && j >= l){
					oSumR += r2Matrix[i*nSnps + j];
					rCount += 1;
				}
			}
		}
	}
	denom = oSum * (1.0/(float) crossCount);
	numer = 1.0 / (float) (lCount + rCount);
	numer *= (oSumL+oSumR);
	return numer/denom;
}

void omega(int nSnps, double *r2Matrix, double *omegaMax){
	*omegaMax = 0.0;
	double tmp;
	int l;

	if (nSnps < 3)
		*omegaMax = 0.0;
	else{
		for (l=3; l<nSnps-2; l++){
			tmp = omegaAtSnp(l, nSnps, r2Matrix);
			if (tmp > *omegaMax)
				*omegaMax = tmp;
		}
	}
}

#ifdef __AVX2__
/*
 * SIMD-optimized pairwiseDiffs using AVX2
 * Processes 8 SNPs at a time for each sample pair
 * Expected speedup: 4-6x over scalar version
 */
void pairwiseDiffs_avx2(int nSamps, int nSnps, int *haps, double *diffLs){
	int i, j, k;
	int pairsSeen = 0;

	int vec_iterations = nSnps / 8;
	int remainder = nSnps % 8;

	__m256i zero = _mm256_setzero_si256();
	__m256i one  = _mm256_set1_epi32(1);

	for(i=0; i<nSamps-1; i++){
		for(j=i+1; j<nSamps; j++){
			int diffs = 0;

			// Vectorized SNP comparison
			__m256i diff_vec = _mm256_setzero_si256();

			for(k=0; k < vec_iterations * 8; k += 8){
				// Load 8 SNPs for sample i and j
				__m256i snps_i = _mm256_set_epi32(
					haps[(k+7)*nSamps + i], haps[(k+6)*nSamps + i],
					haps[(k+5)*nSamps + i], haps[(k+4)*nSamps + i],
					haps[(k+3)*nSamps + i], haps[(k+2)*nSamps + i],
					haps[(k+1)*nSamps + i], haps[(k+0)*nSamps + i]
				);
				__m256i snps_j = _mm256_set_epi32(
					haps[(k+7)*nSamps + j], haps[(k+6)*nSamps + j],
					haps[(k+5)*nSamps + j], haps[(k+4)*nSamps + j],
					haps[(k+3)*nSamps + j], haps[(k+2)*nSamps + j],
					haps[(k+1)*nSamps + j], haps[(k+0)*nSamps + j]
				);

				// Check validity: both must be in [0,1]
				__m256i valid_i = _mm256_and_si256(
					_mm256_cmpgt_epi32(snps_i, _mm256_set1_epi32(-1)),
					_mm256_cmpgt_epi32(_mm256_set1_epi32(2), snps_i)
				);
				__m256i valid_j = _mm256_and_si256(
					_mm256_cmpgt_epi32(snps_j, _mm256_set1_epi32(-1)),
					_mm256_cmpgt_epi32(_mm256_set1_epi32(2), snps_j)
				);
				__m256i valid_both = _mm256_and_si256(valid_i, valid_j);

				// Compare: are they different?
				__m256i different = _mm256_andnot_si256(
					_mm256_cmpeq_epi32(snps_i, snps_j),
					valid_both
				);

				// Accumulate differences
				diff_vec = _mm256_sub_epi32(diff_vec, different);
			}

			// Horizontal sum of diff_vec
			int diff_array[8];
			_mm256_storeu_si256((__m256i*)diff_array, diff_vec);
			for(int lane = 0; lane < 8; lane++){
				diffs += diff_array[lane];
			}

			// Handle remainder SNPs with scalar code
			for(k = vec_iterations * 8; k < nSnps; k++){
				int basei = haps[k*nSamps + i];
				int basej = haps[k*nSamps + j];
				if(basei >= 0 && basei <= 1 && basej >= 0 && basej <= 1){
					if (basei != basej){
						diffs += 1;
					}
				}
			}

			diffLs[pairsSeen] = diffs;
			pairsSeen += 1;
		}
	}
}
#endif

// Scalar version of pairwiseDiffs (original implementation, used as fallback)
void pairwiseDiffs(int nSamps, int nSnps, int *haps, double *diffLs){
	int i, j, k, basei, basej, diffs;
	int pairsSeen = 0;

#ifdef __AVX2__
	// Use AVX2 if compiled with support and CPU has the capability
	static int use_avx2 = -1;
	if (use_avx2 == -1) {
		use_avx2 = has_avx2_support();
	}

	if (use_avx2) {
		pairwiseDiffs_avx2(nSamps, nSnps, haps, diffLs);
		return;
	}
#endif

	// Fallback to scalar version
	for(i=0; i<nSamps-1; i++){
		for(j=i+1; j<nSamps; j++){
			diffs = 0;
			for(k=0; k<nSnps; k++){
				basei = haps[k*nSamps + i];
				basej = haps[k*nSamps + j];
				if(basei >= 0 && basei <= 1 && basej >= 0 && basej <= 1){
					if (basei != basej){
						diffs += 1;
					}
				}
			}
			diffLs[pairsSeen] = diffs;
			pairsSeen += 1;
		}
	}
}

void pairwiseDiffsDiplo(int nSamps, int nSnps, int *haps, double *diffLs){
	int i, j, k, basei, basej, diffs;
	int pairsSeen = 0;
	for(i=0; i<nSamps-1; i++){
		for(j=i+1; j<nSamps; j++){
			diffs = 0;
			for(k=0; k<nSnps; k++){
				basei = haps[k*nSamps + i];
				basej = haps[k*nSamps + j];
				if(basei >= 0 && basei <= 2 && basej >= 0 && basej <= 2){
					if (basei != basej){
						diffs += 1;
					}
				}
			}
			diffLs[pairsSeen] = diffs;
			pairsSeen += 1;
		}
	}
}

void getHaplotypeFreqSpec(int nSamps, int nSnps, int *haps, int *hapCounts)
{
        int i;
        int j;
        int k;
        int haplotype_found;
        int allsame;
        int freq;

        int nHaps = 0;
        int haplotypes[nSnps*nSamps];
        int haplotype_occurrences[nSamps];

        for(i=0; i<nSamps; i++)
        {
		hapCounts[i] = 0;
		haplotype_occurrences[i] = 0;
        }

        for(i=0; i<nSamps; i++)
        {
		haplotype_found = 0;
		for(j=0; j<nHaps; j++)
		{
			allsame = 1;
			for(k=0; k<nSnps; k++)
			{
				if((haplotypes[k*nSamps + j] != haps[k*nSamps + i]) && (haplotypes[k*nSamps + j] != -1) && (haps[k*nSamps + i] != -1))
				{
					allsame = 0;
					break;
			        }
			}

			if(allsame)
			{
			        haplotype_found = 1;
		        	haplotype_occurrences[j]+=1;
			        break;
			}
	    	}

		if(!haplotype_found)
		{
			nHaps++;
			for(j=0; j<nSnps; j++)
			{
			        haplotypes[j*nSamps + nHaps-1] = haps[j*nSamps + i];
			}
			haplotype_occurrences[nHaps-1]=1;
		}
        }

	for (i=0; i<nHaps; i++)
	{
		freq = haplotype_occurrences[i];
		if (freq > 0 && freq <= nSamps)
		{
			hapCounts[freq-1] += 1;
		}
	}
	hapCounts[nSamps] = nHaps;
}
