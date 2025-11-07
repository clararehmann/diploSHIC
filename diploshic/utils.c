#include <math.h>

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

void pairwiseDiffs(int nSamps, int nSnps, int *haps, double *diffLs){
	int i, j, k, basei, basej, diffs;
	int pairsSeen = 0;
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
