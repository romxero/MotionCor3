#pragma once
#include <hip/hip_runtime.h>

#include "../Util/CUtilInc.h"
#include "../MrcUtil/CMrcUtilInc.h"
#include "../Align/CAlignInc.h"
#include <stdio.h>
#include <hipfft/hipfft.h>

namespace MotionCor2
{
namespace MotionDecon
{

class GDeconFrame
{
public:
	GDeconFrame(void);
	~GDeconFrame(void);
	void CleanGpu(void);
	void CalcSinc2Sum
	(  float* pfInFrmShifts,
	   int iNumFrames,
	   int* piCmpSize
	);
	void DirWeightFrame
	(  int iFrame,
	   hipfftComplex* pCmpFrm,
	   bool bGpu
	);
	void WeightFrame
	(  float fShift,
	   hipfftComplex* pCmpFrm,
	   bool bGpu,
	   int* piCmpSize
	);
	int m_aiCmpSize[2];
	int m_iNumFrames;
private:
	hipfftComplex* mH2D
	(  hipfftComplex* pCmpFrm,
	   bool bGpu,
	   int* piCmpSize
	);
	void mD2H
	(  hipfftComplex* gCmpFrm,
	   hipfftComplex* pCmpFrm,
	   bool bGpu,
	   int* piCmpSize
	);
	float* m_gfSinc2Sum;
	float* m_pfInFrmShifts;
};

class GMotionWeight
{
public:
	GMotionWeight(void);
	~GMotionWeight(void);
	void DirWeight
	( float* pfMotion,
	  hipfftComplex* pCmpFrm,
	  int* piCmpSize,
          hipStream_t stream=0
	);
	void Weight
	( float fMotion,
	  hipfftComplex* pCmpFrm,
	  int* piCmpSize,
          hipStream_t stream=0
	);
};

class CInFrameMotion
{
public:
	CInFrameMotion(void);
	~CInFrameMotion(void);
	void SetFullShift(Align::CStackShift* pStackShift);
	void SetPatchShifts(Align::CPatchShifts* pPatchShifts);
	void DoFullMotion
	( int iFrame,
	  hipfftComplex* pCmpFrm,
	  int* piCmpSize,
          hipStream_t stream=0
	);
	void DoLocalMotion
	( int iFrame,
	  hipfftComplex* pCmpFrm,
	  int* piCmpSize,
          hipStream_t stream=0
	);
	void GetFullMotion
	( int iFrame,
	  float* pfMotion // 2 elements
	);
	void GetLocalMotion
	( int iFrame,
	  int iPatch,
	  float* pfMotion // 2 elements
	);
	int m_iNumFrames;
	int m_iNumPatches;
private:
	void mCalculate(Align::CStackShift* pStackShift);
	Align::CStackShift* m_pFullShift;
	Align::CPatchShifts* m_pPatchShifts;
};


}
}

