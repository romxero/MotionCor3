

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include "CMrcUtilInc.h"
#include "../CMainInc.h"
#include "../Util/CUtilInc.h"
#include <hip/hip_runtime.h>

#include <hipfft/hipfft.h>
#include <memory.h>
#include <stdio.h>
#include <nvToolsExt.h>

using namespace MotionCor2;
using namespace MotionCor2::MrcUtil;
using namespace MotionCor2::Util;

static CStackBuffer* s_pStkBuffer = 0L;
static CStackBuffer* s_pTmpBuffer = 0L;
static CStackBuffer* s_pSumBuffer = 0L; 
static bool s_bSplitSum = false;

static void mCalcMean(hipfftComplex* gCmp, int* piCmpSize, int iGpu)
{
	int iSizeX = (piCmpSize[0] - 1) * 2;
	int iPadSize = piCmpSize[0] * 2 * piCmpSize[1];
	float* pfPadBuf = new float[iPadSize];
	hipMemcpy(pfPadBuf, gCmp, iPadSize * sizeof(float),
		hipMemcpyDefault);
	double dMean = 0.0;
	for(int y=0; y<piCmpSize[1]; y++)
	{	int i = y * piCmpSize[0] * 2;
		for(int x=0; x<iSizeX; x++)
		{	dMean += pfPadBuf[i+x];
		}
	}
	dMean /= (iSizeX * piCmpSize[1]);
	printf("Gpu: %d, Mean: %f\n", iGpu, dMean);
	delete[] pfPadBuf;
}

static void sDoSplitSum(void)
{
	if(!s_bSplitSum) return;
	//----------------------
	Util::GAddFrames gAddFrames;
	hipfftComplex* gCmpSum0 = s_pSumBuffer->GetFrame(0, 0);
	hipfftComplex* gCmpSum1 = s_pSumBuffer->GetFrame(0, 1);
	hipfftComplex* gCmpSum2 = s_pSumBuffer->GetFrame(0, 2);
	gAddFrames.DoIt(gCmpSum0, 1.0f, gCmpSum1, -1.0f, gCmpSum2,
	   s_pSumBuffer->m_aiCmpSize);
}

CSumFFTStack::CSumFFTStack(void)
{
}

CSumFFTStack::~CSumFFTStack(void)
{
}

//------------------------------------------------------------------------------
// 1. This function calculates the simple sum(s) without correction of beam
//    induced motion.
// 2. If bSplitSum is true, odd and even sums will be calculated in addition
//    to the whole sum. (SZ: 08-10-2023)
//------------------------------------------------------------------------------ 
void CSumFFTStack::DoIt(int iBuffer, bool bSplitSum)
{
        nvtxRangePushA("CSumFFTStack::DoIt");
	CBufferPool* pBufferPool = CBufferPool::GetInstance();
	s_pStkBuffer = pBufferPool->GetBuffer((EBuffer)iBuffer);
	s_pSumBuffer = pBufferPool->GetBuffer(EBuffer::sum);
	s_pTmpBuffer = pBufferPool->GetBuffer(EBuffer::tmp);
	s_bSplitSum = bSplitSum;
	//--------------------------------------------------
	int iNumGpus = pBufferPool->m_iNumGpus;
	CSumFFTStack* pSumStacks = new CSumFFTStack[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	pSumStacks[i].mSumFrames(i);
	}
	for(int i=0; i<iNumGpus; i++)
	{	pSumStacks[i].mWait();
	}
	delete[] pSumStacks;
	//----------------------------------------------------
	if(iNumGpus == 1) 
	{	sDoSplitSum();
		nvtxRangePop();
		return;
	}
	//----------------------------------------------------
	pBufferPool->SetDevice(0);
	hipfftComplex* gCmpSum0 = s_pSumBuffer->GetFrame(0, 0);
	hipfftComplex* gCmpSum1 = s_pSumBuffer->GetFrame(0, 1);
	hipfftComplex* gCmpBuf = s_pTmpBuffer->GetFrame(0, 0);
	int* piCmpSize = s_pStkBuffer->m_aiCmpSize;
	size_t tBytes = s_pStkBuffer->m_tFmBytes;
	Util::GAddFrames addFrames;
	//--------------------------------------------------------------
	for(int i=1; i<iNumGpus; i++)
	{	//------------------------
		// calculate the whole sum
		//------------------------
		hipfftComplex* gCmpSum = s_pSumBuffer->GetFrame(i, 0);
		hipMemcpy(gCmpBuf, gCmpSum, tBytes, hipMemcpyDefault);
		addFrames.DoIt(gCmpSum0, 1.0f, gCmpBuf, 1.0f, gCmpSum0, 
		   piCmpSize);
		if(!s_bSplitSum) continue;
		//------------------------
		// calculate the even sum
		//------------------------
		gCmpSum = s_pSumBuffer->GetFrame(i, 1);
		hipMemcpy(gCmpBuf, gCmpSum, tBytes, hipMemcpyDefault);
		addFrames.DoIt(gCmpSum1, 1.0f, gCmpBuf, 1.0f, gCmpSum1,
                   piCmpSize);
	}
	sDoSplitSum();
        nvtxRangePop();
}

void CSumFFTStack::mSumFrames(int iNthGpu)
{
	m_iNthGpu = iNthGpu;
	s_pStkBuffer->SetDevice(m_iNthGpu);
	//---------------------------------
	hipStreamCreate(&m_aStreams[0]);
	hipStreamCreate(&m_aStreams[1]);	
	//-------------------------------
	size_t tBytes = s_pStkBuffer->m_tFmBytes;
	hipfftComplex* gCmpSum0 = s_pSumBuffer->GetFrame(m_iNthGpu, 0);
	hipfftComplex* gCmpSum1 = s_pSumBuffer->GetFrame(m_iNthGpu, 1);
	hipMemsetAsync(gCmpSum0, 0, tBytes, m_aStreams[0]);
	if(s_bSplitSum)
	{	hipMemsetAsync(gCmpSum1, 0, tBytes, m_aStreams[0]);
	}
	//--------------------------------------------------
	m_iStartFrm = s_pStkBuffer->GetStartFrame(m_iNthGpu);
	mSumCpuFrames();
	mSumGpuFrames();
}

void CSumFFTStack::mWait(void)
{
	s_pStkBuffer->SetDevice(m_iNthGpu);
	hipStreamSynchronize(m_aStreams[0]);
	hipStreamSynchronize(m_aStreams[1]);
	hipStreamDestroy(m_aStreams[0]);
	hipStreamDestroy(m_aStreams[1]);
}

void CSumFFTStack::mSumGpuFrames(void)
{
	int iNumFrames = s_pStkBuffer->GetNumFrames(m_iNthGpu);
	hipfftComplex* gCmpSum0 = s_pSumBuffer->GetFrame(m_iNthGpu, 0);
	hipfftComplex* gCmpSum1 = s_pSumBuffer->GetFrame(m_iNthGpu, 1);
	hipfftComplex* gCmpFrm = 0L;
	int* piCmpSize = s_pStkBuffer->m_aiCmpSize;
	Util::GAddFrames aGAddFrames;
	//---------------------------
	for(int i=0; i<iNumFrames; i++)
	{	if(!s_pStkBuffer->IsGpuFrame(m_iNthGpu, i)) continue;
		//-------------------------
		// calculate the whole sum
		//-------------------------
		gCmpFrm = s_pStkBuffer->GetFrame(m_iNthGpu, i);
		aGAddFrames.DoIt(gCmpSum0, 1.0f, gCmpFrm, 1.0f, gCmpSum0, 
		   piCmpSize, m_aStreams[0]);
		if(!s_bSplitSum) continue;
		//------------------------
		// calculate the even sum
		//------------------------
		if((i + m_iStartFrm) % 2 != 0) continue;
		aGAddFrames.DoIt(gCmpSum1, 1.0f, gCmpFrm, 1.0f, gCmpSum1,
                   piCmpSize, m_aStreams[0]);
	}
}

void CSumFFTStack::mSumCpuFrames(void)
{
	int iNumFrames = s_pStkBuffer->GetNumFrames(m_iNthGpu);
	hipfftComplex* gCmpSum0 = s_pSumBuffer->GetFrame(m_iNthGpu, 0);
	hipfftComplex* gCmpSum1 = s_pSumBuffer->GetFrame(m_iNthGpu, 1);
	hipfftComplex *gCmpBuf = 0L, *pCmpFrm = 0L;
	int* piCmpSize = s_pStkBuffer->m_aiCmpSize;
	size_t tBytes = s_pStkBuffer->m_tFmBytes;
	Util::GAddFrames aGAddFrames;
	int iCount = 0;
	//-------------
	for(int i=0; i<iNumFrames; i++)
	{	if(s_pStkBuffer->IsGpuFrame(m_iNthGpu, i)) continue;
		int iStream = iCount % 2;
		pCmpFrm = s_pStkBuffer->GetFrame(m_iNthGpu, i);
		gCmpBuf = s_pTmpBuffer->GetFrame(m_iNthGpu, iStream);
		//---------------------------------------------------
		if(iStream == 1) hipStreamSynchronize(m_aStreams[0]);
		hipMemcpyAsync(gCmpBuf, pCmpFrm, tBytes,
		   hipMemcpyDefault, m_aStreams[iStream]);
		if(iStream == 1) hipStreamSynchronize(m_aStreams[1]);
		//----------------------------------------------------
		// calculate the whole sum
		//----------------------------------------------------
		aGAddFrames.DoIt(gCmpSum0, 1.0f, gCmpBuf, 1.0f, gCmpSum0,
		   piCmpSize, m_aStreams[0]);
		if(!s_bSplitSum) continue;
		//------------------------------
		// calculate even sums
		//------------------------------
		if((i + m_iStartFrm) % 2 != 0) continue;
		aGAddFrames.DoIt(gCmpSum1, 1.0f, gCmpBuf, 1.0f, gCmpSum1,
		   piCmpSize, m_aStreams[0]);
	}
}	
