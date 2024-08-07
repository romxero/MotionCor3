

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include "CAlignInc.h"
#include "../CMainInc.h"
#include "../Util/CUtilInc.h"
#include <hip/hip_runtime.h>

#include <hipfft/hipfft.h>
#include <memory.h>
#include <stdio.h>

using namespace MotionCor2;
using namespace MotionCor2::Align;

static EBuffer s_eBuffer = EBuffer::frm;
static CStackShift* s_pStackShift = 0L;
static int s_aiSumRange[2] = {0};

CAlignedSum::CAlignedSum(void)
{
}

CAlignedSum::~CAlignedSum(void)
{
}

void CAlignedSum::DoIt
(	EBuffer eBuffer,
	CStackShift* pStackShift,
	int* piSumRange
)
{	s_eBuffer = eBuffer;
	s_pStackShift = pStackShift;
	if(piSumRange == 0L)
	{	s_aiSumRange[0] = 0;
		s_aiSumRange[1] = pStackShift->m_iNumFrames;
	}
	else memcpy(s_aiSumRange, piSumRange, sizeof(int) * 2);
	//----------------------------------------------------
	CBufferPool* pBufferPool = CBufferPool::GetInstance();
	int iNumGpus = pBufferPool->m_iNumGpus;
	CAlignedSum* pAlignedSums = new CAlignedSum[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	pAlignedSums[i].mDoIt(i);
	}
	for(int i=0; i<iNumGpus; i++)
	{	pAlignedSums[i].mWait();
	}
	delete[] pAlignedSums;
	if(iNumGpus <= 1) return;
	//-----------------------
	pBufferPool->SetDevice(0);
	CStackBuffer* pFrmBuffer = pBufferPool->GetBuffer(eBuffer);
	CStackBuffer* pSumBuffer = pBufferPool->GetBuffer(EBuffer::sum);
	CStackBuffer* pTmpBuffer = pBufferPool->GetBuffer(EBuffer::tmp);
	//--------------------------------------------------------------
	size_t tBytes = pFrmBuffer->m_tFmBytes;
	int* piCmpSize = pFrmBuffer->m_aiCmpSize;
	hipfftComplex* gCmpSum = pSumBuffer->GetFrame(0, 0);
	hipfftComplex* gCmpBuf = pTmpBuffer->GetFrame(0, 0);
	Util::GAddFrames addFrames;
	//-------------------------
	for(int i=1; i<iNumGpus; i++)
	{	hipfftComplex* gSum = pSumBuffer->GetFrame(i, 0);
		hipMemcpy(gCmpBuf, gSum, tBytes, hipMemcpyDefault);
		addFrames.DoIt(gCmpBuf, 1.0f, gCmpSum, 1.0f, 
		   gCmpSum, piCmpSize);
	}
}		

void CAlignedSum::mDoIt(int iNthGpu)
{	
	m_iNthGpu = iNthGpu;
	//------------------
	CBufferPool* pBufferPool = CBufferPool::GetInstance();
	m_pFrmBuffer = pBufferPool->GetBuffer(s_eBuffer);
	m_pSumBuffer = pBufferPool->GetBuffer(EBuffer::sum);
        m_pTmpBuffer = pBufferPool->GetBuffer(EBuffer::tmp);
	//--------------------------------------------------
	m_pFrmBuffer->SetDevice(m_iNthGpu);
	hipStreamCreate(&m_aStreams[0]);
	hipStreamCreate(&m_aStreams[1]);	
	//-------------------------------
	m_gCmpSums[0] = m_pSumBuffer->GetFrame(m_iNthGpu, 0);
	m_gCmpSums[1] = m_pTmpBuffer->GetFrame(m_iNthGpu, 0);
	size_t tBytes = m_pFrmBuffer->m_tFmBytes;
	hipMemsetAsync(m_gCmpSums[0], 0, tBytes, m_aStreams[0]);
	hipMemsetAsync(m_gCmpSums[1], 0, tBytes, m_aStreams[1]);
	//-------------------------------------------------------
	int iStartFrm = m_pFrmBuffer->GetStartFrame(m_iNthGpu);
	int iNumFrames = m_pFrmBuffer->GetNumFrames(m_iNthGpu);
	for(int i=0; i<iNumFrames; i++)
	{	m_iAbsFrm = iStartFrm + i;
		if(m_iAbsFrm < s_aiSumRange[0]) continue;
		else if(m_iAbsFrm > s_aiSumRange[1]) break;
		m_iStream = i % 2;
		mDoFrame(i);
	}		
}

void CAlignedSum::mWait(void)
{
	m_pFrmBuffer->SetDevice(m_iNthGpu);
	//---------------------------------
	hipStreamSynchronize(m_aStreams[1]);
	Util::GAddFrames addFrames;
	addFrames.DoIt(m_gCmpSums[0], 1.0f, m_gCmpSums[1], 1.0f,
	   m_gCmpSums[0], m_pFrmBuffer->m_aiCmpSize, 
	   m_aStreams[0]);
	hipStreamSynchronize(m_aStreams[0]);
	//-----------------------------------
	hipStreamDestroy(m_aStreams[0]);
	hipStreamDestroy(m_aStreams[1]);
}

void CAlignedSum::mDoFrame(int iFrame)
{
	hipfftComplex* gCmpFrm = m_pFrmBuffer->GetFrame(m_iNthGpu, iFrame);
	hipfftComplex* gCmpSum = m_gCmpSums[m_iStream];
	//----------------------------------------------------------------
	float afShift[2] = {0.0f};
	s_pStackShift->GetShift(m_iAbsFrm, afShift, -1.0f);
	//-------------------------------------------------
	bool bSum = true;
	Util::GPhaseShift2D phaseShift;
	phaseShift.DoIt(gCmpFrm, m_pFrmBuffer->m_aiCmpSize,
           afShift, bSum, gCmpSum, m_aStreams[m_iStream]);
}
