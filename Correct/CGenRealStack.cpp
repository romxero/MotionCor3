

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include "CCorrectInc.h"
#include "../CMainInc.h"
#include "../Util/CUtilInc.h"
#include <hip/hip_runtime.h>

#include <hipfft/hipfft.h>
#include <memory.h>
#include <stdio.h>

using namespace MotionCor2;
using namespace MotionCor2::Correct;

static CStackBuffer* s_pFrmBuffer = 0L;
static CStackBuffer* s_pTmpBuffer = 0L;
static Align::CStackShift* s_pStackShift = 0L;
static bool s_bCorrectBilinear = false;
static bool s_bMotionDecon = false;

void CGenRealStack::DoIt
(	EBuffer eBuffer,
	bool bCorrectBilinear,
	bool bMotionDecon,
	Align::CStackShift* pStackShift
)
{	CBufferPool* pBufferPool = CBufferPool::GetInstance();
	s_pFrmBuffer = pBufferPool->GetBuffer(eBuffer);
	s_pTmpBuffer = pBufferPool->GetBuffer(EBuffer::tmp);
	s_pStackShift = pStackShift;
	s_bCorrectBilinear = bCorrectBilinear;
	s_bMotionDecon = bMotionDecon;
	//----------------------------
	int iNumGpus = s_pFrmBuffer->m_iNumGpus;
	CGenRealStack* pThreads = new CGenRealStack[iNumGpus];
	//-----------------------------------------------------------
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].Run(i);
	};
	for(int i=0; i<iNumGpus; i++)
	{	pThreads[i].WaitForExit(-1.0f);
	}
	delete[] pThreads;
}

CGenRealStack::CGenRealStack(void)
{
}

CGenRealStack::~CGenRealStack(void)
{
}

void CGenRealStack::Run(int iNthGpu)
{
	m_iNthGpu = iNthGpu;	
	this->Start();
}

void CGenRealStack::ThreadMain(void)
{
	s_pFrmBuffer->SetDevice(m_iNthGpu);
	//---------------------------------
	hipStreamCreate(&m_aStream[0]);
	hipStreamCreate(&m_aStream[1]);
	//------------------------------
	CBufferPool* pBufferPool = CBufferPool::GetInstance();
	m_pCufft2D = pBufferPool->GetInverseFFT(m_iNthGpu);
	int* piCmpSize = s_pFrmBuffer->m_aiCmpSize;
	m_pCufft2D->CreateInversePlan(piCmpSize, true);
	//---------------------------------------------
	mDoCpuFrames();
	mDoGpuFrames();
	//-------------
	hipStreamSynchronize(m_aStream[0]);
	hipStreamSynchronize(m_aStream[1]);
	hipStreamDestroy(m_aStream[0]);
	hipStreamDestroy(m_aStream[1]);
}

void CGenRealStack::mDoGpuFrames(void)
{
	int iStartFrm = s_pFrmBuffer->GetStartFrame(m_iNthGpu);
	int iNumFrames = s_pFrmBuffer->GetNumFrames(m_iNthGpu);
	//-----------------------------------------------------
	for(int i=0; i<iNumFrames; i++)
	{	if(!s_pFrmBuffer->IsGpuFrame(m_iNthGpu, i)) continue;
		m_iAbsFrm = i + iStartFrm;
		hipfftComplex* gCmpFrm = s_pFrmBuffer->GetFrame(m_iNthGpu, i);
		//-----------------------------------------------------------
		mAlignFrame(gCmpFrm);
		mCorrectBilinear(gCmpFrm);
		mMotionDecon(gCmpFrm);
		m_pCufft2D->Inverse(gCmpFrm, m_aStream[0]);
	}
}

void CGenRealStack::mDoCpuFrames(void)
{
	int iCount = 0;
	int iStartFrm = s_pFrmBuffer->GetStartFrame(m_iNthGpu);
	int iNumFrames = s_pFrmBuffer->GetNumFrames(m_iNthGpu);
	hipfftComplex *gCmpBuf = 0L, *pCmpFrm = 0L;
	size_t tBytes = s_pFrmBuffer->m_tFmBytes;
	//---------------------------------------
	for(int i=0; i<iNumFrames; i++)
	{	if(s_pFrmBuffer->IsGpuFrame(m_iNthGpu, i)) continue;
		int iStream = iCount % 2;
		m_iAbsFrm = i + iStartFrm;
		//------------------------
		pCmpFrm = s_pFrmBuffer->GetFrame(m_iNthGpu, i);
		gCmpBuf = s_pTmpBuffer->GetFrame(m_iNthGpu, iStream);
		//---------------------------------------------------
		hipMemcpyAsync(gCmpBuf, pCmpFrm, tBytes,
			hipMemcpyDefault, m_aStream[iStream]);
		if(iStream == 1) hipStreamSynchronize(m_aStream[1]);
		//---------------------------------------------------
		mAlignFrame(gCmpBuf);
		mCorrectBilinear(gCmpBuf);
		mMotionDecon(gCmpBuf);
		m_pCufft2D->Inverse(gCmpBuf, m_aStream[0]);
		if(iStream == 1) hipStreamSynchronize(m_aStream[0]);
		//---------------------------------------------------
		hipMemcpyAsync(pCmpFrm, gCmpBuf, tBytes,
			hipMemcpyDefault, m_aStream[iStream]);
		iCount += 1;
	}
}

void CGenRealStack::mCorrectBilinear(hipfftComplex* gCmpFrm)
{
	if(!s_bCorrectBilinear) return;
	Util::GCorrLinearInterp gCorrLinearInterp;
	int* piCmpSize = s_pFrmBuffer->m_aiCmpSize;
	gCorrLinearInterp.DoIt(gCmpFrm, piCmpSize, m_aStream[0]);
}

void CGenRealStack::mAlignFrame(hipfftComplex* gCmpFrm)
{
	if(s_pStackShift == 0L) return;
	//-----------------------------	
	float afShift[2] = {0.0f};
	s_pStackShift->GetShift(m_iAbsFrm, afShift, -1.0f);
	int* piCmpSize = s_pFrmBuffer->m_aiCmpSize;
	Util::GPhaseShift2D aGPhaseShift;
	aGPhaseShift.DoIt(gCmpFrm, piCmpSize, afShift, m_aStream[0]);
}

void CGenRealStack::mMotionDecon(hipfftComplex* gCmpFrm)
{	
	CInput* pInput = CInput::GetInstance();
     	if(pInput->m_iInFmMotion == 0) return;
	if(!s_bMotionDecon) return;
	if(s_pStackShift == 0L) return;
	//-----------------------------
	m_aInFrameMotion.SetFullShift(s_pStackShift);
	m_aInFrameMotion.DoFullMotion(m_iAbsFrm, gCmpFrm, 
		s_pFrmBuffer->m_aiCmpSize, m_aStream[0]);
}

