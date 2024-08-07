

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include "CAlignInc.h"
#include "../CMainInc.h"
#include "../Util/CUtilInc.h"
#include <memory.h>
#include <stdio.h>
#include <hip/hip_runtime.h>

#include <hipfft/hipfft.h>
#include <nvToolsExt.h>

using namespace MotionCor2;
using namespace MotionCor2::Align;

CSimpleSum::CSimpleSum(void)
{
}

CSimpleSum::~CSimpleSum(void)
{
}

void CSimpleSum::DoIt(DU::CDataPackage* pPackage)
{
	nvtxRangePushA ("CsimpleSum");
	CAlignBase::Clean();
	CAlignBase::DoIt(pPackage);
	//-------------------------
	CBufferPool* pBufferPool = CBufferPool::GetInstance();
	CStackBuffer* pFrmBuffer = pBufferPool->GetBuffer(EBuffer::frm);
	int* piCmpSize = pFrmBuffer->m_aiCmpSize;
	int aiPadSize[] = {piCmpSize[0] * 2, piCmpSize[1]};
	//-------------------------------------------------
	Util::CMultiGpuBase::mSetGpus(pBufferPool->m_piGpuIDs, 
		pBufferPool->m_iNumGpus);
	Util::CMultiGpuBase::mCreateStreams(2);
	Util::CMultiGpuBase::mCreateForwardFFTs(aiPadSize, true);
	Util::CMultiGpuBase::mCreateInverseFFTs(m_aiCmpSize, true);
	//---------------------------------------------------------
	mCalcSum();
	mGenStack();
	//----------
	Util::CMultiGpuBase::mDeleteStreams();
	Util::CMultiGpuBase::mDeleteForwardFFTs();
	Util::CMultiGpuBase::mDeleteInverseFFTs();
	nvtxRangePop ();
}

void CSimpleSum::mCalcSum(void)
{
	CInput* pInput = CInput::GetInstance();
	MrcUtil::CSumFFTStack sumFFTStack;
	bool bSplitSum = (pInput->m_iSplitSum == 0) ? false : true;
	sumFFTStack.DoIt(EBuffer::frm, bSplitSum);
	//----------------------------------------
	CBufferPool* pBufferPool = CBufferPool::GetInstance();
	CStackBuffer* pSumBuffer = pBufferPool->GetBuffer(EBuffer::sum);
	hipfftComplex* gCmpSum0 = pSumBuffer->GetFrame(0, 0);
	hipfftComplex* gCmpSum1 = pSumBuffer->GetFrame(0, 1);
	hipfftComplex* gCmpSum2 = pSumBuffer->GetFrame(0, 2);
	//--------------------------------------------------
	CStackBuffer* pTmpBuffer = pBufferPool->GetBuffer(EBuffer::tmp);
	hipfftComplex* gCmpBuf0 = pTmpBuffer->GetFrame(0, 0);
	hipfftComplex* gCmpBuf1 = pTmpBuffer->GetFrame(0, 1);
	hipfftComplex* gCmpBuf2 = pTmpBuffer->GetFrame(0, 2);
	//--------------------------------------------------
	pBufferPool->SetDevice(0);
	mCropFrame(gCmpSum0, gCmpBuf0, m_pStreams[0], 0);
	if(bSplitSum)
	{	mCropFrame(gCmpSum1, gCmpBuf1, m_pStreams[0], 0);
		mCropFrame(gCmpSum2, gCmpBuf2, m_pStreams[0], 0);
	}
	//-------------------------------------------------------
	float *gfUnpad0 = 0L, *gfUnpad1 = 0L, *gfUnpad2 = 0L;
	gfUnpad0 = reinterpret_cast<float*>(gCmpSum0);
	mUnpad(gCmpBuf0, gfUnpad0, m_pStreams[0]);
	if(bSplitSum)
	{	gfUnpad1 = reinterpret_cast<float*>(gCmpSum1);
		mUnpad(gCmpBuf1, gfUnpad1, m_pStreams[0]);
		gfUnpad2 = reinterpret_cast<float*>(gCmpSum2);
		mUnpad(gCmpBuf2, gfUnpad2, m_pStreams[0]);
	}
	hipStreamSynchronize(m_pStreams[0]);
	//-----------------------------------
	size_t tBytes = m_pPackage->m_pAlnSums->m_tFmBytes;
	void* pvCropped = m_pPackage->m_pAlnSums->GetFrame(0);
	hipMemcpy(pvCropped, gfUnpad0, tBytes, hipMemcpyDefault);
	if(bSplitSum)
	{	pvCropped = m_pPackage->m_pAlnSums->GetFrame(1);
		hipMemcpy(pvCropped, gfUnpad1, tBytes, hipMemcpyDefault);
		pvCropped = m_pPackage->m_pAlnSums->GetFrame(2);
		hipMemcpy(pvCropped, gfUnpad2, tBytes, hipMemcpyDefault);
	}
}

void CSimpleSum::mGenStack(void)
{
	CInput* pInput = CInput::GetInstance();
	if(pInput->m_aiOutStack[0] == 0) return;
	//-------------------------------------
	CBufferPool* pBufferPool = CBufferPool::GetInstance();
     	for(int i=0; i<pBufferPool->m_iNumGpus; i++)
	{	mCropFrames(i);
	}
}

void CSimpleSum::mCropFrames(int iNthGpu)
{
	CBufferPool* pBufferPool = CBufferPool::GetInstance();
	CStackBuffer* pFrmBuffer = pBufferPool->GetBuffer(EBuffer::frm);
	CStackBuffer* pTmpBuffer = pBufferPool->GetBuffer(EBuffer::tmp);
	hipfftComplex* gCmpBuf = pTmpBuffer->GetFrame(iNthGpu, 0);
	//-------------------------------------------------------
	int iNumFrames = pFrmBuffer->GetNumFrames(iNthGpu);
	int iStartFrame = pFrmBuffer->GetStartFrame(iNthGpu);
	pBufferPool->SetDevice(iNthGpu);
	//------------------------------
	for(int i=0; i<iNumFrames; i=i+1)
	{	hipfftComplex* gCmpFrm = pFrmBuffer->GetFrame(iNthGpu, i);
		//-------------------------------------------------------
		hipStream_t stream = m_pStreams[2 * iNthGpu];
		mCropFrame(gCmpFrm, gCmpBuf, stream, iNthGpu);
		//--------------------------------------------
		float* gfUnpad = reinterpret_cast<float*>(gCmpFrm);
		mUnpad(gCmpBuf, gfUnpad, stream);
		hipStreamSynchronize(stream);
	}
	//------------------------------------
	DU::CMrcStack* pAlnStack = m_pPackage->m_pAlnStack;
	for(int i=0; i<iNumFrames; i++)
	{	hipfftComplex* gCmpFrm = pFrmBuffer->GetFrame(iNthGpu, i);
		float* gfFrm = reinterpret_cast<float*>(gCmpFrm);
		void* pvFrm = pAlnStack->GetFrame(iStartFrame + i);
		hipMemcpy(pvFrm, gfFrm, pAlnStack->m_tFmBytes, 
		   hipMemcpyDefault);
	}
}

void CSimpleSum::mCropFrame
(	hipfftComplex* gCmpFrm,
	hipfftComplex* gCmpBuf,
	hipStream_t stream,
	int iNthGpu
)
{	CBufferPool* pBufferPool = CBufferPool::GetInstance();
	CStackBuffer* pFrmBuffer = pBufferPool->GetBuffer(EBuffer::frm);
	float* gfPadFrm = reinterpret_cast<float*>(gCmpFrm);
	m_pForwardFFTs[iNthGpu].Forward(gfPadFrm, true, stream);
	//------------------------------------------------------
	bool bSum = true;
	Util::GFourierResize2D fourierResize;
	fourierResize.DoIt(gCmpFrm, pFrmBuffer->m_aiCmpSize, 
		gCmpBuf, m_aiCmpSize, !bSum, stream);
	//--------------------------------------------
	m_pInverseFFTs[iNthGpu].Inverse(gCmpBuf, stream);
}

void CSimpleSum::mUnpad
(	hipfftComplex* gCmpPad, 
	float* gfUnpad,
	hipStream_t stream
)
{ 	Util::GPad2D pad2D;
	float* gfPad = reinterpret_cast<float*>(gCmpPad);
	pad2D.Unpad(gfPad, m_aiPadSize, gfUnpad, stream);
}
