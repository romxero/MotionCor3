

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include "CAlignInc.h"
#include "../CMainInc.h"
#include "../Util/CUtilInc.h"
#include <Util/Util_Time.h>
#include <memory.h>
#include <stdio.h>
#include <hip/hip_runtime.h>

#include <hipfft/hipfft.h>

using namespace MotionCor2;
using namespace MotionCor2::Align;

static CStackBuffer* s_pPatBuffer = 0L;
static CStackBuffer* s_pXcfBuffer = 0L;
static CStackBuffer* s_pSumBuffer = 0L;
static int s_aiPatStart[2] = {0};
	
CExtractPatch::CExtractPatch(void)
{
}

CExtractPatch::~CExtractPatch(void)
{
}

void CExtractPatch::DoIt(int iPatch)
{	
	CBufferPool* pBufferPool = CBufferPool::GetInstance();
	s_pPatBuffer = pBufferPool->GetBuffer(EBuffer::pat);
	s_pXcfBuffer = pBufferPool->GetBuffer(EBuffer::xcf);
	s_pSumBuffer = pBufferPool->GetBuffer(EBuffer::sum);
	//--------------------------------------------------
	CPatchCenters* pPatchCenters = CPatchCenters::GetInstance();
	pPatchCenters->GetStart(iPatch, s_aiPatStart);
	//--------------------------------------------
	int iNumGpus = pBufferPool->m_iNumGpus;
	CExtractPatch* pExtractPatch = new CExtractPatch[iNumGpus];
	for(int i=0; i<iNumGpus; i++)
	{	pExtractPatch[i].Run(i);
	}
	for(int i=0; i<iNumGpus; i++)
	{	pExtractPatch[i].Wait();
	}
	if(pExtractPatch != 0L) delete[] pExtractPatch;
}

void CExtractPatch::Run(int iNthGpu)
{	
	m_iNthGpu = iNthGpu;
	s_pPatBuffer->SetDevice(m_iNthGpu);
	//---------------------------------
	hipStreamCreate(&m_aStream[0]);
	hipStreamCreate(&m_aStream[1]);
	mProcessCpuFrames();
	mProcessGpuFrames();
}

void CExtractPatch::Wait(void)
{
	s_pPatBuffer->SetDevice(m_iNthGpu);
	hipStreamSynchronize(m_aStream[0]);
	hipStreamSynchronize(m_aStream[1]);
	hipStreamDestroy(m_aStream[0]);
	hipStreamDestroy(m_aStream[1]);
}

void CExtractPatch::mProcessGpuFrames(void)
{
	int iNumFrames = s_pPatBuffer->GetNumFrames(m_iNthGpu);
	for(int i=0; i<iNumFrames; i++)
	{	if(!s_pPatBuffer->IsGpuFrame(m_iNthGpu, i)) continue;
		mProcessFrame(i, m_aStream[0]);
	}
}

void CExtractPatch::mProcessCpuFrames(void)
{
	int iCount = 0;
	int iNumFrames = s_pPatBuffer->GetNumFrames(m_iNthGpu);
	for(int i=0; i<iNumFrames; i++)
	{	if(s_pPatBuffer->IsGpuFrame(m_iNthGpu, i)) continue;
		int iStream = iCount % 2;
		mProcessFrame(i, m_aStream[iStream]);
		iCount += 1;
	}
}

void CExtractPatch::mProcessFrame(int iFrame, hipStream_t stream)
{
	hipfftComplex* gPatCmp = s_pPatBuffer->GetFrame(m_iNthGpu, iFrame);
	hipfftComplex* gXcfCmp = s_pXcfBuffer->GetFrame(m_iNthGpu, iFrame);
	float* gfPatFrm = reinterpret_cast<float*>(gPatCmp);
	float* gfXcfFrm = reinterpret_cast<float*>(gXcfCmp);
	//--------------------------------------------------
	int iXcfPadX = s_pXcfBuffer->m_aiCmpSize[0] * 2;
	int iOffset = s_aiPatStart[1] * iXcfPadX + s_aiPatStart[0];
	float* gfSrc = gfXcfFrm + iOffset;
	//--------------------------------
	int iCpySizeX = (s_pPatBuffer->m_aiCmpSize[0] - 1) * 2;
	int iPatPadX = s_pPatBuffer->m_aiCmpSize[0] * 2;
	int iPatPadY = s_pPatBuffer->m_aiCmpSize[1];
	int aiPatPadSize[] = {iPatPadX, iPatPadY};
	Util::GPartialCopy::DoIt(gfSrc, iXcfPadX, gfPatFrm, 
		iCpySizeX, aiPatPadSize, stream);
}

