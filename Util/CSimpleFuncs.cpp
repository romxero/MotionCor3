

#include <hip/hip_runtime.h>
#include <hip/hip_runtime.h>
#include "CUtilInc.h"
#include <hip/hip_runtime.h>

#include <hipfft/hipfft.h>
#include <memory.h>
#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <sys/resource.h>

size_t MotionCor2::Util::GetUCharBytes(int* piSize)
{
	size_t tBytes = sizeof(char) * piSize[0];
	tBytes *= piSize[1];
        return tBytes;
}

size_t MotionCor2::Util::GetFloatBytes(int* piSize)
{
	size_t tBytes = sizeof(float) * piSize[0];
	tBytes *= piSize[1];
	return tBytes;
}

size_t MotionCor2::Util::GetCmpBytes(int* piSize)
{
	size_t tBytes = sizeof(hipfftComplex) * piSize[0];
	tBytes *= piSize[1];
	return tBytes;
}

unsigned char* MotionCor2::Util::GGetUCharBuf(int* piSize, bool bZero)
{
	size_t tBytes = MotionCor2::Util::GetUCharBytes(piSize);
	unsigned char* gucBuf = 0L;
	hipMalloc(&gucBuf, tBytes);
	if(bZero) hipMemset(gucBuf, 0, tBytes);
	return gucBuf;
}

unsigned char* MotionCor2::Util::CGetUCharBuf(int* piSize, bool bZero)
{
	int iPixels = piSize[0] * piSize[1];
	unsigned char* pucBuf = new unsigned char[iPixels];
	if(bZero) memset(pucBuf, 0, iPixels * sizeof(char));
	return pucBuf;
}

float* MotionCor2::Util::GGetFloatBuf(int* piSize, bool bZero)
{
	size_t tBytes = MotionCor2::Util::GetFloatBytes(piSize);
	float* gfBuf = 0L;
	hipMalloc(&gfBuf, tBytes);
	if(bZero) hipMemset(gfBuf, 0, tBytes);
	return gfBuf;
}

float* MotionCor2::Util::CGetFloatBuf(int* piSize, bool bZero)
{
	int iPixels = piSize[0] * piSize[1];
	float* pfBuf = new float[iPixels];
	if(bZero) memset(pfBuf, 0, iPixels * sizeof(float));
	return pfBuf;
}

void* MotionCor2::Util::GetPinnedBuf(int* piSize, int iPixelBytes, bool bZero)
{
	void* pvBuf = 0L;
	size_t tBytes = (size_t)piSize[0] * piSize[1] * iPixelBytes;
	hipHostMalloc(&pvBuf, tBytes, hipHostMallocDefault);
	if(bZero) memset(pvBuf, 0, tBytes);
	return pvBuf;
}

void* MotionCor2::Util::GetGpuBuf(int* piSize, int iPixelBytes, bool bZero)
{
	void* gvBuf = 0L;
	size_t tBytes = (size_t)piSize[0] * piSize[1] * iPixelBytes;
	hipMalloc(&gvBuf, tBytes);
	if(bZero) hipMemset(gvBuf, 0, tBytes);
	return gvBuf;
}

hipfftComplex* MotionCor2::Util::GGetCmpBuf(int* piSize, bool bZero)
{
	size_t tBytes = MotionCor2::Util::GetCmpBytes(piSize);
	hipfftComplex* gCmpBuf = 0L;
	hipMalloc(&gCmpBuf, tBytes);
	if(bZero) hipMemset(gCmpBuf, 0, tBytes);
	return gCmpBuf;
}

hipfftComplex* MotionCor2::Util::CGetCmpBuf(int* piSize, bool bZero)
{
	int iPixels = piSize[0] * piSize[1];
	hipfftComplex* pCmpBuf = new hipfftComplex[iPixels];
	if(bZero) memset(pCmpBuf, 0, iPixels * sizeof(hipfftComplex));
	return pCmpBuf;
}

unsigned char* MotionCor2::Util::GCopyFrame(unsigned char* pucSrc, int* piSize, hipStream_t stream)
{
	unsigned char* gucDst = MotionCor2::Util::GGetUCharBuf(piSize, false);
	MotionCor2::Util::CopyFrame(pucSrc, gucDst, piSize, stream);
	return gucDst;
}

unsigned char* MotionCor2::Util::CCopyFrame(unsigned char* pucSrc, int* piSize, hipStream_t stream)
{
	unsigned char* pucDst = MotionCor2::Util::CGetUCharBuf(piSize, false);
	MotionCor2::Util::CopyFrame(pucSrc, pucDst, piSize, stream);
	return pucDst;
}

void MotionCor2::Util::CopyFrame(unsigned char* pucSrc, 
	unsigned char* pucDst, int* piSize, hipStream_t stream)
{
	size_t tBytes = sizeof(char) * piSize[0] * piSize[1];
	hipMemcpyAsync(pucDst, pucSrc, tBytes, hipMemcpyDefault, stream);
}

float* MotionCor2::Util::GCopyFrame(float* pfSrc, int* piSize, hipStream_t stream)
{
	float* gfDst = MotionCor2::Util::GGetFloatBuf(piSize, false);
	MotionCor2::Util::CopyFrame(pfSrc, gfDst, piSize, stream);
	return gfDst;
}

float* MotionCor2::Util::CCopyFrame(float* pfSrc, int* piSize, hipStream_t stream)
{
        float* pfDst = MotionCor2::Util::CGetFloatBuf(piSize, false);
	MotionCor2::Util::CopyFrame(pfSrc, pfDst, piSize, stream);
        return pfDst;
}

void MotionCor2::Util::CopyFrame(float* pfSrc, float* pfDst, int* piSize, hipStream_t stream)
{       
	size_t tBytes = sizeof(float) * piSize[0] * piSize[1];
        hipMemcpyAsync(pfDst, pfSrc, tBytes, hipMemcpyDefault, stream);
}

hipfftComplex* MotionCor2::Util::GCopyFrame(hipfftComplex* pCmpSrc, int* piSize, hipStream_t stream)
{
	hipfftComplex* gCmpDst = MotionCor2::Util::GGetCmpBuf(piSize, false);
	MotionCor2::Util::CopyFrame(pCmpSrc, gCmpDst, piSize, stream);
	return gCmpDst;
}

hipfftComplex* MotionCor2::Util::CCopyFrame(hipfftComplex* pCmpSrc, int* piSize, hipStream_t stream)
{
	hipfftComplex* pCmpDst = MotionCor2::Util::CGetCmpBuf(piSize, false);
	MotionCor2::Util::CopyFrame(pCmpSrc, pCmpDst, piSize, stream);
	return pCmpDst;
}

void MotionCor2::Util::CopyFrame(hipfftComplex* pCmpSrc,
	hipfftComplex* pCmpDst, int* piSize, hipStream_t stream)
{
	size_t tBytes = sizeof(hipfftComplex) * piSize[0] * piSize[1];
	hipMemcpyAsync(pCmpDst, pCmpSrc, tBytes, hipMemcpyDefault, stream);
}

size_t MotionCor2::Util::GetGpuMemory(int iGpuId)
{	
	hipDeviceProp_t aDeviceProp;
	hipGetDeviceProperties(&aDeviceProp, iGpuId);
	return aDeviceProp.totalGlobalMem;
}

int MotionCor2::Util::CalcNumGpuFrames
(	int* piFrmSize,
	int iGpuId,
	double dOccupancy
)
{	size_t tGpuMem = MotionCor2::Util::GetGpuMemory(iGpuId);
	size_t tFrmMem = sizeof(float) * piFrmSize[0] * piFrmSize[1];
	int iNumFrames = (int)(tGpuMem * dOccupancy / tFrmMem);
	return iNumFrames;
}

void MotionCor2::Util::PrintGpuMemoryUsage(const char* pcInfo)
{
	size_t tTotal = 0, tFree = 0;
	hipError_t tErr = hipMemGetInfo(&tFree, &tTotal);
	//-------------------------------------------------
	tTotal /= (1024 * 1024);
	tFree /= (1024 * 1024);
	double dUsed = 100.0 - tFree * 100.0 / tTotal;
	printf("%s", pcInfo);
	printf("  total GPU memory: %ld MB\n", tTotal);
	printf("  free GPU memory:  %ld MB\n", tFree);
	printf("  used GPU memory: %8.2f%%\n\n", dUsed);
}

float MotionCor2::Util::GetGpuMemoryUsage(void)
{
	size_t tTotal = 0, tFree = 0;
	hipError_t tErr = hipMemGetInfo(&tFree, &tTotal);
	double dUsed = 1.0 - tFree * 1.0 / tTotal;
	return (float)dUsed;
}

void MotionCor2::Util::CheckCudaError(const char* pcLocation)
{
	hipError_t cuErr = hipGetLastError();
	if(cuErr == hipSuccess) return;
	//------------------------------
	fprintf(stderr, "%s: %s\n\t\n\n", pcLocation,
		hipGetErrorString(cuErr));
	hipDeviceReset();
	assert(0);
}

void MotionCor2::Util::CheckRUsage(const char* pcLocation)
{
	struct rusage resUsage;
	int iRet = getrusage(RUSAGE_SELF, &resUsage);
	double dMemUsageGB = resUsage.ru_maxrss / (1024.0 * 1024);
	printf("%s: total memory allocation %.2f GB\n", 
	   pcLocation, dMemUsageGB);
}
