#pragma once
#include <hip/hip_runtime.h>


class Util_String
{
public:

	static void RemoveSpace(char* pcString);

	static char* GetCopy(char* pcString);

};
