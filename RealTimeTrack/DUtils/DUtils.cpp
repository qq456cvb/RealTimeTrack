/*
 * DUtils.cpp
 *
 *  Created on: May 8, 2013
 *      Author: tdngo
 */

#include "DUtils.h"
#include <opencv2/core.hpp>

double DUtils::FPSCalculation()
{
	static int64 currentTime, lastTime = cv::getTickCount();
	static double fpsCounter = 0;
	static double fps = 0;

	currentTime = cv::getTickCount();
	fpsCounter++;

	const static int64 oneSecondTick = 1e6 * cv::getTickFrequency();

	// If 1 second has passed since the last FPS estimation, update the fps
	if (currentTime - lastTime > oneSecondTick)
	{
		fps = fpsCounter / ((currentTime - lastTime) / (double)(oneSecondTick));
		lastTime = currentTime;
		fpsCounter = 0;
	}

	return fps;
}

std::string DUtils::GetFullPath(std::string relativePath)
{
	char* absPath = realpath(relativePath.c_str(), NULL);
	std::string result = std::string(absPath);

	delete absPath;
	return result;
}
