#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <yaml-cpp/yaml.h>

class CameraParam
{
public:
	CameraParam(const double fx, const double fy, const double cx, const double cy,
		const double k1, const double k2, const double p1, const double p2, const double k3)
		: fx(fx), fy(fy), cx(cx), cy(cy), k1(k1), k2(k2), p1(p1), p2(p2), k3(k3) {};

private:
	const double fx;
	const double fy;
	const double cx;
	const double cy;
	const double k1;
	const double k2;
	const double p1;
	const double p2;
	const double k3;

public:
	cv::Mat getCVcamMat()
	{
		cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
		cameraMatrix.at<double>(0, 0) = fx;
		cameraMatrix.at<double>(1, 1) = fy;
		cameraMatrix.at<double>(0, 2) = cx;
		cameraMatrix.at<double>(1, 2) = cy;

		return cameraMatrix;
	}

	cv::Mat getCVdistCoeffs()
	{
		cv::Mat distCoeffs = cv::Mat::zeros(5, 1, CV_64F);
		distCoeffs.at<double>(0) = k1;
		distCoeffs.at<double>(1) = k2;
		distCoeffs.at<double>(2) = p1;
		distCoeffs.at<double>(3) = p2;
		distCoeffs.at<double>(4) = k3;
		
		return distCoeffs;
	}

};

std::vector<cv::Mat> LoadAllImages_inDirectory(const std::string& dir_path)
{
	std::vector<std::string> fileNames;
	for (const auto& file : std::filesystem::directory_iterator(dir_path))
	{
		fileNames.push_back(file.path().string());
	}

	std::vector<cv::Mat> imgs;
	for (const auto& filename : fileNames)
	{
		imgs.push_back(cv::imread(filename));
	}

	return imgs;
}

void saveCameraParam(const std::string& file_dir, const cv::Mat& cameraMatrix, const cv::Mat& distCoeffs, const double& reproErr)
{
	YAML::Node camParam;
	camParam["CameraParam"]["fx"] = cameraMatrix.at<double>(0, 0);
	camParam["CameraParam"]["fy"] = cameraMatrix.at<double>(1, 1);
	camParam["CameraParam"]["cx"] = cameraMatrix.at<double>(0, 2);
	camParam["CameraParam"]["cy"] = cameraMatrix.at<double>(1, 2);
	camParam["DistCoeffs"]["k1"] = distCoeffs.at<double>(0);
	camParam["DistCoeffs"]["k2"] = distCoeffs.at<double>(1);
	camParam["DistCoeffs"]["p1"] = distCoeffs.at<double>(2);
	camParam["DistCoeffs"]["p2"] = distCoeffs.at<double>(3);
	camParam["DistCoeffs"]["k3"] = distCoeffs.at<double>(4);
	camParam["ReproErr"] = reproErr;

	YAML::Emitter out;
	out << camParam;

	std::ofstream ofs(file_dir);
	ofs << out.c_str();
	ofs.close();
}

CameraParam loadCameraParam(const std::string& file_dir)
{
	YAML::Node camParam = YAML::LoadFile(file_dir);
	const double fx = camParam["CameraParam"]["fx"].as<double>();
	const double fy = camParam["CameraParam"]["fy"].as<double>();
	const double cx = camParam["CameraParam"]["cx"].as<double>();
	const double cy = camParam["CameraParam"]["cy"].as<double>();
	const double k1 = camParam["DistCoeffs"]["k1"].as<double>();
	const double k2 = camParam["DistCoeffs"]["k2"].as<double>();
	const double p1 = camParam["DistCoeffs"]["p1"].as<double>();
	const double p2 = camParam["DistCoeffs"]["p2"].as<double>();
	const double k3 = camParam["DistCoeffs"]["k3"].as<double>();

	CameraParam cparam(fx, fy, cx, cy, k1, k2, p1, p2, k3);
	return cparam;
}

int main(void)
{

	CameraParam camParam = loadCameraParam("../../chArUcoCalibration/cameraParam.yaml");
	std::string img_path = "../../chArUcoCalibration/Capdata/tracking";

	std::vector<cv::Mat> tracking_img = LoadAllImages_inDirectory(img_path);
	const cv::Mat cameraMatrix = camParam.getCVcamMat();
	const cv::Mat distCoeffs = camParam.getCVdistCoeffs();

	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100);
	cv::Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();

	const int gridX = 5;
	const int gridY = 5;
	const double markerLength = 0.050;
	const double separateLength = markerLength / 6;
	const int firstMarker = 0;
	cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(gridX, gridY, markerLength, separateLength, dictionary, firstMarker);


	bool refindStrategy = true;

	for (const auto& img : tracking_img)
	{
		std::vector<int> ids;
		std::vector<std::vector<cv::Point2f>> corners, rejected;

		cv::aruco::detectMarkers(img, dictionary, corners, ids, detectorParams, rejected);
		if (refindStrategy) cv::aruco::refineDetectedMarkers(img, board, corners, ids, rejected);
		cv::Mat imageCopy;
		img.copyTo(imageCopy);
		cv::Mat rvec, tvec;
		cv::aruco::estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, rvec, tvec);
		if (ids.size() > 0)
		{
			cv::aruco::drawDetectedMarkers(imageCopy, corners);
			cv::aruco::drawAxis(imageCopy, cameraMatrix, distCoeffs, rvec, tvec, 0.05);
			cv::imshow("img", imageCopy);
			cv::waitKey(0);
		}
	}

	return 0;
}