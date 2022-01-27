#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/aruco/charuco.hpp>
#include <yaml-cpp/yaml.h>


std::vector<cv::Mat> LoadAllImages_inDirectory(const std::string& dir_path)
{
	std::vector<std::string> fileNames;
	for (const auto& file : std::filesystem::directory_iterator(dir_path))
	{
		fileNames.push_back(file.path().string());
	}

	std::vector<cv::Mat> calib_imgs;
	for (const auto& filename : fileNames)
	{
		calib_imgs.push_back(cv::imread(filename));
	}

	return calib_imgs;
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

int main(void)
{
	//フォルダ内ファイルパスの読み込み
	std::string directory_path = "../../chArUcoCalibration/Capdata/calib/calib1";
	std::vector<cv::Mat> calib_imgs = LoadAllImages_inDirectory(directory_path);

	cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_100);
	cv::Ptr<cv::aruco::DetectorParameters> detectorParams = cv::aruco::DetectorParameters::create();


	const int squaresX = 10;
	const int squaresY = 7;
	const float squareLength = 0.047;
	const float markerLength = squareLength * 3 / 5;
	cv::Ptr<cv::aruco::CharucoBoard> charucoBoard = cv::aruco::CharucoBoard::create(squaresX, squaresY, squareLength, markerLength, dictionary);
	cv::Ptr<cv::aruco::Board> board = charucoBoard.staticCast<cv::aruco::Board>();

	bool refindStrategy = true;

	std::vector<std::vector<std::vector<cv::Point2f>>> allCorners;
	std::vector<std::vector<int>> allIds;
	std::vector<cv::Mat> allImgs;

	for (const auto& img : calib_imgs)
	{
		std::vector<int> ids;
		std::vector<std::vector<cv::Point2f>> corners, rejected;
		
		cv::aruco::detectMarkers(img, dictionary, corners, ids, detectorParams, rejected);
		if (refindStrategy) cv::aruco::refineDetectedMarkers(img, board, corners, ids, rejected);
		cv::Mat imageCopy;
		img.copyTo(imageCopy);
		cv::Mat currentCharucoCorners, currentCharucoIds;
		if (ids.size() > 0)
		{
			cv::aruco::interpolateCornersCharuco(corners, ids, img, charucoBoard, currentCharucoCorners, currentCharucoIds);
		}

		if (ids.size() > 0)
		{
			cv::aruco::drawDetectedMarkers(imageCopy, corners);
		}

		if (currentCharucoCorners.total() > 0){
			cv::aruco::drawDetectedCornersCharuco(imageCopy, currentCharucoCorners, currentCharucoIds);
		}

		//cv::imshow("img", imageCopy);
		//cv::waitKey(0);

		allCorners.push_back(corners);
		allIds.push_back(ids);
		allImgs.push_back(img);
	}

	if (allIds.size() < 1)
	{
		std::cerr << "Not enough captures for calibration." << std::endl;
	}

	std::vector<std::vector<cv::Point2f>> allCornersConcatenated;
	std::vector<int> allIdsConcatenated;
	std::vector<int> markerCounterPerFrame;
	markerCounterPerFrame.reserve(allCorners.size());

	for (unsigned int i = 0; i < allCorners.size(); i++)
	{
		markerCounterPerFrame.push_back((int)allCorners[i].size());
		for (unsigned int j = 0; j < allCorners[i].size(); j++)
		{
			allCornersConcatenated.push_back(allCorners[i][j]);
			allIdsConcatenated.push_back(allIds[i][j]);
		}
	}

	cv::Size imgSize = allImgs[0].size();
	cv::Mat cameraMatrix, distCoeffs;
	double reproErr_aruco = 0;
	reproErr_aruco = cv::aruco::calibrateCameraAruco(allCornersConcatenated, allIdsConcatenated, markerCounterPerFrame, board, imgSize, cameraMatrix, distCoeffs, cv::noArray(), cv::noArray());
	//std::cout << cameraMatrix << std::endl;
	//std::cout << distCoeffs << std::endl;

	const int frame_num = allCorners.size();
	std::vector<cv::Mat> allCharucoCorners;
	std::vector<cv::Mat> allCharucoIds;
	std::vector<cv::Mat> filteredImages;
	allCharucoCorners.reserve(frame_num);
	allCharucoIds.reserve(frame_num);

	for (int i = 0; i < frame_num; i++)
	{
		cv::Mat currentCharucoCorners, currentCharucoIds;
		cv::aruco::interpolateCornersCharuco(allCorners[i], allIds[i], allImgs[i], charucoBoard, currentCharucoCorners, currentCharucoIds, cameraMatrix, distCoeffs);
		allCharucoCorners.push_back(currentCharucoCorners);
		allCharucoIds.push_back(currentCharucoIds);
		filteredImages.push_back(allImgs[i]);
	}

	if (allCharucoCorners.size() < 4) {
		std::cerr << "Not enough corners for calibration" << std::endl;
		return 0;
	}

	std::vector<cv::Mat> rvecs, tvecs;
	double reproErr = 0;
	reproErr = cv::aruco::calibrateCameraCharuco(allCharucoCorners, allCharucoIds, charucoBoard, imgSize, cameraMatrix, distCoeffs, rvecs, tvecs);

	//検出結果の可視化
	for (unsigned int i = 0; i < filteredImages.size(); i++)
	{
		cv::Mat imageCopy = filteredImages[i].clone();
		if (allIds[i].size() > 0) {
			if (allCharucoCorners[i].total() > 0)
			{
				cv::aruco::drawDetectedCornersCharuco(imageCopy, allCharucoCorners[i], allCharucoIds[i]);
			}
		}

		cv::imshow("image", imageCopy);
		cv::waitKey(0);
	}

	//カメラパラメータの書き出し
	saveCameraParam("../../chArUcoCalibration/cameraParam.yaml", cameraMatrix, distCoeffs, reproErr);

	return 0;
}