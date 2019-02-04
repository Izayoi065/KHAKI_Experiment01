// KHAKI_Experiment01.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
////

#include "pch.h"

/* プロトタイプ宣言 */
void handExtractor(cv::InputArray inImage_, cv::OutputArray outImage_, cv::Scalar hsv_min, cv::Scalar hsv_max);
void newHandExtractor(cv::InputArray inImage_, cv::OutputArray outImage_, cv::OutputArray outMaskImage_, cv::Scalar hsv_min, cv::Scalar hsv_max, cv::Scalar hsv_Fourier_min, cv::Scalar hsv_Fourier_max);
void newHandExtractor(cv::InputArray inImage_, cv::OutputArray outImage_, cv::OutputArray outMaskImage_, cv::Scalar hsv_min, cv::Scalar hsv_max, cv::Scalar hsv_Fourier_min, cv::Scalar hsv_Fourier_max, double maskSize);
void shiftDft(cv::Mat &src, cv::Mat &dst);
void encodeImage(cv::InputArray A_, cv::OutputArray dest_);
void decodeImage(cv::InputArray A_, cv::OutputArray dest_);
void genMagImage(cv::InputArray A_, cv::OutputArray dest_);
void checkMatType(cv::InputArray inImage_);
unsigned int getDigit(unsigned int num);

int main()
{
	// test_TK005.avi
	/* 初期設定 */
	//cv::Scalar hsv_min = cv::Scalar(0, 101, 50);
	//cv::Scalar hsv_max = cv::Scalar(15, 180, 255);
	cv::Scalar hsv_min = cv::Scalar(0, 101, 50);
	cv::Scalar hsv_max = cv::Scalar(15, 180, 255);

	cv::Scalar hsv_Fourier_min = cv::Scalar(0, 0, 0);
	cv::Scalar hsv_Fourier_max = cv::Scalar(179, 255, 255);

	std::string inputstr;
	std::cout << "入力とする.aviファイル（絶対パス）：";
	std::cin >> inputstr;
	std::string inputFilePass = "data/resource/" + inputstr;
	// 入力となる動画ファイルを入力
	cv::VideoCapture video(inputFilePass);
	if (!video.isOpened()) {
		// 入力ファイルの読み込みに失敗した場合
		std::cout << "入力ファイルの読み込みに失敗しました．" << std::endl;
		return -1;
	}

	/* 入力ファイルのプロパティ */
	const int Width = video.get(CV_CAP_PROP_FRAME_WIDTH);		// 幅
	const int Heignt = video.get(CV_CAP_PROP_FRAME_HEIGHT);		// 高さ
	const int frameCount = video.get(CV_CAP_PROP_FRAME_COUNT);	// 総フレーム数

	std::string inputLog;
	std::cout << "出力ログのファイル名：";
	std::cin >> inputLog;
	std::string filename = "data/result/" + inputLog;
	std::ofstream writing_file;
	writing_file.open(filename, std::ios::out);
	writing_file << "従来手法の総ピクセル数,提案手法の総ピクセル数,従来手法の総ピクセル数-提案手法の総ピクセル数" << std::endl;
	std::cout << "writing " << filename << "..." << std::endl;

	const int digit = getDigit(frameCount);
	/* 入力ファイルの総フレーム数だけ探索を繰り返す */
	for (unsigned int i = 0; i < frameCount; i++) {
		cv::Mat frame, oldMethodImage, newMethodImage, MaskImage;
		video >> frame;	// 入力ファイルからフレーム画像を取得
		handExtractor(frame, oldMethodImage, hsv_min, hsv_max);	// 従来手法による手指領域の抽出
		int oldMethodZeroPixel = cv::countNonZero(oldMethodImage);
		newHandExtractor(frame, newMethodImage, MaskImage, hsv_min, hsv_max, hsv_Fourier_min, hsv_Fourier_max);	// 提案手法による手指領域の抽出
		int newMethodZeroPixel = cv::countNonZero(newMethodImage);

		writing_file << oldMethodZeroPixel << "," << newMethodZeroPixel << "," << oldMethodZeroPixel - newMethodZeroPixel << std::endl;	// csvファイルに結果を出力する，

		std::ostringstream ss;
		ss << std::setw(digit) << std::setfill('0') << i;
		std::string num(ss.str());

		std::string imgOrg = "data/result/images/image" + num + "(Org).png";
		cv::imwrite(imgOrg, frame);
		std::string imgOldMethod = "data/result/images/image" + num + "(OldMethod).png";
		cv::imwrite(imgOldMethod, oldMethodImage);
		std::string imgNewMethod = "data/result/images/image" + num + "(NewMethod).png";
		cv::imwrite(imgNewMethod, newMethodImage);
		std::string imgMaskImage = "data/result/images/image" + num + "(MaskImage).png";
		cv::imwrite(imgMaskImage, MaskImage);
	}
	std::cout << std::endl;
	std::cout << "Finish!" << std::endl;

	return 0;
}

/** @brief HSV色空間を利用したカラーベースの抽出手法にて，手指領域の抽出処理を実行する．
@note この関数は，入力画像から手指領域を抽出処理を適用する．出力では，抽出された手指領域の二値化画像が出力される．
@param image_		 カメラからの入力画像
@param outImage_	抽出された手指領域の二値化画像
@sa	Render
**/
void handExtractor(cv::InputArray inImage_, cv::OutputArray outImage_, cv::Scalar hsv_min, cv::Scalar hsv_max)
{
	cv::Mat inImage = inImage_.getMat();
	cv::Mat hsvImage, hsv_mask, dstImage, singlechannels[3];
	cv::cvtColor(inImage, hsvImage, CV_BGR2HSV);

	cv::inRange(hsvImage, hsv_min, hsv_max, hsv_mask);
	
	cv::morphologyEx(hsv_mask, hsv_mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)), cv::Point(-1, -1), 1);	// オープニング処理：内部ノイズの穴埋め
	cv::morphologyEx(hsv_mask, hsv_mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)), cv::Point(-1, -1), 1);	// クロージング処理：ゴマ塩ノイズの除去


	hsv_mask.copyTo(outImage_);
}

/** @brief 手指領域の抽出処理を実行する．
@note この関数は，入力画像から手指領域を抽出処理を適用する．出力では，抽出された手指領域の二値化画像が出力される．
@param image_		 カメラからの入力画像
@param outImage_	抽出された手指領域の二値化画像
@sa	Render
**/
void newHandExtractor(cv::InputArray inImage_, cv::OutputArray outImage_, cv::OutputArray outMaskImage_, cv::Scalar hsv_min, cv::Scalar hsv_max, cv::Scalar hsv_Fourier_min, cv::Scalar hsv_Fourier_max) {
	cv::Mat inImage = inImage_.getMat();
	cv::Mat hsvImage, hsv_mask, hsv_Fourier_mask, dft_dst, dft_dst_mask, spectreImg, mag_img, result, temp, result_mask, hsv_mask_org, finalMask, tempImage;
	handExtractor(inImage, hsv_mask, hsv_min, hsv_max);	// 従来手法に基づく手指領域のマスク画像を取得
	//handExtractor(inImage, hsv_Fourier_mask, hsv_Fourier_min, hsv_Fourier_max);	// 従来手法に基づく手指領域のマスク画像を取得

	// 入力画像にフーリエ変換
	encodeImage(inImage, dft_dst);

	// スペクトル画像を保存
	genMagImage(dft_dst, tempImage);
	tempImage.convertTo(tempImage, CV_32FC3, 255);	// スケーリング処理	0-1→0-255
	std::string fourierSpecImage = "data/result/images/image(FourSpec).png";
	//cv::imwrite(fourierSpecImage, tempImage);

	// ローパスフィルタに使用するマスク画像
	cv::Mat circleMask = cv::Mat::ones(dft_dst.size(), CV_8UC1) * 0;
	cv::circle(circleMask, cv::Point(dft_dst.cols / 2, dft_dst.rows / 2), 400, cv::Scalar(255), -1, 4);
	cv::circle(circleMask, cv::Point((dft_dst.cols / 2), (dft_dst.rows / 2)), 0.5, cv::Scalar(0), -1, 4);
	dft_dst.copyTo(dft_dst_mask, circleMask);	// スペクトル画像に対して低周波フィルタリングを施した画像

	// フィルタ通過後のスペクトル画像を保存
	genMagImage(dft_dst_mask, tempImage);
	tempImage.convertTo(tempImage, CV_32FC3, 255);	// スケーリング処理	0-1→0-255
	std::string fourierMagSpecImage = "data/result/images/image(FourMagSpec).png";
	//cv::imwrite(fourierMagSpecImage, tempImage);

	// 撮影範囲外に手指領域が残るはずがないため，事前にカット
	//cv::Mat DomeMask = cv::Mat::ones(dft_dst.size(), CV_32FC1) * 255;
	//cv::circle(DomeMask, cv::Point(dft_dst.cols / 2, dft_dst.rows / 2), 0, cv::Scalar(0), -1, 4);

	// スペクトルから画像の復元
	decodeImage(dft_dst_mask, result);
	result.convertTo(temp, CV_32FC1);
	std::string decodeImage = "data/result/images/image(decoded).png";
	temp.convertTo(tempImage, CV_32FC3, 255);	// スケーリング処理	0-1→0-255
	//cv::imwrite(decodeImage, tempImage);
	cv::threshold(temp, result_mask, 0.15, 255, cv::THRESH_BINARY);
	//bitwise_and(temp, DomeMask, result_mask);
	result_mask.convertTo(result_mask, CV_8UC1, 255);
	cv::resize(result_mask, result_mask, hsv_mask.size(), cv::INTER_CUBIC);	// 512 -> 504に変更
	cv::threshold(result_mask, result_mask, 250, 255, cv::THRESH_BINARY);
	//bitwise_and(result_mask, hsv_Fourier_mask, result_mask);	// 従来手法（HSVベース）と空間周波数フィルタリングでマスキング処理

	cv::morphologyEx(result_mask, result_mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)), cv::Point(-1, -1), 1);	// クロージング処理：ゴマ塩ノイズの除去
	cv::morphologyEx(result_mask, result_mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)), cv::Point(-1, -1), 1);	// オープニング処理：内部ノイズの穴埋め

	cv::Mat affineMat = (cv::Mat_<double>(2, 3) << 1.0, 0.0, 3, 0.0, 1.0, 3);
	cv::warpAffine(result_mask, result_mask, affineMat, result_mask.size(), CV_INTER_LINEAR, cv::BORDER_TRANSPARENT);
	result_mask.copyTo(outMaskImage_);

	bitwise_and(result_mask, hsv_mask, finalMask);	// 従来手法（HSVベース）と空間周波数フィルタリングでマスキング処理

	cv::morphologyEx(result_mask, result_mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)), cv::Point(-1, -1), 3);	// クロージング処理：ゴマ塩ノイズの除去
	cv::morphologyEx(result_mask, result_mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)), cv::Point(-1, -1), 1);	// オープニング処理：内部ノイズの穴埋め

	finalMask.copyTo(outImage_);
}

void newHandExtractor(cv::InputArray inImage_, cv::OutputArray outImage_, cv::OutputArray outMaskImage_, cv::Scalar hsv_min, cv::Scalar hsv_max, cv::Scalar hsv_Fourier_min, cv::Scalar hsv_Fourier_max, double maskSize) {
	cv::Mat inImage = inImage_.getMat();
	cv::resize(inImage, inImage, cv::Size(512, 512), cv::INTER_CUBIC);	// 504 -> 512に変更
	cv::Mat hsvImage, hsv_mask, hsv_Fourier_mask, dft_dst, dft_dst_mask, spectreImg, mag_img, result, temp, result_mask, hsv_mask_org, finalMask;
	handExtractor(inImage, hsv_mask, hsv_min, hsv_max);	// 従来手法に基づく手指領域のマスク画像を取得
	//handExtractor(inImage, hsv_Fourier_mask, hsv_Fourier_min, hsv_Fourier_max);	// 従来手法に基づく手指領域のマスク画像を取得

	// 入力画像にフーリエ変換
	encodeImage(inImage, dft_dst);

	// ローパスフィルタに使用するマスク画像
	cv::Mat circleMask = cv::Mat::ones(dft_dst.size(), CV_8UC1) * 0;
	cv::circle(circleMask, cv::Point(dft_dst.cols / 2, dft_dst.rows / 2), maskSize, cv::Scalar(255), -1, 4);
	cv::circle(circleMask, cv::Point(dft_dst.cols / 2, dft_dst.rows / 2), 0, cv::Scalar(0), -1, 4);
	dft_dst.copyTo(dft_dst_mask, circleMask);	// スペクトル画像に対して低周波フィルタリングを施した画像

	// 撮影範囲外に手指領域が残るはずがないため，事前にカット
	cv::Mat DomeMask = cv::Mat::ones(dft_dst.size(), CV_32FC1) * 255;
	cv::circle(DomeMask, cv::Point(dft_dst.cols / 2, dft_dst.rows / 2), 0, cv::Scalar(0), -1, 4);

	// スペクトル画像を
	//genMagImage();
	// スペクトルから画像の復元
	decodeImage(dft_dst_mask, result);
	result.convertTo(temp, CV_32FC1);
	cv::threshold(temp, temp, 0.15, 255, cv::THRESH_BINARY);
	bitwise_and(temp, DomeMask, result_mask);
	result_mask.convertTo(result_mask, CV_8UC1, 255);
	cv::resize(result_mask, result_mask, hsv_mask.size(), cv::INTER_CUBIC);	// 512 -> 504に変更
	//bitwise_and(result_mask, hsv_Fourier_mask, result_mask);	// 従来手法（HSVベース）と空間周波数フィルタリングでマスキング処理

	cv::morphologyEx(result_mask, result_mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)), cv::Point(-1, -1), 1);	// クロージング処理：ゴマ塩ノイズの除去
	cv::morphologyEx(result_mask, result_mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)), cv::Point(-1, -1), 1);	// オープニング処理：内部ノイズの穴埋め

	result_mask.copyTo(outMaskImage_);

	bitwise_and(result_mask, hsv_mask, finalMask);	// 従来手法（HSVベース）と空間周波数フィルタリングでマスキング処理

	cv::morphologyEx(result_mask, result_mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)), cv::Point(-1, -1), 1);	// クロージング処理：ゴマ塩ノイズの除去
	cv::morphologyEx(result_mask, result_mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)), cv::Point(-1, -1), 1);	// オープニング処理：内部ノイズの穴埋め

	finalMask.copyTo(outImage_);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// shiftDft
/// ref:http://www.cellstat.net/fourier/
void shiftDft(cv::Mat &src, cv::Mat &dst) {
	cv::Mat tmp;

	int cx = src.cols / 2;
	int cy = src.rows / 2;

	for (int i = 0; i <= cx; i += cx) {
		cv::Mat qs(src, cv::Rect(i^cx, 0, cx, cy));
		cv::Mat qd(dst, cv::Rect(i, cy, cx, cy));
		qs.copyTo(tmp);
		qd.copyTo(qs);
		tmp.copyTo(qd);
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// encodeImage
/// ref:http://www.cellstat.net/fourier/
void encodeImage(cv::InputArray A_, cv::OutputArray dest_) {
	cv::Mat A = A_.getMat();

	cv::cvtColor(A, A, cv::COLOR_BGR2GRAY);

	cv::Mat Re_img, Im_img, Complex_img, dft_src, dft_dst, dft_dst_p, mag_img;
	std::vector<cv::Mat> mv;

	cv::Size s_size = A.size();
	Re_img = cv::Mat(s_size, CV_64F);
	Im_img = cv::Mat::zeros(s_size, CV_64F);
	Complex_img = cv::Mat(s_size, CV_64FC2);

	A.convertTo(Re_img, CV_64F);
	mv.push_back(Re_img);
	mv.push_back(Im_img);
	merge(mv, Complex_img);

	int dft_rows = cv::getOptimalDFTSize(A.rows);
	int dft_cols = cv::getOptimalDFTSize(A.cols);

	dft_src = cv::Mat::zeros(dft_rows, dft_cols, CV_64FC2);

	cv::Mat roi(dft_src, cv::Rect(0, 0, A.cols, A.rows));
	Complex_img.copyTo(roi);

	dft(dft_src, dft_dst);
	shiftDft(dft_dst, dft_dst);

	dft_dst.copyTo(dest_);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// decodeImage
/// ref:http://www.cellstat.net/fourier/
void decodeImage(cv::InputArray A_, cv::OutputArray dest_) {
	cv::Mat A = A_.getMat();

	cv::Mat dft_src, idft_img;
	std::vector<cv::Mat> mv;
	double min, max;

	cv::Mat dft_dst_clone = A.clone();
	shiftDft(dft_dst_clone, dft_dst_clone);

	idft(dft_dst_clone, dft_src);
	split(dft_src, mv);
	minMaxLoc(mv[0], &min, &max);
	idft_img = cv::Mat(mv[0] * 1.0 / max);

	idft_img.copyTo(dest_);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// genMagImage
/// ref:http://www.cellstat.net/fourier/
void genMagImage(cv::InputArray A_, cv::OutputArray dest_) {
	cv::Mat A = A_.getMat();

	cv::Mat mag_img;
	std::vector<cv::Mat> mv;
	split(A, mv);

	magnitude(mv[0], mv[1], mag_img);
	log(mag_img + 1, mag_img);
	normalize(mag_img, mag_img, 0, 1, CV_MINMAX);

	mag_img.copyTo(dest_);
}

/* 受け取ったMat型のデータの型を出力 */
void checkMatType(cv::InputArray inImage_) {
	/* 受け取ったMat型のデータの型を出力 */
	// 要素の型とチャンネル数の組み合わせ。
	// 紙面の都合により、サンプルで使用する値のみ記述
	cv::Mat src_img = inImage_.getMat();
	std::cout << "type: " << (							// A Mapping of Type to Numbers in OpenCV
		src_img.type() == CV_8UC1 ? "CV_8UC1" :		// 0
		src_img.type() == CV_8SC1 ? "CV_8SC1" :		// 1
		src_img.type() == CV_16UC1 ? "CV_16UC1" :	// 2
		src_img.type() == CV_16SC1 ? "CV_16SC1" :	// 3
		src_img.type() == CV_32SC1 ? "CV_32SC1" :	// 4
		src_img.type() == CV_32FC1 ? "CV_32FC1" :	// 5
		src_img.type() == CV_64FC1 ? "CV_64FC1" :	// 6
		src_img.type() == CV_8UC2 ? "CV_8UC2" :		// 8
		src_img.type() == CV_8SC2 ? "CV_8SC2" :		// 9
		src_img.type() == CV_16UC2 ? "CV_16UC2" :	// 10
		src_img.type() == CV_16SC2 ? "CV_16SC2" :	// 11
		src_img.type() == CV_32SC2 ? "CV_32SC2" :	// 12
		src_img.type() == CV_32FC2 ? "CV_32FC2" :	// 13
		src_img.type() == CV_64FC2 ? "CV_64FC2" :	// 14
		src_img.type() == CV_8UC3 ? "CV_8UC3" :		// 16
		src_img.type() == CV_8SC3 ? "CV_8SC3" :		// 17
		src_img.type() == CV_16UC3 ? "CV_16UC3" :	// 18
		src_img.type() == CV_16SC3 ? "CV_16SC3" :	// 19
		src_img.type() == CV_32SC3 ? "CV_32SC3" :	// 20
		src_img.type() == CV_32FC3 ? "CV_32FC3" :	// 21
		src_img.type() == CV_64FC3 ? "CV_64FC3" :	// 22
		src_img.type() == CV_8UC4 ? "CV_8UC4" :		// 24
		src_img.type() == CV_8SC4 ? "CV_8SC4" :		// 25
		src_img.type() == CV_16UC4 ? "CV_16UC4" :	// 26
		src_img.type() == CV_16SC4 ? "CV_16SC4" :	// 27
		src_img.type() == CV_32SC4 ? "CV_32SC4" :	// 28
		src_img.type() == CV_32FC4 ? "CV_32FC4" :	// 29
		src_img.type() == CV_64FC4 ? "CV_64FC4" :	// 30
		"other"
		) << std::endl;
}

unsigned int getDigit(unsigned int num) {
	unsigned digit = 0;
	while (num != 0) {
		num /= 10;
		digit++;
	}
	return digit;
}