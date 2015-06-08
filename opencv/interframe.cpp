#include <string>  
#include <iostream>  
#include <math.h>  
#include <vector>  
#include <map>  


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui_c.h> 

#define ClusterNum (5)  

using namespace cv;
using namespace std;

//string filename = "1.jpg";
map<int, Vec2i> avg, newavg;       //map<id,color>  
Mat outcenter, newoutcenter;   //全局center变量
int blocksize = 16; //block 大小
int searchlength = 16; //搜索长度

int vectorx[1081][1921], vectory[1081][1921];
int storex[1081][1921], storey[1081][1921];

float myabs(float a)
{
	if (a > 0) return a;
	return -a;
}
int nomal(int value, int top)
{
	if (value <= 0) return 0;
	if (value >= top) return top;
	return value;
}

Vec2i faim(int searchaim, int length)
{
	int lengthy = 0;
	int lengthx = 0;
	if (searchaim == 0)
	{
		lengthy = -length;
		lengthx = 0;
	}
	if (searchaim == 1)
	{
		lengthy = -length;
		lengthx = +length;
	}
	if (searchaim == 2)
	{
		lengthy = 0;
		lengthx = length;
	}
	if (searchaim == 3)
	{
		lengthy = +length;
		lengthx = +length;
	}
	if (searchaim == 4)
	{
		lengthy = +length;
		lengthx = 0;
	}
	if (searchaim == 5)
	{
		lengthy = +length;
		lengthx = -length;
	}
	if (searchaim == 6)
	{
		lengthy = 0;
		lengthx = -length;
	}
	if (searchaim == 7)
	{
		lengthy = -length;
		lengthx = -length;
	}
	return (lengthy, lengthx);
}
int minlengthset[1000][1000], minsearchaimset[1000][1000];
int minblocknumset[1000][1000];// 最好这些可以换成参数形式
Mat rebuilt(Mat frame0, Mat frame1)
{
	int row = frame0.rows;
	int col = frame0.cols;
	Mat aimframe(frame0.size(), CV_8UC3, Scalar(0, 0, 0));
	for (int y = 0; y < row / blocksize; y++)
		for (int x = 0; x < col / blocksize; x++)
		{
		//myabs(framemiss.at<Vec3b>(sy + y*blocksize, sx + x*blocksize)[0]
		//	- frame1.at<Vec3b>(sy + lengthy + y*blocksize, sx + lengthx + x*blocksize)[0])
		int lengthy = faim(minsearchaimset[y][x], minlengthset[y][x]).val[0]
			, lengthx = faim(minsearchaimset[y][x], minlengthset[y][x]).val[1];

		for (int sy = 0; sy < blocksize; sy++)
			for (int sx = 0; sx < blocksize; sx++)
			{
			if (minblocknumset[y][x] == 0)
			{
				aimframe.at<Vec3b>(y*blocksize + sy, x*blocksize + sx) = frame0.at<Vec3b>(nomal(y*blocksize + sy + lengthy, row), nomal(x*blocksize + sx + lengthx, col));
			}
			else
			{
				aimframe.at<Vec3b>(y*blocksize + sy, x*blocksize + sx) = frame1.at<Vec3b>(nomal(y*blocksize + sy + lengthy, row), nomal(x*blocksize + sx + lengthx, col));
			}
			}

		}
	return aimframe;
}

Mat clustering(Mat src)
{
	int row = src.rows;
	int col = src.cols;
	unsigned long int size = row*col;

	Mat clusters(size, 1, CV_32SC1);    //clustering Mat, save class label at every location;  

	//convert src Mat to sample srcPoint.  
	Mat srcPoint(size, 1, CV_32FC3);
	map<int, int> count;       //map<id,num>  


	Vec3f* srcPoint_p = (Vec3f*)srcPoint.data;//////////////////////////////////////////////  
	Vec3f* src_p = (Vec3f*)src.data;
	unsigned long int i;

	for (i = 0; i < size; i++)
	{
		*srcPoint_p = *src_p;
		srcPoint_p++;
		src_p++;
	}
	Mat center(ClusterNum, 1, CV_32FC3);
	double compactness;//compactness to measure the clustering center dist sum by different flag  
	compactness = kmeans(srcPoint, ClusterNum, clusters,
		cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 0.1), 2,
		KMEANS_PP_CENTERS, center);
	outcenter = center;
	cout << "center row:" << center.rows << " col:" << center.cols << endl;

	/*
	for (int y = 0; y < center.rows; y++)
	{
	Vec3f* imgData = center.ptr<Vec3f>(y);
	for (int x = 0; x < center.cols; x++)
	{
	cout << center.at<Vec2f>(x, y);
	//cout << imgData[x].val[0] << " " << imgData[x].val[1] << " " << imgData[x].val[2] << endl;
	}
	cout << endl;
	}*/


	double minH, maxH;
	minMaxLoc(clusters, &minH, &maxH);          //remember must use "&"  
	cout << "H-channel min:" << minH << " max:" << maxH << endl;

	int* clusters_p = (int*)clusters.data;
	//show label mat  
	Mat label(src.size(), CV_32SC1);
	int* label_p = (int*)label.data;
	//assign the clusters to Mat label  
	for (i = 0; i < size; i++)
	{
		*label_p = *clusters_p;
		label_p++;
		clusters_p++;
	}



	Mat label_show;
	label.convertTo(label_show, CV_8UC1);
	normalize(label_show, label_show, 255, 0, CV_MINMAX);
	//	imshow("label", label_show);





	//compute average color value of one label  
	for (int y = 0; y < row; y++)
	{
		//const Vec3f* imgData = src.ptr<Vec3f>(y);
		int* idx = label.ptr<int>(y);
		for (int x = 0; x < col; x++)
		{

			avg[idx[x]][0] += x;
			avg[idx[x]][1] += y;
			count[idx[x]] ++;
		}
	}
	//output the average value (clustering center)  
	//计算所得的聚类中心与kmean函数中center的第一列一致，  
	//以后可以省去后面这些繁复的计算，直接利用center,  
	//但是仍然不理解center的除第一列以外的其他列所代表的意思  
	Mat showImg(src.size(), CV_32FC3);

	for (i = 0; i < ClusterNum; i++)
	{
		avg[i] /= count[i];
		//cout << avg[i] << endl;
		if (avg[i].val[0]>0 && avg[i].val[1]>0)
		{
			//		cout << i << ": " << avg[i].val[0] << " " << avg[i].val[1] << " count:" << count[i] << endl;

		}

	}
	//show the clustering img;  

	for (int y = 0; y < row; y++)
	{
		Vec3f* imgData = showImg.ptr<Vec3f>(y);
		int* idx = label.ptr<int>(y);
		for (int x = 0; x < col; x++)
		{
			int id = idx[x];
			imgData[x].val[0] = center.at<float>(id, 0);
			imgData[x].val[1] = center.at<float>(id, 1);
			imgData[x].val[2] = center.at<float>(id, 2);
		}
	}

	for (i = 0; i < ClusterNum; i++)
	{

		circle(showImg, avg[i], 30, Scalar(0, 0, 255), -1, 8);
	}

	normalize(showImg, showImg, 1, 0, CV_MINMAX);
	//imshow("show", showImg);
	//waitKey();
	return label;
}


int main()
{
	string inputname;
	string outputname;
	/*
	cout << "Please input file name" << endl;
	cin >> inputname;
	cout << "Please input outputfile name" << endl;
	cin >> outputname;
	*/
	//outputname = outputname + ".avi";
	outputname = "output.avi";

	//VideoCapture capture("demoinput.mp4");
	VideoCapture capture("1.mkv");
	float hranges[] = { 0, 255 };
	const float* ranges[] = { hranges };


	//输出视频格式

	int outputfps = 30;
	int ex = static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));     // 得到编码器的int表达式
	char EXT[] = { ex & 0XFF, (ex & 0XFF00) >> 8, (ex & 0XFF0000) >> 16, (ex & 0XFF000000) >> 24, 0 };
	VideoWriter outputVideo;
	Size S = Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH),    //获取输入尺寸
		(int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));





	CvMemStorage *storage = cvCreateMemStorage(0);
	//create 动态存储


	Mat mainframe, keyframe[10];


	outputVideo.open(outputname, -1, 30, S, true);

	if (!capture.isOpened())
		cout << "fail to open!" << endl;
	long totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);
	cout << "整个视频共" << totalFrameNumber << "帧" << endl;

	//设置开始帧()
	long frameToStart = 0;




	capture.set(CV_CAP_PROP_POS_FRAMES, frameToStart);
	cout << "从第" << frameToStart << "帧开始读" << endl;

	//获取帧率
	double rate = capture.get(CV_CAP_PROP_FPS);
	//	double rate = 30;

	cout << "帧率为:" << rate << endl;

	//定义一个用来控制读取视频循环结束的变量
	bool stop = false;
	//承载每一帧的图像

	Mat hist_r, hist_g, hist_b, histold;
	int copyframe = 0;
	//namedWindow("MainU");
	//两帧间的间隔时间:
	int delay = 1000 / rate;

	//利用while循环读取帧
	//currentFrame是在循环体中控制读取到指定的帧后循环结束的变量
	long currentFrame = frameToStart;
	int keyfarmenum = 0;
	Mat frame0, framemiss, frame1;

	Mat framecul;
	Mat newpixId, pixId;
	Mat dst;
	RNG rng; //随机数产生器
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	double minj = 0, mink = 0;;
	for (int i = 0; i <= totalFrameNumber / 2; i++)
	{
		capture.read(frame0);



		if (i != 0) //第一次循环不输出 计算出的帧
		{

			/*
			Mat img = frame0.clone();
			GaussianBlur(img, img, Size(3, 3), 0);
			img.convertTo(img, CV_32FC3);
			pixId = clustering(img);

			//	Mat aimframe(frame0.size(), CV_32FC3);
			int motivex[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 }
			, motivey[10] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 }; //运动向量
			for (int j = 0; j < ClusterNum; j++)
			{
			for (int k = 0; k < ClusterNum; k++)
			{
			if (myabs(avg[j].val[0] - newavg[k].val[0]) < 100 && myabs(avg[j].val[1] - newavg[k].val[1]) < 100
			&& myabs(outcenter.at<float>(j, 0) - outcenter.at<float>(k, 0)) < 30          //三个通道颜色差
			&& myabs(outcenter.at<float>(j, 1) - outcenter.at<float>(k, 2)) < 30
			&& myabs(outcenter.at<float>(j, 2) - outcenter.at<float>(k, 2)) < 30
			)
			{
			motivex[j] = (newavg[k].val[0] - avg[j].val[0]) / 2;
			motivey[j] = (newavg[k].val[1] - avg[j].val[1]) / 2;
			break;
			}

			}
			//cout << motivex[j] << "　" << motivey[j] << endl;
			}

			//system("pause");
			int row = frame0.rows;
			int col = frame0.cols;
			//Mat temp = frame0.clone();
			Mat aimframe(frame0.size(), CV_8UC3, Scalar(0, 0, 0));
			//Mat m(3, 5, CV_32FC3,);
			//Mat aimframe = frame0.clone();
			//cvZero(&aimframe);
			//temp.convertTo(temp, CV_32FC3);

			//imshow("MainU", temp);
			//waitKey();
			for (int y = 0; y < row; y++)
			{
			Vec3f* imgData = frame0.ptr<Vec3f>(y);
			int* idx = pixId.ptr<int>(y);
			for (int x = 0; x < col; x++)
			{
			int id = idx[x];
			//Vec3b test = temp.at<Vec3b>(y, x).val[0];

			// cout << temp << endl;
			//cout << temp << endl;
			aimframe.at<Vec3b>(nomal(y + motivey[id], row - 1), nomal(x + motivex[id], col - 1)) = frame0.at<Vec3b>(y, x);;
			//temp.at<Vec3f>(y, x);
			// system("pause");
			//imgData[x];



			// aimframe.at<Vec3f>(nomal(y + motivey[id], frame0.rows - 1), nomal(x + motivex[id], frame0.cols - 1)) = (1,1,1);


			//aimframe.at<float>(y + motive[id].val[1], x + motive[id].val[0]) = imgData[x].val[1];
			// aimframe.at<float>(y + motive[id].val[1], x + motive[id].val[0]) = imgData[x].val[2];
			// imgData[x].val[1] = center.at<float>(id, 1);
			//imgData[x].val[2] = center.at<float>(id, 2);
			}
			}
			*/

			Mat aimframe(frame0.size(), CV_8UC3, Scalar(0, 0, 0));
			int row = frame0.rows;
			int col = frame0.cols;
			int minlength, minsearchaim, minblocknum;


			int searchlength = 25;
			int failedtime = 0;
			int failflag = 0;
			for (int y = 0; y < row; y++)
				for (int x = 0; x < col; x++)
				{
				int value, value2;
				int tempx, tempy;
				int tempx2, tempy2;
				/*
				if (x == 0)
				{
				if (y == 0)
				{
				tempx = 0;
				tempy = 0;
				}
				else
				{
				tempx = vectorx[y - 1][x];
				tempy = vectory[y - 1][x];
				}

				}
				else
				{
				tempx = vectorx[y][x-1];
				tempy = vectory[y][x - 1];
				}*/
				if (x % 2 == 0)
				{
					if (y == 0)
					{
						tempx = 0;
						tempy = 0;
					}
					else
					{
						tempx = vectorx[y - 1][x];
						tempy = vectory[y - 1][x];
					}
				}
				else
				{
					if (y == 0)
					{
						tempx = vectorx[y][x - 1];
						tempy = vectory[y][x - 1];
					}
					if (y != 0)
					{
						tempx = vectorx[y][x - 1];
						tempy = vectory[y][x - 1];
					}

				}
				value = myabs(framemiss.at<Vec3b>(y, x)[0]
					- frame0.at<Vec3b>(nomal(y + tempy, row), nomal(x + tempx, col))[0])
					+ myabs(framemiss.at<Vec3b>(y, x)[1]
					- frame0.at<Vec3b>(nomal(y + tempy, row), nomal(x + tempx, col))[1])
					+ myabs(framemiss.at<Vec3b>(y, x)[2]
					- frame0.at<Vec3b>(nomal(y + tempy, row), nomal(x + tempx, col))[2]);

				if (vectorx[x][y])
				{
					tempy2 = vectory[x][y];
					tempx2 = vectorx[x][y];
				}

				value2 = myabs(framemiss.at<Vec3b>(y, x)[0]
					- frame0.at<Vec3b>(nomal(y + tempy2, row), nomal(x + tempx2, col))[0])
					+ myabs(framemiss.at<Vec3b>(y, x)[1]
					- frame0.at<Vec3b>(nomal(y + tempy2, row), nomal(x + tempx2, col))[1])
					+ myabs(framemiss.at<Vec3b>(y, x)[2]
					- frame0.at<Vec3b>(nomal(y + tempy2, row), nomal(x + tempx2, col))[2]);





				if (value <= 40 || value2 <= 40)
				{
					if (value <= 40)
					{
						vectorx[y][x] = tempx;
						vectory[y][x] = tempy;
						storex[y][x] = 0;
						storey[y][x] = 0;
					}
					else
					{
						vectorx[y][x] = tempx2;           //this is also a fail
						vectory[y][x] = tempy2;
						if (x % 2 == 0)
						{
							if (y == 0)
							{
								storex[y][x] = tempx2;
								storey[y][x] = tempy2;
							}
							else
							{
								storex[y][x] = tempx2 - vectorx[y - 1][x];
								storex[y][x] = tempy2 - vectory[y - 1][x];
							}

						}
						else
						{
							storex[y][x] = tempx2 - vectorx[y][x - 1];
							storex[y][x] = tempy2 - vectory[y][x - 1];
						}
					}

				}
				else
				{

					failedtime++;
					if (failedtime > 0.021*col*row)
					{
						failflag = 1;
						break;
					}
					int findvalue;
					int rememberx, rememvery;
					int mini = 9999999;
					for (int sy = -searchlength; sy < searchlength; sy++)
						for (int sx = -searchlength; sx < searchlength; sx++)
						{
						if (sy + y >= row || sy + y < 0 || sx + x >= col || sx + x < 0) continue;
						findvalue = myabs(framemiss.at<Vec3b>(y, x)[0]
							- frame0.at<Vec3b>(nomal(sy + y, row), nomal(sx + x, col))[0])
							+ myabs(framemiss.at<Vec3b>(y, x)[1]
							- frame0.at<Vec3b>(nomal(sy + y, row), nomal(sx + x, col))[1])
							+ myabs(framemiss.at<Vec3b>(y, x)[2]
							- frame0.at<Vec3b>(nomal(sy + y, row), nomal(sx + x, col))[2]);
						if (findvalue <mini)
						{
							mini = findvalue;
							rememberx = sx;
							rememvery = sy;
						}
						if (findvalue == 0) break;

						}
					vectorx[y][x] = rememberx;
					vectory[y][x] = rememvery;
					if (x % 2 == 0)
					{
						if (y == 0)
						{
							storex[y][x] = rememberx;
							storey[y][x] = rememvery;
						}
						else
						{
							storex[y][x] = rememberx - vectorx[y - 1][x];
							storex[y][x] = rememvery - vectory[y - 1][x];
						}

					}
					else
					{
						storex[y][x] = rememberx - vectorx[y][x - 1];
						storex[y][x] = rememvery - vectory[y][x - 1];
					}
				}


				}



			if (failflag == 1)
			{

				for (int y = 0; y < row / blocksize; y++)
					for (int x = 0; x < col / blocksize; x++)
					{

					int minsumvalue = 999999999;
					int value = 0;
					int lastvalue = 0;

					for (int blocknum = 0; blocknum <= 1; blocknum++) //搜索前块儿活后块儿
						for (int searchaim = 0; searchaim < 8; searchaim++) //搜索方向八个方向
							for (int length = 0; length <= searchlength; length = length + 2)  //搜索距离
							{
						value = 0;
						lastvalue = 999999999;  //方向正确会趋近单调低价;
						if (length == 0 && searchaim == 0 && blocknum == 0 && x + y != 0)
						{

							length = minlength;
							searchaim = minsearchaim;
							blocknum = minblocknum;
							for (int sy = 0; sy < blocksize; sy++)
								for (int sx = 0; sx < blocksize; sx++)
								{
								int lengthy = faim(searchaim, length).val[0];
								int lengthx = faim(searchaim, length).val[1];


								if (blocknum == 0)
								{
									value = value + myabs(framemiss.at<Vec3b>(sy + y*blocksize, sx + x*blocksize)[0]
										- frame0.at<Vec3b>(nomal(sy + lengthy + y*blocksize, row), nomal(sx + lengthx + x*blocksize, col))[0])
										+ myabs(framemiss.at<Vec3b>(sy + y*blocksize, sx + x*blocksize)[1]
										- frame0.at<Vec3b>(nomal(sy + lengthy + y*blocksize, row), nomal(sx + lengthx + x*blocksize, col))[1])
										+ myabs(framemiss.at<Vec3b>(sy + y*blocksize, sx + x*blocksize)[2]
										- frame0.at<Vec3b>(nomal(sy + lengthy + y*blocksize, row), nomal(sx + lengthx + x*blocksize, col))[2]);
								}
								if (blocknum == 1)
								{
									value = value + myabs(framemiss.at<Vec3b>(sy + y*blocksize, sx + x*blocksize)[0]
										- frame1.at<Vec3b>(nomal(sy + lengthy + y*blocksize, row), nomal(sx + lengthx + x*blocksize, col))[0])
										+ myabs(framemiss.at<Vec3b>(sy + y*blocksize, sx + x*blocksize)[1]
										- frame1.at<Vec3b>(nomal(sy + lengthy + y*blocksize, row), nomal(sx + lengthx + x*blocksize, col))[1])
										+ myabs(framemiss.at<Vec3b>(sy + y*blocksize, sx + x*blocksize)[2]
										- frame1.at<Vec3b>(nomal(sy + lengthy + y*blocksize, row), nomal(sx + lengthx + x*blocksize, col))[2]);
								}

								if (value > minsumvalue) break;
								}
							if (value < minsumvalue)
							{
								minsumvalue = value;
								minlength = length;
								minsearchaim = searchaim;
								minblocknum = blocknum;
							}
							length = 0;
							searchaim = 0;
							blocknum = 0;
						}






						for (int sy = 0; sy < blocksize; sy++)
							for (int sx = 0; sx < blocksize; sx++)
							{
							int lengthy = faim(searchaim, length).val[0];
							int lengthx = faim(searchaim, length).val[1];


							if (blocknum == 0)
							{
								value = value + myabs(framemiss.at<Vec3b>(sy + y*blocksize, sx + x*blocksize)[0]
									- frame0.at<Vec3b>(nomal(sy + lengthy + y*blocksize, row), nomal(sx + lengthx + x*blocksize, col))[0])
									+ myabs(framemiss.at<Vec3b>(sy + y*blocksize, sx + x*blocksize)[1]
									- frame0.at<Vec3b>(nomal(sy + lengthy + y*blocksize, row), nomal(sx + lengthx + x*blocksize, col))[1])
									+ myabs(framemiss.at<Vec3b>(sy + y*blocksize, sx + x*blocksize)[2]
									- frame0.at<Vec3b>(nomal(sy + lengthy + y*blocksize, row), nomal(sx + lengthx + x*blocksize, col))[2]);
							}
							if (blocknum == 1)
							{
								value = value + myabs(framemiss.at<Vec3b>(sy + y*blocksize, sx + x*blocksize)[0]
									- frame1.at<Vec3b>(nomal(sy + lengthy + y*blocksize, row), nomal(sx + lengthx + x*blocksize, col))[0])
									+ myabs(framemiss.at<Vec3b>(sy + y*blocksize, sx + x*blocksize)[1]
									- frame1.at<Vec3b>(nomal(sy + lengthy + y*blocksize, row), nomal(sx + lengthx + x*blocksize, col))[1])
									+ myabs(framemiss.at<Vec3b>(sy + y*blocksize, sx + x*blocksize)[2]
									- frame1.at<Vec3b>(nomal(sy + lengthy + y*blocksize, row), nomal(sx + lengthx + x*blocksize, col))[2]);
							}
							//cout << minsumvalue << endl;

							if (value > minsumvalue) break;
							//if (value > 2000) break;
							}

						if (value > lastvalue)
						{
							break;
						}
						lastvalue = value;
						if (value < minsumvalue)
						{
							minsumvalue = value;
							minlength = length;
							minsearchaim = searchaim;
							minblocknum = blocknum;
						}

							}
					//cout << minlength << endl;
					minlengthset[y][x] = minlength;   //记录最小值
					minsearchaimset[y][x] = minsearchaim; //记录最小值；
					minblocknumset[y][x] = minblocknum; //记录最小值
					}






				cout << "Use block way" << endl;
				framecul = rebuilt(frame0, frame1); // 计算中间帧
				//imshow("MainU", framecul);
				// waitKey(0);

			}
			else                       //fail not happened.
			{

				int  num = 0;

				cout << failedtime << endl;
				//cout << col << "　　" << row << endl;
				for (int y = 0; y < row; y++)
					for (int x = 0; x < col; x++)
					{
					aimframe.at<Vec3b>(y, x) = frame0.at<Vec3b>(nomal(vectory[y][x] + y, row), nomal(vectorx[y][x] + x, col));
					}

				framecul = aimframe;

			}





			/*
			for (int y = 0; y < row; y++)
			{
			if (y % 2 == 1)
			{
			aimframe.row(y) = frame0.row(y)*1;
			}
			else
			{
			aimframe.row(y) = frame1.row(y)*1;
			}
			}
			framecul = aimframe;*/
			//imshow("MainU", framecul);
			//waitKey(0);
			//imshow("MainU", framemiss);
			//waitKey(0);
			//cout << subnum(frame0,frame1) << endl;


			outputVideo << framecul; //输出已知帧

			frame1.release();
			//newpixId.release();
			//newoutcenter.release();
			//img.release();

		}

		capture.read(framemiss);
		frame1 = frame0.clone();

		newpixId = pixId.clone();
		newavg = avg;
		newoutcenter = outcenter.clone();

		cout << i * 2 << "/" << totalFrameNumber << endl;
		outputVideo << frame0;
	}



	outputVideo.release();
	//debug for out put;
	//stop = true;


	//capture.open("output.avi");
	//capture.open("2.mkv");
	/*
	capture.set(CV_CAP_PROP_POS_FRAMES, 0);

	while (!stop)
	{
	//读取下一帧
	if (!capture.read(frame0))
	{
	cout << "读取视频失败" << endl;
	return -1;
	}

	long long valuesub = 0;
	//	system("pause");


	imshow("MainU", frame0);
	//cout << "正在读取第" << currentFrame << "帧" << endl;

	int c = waitKey(delay);
	//按下ESC或者到达指定的结束帧后退出读取视频
	if ((char)c == 27 || currentFrame > frameToStop)
	{
	stop = true;
	}
	//按下按键后会停留在当前帧，等待下一次按键
	if (c >= 0)
	{
	waitKey(0);
	}


	currentFrame = currentFrame + 1;


	}
	//关闭视频文件
	*/
	capture.release();

	//waitKey(0);
	system("pause");
	return 0;
}


