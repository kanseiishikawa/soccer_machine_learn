#include <vector>
#include <iostream>
#include <string>
#include <fstream>

#include "rione_cnn.h"

bool
rione_cnn::cnn_get_weight(std::string file_name)
{
    //重みの書かれているファイルを開く
    std::ifstream file(file_name);

    if(!file)
    {
        std::cout<< "file_open_error" <<"\n";
        return false;
    }
    
    int layer_num, conv_num, conv4, conv3, conv2, conv1;
    file >> layer_num >> conv_num;
    
    for(int i = 0; i < conv_num; i++)
    {
        //畳み込み層の重みを保存する構造体の定義
        filter fil;
	
        if(i != layer_num - 2)
        {
            fil.last = false;
        }
        else
        {
            fil.last = true;
        }

        //4次元配列の各次元を取り出す
        file >> conv4 >> conv3 >> conv2 >> conv1;
        //フィルターの重みを格納する４次元配列を定義する
        fil.weight = std::vector<std::vector<std::vector<std::vector<double> > > >(conv4, std::vector<std::vector<std::vector<double> > >(conv3, std::vector<std::vector<double> >(conv2, std::vector<double>(conv1, 0))));

        //std::cout<< conv4 << " " << conv3 << " " << conv2 << " " << conv1 <<"\n";
        for(int c4 = 0; c4 < conv4; c4++)
        {
            for(int c3 = 0; c3 < conv3; c3++)
            {
                for(int c2 = 0; c2 < conv2; c2++)
                {
                    for(int c1 = 0; c1 < conv1; c1++)
                    {
                        file >> fil.weight[c4][c3][c2][c1];
                        //std::cout<< fil.weight[c4][c3][c2][c1] <<"\n";
                    }
                }
            }
        }
        conv2d_w.push_back(fil);
    }

    for(int i = 0; i < layer_num - conv_num; i++)
    {
        file >> conv2 >> conv1;
        fully fil;
        fil.weight = std::vector<std::vector<double> >(conv2, std::vector<double>(conv1, 0));
	
        for(int c2 = 0; c2 < conv2; c2++)
        {
            for(int c1 = 0; c1 < conv1; c1++)
            {
                file >> fil.weight[c2][c1];
            }
        }
        fully_w.push_back(fil);
    }

    for(int i = 0; i < fully_w.size(); i++)
    {
        file >> conv1;
        fully_w[i].bias = std::vector<double>(conv1, 0);
        for(int r = 0; r < conv1; r++)
        {
            file >> fully_w[i].bias[r];
        }
    }
    return false;
}

std::vector<double>
rione_cnn::cnn_propagation(std::vector<std::vector<std::vector<double> > > &input_data)
{
    std::vector<std::vector<std::vector<double> > > next_data = input_data;
    for(int i = 0; i < conv2d_w.size(); i++)
    {
        //次の層の大きさを確認して定義する
        std::vector<std::vector<std::vector<double> > > process_data = std::vector<std::vector<std::vector<double> > >(conv2d_w[i].weight.size(), std::vector<std::vector<double> >(next_data[0].size(), std::vector<double>(next_data[0][0].size(), 0)));
	
        for(int r = 0; r < conv2d_w[i].weight.size(); r++)
        {  
            for(int s = 0; s < conv2d_w[i].weight[r].size(); s++)
            {
                //フィルターをかける
                std::vector<std::vector<double> > instance = filter_prg(next_data[s], conv2d_w[i].weight[r][s]);
                //結果を代入する
                for(int c1 = 0; c1 < instance.size(); c1++)
                {
                    for(int c2 = 0; c2 < instance[c1].size(); c2++)
                    {
                        process_data[r][c1][c2] += instance[c1][c2];
                        //if(i==0 && r == 0 && s == 0 && c1 == 0)
                        //std::cout<< instance[c1][c2] <<"\n";
                    }
                }
            }
        }

        for(int a = 0; a < process_data.size(); a++)
        {
            for(int b = 0; b < process_data[a].size(); b++)
            {
                for(int c = 0; c < process_data[a][b].size(); c++)
                {
                    //各値にrelu関数を通す
                    process_data[a][b][c] = relu(process_data[a][b][c]);
                }
            }
        }
	
        next_data = process_data;
        for(int math = 0; math < process_data.size(); math++)
        {
            //サイズ２でmaxpoolingを行う
            next_data[math] = max_pooling(process_data[math], 2);
        }
    }

    std::vector<double> fully_connect;
    for(int i = 0; i < next_data.size(); i++)
    {
        for(int r = 0; r < next_data[i].size(); r++)
        {
            for(int t = 0; t < next_data[i][r].size(); t++)
            {
                //全結合層に代入するため1次元に戻す
                fully_connect.push_back(next_data[i][r][t]);
            }
        }
    }

    //全結合層から出力層までの計算を行う
    std::vector<double> result = fully_connect;
    for(int i = 0; i < fully_w.size(); i++)
    {
        std::vector<double> next_fully = std::vector<double>(fully_w[i].weight.size(), 0);
        for(int r = 0; r < fully_w[i].weight.size(); r++)
        {
            for(int t = 0; t < fully_w[i].weight[r].size(); t++)
            {
                next_fully[r] += fully_w[i].weight[r][t] * result[t];
            }
        }
	
        for(int b = 0; b < fully_w[i].bias.size(); b++)
        {
            next_fully[b] += fully_w[i].bias[b];
        }

        result = next_fully;
    }

    return result;
}

std::vector<std::vector<double> > 
rione_cnn::filter_prg(std::vector<std::vector<double> > &data, std::vector<std::vector<double> > &filter)
{
    //フィルタに入れた後の結果を保存するvectorを定義する
    std::vector<std::vector<double> > next_data = std::vector<std::vector<double> >(data.size(), std::vector<double>(data[0].size(), 0));

    //入力データの各値にフィルターをかけていく
    for(int i = 0; i < data.size(); i++)
    {
        for(int r = 0; r < data[i].size(); r++)
        {
            int size1 = filter.size();
            for(int ic = -1; ic < size1 - 1; ic++)
            {
                int size2 = filter[ic + 1].size(); 
                for(int rc = -1; rc < size2 - 1; rc++)
                {
                    int ip = i + ic;
                    int rp = r + rc;
                    bool zero_puropaiting = false;//配列外になるときはゼロプロパイティングを採用する

                    //ゼロプロパイティングが適用されるかどうかを調べる
                    if(ip < 0)
                    {
                        //ip += 1;
                        zero_puropaiting = true;
                    }
                    else if(ip >= data.size())
                    {
                        //ip -= 1;
                        zero_puropaiting = true;
                    }
                    else if(rp < 0)
                    {
                        //rp += 1;
                        zero_puropaiting = true;
                    }
                    else if (rp >= data[i].size())
                    {
                        //rp -= 1;
                        zero_puropaiting = true;
                    }

                    //zero_puropaiting = true;
                    if(zero_puropaiting == false)
                    {
                        next_data[i][r] += data[ip][rp] * filter[ic + 1][rc + 1];//フィルターをかけて足していく
                    }
                }
            }
        }
    }
    return next_data;
}

std::vector<std::vector<double> >
rione_cnn::max_pooling(std::vector<std::vector<double> > &data, int pooling_size)
{
    int high, side, high_r, side_r;

    //poolingをした後の大きさを調べる
    if(data.size() % pooling_size  !=  0)
    {
        high = data.size() / pooling_size + 1;
    }
    else
    {
        high = data.size() / pooling_size;
    }

    if(data[0].size() % pooling_size != 0)
    {
        side = data[0].size() / pooling_size + 1;
    }
    else
    {
        side = data[0].size() / pooling_size;
    }

    //poolingをした後の結果を保存するvectorを定義する
    std::vector<std::vector<double> > result = std::vector<std::vector<double> >(high, std::vector<double>(side, 0));

    //各値にpoolingをかけていく
    for(int i = 0; i < result.size(); i++)
    {
        for(int r = 0; r < result[i].size(); r++)
        {
            //対象範囲でもっとも大きい数字を取り出す
            double max = -100000;
            for(int s = 0; s < pooling_size; s++)
            {
                for(int e = 0; e < pooling_size; e++)
                {
                    int check_y = i*pooling_size + s;
                    int check_x = r*pooling_size + e;
		    
                    if(check_y < data.size() && check_x < data[i].size())
                    {
                        if(data[check_y][check_y] > max)
                        {
                            max = data[check_y][check_x];
                        }
                    }
                }
            }
            //対象範囲でもっとも大きい数字を代入する
            result[i][r] = max;
        }
    }
    
    return result;
}

//relu関数の処理を行う
double
rione_cnn::relu(double x)
{
    if(x < 0)
    {
        x = 0;
    }

    return x;
}

void
rione_cnn::debug()
{
    std::vector<std::vector<std::vector<double> > > x = std::vector<std::vector<std::vector<double> > >(4, std::vector<std::vector<double> >(34, std::vector<double>(53, 1)));

    int count = 0;
    for(int i = 0; i < x.size(); i++)
    {
        for(int r = 0; r < x[i].size(); r++)
        {
            for(int t = 0; t < x[i][r].size(); t++)
            {
                x[i][r][t] = count;
                count += 1;
            }
        }
    }
    
    cnn_get_weight("data.txt");
    clock_t start = clock();
    std::vector<double> ac = cnn_propagation(x);
    clock_t end = clock();
    double a = (double)(end - start) / CLOCKS_PER_SEC;
    std::cout<< a <<"\n";
    for(int i = 0;i < ac.size(); i++)
    {
        std::cout<< ac[i] <<"\n";
    }
}
