#include <vector>       // ヘッダファイルインクルード
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>    
#include <iterator>
#include <random>
#include <cmath>

#include "machine_learning.h"

#define RAND_FLOAT(LO, HI) LO + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(HI - LO)))

static bool file_open = true;
using namespace std;

//機械学習のメイン関数
//第一引数に出力層の答え
//第二引数に入力層の値
//第三引数に学習回数
//第四引数にバッチサイズ
//第五引数に学習率
//どちらも二次元配列で行う
void
rione_learn::machine_learn(vector<vector<long double> > answer, vector<vector<long double> > data, int epoch, int batch_size, long double learning_rate , string loss)
{
    loss_func = loss;
    int batch =  answer.size()/batch_size;
    bool shhufle_need = true;
    
    for(int i = 0; i < epoch; i++)//学習回数分だけループを回す
    {
	long double loss = 0;
	shhufle(data, answer);//バッチサイズに備えてデータをシャッフルする
	for(int s = 0; s < batch_size; s++)
	{   
	    //バッチサイズの数だけ微分した値を保存する(初期化)
	    vector<vector<vector<long double> > > d_w = weights;
	    vector<vector<long double> > d_b = bias;
	    d_b = initialization_2(d_b);
	    d_w = initialization_3(d_w);

	    for(int r = 0; r < batch; r++)//バッチサイズの数だけループを回す
	    {
		//  int select = randsize(mt);//どのデータで計算するか決める
	 	propagation(all_data[0]);//指定データの計算を行う
		loss += square_sum_error(result,all_answer[0]);
		if(s==0)
		{
		    //cout<< result[0] <<"\n";
		}
		counterpropagation(all_answer[0]);//結果にしたがって微分を行う
	
		//使ったデータを削除する
		all_data.erase(all_data.begin());
		all_answer.erase(all_answer.begin());
		
		//微分して得られた重みの値を加算する
		for(int t = 0; t < d_w.size(); t++)
		{
		    for(int e = 0; e < d_w[t].size(); e++)
		    {
			for(int u = 0; u < d_w[t][e].size(); u++)
			{
			    d_w[t][e][u] += d_weights[t][e][u];
			}
		    }
		}

		//微分して得られたバイアスの値を加算する
		for(int t = 0; t < d_b.size(); t++)
		{
		    for(int e = 0; e < d_b[t].size(); e++)
		    { 
			d_b[t][e] += d_bias[t][e];
		    }
		}
	    }
	    //cout<< loss <<"\n";
	    //重みとバイアスの更新
	    update(d_b, d_w, batch_size ,learning_rate);
	}
	
	cout<< loss / (batch_size * batch) <<"\n";
    }
    
}

//重みとバイアスをランダムで設定する
bool
rione_learn::random_first_weight(vector<int> layer, vector<string> act, string file_name)
{
    if(layer.size() != act.size())
    {
	return false;
    }

    ofstream file(file_name);

    file<< layer.size() - 1 << "\n";
    
    for(int i = 0; i < layer.size() -  1; i++)
    {
	file<< layer[i] << " " << layer[i+1] <<"\n";
	for(int r = 0; r < layer[i]; r++)
	{
	    for(int t = 0; t < layer[i + 1]; t++)
	    {
		double select = RAND_FLOAT(-2.0, 2.0);
		file<< select <<" ";
	    }
	    file<< "\n";
	}
    }


    for(int i = 1; i < layer.size(); i++)
    {
	file<< layer[i] << "\n";
	for(int r = 0; r < layer[i]; r++)
	{
	    double select = RAND_FLOAT(-2.0, 2.0);
	    file<< select <<" ";
	}
	file << "\n";
    }

    for(int i = 1; i < act.size(); i++)
    {
	file<< act[i] <<"\n";
    }

    file.close();
    
    return true;
}

//テキストファイルから重みとバイアスをセットする
bool
rione_learn::get_first_weight(string file_name)
{
    ifstream ifs(file_name,ios::in);

    if(file_open == false)
    {
	//cout<< "not_open" <<"\n";
	return false;
    }
  
    if (ifs.fail())
    {
	std::cerr << "失敗" << std::endl;
	return false;
    }

    //層の数を取得
    ifs >> layerNum;

    //重みの取得
    for(int i = 0;i < layerNum; i++)
    {
	int high, width;
	ifs >> high >> width;
	vector<vector<vector<long double> > > w(1, vector<vector<long double> >(high, vector<long double>(width)));
    
	for(int r = 0; r < high; r++)
	{
	    for(int t = 0; t < width; t++)
	    {
		ifs >> w[0][r][t];
	    }
	}
    
	if(i == 0)
	{
	    weights = w;
	}
	else
	{
	    weights.insert(weights.end(), w.begin() ,w.end());
	}
    }

    //バイアスを取得
    for(int i = 0; i < layerNum; i++)
    {
	int length;
	ifs >> length;
	vector<vector<long double> > b;

	b = vector<vector<long double> >(1, vector<long double>(length, 0));
	
	for(int r = 0; r < length; r++)
	{
	    ifs >> b[0][r];
	}

	if(i == 0)
	{
	    bias = b;
	}
	else
	{
	    bias.insert(bias.end(), b.begin(), b.end());
	}
    }

    //活性化関数の情報を取得
    for(int i = 0; i < layerNum; i++)
    {
	string name;
	ifs >> name;
	func_name.push_back(name);
    }

    d_weights = weights;
    d_bias = bias;
    
    ifs.close();
    file_open = false;
    return true;
}

bool
rione_learn::write_data(string file_name)
{
    ofstream file(file_name);
    
    if(!file)
    {
	return false;
    }

    file<< layerNum <<"\n";
    
    for(int i = 0; i < weights.size(); i++)
    {
	file<< weights[i].size() << " " << weights[i][0].size() <<"\n";
	for(int r = 0; r < weights[i].size(); r++)
	{
	    for(int t = 0; t < weights[i][r].size(); t++)
	    {
		file<< weights[i][r][t] <<" ";
	    }
	    file<< "\n";
	}
    }

    for(int i = 0; i < bias.size(); i++)
    {
	file<< bias[i].size() <<"\n";
	for(int r = 0; r < bias[i].size(); r++)
	{
	    file<< bias[i][r] <<" ";
	}
	file<< "\n";
    }

    for(int i = 0; i < func_name.size(); i++)
    {
	file<< func_name[i] <<"\n";
    }

    file_open = true;
    return true;
  
}
//引数が入力層の値になって、順伝播の計算を行う
void
rione_learn::propagation(vector<long double> data)
{

    input_data = data;
    
    for(int i = 0; i < layerNum; i++)
    {
	data = propagation_matrix_calculation(data, weights[i]);//重さとの行列計算
	//cout<< data[0] <<"\n";
	//一時保存する各中間層の値
	vector<vector<long double> > hidden_layer_once;
	hidden_layer_once = vector<vector<long double> >(1, vector<long double>(data.size(), 0));

	for(int r = 0; r < data.size(); r++)
	{
	    data[r] += bias[i][r];//バイアスを加える
	}

	hidden_layer_once[0] = data;
	
	if(i == 0)
	{
	    hidden_layer = hidden_layer_once;
	}
	else
	{
	    hidden_layer.insert(hidden_layer.end(), hidden_layer_once.begin(), hidden_layer_once.end());
	}

	for(int r = 0; r < data.size(); r++)
	{
	    data[r] = activation(data[r], func_name[i], false ,i);//活性化関数に代入する
	}
    }

    result = data;//結果を保存する
    
    d_hidden_layer = hidden_layer;
}

//結果を使って損失関数を中心に重みとバイアスの微分を行う
void
rione_learn::counterpropagation(vector<long double> answer)
{
    for(int i = 0; i < d_hidden_layer[layerNum - 1].size(); i++)
    {
	d_hidden_layer[layerNum - 1][i] = d_loss(result[i], answer[i], loss_func);//出力層の値を損失関数で微分する
  
	d_bias[layerNum - 1][i] = d_hidden_layer[layerNum - 1][i] * activation(hidden_layer[layerNum - 1][i] , func_name[layerNum - 1] , true , layerNum - 1);//出力層のバイアスを微分する
    }
    
    //出力層の重みを微分する
    for(int i = 0; i < d_weights[layerNum - 1].size(); i++)
    {
	for(int r = 0; r < d_weights[layerNum - 1][i].size() ; r++)
	{
	    d_weights[layerNum - 1][i][r] = d_bias[layerNum - 1][r] * activation(hidden_layer[layerNum - 2][i], func_name[layerNum - 1] , false ,layerNum - 1);
	}
    }
    
    //隠れ層の微分を行う
    for(int i = layerNum - 2; i > -1; i--)
    {
	for(int r = 0; r < d_hidden_layer[i].size(); r++)
	{
	    d_hidden_layer[i][r] = 0;
	    for(int t = 0; t < d_bias[i + 1].size(); t++)
	    {
		//隠れ層の出力部分の微分を行う
		d_hidden_layer[i][r] += d_bias[i + 1][t] * weights[i + 1][r][t];
	    }
	}
	
	for(int r = 0; r < d_bias[i].size(); r++)
	{
	    //隠れ層のバイアスの微分を行う
	    d_bias[i][r] = d_hidden_layer[i][r] * activation(hidden_layer[i][r] , func_name[i] ,true, i);
	}
	for(int r = 0; r < d_weights[i].size(); r++)
	{
	    for(int t = 0; t < d_weights[i][r].size(); t++)
	    {
		//隠れ層の重みの微分を行う
		if(i != 0)
		{
		    d_weights[i][r][t] = d_bias[i][r] * activation(hidden_layer[i - 1][t] , func_name[i] ,false ,i);
		}
		else
		{
		    d_weights[i][r][t] =  d_bias[i][r] * input_data[t];
		}
	    }
	}
    }
}

vector<vector<vector<long double> > >
rione_learn::initialization_3(vector<vector<vector<long double> > > a)
{
    for(int i = 0; i < a.size(); i++)
    {
	for(int r = 0; r < a[i].size(); r++)
	{
	    for(int t = 0; t < a[i][r].size(); t++)
	    {
		a[i][r][t] = 0;
	    }
	}
    }
    return a;
}

vector<vector<long double> >
rione_learn::initialization_2(vector<vector<long double> > a)
{
    for(int i = 0; i < a.size(); i++)
    {
	for(int r = 0; r < a[i].size(); r++)
	{
	    a[i][r] = 0;
	}
    }
  
    return a;
}

//順伝播を計算するときに使う行列計算の関数
vector<long double>
rione_learn::propagation_matrix_calculation(vector<long double> front, vector<vector<long double> > after)
{
    vector<long double> last(after[0].size(),0);
    
    for(int i = 0; i < after[0].size(); i++)
    {
	for(int r = 0; r < after.size(); r++)
	{
	    last[i] += front[r] * after[r][i];
	    //cout<< last[i] <<"\n";
	}
    }
  
    return last;
}

void
rione_learn::update(vector<vector<long double> > differential_bias, vector<vector<vector<long double> > > differential_weights, int batch_size, long double learnig_rate)
{
    for(int i = 0; i < differential_bias.size(); i++)
    {
	for(int r = 0; r < differential_bias[i].size(); r++)
	{
	    bias[i][r] = bias[i][r] - differential_bias[i][r] / batch_size * learnig_rate;
	}
    }

    for(int i = 0; i < differential_weights.size(); i++)
    {
	for(int r = 0; r < differential_weights[i].size(); r++)
	{
	    for(int t = 0; t < differential_weights[i][r].size(); t++)
	    {
		weights[i][r][t] = weights[i][r][t] - differential_weights[i][r][t] / batch_size * learnig_rate;
	    }
	}
    }
}

vector<long double>
rione_learn::predict(vector<long double> data , bool show)
{
    propagation(data);
  
    if(show == false)
    {
	return result;
    }
    else
    {
	cout<< "------" <<"\n";
	for(int i = 0; i < result.size(); i++)
	{
	    cout<< result[i] <<"\n";
	}
	cout<< "------" <<"\n";
    }
  
    return result;
}

void
rione_learn::shhufle(vector<vector<long double> > data, vector<vector<long double> > answer)
{
    vector<vector<long double> > shhufle_data;
    vector<vector<long double> > shhufle_answer;
    shhufle_data =  vector<vector<long double> >(1, vector<long double>(data[0].size(), 0));
    shhufle_answer =  vector<vector<long double> >(1, vector<long double>(answer[0].size(), 0));
  
    vector<vector<long double> > once_data;
    once_data =  vector<vector<long double> >(1, vector<long double>(data[0].size(), 0));

    vector<vector<long double> > once_answer;
    once_answer =  vector<vector<long double> >(1, vector<long double>(answer[0].size(), 0));
    int size = data.size();
  
    while(1)
    {
	random_device rnd;     // 非決定的な乱数生成器を生成
	mt19937 mt(rnd());     //  メルセンヌ・ツイスタの32ビット版、引数は初期シード値
	uniform_int_distribution<int> randsize(0, data.size()-1);//指定範囲の一様乱数
	int select = randsize(mt);

	if(size == data.size())
	{
	    shhufle_data[0] = data[select];
	    shhufle_answer[0] = answer[select];
	}
	else
	{
	    once_data[0] = data[select];
	    once_answer[0] = answer[select];
	    shhufle_data.insert(shhufle_data.end(), once_data.begin() , once_data.end());
	    shhufle_answer.insert(shhufle_answer.end(), once_answer.begin() , once_answer.end());
	}

	data.erase(data.begin() + select);
	answer.erase(answer.begin() + select);

	if(data.size() == 1)
	{
	    once_data[0] = data[0];
	    once_answer[0] = answer[0];
	    shhufle_data.insert(shhufle_data.end(), once_data.begin() , once_data.end());
	    shhufle_answer.insert(shhufle_answer.end(), once_answer.begin(), once_answer.end());
	    break;
	}
    }

    all_data = shhufle_data;
    all_answer = shhufle_answer;
}

long double
rione_learn::activation(long double x, string function_name, bool differential, int number)
{
    if(function_name == "sigmoid" && differential == true)
    {
	return d_sygmoid(x);
    }
    else if(function_name == "sigmoid" && differential == false)
    {   
	return  sygmoid(x);
    }
    else if(function_name == "softmax" && differential == true)
    {
	return d_softmax(x ,number);
    }
    else if(function_name == "softmax" && differential == false)
    {
	return softmax(x, number);
    }
    else if(function_name == "softsign" && differential == true)
    {
	return soft_sign(x);
    }
    else if(function_name == "softsign" && differential == false)
    {
	return d_soft_sign(x);
    }
  
    return 0;
}

long double
rione_learn::d_loss(long double x, long double y, string loss_name)
{
    if(loss_name == "mean_square")
    {
	return d_square_sum_error(x, y);
    }
    if(loss_name == "entropy")
    {
	return d_entropy(x, y);
    }
  
  return 0;
}

long double
rione_learn::sygmoid(long double x)
{
    return 1/(1+exp(-x));
}

long double
rione_learn::d_sygmoid(long double x)
{
    return (1 - sygmoid(x)) * sygmoid(x);
}

long double
rione_learn::softmax(long double x, int number)
{ 
    long double total = 0;

    for(int i = 0; i < hidden_layer[number].size(); i++)
    {	
	total += exp(hidden_layer[number][i]);
    }

    return exp(x) / total;
}

long double
rione_learn::d_softmax(long double x, int number)
{
    return softmax(x, number) * (1 - softmax(x, number));
}

long double
rione_learn::soft_sign(long double x)
{
    return x / (1 + abs(x));
}

long double
rione_learn::d_soft_sign(long double x)
{
    return 1 / pow(1 + abs(x), 2);
}

long double
rione_learn::square_sum_error(vector<long double> x, vector<long double> y)
{
    long double a = 0;
  
    for(int i = 0; i < x.size(); i++)
    {
	a += pow(x[i] - y[i], 2);
    
    }
  
    return a/2;
}

long double
rione_learn::d_square_sum_error(long double x , long double y)
{
    return (x - y);
}

long double
rione_learn::d_entropy(long double x, long double y)
{
    return -(y/x) + (1-y)/(1-x);
}

void
rione_learn::debug()
{
    /*
  vector<vector<long double> > a;
  vector<vector<long double> > b;
  a = vector<vector<long double> >(3, vector<long double>(3, 0));
  b = vector<vector<long double> >(3, vector<long double>(3, 0));
  vector<vector<long double> > c;
  vector<vector<long double> > d;
  a[0][0] = 1;
  a[1][1] = 1;
  a[2][2] = 1;

  b[0][1] = 1;
  b[1][2] = 1;
  b[2][0] = 1;

  c = a;
  d = b;
  for(int i = 0; i < 1000; i++)
  {
    c.insert(c.end(), a.begin(), a.end());
    d.insert(d.end(), b.begin(), b.end());
  }

  predict(a[0], true);
  predict(a[1], true);
  predict(a[2], true);
  
  machine_learn(d, c, 25, 10, 0.2, "entropy");
  //machine_learn(d, c, 25, 10, 0.2, "mean_square");
  predict(a[0], true);
  predict(a[1], true);
  predict(a[2], true);
  //write_data("data.txt");
  */

    vector<int> a;
    vector<string> b;

    a.push_back(38);
    a.push_back(16);
    a.push_back(8);
    a.push_back(1);

    b.push_back("sigmoid");
    b.push_back("sigmoid");
    b.push_back("sigmoid");
    b.push_back("softsign");

    random_first_weight(a,b,"point_learn.txt");
  }

