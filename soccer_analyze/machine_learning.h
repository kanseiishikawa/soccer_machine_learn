#include <vector>       // ヘッダファイルインクルード
#include <iostream>
#include <string>

#ifndef MACHINE_LEARNING_H
#define MACHINE_LEARNING_H

//using namespace std;

class rione_learn
{
  private:
    std::vector<std::vector<std::vector<long double> > > weights;//重みを保存する
    std::vector<std::vector<long double> > bias;//バイアスを保存する
    std::vector<std::vector<std::vector<long double> > > d_weights;//各重みの微分後の値を保存する
    std::vector<std::vector<long double> > d_bias;//各バイアスの微分後の値を保存する

    std::vector<std::vector<long double> > hidden_layer;//各隠れ層の値を保存する
    std::vector<std::vector<long double> > d_hidden_layer;//各隠れ層の微分後の値を保存する

    std::vector<long double> result;//ニューラルネットワークの計算の結果を保存する

    std::vector<std::string> func_name;//活性化関数を保存する

    std::vector<long double> input_data;//入力層の値を保存する

    std::vector<std::vector<long double> > all_data;
    std::vector<std::vector<long double> > all_answer;

    int layerNum;//レイヤーの数を保存

    std::string loss_func;

    std::vector<long double> propagation_matrix_calculation(std::vector<long double> front, std::vector<std::vector<long double> > after);

    long double activation(long double x, std::string function_name, bool differential, int number);
    long double d_loss(long double x, long double y, std::string loss_name);
    long double soft_sign(long double x);
    long double d_soft_sign(long double x);
    long double sygmoid(long double x);
    long double d_sygmoid(long double x);
    long double softmax(long double x ,int number);
    long double d_softmax(long double x, int number);
    long double d_square_sum_error(long double x , long double y);
    long double square_sum_error(std::vector<long double> x, std::vector<long double> y);
    long double d_entropy(long double x, long double y);

    void counterpropagation(std::vector<long double> answer);
    
    std::vector<std::vector<std::vector<long double> > > initialization_3(std::vector<std::vector<std::vector<long double> > > a);
    std::vector<std::vector<long double> > initialization_2(std::vector<std::vector<long double> > a);
    void shhufle(std::vector<std::vector<long double> > data, std::vector<std::vector<long double> > answer);

    void update(std::vector<std::vector<long double> > differential_bias, std::vector<std::vector<std::vector<long double> > > differential_weights, int batch_size, long double learnig_rate);

    void propagation(std::vector<long double> data);//順伝播の計算を行う
  
public:
    bool get_first_weight(std::string file_name);

    bool random_first_weight(std::vector<int> layer, std::vector<std::string> act, std::string file_name);

    void machine_learn(std::vector<std::vector<long double> > answer, std::vector<std::vector<long double> > data, int epoch, int batch_size, long double learning_rate, std::string loss);

    std::vector<long double> predict(std::vector<long double> data, bool show);

    bool write_data(std::string file_name);
    void debug();
  
  
};
#endif
