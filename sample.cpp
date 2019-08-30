#include <vector>       // ヘッダファイルインクルード
#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>    
#include <iterator>
#include <random>
#include <cmath>

#include "machine_learning.h"

void first();
void second();
void third();

int main()
{
    //first();
    //second();//主にpythonの学習結果を乗せるのにはこれを使う。(多分これしか使わないと思う)
    //third();すいませんなぜか動きません:解決したら報告します。
}

//最初の乱数生成のsample
void first()
{
    rione_learn rl;//クラスを宣言
    std::vector< int > a;
    std::vector< std::string > b;

    //int型のvectorを用意して入力層->中間層->出力層の順に各層の大きさを入れる
    a.push_back( 3 );
    a.push_back( 3 );
    a.push_back( 3 );

    //string型のvectorを用意して各層で使う活性化関数を指定する
    //設計のミスで最初は入力層なので活性化関数は必要ないけど何か文字列を入れてください。バグります
    //指定できる活性化関数はdocument.txtを参照
    b.push_back( "soid" );
    b.push_back( "sigmoid" );
    b.push_back( "softsign" );

    //乱数を用いて初期の重みとバイアスを生成する( 第三引数に重みとバイアスを書き込むファイル名を指定)
    rl.random_first_weight( a, b, "point_learn.txt" );
}

//もう学習が済んでいて重みやバイアスがテキストファイルで持っている時
void second()
{
    rione_learn rl;

    //ファイル名を指定してファイルを読み込むファイルがなかったらfalseが返ってくる
    //これでrlに重みとバイアスがセットされる
    if( ! rl.get_first_weight( "point_learn.txt" ) )
    {
        return;
    }

    //入力層に一次元のvector<long double>を宣言(入力層と同じ大きさでないといけない)
    std::vector< long double > test( 3, 2 );

    //第一引数に入力層の値 第二引数に予測結果を表示するかどうか
    std::vector< long double > result = rl.predict( test, true );
}


void third()
{
    first();//重みとバイアスの初期設定

    rione_learn rl;

    if( ! rl.get_first_weight( "point_learn.txt" ) )
    {
        return;
    }
    std::vector<std::vector< long double> > a;
    std::vector<std::vector< long double> > b;
    a = std::vector<std::vector< long double> >(3, std::vector< long double >(3, 0));
    b = std::vector<std::vector< long double> >(3, std::vector< long double >(3, 0));
    std::vector<std::vector< long double> > c;
    std::vector<std::vector< long double> > d;
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
    
    rl.predict(a[0], true);
    rl.predict(a[1], true);
    rl.predict(a[2], true);
    
    rl.machine_learn(d, c, 10, 3, 0.1, "entropy");
    //rl.machine_learn(d, c, 25, 10, 0.2, "mean_square");
    rl.predict(a[0], true);
    rl.predict(a[1], true);
    rl.predict(a[2], true);
    
}
