#include <vector>
#include <iostream>
#include <string>

#ifndef RIONE_CNN_H
#define RIONE_CNN_H

struct filter
{
    std::vector<std::vector<std::vector<std::vector<double> > > > weight;
    bool last;
};

struct fully
{
    std::vector<std::vector<double> > weight;
    std::vector<double> bias;
};


class rione_cnn
{
private:
    double instance_math;
    std::vector<std::vector<double> > instance_2;
    std::vector<filter> conv2d_w;
    std::vector<fully> fully_w;
    std::vector<std::vector<double> > filter_prg(std::vector<std::vector<double> > &data, std::vector<std::vector<double> > &filter);
    void relu(double& x);

public:
    bool cnn_get_weight(std::string file_name);
    std::vector<double> cnn_propagation(std::vector<std::vector<std::vector<double> > > &input_data);
    std::vector<std::vector<double> > max_pooling(std::vector<std::vector<double> > &data, int pooling_size);
    void debug();
    //bool cnn_propagation();
};
#endif
