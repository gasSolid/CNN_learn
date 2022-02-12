#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <math.h>
using namespace std;
void random_init_arr(vector<float> &arr)
{
    int len = arr.size();
    for (int i = 0; i < len; i++)
    {
        arr[i] = 0.5 - random() / RAND_MAX;
    }
}

void preprocess_func(const vector<float> &layer1, vector<float> &layer2, const vector<float> &weight1_2)
{
    int cnt_layer1 = layer1.size();
    int cnt_layer2 = layer2.size();
    for (int i = 0; i < cnt_layer2; i++)
    {
        float tmp = 0;
        for (int j = 0; j < cnt_layer1; j++)
        {
            tmp += layer1[j] * weight1_2[i * cnt_layer1 + j];
        }
        layer2[i] = tmp;
    }
}

void sigma_func(vector<float> &arr)
{
    int len = arr.size();
    for (int i = 0; i < len; i++)
    {
        arr[i] = (1.0 / (1.0 + exp(-arr[i])));
    }
}

void backTracing_func(vector<float> &weight, const vector<float> &layer1, const vector<float> &layer2, vector<float> &d_weight1, const vector<float> &d_weight2)
{
    int len1 = layer1.size();
    int len2 = layer2.size();
    for (int i = 0; i < len2; i++)
    {
        for (int j = 0; j < len1; j++)
        {
            d_weight1[i * len2 + j] = d_weight2[i] * layer2[i] * (1.0 - layer2[i]) * layer1[j];
            //cout<<d_weight2[i] <<" "<< layer2[i] <<" "<< layer1[j]<<endl;
            weight[i * len2 + j] -= d_weight1[i * len2 + j];
        }
    }
}

void print_arr(const vector<float> &arr)
{
    copy(arr.begin(), arr.end(), ostream_iterator<float>(cout, " "));
    cout << endl;
}

int main()
{
    int first_layer_node = 2;
    int second_layer_node = 2;
    int out_layer_node = 1;
    vector<float> first_layer(first_layer_node);
    first_layer[0] = 0.35;
    first_layer[1] = 0.9;
    vector<float> second_layer(first_layer_node, 0);
    vector<float> out(out_layer_node);
    out[0] = 0.5;
    vector<float> out_layer(out_layer_node);
    vector<float> weight1_2(first_layer_node * second_layer_node);
    vector<float> d_weight1_2(first_layer_node * second_layer_node);
    // random_init_arr(weight1_2);
    weight1_2 = {0.1, 0.8, 0.4, 0.6};

    vector<float> weight2_3(second_layer_node * out_layer_node);
    vector<float> d_weight2_3(second_layer_node * out_layer_node);
    // random_init_arr(weight2_3);
    weight2_3 = {0.3, 0.9};
    vector<float> d_weight_out(out_layer_node);
    int idx = 0;
    while (true)
    {
        // pre process l1
        preprocess_func(first_layer, second_layer, weight1_2);
        //print_arr(second_layer);
        sigma_func(second_layer);
        // pre process l2
        preprocess_func(second_layer, out_layer, weight2_3);
        sigma_func(out_layer);
        // print_arr(out_layer);

        // bp l2
        d_weight_out[0] = out_layer[0] - out[0];
        if (d_weight_out[0] < 0.0000001){
            cout<<out_layer[0]<<endl;
            break;
        }
        backTracing_func(weight2_3, second_layer, out_layer, d_weight2_3, d_weight_out);
        backTracing_func(weight1_2, first_layer, second_layer, d_weight1_2, d_weight2_3);
        idx++;
    }
    cout<<idx<<endl;

    print_arr(weight1_2);
    print_arr(weight);
    return 0;
}
