#ifndef RUN_MODE_H
#define RUN_MODE_H


#include <fstream>
#include <string>
#include <armadillo>

using namespace std;
using namespace arma;


template < typename T>
arma::Mat<T> load_mat(std::ifstream &file, const std::string &keyword);

float usr_leaky_relu(float x);

inline int positive_modulo(int i, int n);

vector<int> decToBinary(int n, int bitWidth);

vector<int> intsToBits(vector<int> ints, int bitWidth);

//load a model from file into a vector of matrix
vector<arma::Mat<float>> load_model_matrix(const char* filename, int modelSize);

//run a model stored in a file with the input stored in a matrix.
vector<int> run_model_matrix(arma::Mat<float> input, const char* filename, int modelSize);

void matrix_ops_examples();
#endif 