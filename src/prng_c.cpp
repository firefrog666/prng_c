#include <iostream>
#include <fstream>
#include <string>
#include <math.h> 
#include <armadillo>
#include "run_model.h"

#define MODEL_NAME "model_matrix.txt"
#define INTBITWIDTH 16
#define RNG_INPUT_INT_NUM 17


int
main(int argc, char** argv)
  {
	arma::Mat<float> input(1, RNG_INPUT_INT_NUM);
	if (argc == 1) {		
		cout << "No args. Initiate inputs to zerossss" << endl;
		input.fill(0);
		
	}
	else {
		cout << "loading input from " << argv[1] << endl;
		input.load(argv[1]);
	}
	input.print("input: ");
	vector<int> rng = run_model_matrix(input, MODEL_NAME, 13);

	vector<int> bits = intsToBits(rng, INTBITWIDTH);
	cout << "rng generated: "<<endl;
	for (int i = 0; i < bits.size(); i++) {
		cout << bits[i];
	}
	cout << endl;

	return 0;	
  }

