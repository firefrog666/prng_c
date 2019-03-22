HOW TO COMPILE:

	Open src/prng_c.sln with visual studio 2017. Build the project.

HOW TO RUN:

	In comand line, use
	cd src/x64/Release
	./prng_c.exe intput_in_bits.txt
	to generate rng numbers from a text file which contains 272 bits of 0/1.
	use
	./prng_c.exe intput_ints.txt
	to generate rng numbers from a text file which contains 17 integers.

SOME NOTES:

	The mode matrices are stored in src/x64/Release/model_matrix.txt. 
	The functions for load and caculate the model are in the src/run_model.h and src/run_model.cpp files.