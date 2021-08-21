// export LD_LIBRARY_PATH=/home/heisenbug/mlpack/mlpack/mlpack-master/build/lib/
// export LD_LIBRARY_PATH=/home/heisenbug/mlpack/mlpack/mlpack-local/build/lib/
#include <iostream>
#include <vector>
#include <mlpack/core.hpp>
#include "armadillo"

//! Creates a csv file of given dimensions with random values
void create_csv(size_t n_rows, size_t n_cols)
{
  arma::mat matrix(n_rows, n_cols);
  matrix.randu();
  matrix.save("data/test.csv", arma::file_type::csv_ascii);
}

int main()
{
  std::string path = "data/test.csv";
  int START_ROW = 1, STOP_ROW = 1000000, STEP_ROW = 5000;

  arma::vec row_sizes = arma::regspace(START_ROW, STEP_ROW, STOP_ROW);
  arma::vec col_sizes = {5, 15, 25};

  arma::mat log;
  log.set_size(row_sizes.n_elem * col_sizes.n_elem, 5);
  arma::fmat test;

  std::cout << "Total combinations: " << log.n_rows << '\n';

  int counter = 0;
  for (int i = 1; i < row_sizes.n_elem; i++)
  {
    for (int j = 1; j < col_sizes.n_elem; j++)
    {
      if(counter % 1000 == 0)
        std::cout << "Count: " << counter << '\n';

      // log row size and column size
      log(counter, 0) = row_sizes(i);
      log(counter, 1) = col_sizes(j);

      // creating test.csv
      create_csv(row_sizes(i), col_sizes(j));

      // Load using armadillo's csv parser
      auto start = std::chrono::high_resolution_clock::now();
      test.load(path, arma::file_type::csv_ascii);
      auto stop = std::chrono::high_resolution_clock::now();
      auto time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
      log(counter, 2) = time_taken.count();
      
      // Load using mlpack's new csv parser
      start = std::chrono::high_resolution_clock::now();
      mlpack::data::Load(path, test);
      stop = std::chrono::high_resolution_clock::now();
      time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
      log(counter, 3) = time_taken.count();

      // Load using mlpack's new custom csv parser without boost
      mlpack::data::DatasetInfo info;
      start = std::chrono::high_resolution_clock::now();
      mlpack::data::Load(path, test, info);
      stop = std::chrono::high_resolution_clock::now();
      time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
      log(counter, 4) = time_taken.count();
      counter++;
    }
  }
  log.save("logs/old-parser.csv", arma::file_type::csv_ascii);
}
