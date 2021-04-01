#include "rapidcsv.h"
#include "csv.h"
#include <iostream>
#include <vector>
#include <mlpack/core.hpp>
#include "armadillo"

//! Creates a csv file of given dimensions with random values
void create_csv(size_t n_rows, size_t n_cols)
{
  arma::mat matrix(n_rows, n_cols);
  matrix.randu();
  matrix.save("test.csv", arma::file_type::csv_ascii);
}

int main()
{
  int START_ROW = 1, STOP_ROW = 1000000, STEP_ROW = 5000;

  arma::vec row_sizes = arma::regspace(START_ROW, STEP_ROW, STOP_ROW);
  arma::vec col_sizes = {5, 15, 25};

  arma::mat log;
  log.set_size(row_sizes.n_elem * col_sizes.n_elem, 5);
  arma::mat test;

  std::cout << "Total combinations: " << log.n_rows << '\n';

  int counter = 0;
  for (int i = 0; i < row_sizes.n_elem; i++)
  {
    for (int j = 0; j < col_sizes.n_elem; j++)
    {
      std::cout << "Count: " << counter << '\n';

      log(counter, 0) = row_sizes(i);
      log(counter, 1) = col_sizes(j);

      create_csv(row_sizes(i), col_sizes(j));

      // Load using default csv parser
      auto start = std::chrono::high_resolution_clock::now();
      test.load("test.csv", arma::file_type::csv_ascii);
      auto stop = std::chrono::high_resolution_clock::now();
      auto time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
      log(counter, 2) = time_taken.count();

      // Load using mlpack's custom csv parser
      mlpack::data::DatasetInfo info;
      start = std::chrono::high_resolution_clock::now();
      mlpack::data::Load("test.csv", test, info, false, true);
      stop = std::chrono::high_resolution_clock::now();
      time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
      log(counter, 3) = time_taken.count();

      // Load using rapidcsv
      start = std::chrono::high_resolution_clock::now();
      rapidcsv::Document doc("test.csv", rapidcsv::LabelParams(-1, -1));
      arma::mat data(doc.GetRowCount(), doc.GetColumnCount(), arma::fill::ones);
      std::vector<float> col;
      for (int i = 0; i < doc.GetRowCount(); i++)
      {
        col = doc.GetRow<float>(i);
        for (int j = 0; j < doc.GetColumnCount(); j++)
        {
          data(i, j) = col[j];
        }
      }
      stop = std::chrono::high_resolution_clock::now();
      time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
      log(counter, 4) = time_taken.count();

      counter++;
    }
  }
  log.save("log.csv", arma::file_type::csv_ascii);
}
