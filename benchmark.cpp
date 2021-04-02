// use the above line everytime
// export LD_LIBRARY_PATH="/home/heisenbug/Dev/mlpack/mlpack-lib/build/lib/:$LD_LIBRARY_PATH"

#include <rapidcsv.h>
#include <csv.h>
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
  int START_ROW = 1, STOP_ROW = 1000000, STEP_ROW = 5000;

  arma::vec row_sizes = arma::regspace(START_ROW, STEP_ROW, STOP_ROW);
  arma::vec col_sizes = {5, 15, 25};

  arma::mat log;
  log.set_size(row_sizes.n_elem * col_sizes.n_elem, 6);
  arma::mat test;

  std::cout << "Total combinations: " << test.n_rows << '\n';

  int counter = 0;
  for (int i = 0; i < row_sizes.n_elem; i++)
  {
    for (int j = 0; j < col_sizes.n_elem; j++)
    {
      std::cout << "Count: " << counter << '\n';

      // log row size and column size
      log(counter, 0) = row_sizes(i);
      log(counter, 1) = col_sizes(j);

      // creating test.csv
      create_csv(row_sizes(i), col_sizes(j));

      // Load using default csv parser
      auto start = std::chrono::high_resolution_clock::now();
      test.load("data/test.csv", arma::file_type::csv_ascii);
      auto stop = std::chrono::high_resolution_clock::now();
      auto time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
      log(counter, 2) = time_taken.count();

      // Load using mlpack's custom csv parser
      mlpack::data::DatasetInfo info;
      start = std::chrono::high_resolution_clock::now();
      mlpack::data::Load("data/test.csv", test, info, false, true);
      stop = std::chrono::high_resolution_clock::now();
      time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
      log(counter, 3) = time_taken.count();

      // Load using rapidcsv
      start = std::chrono::high_resolution_clock::now();
      // Intialize without header
      rapidcsv::Document doc_rapid_csv("data/test.csv", rapidcsv::LabelParams(-1, -1));
      // is it safe or recommonded to fill mat at the time of initialization?
      arma::fmat mat_rcsv(doc_rapid_csv.GetRowCount(), doc_rapid_csv.GetColumnCount());
      std::vector<float> column;

      for(int i = 0; i < doc_rapid_csv.GetColumnCount(); i++)
      {
        column = doc_rapid_csv.GetColumn<float>(i);
        arma::fvec column_vector(column);
        mat_rcsv.col(i) = column_vector;
      }

      stop = std::chrono::high_resolution_clock::now();
      time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
      log(counter, 4) = time_taken.count();

      // Load using fastcsv
      start = std::chrono::high_resolution_clock::now();
      io::CSVReader<50> doc_fast_csv("data/test.csv");
      std::vector<float> row_elements;
      std::stringstream row_ss;
      arma::fmat mat_fcsv(row_sizes(i), col_sizes(j));
      int i = 0;
      while(char*line = doc_fast_csv.next_line())
      {
        std::stringstream row_ss(line);
        
        while (row_ss.good()) {
          std::string substr;
          getline(row_ss, substr, ',');
          row_elements.push_back(std::stod(substr));
        }
        arma::frowvec row_vector(row_elements);
        row_elements.clear();
        mat_fcsv.row(i) = row_vector;
        i++;
      }
      stop = std::chrono::high_resolution_clock::now();
      time_taken = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
      log(counter, 5) = time_taken.count();

      counter++;
    }
  }
  log.save("logs/log.csv", arma::file_type::csv_ascii);
}
