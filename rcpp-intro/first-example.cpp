#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
NumericVector square_root(NumericVector x) {
  return sqrt(x);
}

