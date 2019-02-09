#include <RcppEigen.h>
// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;
using namespace Eigen;

// [[Rcpp::export]]
VectorXd mv_normal(VectorXd mu, MatrixXd sigma){
  int r = mu.size();
  VectorXd standard_normal(r);
  VectorXd result(r);
  
  standard_normal = as<VectorXd>(rnorm(r, 0, 1));
  result = mu + sigma.llt().matrixL() * standard_normal;
  return result;
}

