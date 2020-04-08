// Copyright 2018 The Simons Foundation, Inc. - All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NETKET_RBM_DENSITY_MATRIX_HPP
#define NETKET_RBM_DENSITY_MATRIX_HPP

#include <cmath>

#include "Machine/abstract_machine.hpp"

namespace netket {

/** Restricted Boltzmann machine class with spin 1/2 hidden units.
 *
 */
class RbmDensityMatrix : public AbstractMachine {
  // number of visible units
  int nv_;

  // number of hidden units
  int nh_;

  // number of auxilary units
  int na_;

  // number of parameters
  int npar_;

  // weights between visible and hidden units
  MatrixType W_;

  // weights between visible and auxilary units
  MatrixType U_;

  // visible units bias
  VectorType a_;

  // hidden units bias
  VectorType b_;

  // auxilary units bias
  VectorType c_;

  VectorType gammaplus_;
  VectorType gammaminus_;
  VectorType pimatrix_;
  VectorType newgammaplus_;
  VectorType newgammaminus_;
  VectorType newpimatrix_;

  bool usea_;
  bool useb_;
  bool usec_;

 public:
  RbmDensityMatrix(std::shared_ptr<const AbstractHilbert> hilbert, int nhidden = 0,
          int alpha = 0, int beta = 0, 
          bool usea = true, bool useb = true, bool usec = true);

  int Nvisible() const override;
  int Npar() const override;
  /*constexpr*/ int Nhidden() const noexcept { return nh_; }
  /*constexpr*/ int Nauxillary() const noexcept { return na_;}

  void InitRandomPars(int seed, double sigma) override;
  void InitLookup(VisibleConstType v, LookupType &lt) override;
  void UpdateLookup(VisibleConstType v, const std::vector<int> &tochange,
                    const std::vector<double> &newconf,
                    LookupType &lt) override;
  VectorType DerLog(VisibleConstType v) override;
  VectorType DerLog(VisibleConstType v, const LookupType &lt) override;
  Complex LogVal(VisibleConstType v) override;
  Complex LogVal(VisibleConstType v, const LookupType &lt) override;

  VectorType GetParameters() override;
  void SetParameters(VectorConstRefType pars) override;

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped
  VectorType LogValDiff(
      VisibleConstType v, const std::vector<std::vector<int>> &tochange,
      const std::vector<std::vector<double>> &newconf) override;

  // Difference between logarithms of values, when one or more visible variables
  // are being flipped Version using pre-computed look-up tables for efficiency
  // on a small number of spin flips
  Complex LogValDiff(VisibleConstType v, const std::vector<int> &tochange,
                     const std::vector<double> &newconf,
                     const LookupType &lt) override;

  void Save(const std::string &filename) const override;
  void Load(const std::string &filename) override;

  bool IsHolomorphic() const noexcept override;

  static double lncosh(double x) {
    const double xp = std::abs(x);
    if (xp <= 12.) {
      return std::log(std::cosh(xp));
    } else {
      const static double log2v = std::log(2.);
      return xp - log2v;
    }
  }

  // ln(cos(x)) for std::complex argument
  // the modulus is computed by means of the previously defined function
  // for real argument
  static Complex lncosh(Complex x) {
    const double xr = x.real();
    const double xi = x.imag();

    Complex res = RbmSpin::lncosh(xr);
    res += std::log(Complex(std::cos(xi), std::tanh(xr) * std::sin(xi)));

    return res;
  }

  static void tanh(VectorConstRefType x, VectorType &y) {
    assert(y.size() >= x.size());
    y = Eigen::tanh(x.array());
  }

  static void tanh(RealVectorConstRefType x, RealVectorType &y) {
    assert(y.size() >= x.size());
    y = Eigen::tanh(x.array());
  }

  static void lncosh(VectorConstRefType x, VectorType &y) {
    assert(y.size() >= x.size());
    for (int i = 0; i < x.size(); i++) {
      y(i) = lncosh(x(i));
    }
  }

  static void lncosh(RealVectorConstRefType x, RealVectorType &y) {
    assert(y.size() >= x.size());
    for (int i = 0; i < x.size(); i++) {
      y(i) = lncosh(x(i));
    }
  }

 private:
  inline void Init();
};

}  // namespace netket

#endif
