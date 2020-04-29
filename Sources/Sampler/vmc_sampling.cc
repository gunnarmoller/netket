#include "vmc_sampling.hpp"

namespace netket {
namespace vmc {
    
const Complex I_(0, 1);

Result ComputeSamples(AbstractSampler &sampler, Index nsamples, Index ndiscard,
                      bool compute_logderivs) {
  sampler.Reset();

  for (Index i = 0; i < ndiscard; i++) {
    sampler.Sweep();
  }

  const Index nvisible = sampler.GetMachine().Nvisible();
  const Index npar = sampler.GetMachine().Npar();
  MatrixXd samples(nvisible, nsamples);

  nonstd::optional<MatrixXcd> log_derivs;
  if (compute_logderivs) {
    log_derivs.emplace(npar, nsamples);
    for (Index i = 0; i < nsamples; i++) {
      sampler.Sweep();
      samples.col(i) = sampler.Visible();
      log_derivs->col(i) = sampler.DerLogVisible();
    }

    // Compute "centered" log-derivatives, i.e., O_k ↦ O_k - ⟨O_k⟩
    VectorXcd log_der_mean = log_derivs->rowwise().mean();
    MeanOnNodes<>(log_der_mean);
    log_derivs = log_derivs->colwise() - log_der_mean;
  } else {
    for (Index i = 0; i < nsamples; i++) {
      sampler.Sweep();
      samples.col(i) = sampler.Visible();
    }
  }
  return Result(std::move(samples), std::move(log_derivs));
}

Complex LocalValue(const AbstractOperator &op, AbstractMachine &psi,
                   Eigen::Ref<const VectorXd> v) {
  AbstractOperator::ConnectorsType tochange;
  AbstractOperator::NewconfsType newconf;
  AbstractOperator::MelType mels;

  op.FindConn(v, mels, tochange, newconf);

  auto logvaldiffs = psi.LogValDiff(v, tochange, newconf);

  assert(mels.size() == std::size_t(logvaldiffs.size()));

  Complex result = 0.0;
  for (Index i = 0; i < logvaldiffs.size(); ++i) {
    result += mels[i] * std::exp(logvaldiffs(i));  // O(v,v') * Ψ(v)/Ψ(v')
  }

  return result;
}

VectorXcd LocalValues(const AbstractOperator &op, AbstractMachine &psi,
                      Eigen::Ref<const MatrixXd> vs) {
  VectorXcd out(vs.cols());
  for (Index i = 0; i < vs.cols(); ++i) {
    out(i) = LocalValue(op, psi, vs.col(i));
  }
  return out;
}

VectorXcd LocalValueDeriv(const AbstractOperator &op, AbstractMachine &psi,
                          Eigen::Ref<const VectorXd> v) {
  AbstractOperator::ConnectorsType tochange;
  AbstractOperator::NewconfsType newconf;
  AbstractOperator::MelType mels;
  op.FindConn(v, mels, tochange, newconf);

  auto logvaldiffs = psi.LogValDiff(v, tochange, newconf);
  auto log_deriv = psi.DerLog(v);

  VectorXcd grad(v.size());
  for (int i = 0; i < logvaldiffs.size(); i++) {
    const auto melval = mels[i] * std::exp(logvaldiffs(i));
    const auto log_deriv_prime = psi.DerLogChanged(v, tochange[i], newconf[i]);
    grad += melval * (log_deriv - log_deriv_prime);
  }

  return grad;
}

Stats Expectation(const Result &result, AbstractMachine &psi,
                  const AbstractOperator &op) {
  Binning<double> bin;
  for (Index i = 0; i < result.NSamples(); ++i) {
    const Complex loc = LocalValue(op, psi, result.Sample(i));
    bin << loc.real();
  }
  return bin.AllStats();
}

Stats Expectation(const Result &result, AbstractMachine &psi,
                  const AbstractOperator &op, VectorXcd &locvals) {
  locvals.resize(result.NSamples());

  Binning<double> bin;
  for (Index i = 0; i < result.NSamples(); ++i) {
    const Complex loc = LocalValue(op, psi, result.Sample(i));
    locvals(i) = loc;
    bin << loc.real();
  }

  return bin.AllStats();
}



Stats Variance(const Result &result, AbstractMachine &psi,
               const AbstractOperator &op) {
  VectorXcd locvals;
  auto ex = Expectation(result, psi, op, locvals);
  return Variance(result, psi, op, ex.mean, locvals);
}

Stats Variance(const Result & /*result*/, AbstractMachine & /*psi*/,
               const AbstractOperator & /*op*/, double expectation_value,
               const VectorXcd &locvals) {
  Binning<double> bin_var;
  for (Index i = 0; i < locvals.size(); ++i) {
    bin_var << std::norm(locvals(i) - expectation_value);
  }
  return bin_var.AllStats();
}

VectorXcd Gradient(const Result &result, AbstractMachine &psi,
                   const AbstractOperator &op) {
  if (!result.LogDerivs().has_value()) {
    throw std::runtime_error{
        "vmc::Result does not contain log-derivatives, which are required to "
        "compute gradients."};
  }

  VectorXcd locvals;
  Expectation(result, psi, op, locvals);

  VectorXcd grad =
      result.LogDerivs()->conjugate() * locvals / double(result.NSamples());
  MeanOnNodes<>(grad);

  return grad;
}

VectorXcd Gradient(const Result &result, AbstractMachine & /*psi*/,
                   const AbstractOperator & /*op*/, const VectorXcd &locvals) {
  if (!result.LogDerivs().has_value()) {
    throw std::runtime_error{
        "vmc::Result does not contain log-derivatives, which are required to "
        "compute gradients."};
  }

  VectorXcd grad =
      result.LogDerivs()->conjugate() * locvals / double(result.NSamples());
  MeanOnNodes<>(grad);
  return grad;
}

VectorXcd GradientOfVariance(const Result &result, AbstractMachine &psi,
                             const AbstractOperator &op) {
  // TODO: This function can probably be implemented more efficiently (e.g., by
  // computing local values and their gradients at the same time or reusing
  // already computed local values)
  MatrixXcd locval_deriv(psi.Npar(), result.NSamples());

  for (int i = 0; i < result.NSamples(); i++) {
    locval_deriv.col(i) = LocalValueDeriv(op, psi, result.Sample(i));
  }
  VectorXcd locval_deriv_mean = locval_deriv.colwise().mean();
  MeanOnNodes<>(locval_deriv_mean);
  locval_deriv = locval_deriv.colwise() - locval_deriv_mean;

  VectorXcd grad = locval_deriv.conjugate() *
                   LocalValues(op, psi, result.SampleMatrix()) /
                   double(result.NSamples());
  MeanOnNodes<>(grad);
  return grad;
}

// ----------------------------------------------------------------------------
/** TTComment:
 * The functions for density matrix
 */

/** TTComment:
 * C1 local value used to derive the gradient of cost function
 */
Complex C1LocalValue(const AbstractOperator &op, AbstractDensityMatrix &rho,
                     Eigen::Ref<const VectorXd> v, 
                     Eigen::Ref<const VectorXd> gamma)
{
  int n_vis = rho.Nvisible();  
  Eigen::Ref<const VectorXd> vr = v.head(n_vis);
  Eigen::Ref<const VectorXd> vc = v.tail(n_vis);
  
  AbstractOperator::ConnectorsType tochanger;
  AbstractOperator::ConnectorsType tochangec;
  AbstractOperator::NewconfsType newconfr;
  AbstractOperator::NewconfsType newconfc;
  AbstractOperator::MelType melsr;
  AbstractOperator::MelType melsc;
  
  op.FindConn(vr, melsr, tochanger, newconfr);
  op.FindConn(vc, melsc, tochangec, newconfc);
  
  for (auto &el1 : tochangec)
    for (auto &el2 : el1)
      el2 += n_vis;
  
  auto logvaldiffsr = rho.LogValDiff(v, tochanger, newconfr);
  auto logvaldiffsc = rho.LogValDiff(v, tochangec, newconfc);
  
  assert(melsr.size() == std::size_t(logvaldiffsr.size()));
  assert(melsc.size() == std::size_t(logvaldiffsc.size()));
  
  Complex result = 0.0;
  // Values from hamiltonian part
  for (Index i = 0; i < logvaldiffsr.size(); ++i)
  {
    result += -I_ * melsr[i] * std::exp(-logvaldiffsr(i));
  }
  for (Index i = 0; i < logvaldiffsc.size(); ++i)
  {
    result += I_ * std::conj(melsc[i]) * std::exp(-logvaldiffsc(i));    
  }
  
  decltype(logvaldiffsr) templogvaldiff;
  AbstractOperator::NewconfsType tempnewconf;
  AbstractOperator::ConnectorsType temptochange;
  // Values from jump operator part (only for spin 1/2)
  for (int j = 0; j < n_vis; ++j)
  {
    if (vc[j] == -1 && vr[j] == -1)
    {
      tempnewconf = {{1, 1}};
      temptochange = {{j, j + n_vis}};
      templogvaldiff = rho.LogValDiff(v, temptochange, tempnewconf);
      result += gamma[j] / 2 * std::exp(-templogvaldiff(0));  
    }
    if (vr[j] == +1)
    {
      result -= gamma[j] / 2;  
    }
    if (vc[j] == +1)
    {
      result -= gamma[j] / 2;
    }
  }
  
  return result;
}

/** TTComment:
 * C2 local value used to derive the gradient of cost function
 */
VectorType C2LocalValue(const AbstractOperator &op, AbstractDensityMatrix &rho,
                     Eigen::Ref<const VectorXd> v, 
                     Eigen::Ref<const VectorXd> gamma)
{
  int n_vis = rho.Nvisible();  
  int n_par = rho.Npar();
  Eigen::Ref<const VectorXd> vr = v.head(n_vis);
  Eigen::Ref<const VectorXd> vc = v.tail(n_vis);
  Eigen::VectorXd temp_v;
  int h = 0;
  
  AbstractOperator::ConnectorsType tochanger;
  AbstractOperator::ConnectorsType tochangec;
  AbstractOperator::NewconfsType newconfr;
  AbstractOperator::NewconfsType newconfc;
  AbstractOperator::MelType melsr;
  AbstractOperator::MelType melsc;
  
  op.FindConn(vr, melsr, tochanger, newconfr);
  op.FindConn(vc, melsc, tochangec, newconfc);
  
  for (auto &el1 : tochangec)
    for (auto &el2 : el1)
      el2 += n_vis;
  
  auto logvaldiffsr = rho.LogValDiff(v, tochanger, newconfr);
  auto logvaldiffsc = rho.LogValDiff(v, tochangec, newconfc);
  
  assert(melsr.size() == std::size_t(logvaldiffsr.size()));
  assert(melsc.size() == std::size_t(logvaldiffsc.size()));
  
  VectorType result = VectorType::Zero(n_par);
  // Values from hamiltonian part
  for (Index i = 0; i < logvaldiffsr.size(); ++i)
  {
    temp_v = v;
    h = 0;
    for (auto &el2 : tochanger[i])
    {
      temp_v[el2] = newconfr[i][h];
      ++h;
    }
    result += -I_ * melsr[i] * std::exp(-logvaldiffsr(i)) * 
                      rho.DerLog(temp_v);
  }
  for (Index i = 0; i < logvaldiffsc.size(); ++i)
  {
    temp_v = v;
    h = 0;
    for (auto &el2 : tochangec[i])
    {
      temp_v[el2] = newconfc[i][h];
      ++h;
    }  
    result += I_ * std::conj(melsc[i]) * std::exp(-logvaldiffsc(i)) *
                    rho.DerLog(temp_v);    
  }
  
  decltype(logvaldiffsr) templogvaldiff;
  AbstractOperator::NewconfsType tempnewconf;
  AbstractOperator::ConnectorsType temptochange;
  // Values from jump operator part (only for spin 1/2)
  /// TODO: implement jump operator for any system
  for (int j = 0; j < n_vis; ++j)
  {
    if (vc[j] == -1 && vr[j] == -1)
    {
      temp_v = v;
      temp_v[j] = 1;
      temp_v[j + n_vis] = 1;
      tempnewconf = {{1, 1}};
      temptochange = {{j, j + n_vis}};
      templogvaldiff = rho.LogValDiff(v, temptochange, tempnewconf);
      result += gamma[j] / 2 * std::exp(-templogvaldiff(0)) * 
                        rho.DerLog(temp_v);  
    }
    if (vr[j] == +1)
    {
      result -= gamma[j] / 2 * rho.DerLog(v);  
    }
    if (vc[j] == +1)
    {
      result -= gamma[j] / 2 * rho.DerLog(v);
    }
  }
  
  return result;
}

/** TTComment:
 * calculating gradient of cost function for density matrix
 */
VectorXcd GradientDM(const Result &result, AbstractDensityMatrix &rho,
                     const AbstractOperator &op,
                     Eigen::Ref<const VectorXd> gamma)
{
    int n_par = rho.Npar();
    int n_samples = result.NSamples();
    
    Complex locC1_value = 0.0;
    VectorType locC2_value = VectorType::Zero(n_par);
    
    Complex c1_value = 0.0;
    VectorType o_value = VectorType::Zero(n_par);
    Complex c2_value = VectorType::Zero(n_par);
    
    for (Index i = 0; i < n_samples; ++i)
    {
        locC1_value = C1LocalValue(op, rho, result.Sample(i), gamma);
        locC2_value = C2LocalValue(op, rho, result.Sample(i), gamma);
        c1_value += pow(abs(locC1_value), 2);
        o_value += result.LogDerivs();
        c2_value += conj(locC1_value) * locC2_value;
    }
    VectorType result = - c1_value * o_value / (n_samples * n_samples) + 
                          c2_value / n_samples;
    
    return 2*result.real()
}

/** TTComment:
 * calculating expectation values for a given density matrix and 
 * operator
 */
Complex ExpectationDM(const Result &result, AbstractDensityMatrix &rho,
                      const AbstractOperator &op)
{
    int n_samples = result.NSamples();
    int n_vis = rho.Nvisible();
    Complex result = 0.0;
    VectorType v = VectorType::Zero(2 * n_vis);
    VectorType vr = VectorType::Zero(n_vis);
    VectorType vc = VectorType::Zero(n_vis);
    
    for (Index i = 0; i < n_samples; ++i)
    {
        v = result.Sample(i)
        vr = v.head(n_vis);
        vc = v.tail(n_vis);
        result += op.FindAllMatrixElements(vr, vc) / rho.LogVal(v);        
    }
    return result;
}

}  // namespace vmc
}  // namespace netket
