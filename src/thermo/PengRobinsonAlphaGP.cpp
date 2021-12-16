//! @file PengRobinsonAlphaGP.cpp

// This file is part of Cantera. See License.txt in the top-level directory or
// at https://cantera.org/license.txt for license and copyright information.

#include "cantera/thermo/PengRobinsonAlphaGP.h"
#include "cantera/thermo/ThermoFactory.h"
#include "cantera/thermo/Species.h"
#include "cantera/base/stringUtils.h"
#include "cantera/base/ctml.h"

#include <boost/math/tools/roots.hpp>

using namespace std;
namespace bmt = boost::math::tools;

namespace Cantera
{

const double PengRobinsonAlphaGP::omega_a = 4.5723552892138218E-01;
const double PengRobinsonAlphaGP::omega_b = 7.77960739038885E-02;
const double PengRobinsonAlphaGP::omega_vc = 3.07401308698703833E-01;

GaussianProcess::GaussianProcess(const std::string &datapath, const std::string &species)
{
    // handling AlphaGP parameters
    std::cout << "Reading " << datapath << " of species para" << species << std::endl;
    readGPPara(datapath+species+"_para.csv", m_BasisTheta, m_KernelGamma, m_KernelSigmaF, m_HessianMatrix ,m_yscale);
    
    // handling AlphaGP data
    std::cout << "Reading Alpha data of species " << species << std::endl;  
    readGPData(datapath+species+".csv", m_X, m_y);
    
    int N = m_X.rows();
    
	std::cout << "Calculating m_F ..." << std::endl;

	m_F.resize(N,1);
	basisFunc(m_X, m_F);
    
	std::cout << "Calculating kernelFunc ..." << std::endl;

	m_K.resize(N,N);
    m_Kx.resize(N,1);
    kernelFunc(m_X, m_X, m_K);

	std::cout << "Calculating m_Ki ..." << std::endl;

	MatrixXd I = MatrixXd::Constant(N,N,1e-4).diagonal().asDiagonal();
    m_Ki = (m_K+I).inverse();
}

void GaussianProcess::readCSV(const std::string filename, std::vector< std::vector<double> > &output, int &nrow, int &ncol)
{
    int ndata = 0;
    nrow = 0;
    
    fstream input(filename, ios::in);
    if(!input.is_open()){
        std::cout << "GaussianProcess::readCSV file " 
                  << filename << "not found!" << std::endl;
        return;
    }
    
    std::string csvLine;
    // read every line from the stream
    while(std::getline(input, csvLine))
    {
        istringstream csvStream(csvLine);
        std::vector<double> csvColumn;
        std::string csvElement;
        while( getline(csvStream, csvElement, ',') )
        {
            csvColumn.push_back(std::stod(csvElement));
            ndata++;
        }
        output.push_back(csvColumn);
        nrow++;
    }
    ncol = ndata/nrow;
}


void GaussianProcess::readGPData(const std::string filename, MatrixXd &X, MatrixXd &y)
{
    int nrow = 0;
    int ncol = 0;
    std::vector< std::vector<double> > data;
    
    readCSV(filename, data, nrow, ncol);

    X.resize(nrow, ncol-1);
    y.resize(nrow, 1);

    for(int i=0; i<nrow; ++i){
        for(int j=0; j<ncol-1; ++j){
            X(i,j) = data[i][j];
        }
        y(i,0) = data[i][ncol-1] / m_yscale;
    }
}

void GaussianProcess::readGPPara(const std::string filename, VectorXd &BasisTheta, VectorXd &KernelGamma, double &KernelSigmaF, MatrixXd &HessianMatrix, double &yscale)
{
    int nrow = 0;
    int ncol = 0;
    std::vector< std::vector<double> > para;
    
    readCSV(filename, para, nrow, ncol);
    
    KernelGamma.resize(ncol-1);
    int i=0;
    for(int j=0; j<ncol-1; ++j){
        KernelGamma[j] = para[i][j];
    }
    KernelSigmaF = para[i][ncol-1];
    
    BasisTheta.resize(ncol);
    if(nrow>=1){
        i=1;
        for(int j=0; j<ncol; ++j)
            BasisTheta[j] = para[i][j];
    }else{
        for(int j=0; j<ncol; ++j)
            BasisTheta[j] = 1;
    }
	
	HessianMatrix.resize(ncol,ncol);
	for(int i=0; i<ncol; ++i){
		for(int j=0; j<ncol; ++j){
			HessianMatrix(i,j)=para[i+2][j];
		}
	}
    std::cout << "nrow=" << nrow << " ncol=" << ncol << std::endl; 
    std::cout << "KernelSigmaF=" << KernelSigmaF << std::endl;

    yscale = nrow <= ncol+2 ? 1 : para[ncol+2][0];

    std::cout << "yscale=" << yscale << std::endl;
    
	//cout<<std::right << setw(3) <<BasisTheta<<","<<HessianMatrix;
}

void GaussianProcess::basisFunc(const MatrixXd &X1, MatrixXd &F) const
{
    int ndim = X1.cols();
    int nrow = X1.rows();
	for(int j=0; j<nrow; ++j){
		F(j,0) = m_BasisTheta[0] + m_BasisTheta[1]*X1(j,0);
		//cout<<X1(j,0)<<","<<X1(j,1)<<endl;
	}
}

MatrixXd GaussianProcess::dfdtheta(const MatrixXd &X1) const
{
    int ndim = X1.cols();
    int nrow = X1.rows();

	MatrixXd df(nrow,ndim+1);
	for(int j=0; j<nrow; ++j){
		df(j,0) = 1;
		df(j,1) = X1(j,0);
	}
	return df;
}

void GaussianProcess::kernelFunc0(const MatrixXd &X1, const MatrixXd &X2,  MatrixXd &K) const
{
    int ndim = X1.cols();
    int nrow = X1.rows();
    int ncol = X2.rows();
    //cout<<X2<<endl;
    double sigma2 = m_KernelSigmaF * m_KernelSigmaF;
    VectorXd gamma2 = m_KernelGamma * m_KernelGamma;
    for(int i=0; i<nrow; ++i){
        for(int j=0; j<ncol; ++j){
            double val = 0;
            for(int k=0; k<ndim; ++k){
                val += - (X1(i,k)-X2(j,k)) * (X1(i,k)-X2(j,k)) / 2 / gamma2[k];
            }
            K(i,j) = sigma2 * exp(val);
        }
    }
}

MatrixXd GaussianProcess::hmatrix(const MatrixXd &X1, const MatrixXd &X2) const
{
    int ndim = X1.cols();
    int N = X1.rows();
	int M = X2.rows();
	
	MatrixXd Kx(N, M);
	kernelFunc0(X1, X2, Kx); // Kx, MxN
	
	MatrixXd grad_f = dfdtheta(X1); // Nx3
	MatrixXd h = grad_f.transpose() * Kx; // 3xM
	return h;
}

void GaussianProcess::kernelFunc(const MatrixXd &X1, const MatrixXd &X2,  MatrixXd &K) const
{
    kernelFunc0(X1, X2, K);

	// compensation terms from basis function
	MatrixXd hXx = hmatrix(X1, X2); // 3xM
	MatrixXd hXX = hmatrix(X1, X1); // 3xN

	MatrixXd Hi =  m_HessianMatrix.completeOrthogonalDecomposition().pseudoInverse();
	// std::cout << "Hi = " << setw(4) << Hi << std::endl;
	// std::cout << "hXx = " << setw(4) << hXx << std::endl;
    K = K + hXX.transpose() * Hi * hXx; // NxM
}

double GaussianProcess::predictGP(const MatrixXd &Xnew)
{	
	int N = m_X.rows();
	MatrixXd K(N,N);
	kernelFunc0(m_X, m_X, K);
	MatrixXd I = MatrixXd::Constant(N,N,1e-4).diagonal().asDiagonal();
    MatrixXd Ki = (K+I).inverse();
	MatrixXd Kx(N,1);
	kernelFunc0(m_X, Xnew, Kx);
	double y = (Kx.transpose() * Ki * m_y).value()*m_yscale;

	//  std::cout << "predict Xnew=" << setw(4) << Xnew << std::endl;
	//  std::cout << "predict y=" << y << std::endl;
    return y;
}


double GaussianProcess::predict(const MatrixXd &Xnew)
{
    // get Kx
	MatrixXd f(1,1);
    
	kernelFunc(m_X, Xnew, m_Kx);
	basisFunc(Xnew, f);

    //y(x) = f(x) + Kx.T * Ki * (y - f(X))
	double y = ( f + m_Kx.transpose() * m_Ki * (m_y-m_F) ).value()*m_yscale;
	// std::cout << "predict Xnew=" << setw(4) << Xnew << std::endl;
	// std::cout << "predict Kx=" << setw(4) << m_Kx << std::endl;
	// std::cout << "predict f=" << f << std::endl;
	// std::cout << "predict y=" << y << std::endl;
    return y;
}

PengRobinsonAlphaGP::PengRobinsonAlphaGP(const std::string& infile, const std::string& id_) :
    m_b(0.0),
    m_a(0.0),
    m_aAlpha_mix(0.0),
    m_NSolns(0),
    m_dpdV(0.0),
    m_dpdT(0.0)
{
    fill_n(m_Vroot, 3, 0.0);
    initThermoFile(infile, id_);
}

inline double PengRobinsonAlphaGP::GPpredict(GaussianProcess GP, double T, int k) const
{    
    MatrixXd AlphaXnew(1,2);
    double Tc = speciesCritTemperature(m_a_coeffs(k,k), m_b_coeffs[k]);
    double Tr = T / Tc;
    AlphaXnew << Tr;
	return GP.predict(AlphaXnew);
}

void PengRobinsonAlphaGP::updateAlpha(double T) const
{
    for(size_t k = 0; k < m_kk; ++k){
        m_alpha[k] = GPpredict(m_alphaGP[k], T, k);
    }
}

void PengRobinsonAlphaGP::mixAlpha() const
{
    // update alpha mix
    for(size_t k = 0; k < m_kk; ++k){
        m_aAlpha_binary(k,k) = m_a_coeffs(k,k) * m_alpha[k];
    }
    // standard mixing rule for cross-species interaction term
    for(size_t k = 0; k < m_kk; ++k){
        for (size_t j = 0; j < m_kk; j++) {
            if (k == j) {
                continue;
            }
            double a_Alpha_kj = m_a_coeffs(k,j)*sqrt(m_alpha[j] * m_alpha[k]);
            m_aAlpha_binary(j, k) = a_Alpha_kj;
            m_aAlpha_binary(k, j) = a_Alpha_kj;
        }
    }
}

void PengRobinsonAlphaGP::setSpeciesCoeffs(const std::string& species, double a, double b, double w)
{
    size_t k = speciesIndex(species);
    if (k == npos) {
        throw CanteraError("PengRobinsonAlphaGP::setSpeciesCoeffs",
            "Unknown species '{}'.", species);
    }
    
    m_alphaGP.push_back(GaussianProcess("mech2/Alpha/", species));
	m_cpGP.push_back(GaussianProcess("mech2/Cpmass/", species));
    m_cpmoleGP.push_back(GaussianProcess("mech2/Cpmole/", species));
    m_denGP.push_back(GaussianProcess("mech2/Density/", species));
    m_hresidGP.push_back(GaussianProcess("mech2/Hmole/", species));
    m_sresidGP.push_back(GaussianProcess("mech2/Smole/", species));
    
    // Calculate value of kappa (independent of temperature)
    // w is an acentric factor of species
    if (w <= 0.491) {
        m_kappa[k] = 0.37464 + 1.54226*w - 0.26992*w*w;
    } else {
        m_kappa[k] = 0.374642 + 1.487503*w - 0.164423*w*w + 0.016666*w*w*w;
    }

    //Calculate alpha (temperature dependent interaction parameter)
    double critTemp = speciesCritTemperature(a, b); // critical temperature of individual species
    double sqt_T_r = sqrt(temperature() / critTemp);
    double sqt_alpha = 1 + m_kappa[k] * (1 - sqt_T_r);
    m_alpha[k] = sqt_alpha*sqt_alpha;
    m_a_coeffs(k,k) = a;
    m_b_coeffs[k] = b;
}

void PengRobinsonAlphaGP::setBinaryCoeffs(const std::string& species_i,
        const std::string& species_j, double a0)
{
    size_t ki = speciesIndex(species_i);
    if (ki == npos) {
        throw CanteraError("PengRobinsonAlphaGP::setBinaryCoeffs",
            "Unknown species '{}'.", species_i);
    }
    size_t kj = speciesIndex(species_j);
    if (kj == npos) {
        throw CanteraError("PengRobinsonAlphaGP::setBinaryCoeffs",
            "Unknown species '{}'.", species_j);
    }

    m_a_coeffs(ki, kj) = m_a_coeffs(kj, ki) = a0;
    // Calculate alpha_ij
    double alpha_ij = m_alpha[ki] * m_alpha[kj];
    
    // added by Xingyu 2021/11/05: this is alpha_ij * aij
    m_aAlpha_binary(ki, kj) = m_aAlpha_binary(kj, ki) = a0*alpha_ij; 
}

// ------------Molar Thermodynamic Properties -------------------------

double PengRobinsonAlphaGP::cp_mole() const
{
    // _updateReferenceStateThermo();
    double cp_mole = 0;
    for(size_t k = 0; k < m_kk; ++k){
        double T =temperature();
        // std::cout<<m_cpmole[k]<<std::endl;
        cp_mole =cp_mole +  moleFractions_[k] * GPpredict(m_cpmoleGP[k], T, k);
    }
    return cp_mole;//J/kmol/K

    // double mv = molarVolume();
    // double vpb = mv + (1 + Sqrt2) * m_b;
    // double vmb = mv + (1 - Sqrt2) * m_b;
    // calculatePressureDerivatives();
    // double cpref = GasConstant * mean_X(m_cp0_R);
    // double dHdT_V = cpref + mv * m_dpdT - GasConstant
    //                 + 1.0 / (2.0 * Sqrt2 * m_b) * log(vpb / vmb) * T * d2aAlpha_dT2();
    //return dHdT_V - (mv + T * m_dpdT / m_dpdV) * m_dpdT;
}

double PengRobinsonAlphaGP::cv_mole() const
{
    _updateReferenceStateThermo();
    double T = temperature();
    calculatePressureDerivatives();
    return (cp_mole() + T * m_dpdT * m_dpdT / m_dpdV);
}

double PengRobinsonAlphaGP::pressure() const
{
    _updateReferenceStateThermo();
    //  Get a copy of the private variables stored in the State object
    double T = temperature();
    double mv = molarVolume();
    double denom = mv * mv + 2 * mv * m_b - m_b * m_b;
    double pp = GasConstant * T / (mv - m_b) - m_aAlpha_mix / denom;
    return pp;
}

double PengRobinsonAlphaGP::standardConcentration(size_t k) const
{
    getStandardVolumes(m_tmpV.data());
    return 1.0 / m_tmpV[k];
}

void PengRobinsonAlphaGP::getActivityCoefficients(double* ac) const
{
    double mv = molarVolume();
    double vpb2 = mv + (1 + Sqrt2) * m_b;
    double vmb2 = mv + (1 - Sqrt2) * m_b;
    double vmb = mv - m_b;
    double pres = pressure();

    for (size_t k = 0; k < m_kk; k++) {
        m_pp[k] = 0.0;
        for (size_t i = 0; i < m_kk; i++) {
            m_pp[k] += moleFractions_[i] * m_aAlpha_binary(k, i);
        }
    }
    double num = 0;
    double denom = 2 * Sqrt2 * m_b * m_b;
    double denom2 = m_b * (mv * mv + 2 * mv * m_b - m_b * m_b);
    double RTkelvin = RT();
    for (size_t k = 0; k < m_kk; k++) {
        num = 2 * m_b * m_pp[k] - m_aAlpha_mix * m_b_coeffs[k];
        ac[k] = (-RTkelvin * log(pres * mv/ RTkelvin) + RTkelvin * log(mv / vmb)
                 + RTkelvin * m_b_coeffs[k] / vmb
                 - (num /denom) * log(vpb2/vmb2)
                 - m_aAlpha_mix * m_b_coeffs[k] * mv/denom2
                );
    }
    for (size_t k = 0; k < m_kk; k++) {
        ac[k] = exp(ac[k]/ RTkelvin);
    }
}

// ---- Partial Molar Properties of the Solution -----------------

void PengRobinsonAlphaGP::getChemPotentials(double* mu) const
{
    getGibbs_ref(mu);
    double RTkelvin = RT();
    for (size_t k = 0; k < m_kk; k++) {
        double xx = std::max(SmallNumber, moleFraction(k));
        mu[k] += RTkelvin * (log(xx));
    }

    double mv = molarVolume();
    double vmb = mv - m_b;
    double vpb2 = mv + (1 + Sqrt2) * m_b;
    double vmb2 = mv + (1 - Sqrt2) * m_b;

    for (size_t k = 0; k < m_kk; k++) {
        m_pp[k] = 0.0;
        for (size_t i = 0; i < m_kk; i++) {
            m_pp[k] += moleFractions_[i] * m_aAlpha_binary(k, i);
        }
    }
    double pres = pressure();
    double refP = refPressure();
    double denom = 2 * Sqrt2 * m_b * m_b;
    double denom2 = m_b * (mv * mv + 2 * mv * m_b - m_b * m_b);

    for (size_t k = 0; k < m_kk; k++) {
        double num = 2 * m_b * m_pp[k] - m_aAlpha_mix * m_b_coeffs[k];

        mu[k] += (RTkelvin * log(pres/refP) - RTkelvin * log(pres * mv / RTkelvin)
                  + RTkelvin * log(mv / vmb)
                  + RTkelvin * m_b_coeffs[k] / vmb
                  - (num /denom) * log(vpb2/vmb2)
                  - m_aAlpha_mix * m_b_coeffs[k] * mv/denom2
                 );
    }
}

void PengRobinsonAlphaGP::getPartialMolarEnthalpies(double* hbar) const
{
    // First we get the reference state contributions
    getEnthalpy_RT_ref(hbar);
    scale(hbar, hbar+m_kk, hbar, RT());
    vector_fp tmp;
    tmp.resize(m_kk,0.0);

    // We calculate m_dpdni
    double T = temperature();
    double mv = molarVolume();
    double vmb = mv - m_b;
    double vpb2 = mv + (1 + Sqrt2) * m_b;
    double vmb2 = mv + (1 - Sqrt2) * m_b;
    double daAlphadT = daAlpha_dT();

    for (size_t k = 0; k < m_kk; k++) {
        m_pp[k] = 0.0;
        tmp[k] = 0.0;
        for (size_t i = 0; i < m_kk; i++) {
            double grad_aAlpha = m_dalphadT[i]/m_alpha[i] + m_dalphadT[k]/m_alpha[k];
            m_pp[k] += moleFractions_[i] * m_aAlpha_binary(k, i);
            tmp[k] +=moleFractions_[i] * m_aAlpha_binary(k, i) * grad_aAlpha;
        }
    }

    double denom = mv * mv + 2 * mv * m_b - m_b * m_b;
    double denom2 = denom * denom;
    double RTkelvin = RT();
    for (size_t k = 0; k < m_kk; k++) {
        m_dpdni[k] = RTkelvin / vmb + RTkelvin * m_b_coeffs[k] / (vmb * vmb) - 2.0 * m_pp[k] / denom
                    + 2 * vmb * m_aAlpha_mix * m_b_coeffs[k] / denom2;
    }

    double fac = T * daAlphadT - m_aAlpha_mix;
    calculatePressureDerivatives();
    double fac2 = mv + T * m_dpdT / m_dpdV;
    double fac3 = 2 * Sqrt2 * m_b * m_b;
    double fac4 = 0;
    for (size_t k = 0; k < m_kk; k++) {
        fac4 = T*tmp[k] -2 * m_pp[k];
        double hE_v = mv * m_dpdni[k] - RTkelvin - m_b_coeffs[k] / fac3  * log(vpb2 / vmb2) * fac
                     + (mv * m_b_coeffs[k]) /(m_b * denom) * fac
                     + 1/(2 * Sqrt2 * m_b) * log(vpb2 / vmb2) * fac4;
        hbar[k] = hbar[k] + hE_v;
        hbar[k] -= fac2 * m_dpdni[k];
    }
}

void PengRobinsonAlphaGP::getPartialMolarEntropies(double* sbar) const
{
    // Using the identity : (hk - T*sk) = gk
    double T = temperature();
    getPartialMolarEnthalpies(sbar);
    getChemPotentials(m_tmpV.data());
    for (size_t k = 0; k < m_kk; k++) {
        sbar[k] = (sbar[k] - m_tmpV[k])/T;
    }
}

void PengRobinsonAlphaGP::getPartialMolarIntEnergies(double* ubar) const
{
    // u_i = h_i - p*v_i
    double p = pressure();
    getPartialMolarEnthalpies(ubar);
    getPartialMolarVolumes(m_tmpV.data());
    for (size_t k = 0; k < m_kk; k++) {
        ubar[k] = ubar[k] - p*m_tmpV[k];
    }
}

void PengRobinsonAlphaGP::getPartialMolarCp(double* cpbar) const
{
    throw NotImplementedError("PengRobinsonAlphaGP::getPartialMolarCp");
}

void PengRobinsonAlphaGP::getPartialMolarVolumes(double* vbar) const
{
    for (size_t k = 0; k < m_kk; k++) {
        m_pp[k] = 0.0;
        for (size_t i = 0; i < m_kk; i++) {
            m_pp[k] += moleFractions_[i] * m_aAlpha_binary(k, i);
        }
    }

    double mv = molarVolume();
    double vmb = mv - m_b;
    double vpb = mv + m_b;
    double fac = mv * mv + 2 * mv * m_b - m_b * m_b;
    double fac2 = fac * fac;
    double RTkelvin = RT();

    for (size_t k = 0; k < m_kk; k++) {
        double num = (RTkelvin + RTkelvin * m_b/ vmb + RTkelvin * m_b_coeffs[k] / vmb
                      + RTkelvin * m_b * m_b_coeffs[k] /(vmb * vmb)
                      - 2 * mv * m_pp[k] / fac
                      + 2 * mv * vmb * m_aAlpha_mix * m_b_coeffs[k] / fac2
                     );
        double denom = (pressure() + RTkelvin * m_b / (vmb * vmb)
                        + m_aAlpha_mix/fac
                        - 2 * mv* vpb * m_aAlpha_mix / fac2
                       );
        vbar[k] = num / denom;
    }
}

double PengRobinsonAlphaGP::speciesCritTemperature(double a, double b) const
{
    if (b <= 0.0) {
        return 1000000.;
    } else if (a <= 0.0) {
        return 0.0;
    } else {
        return a * omega_b / (b * omega_a * GasConstant);
    }
}

double PengRobinsonAlphaGP::speciesCritPressure(double a, double b) const
{
    if (b <= 0.0) {
        return 1000000.;
    } else if (a <= 0.0) {
        return 0.0;
    } else {
        return a * omega_b * omega_b / (b * b * omega_a);
    }
}

bool PengRobinsonAlphaGP::addSpecies(shared_ptr<Species> spec)
{
    bool added = MixtureFugacityTP::addSpecies(spec);
    if (added) {
        m_a_coeffs.resize(m_kk, m_kk, 0.0);
        m_b_coeffs.push_back(0.0);
        m_aAlpha_binary.resize(m_kk, m_kk, 0.0);
        m_kappa.push_back(0.0);

        m_alpha.push_back(0.0);
        m_dalphadT.push_back(0.0);
        m_d2alphadT2.push_back(0.0);

        m_pp.push_back(0.0);
        m_partialMolarVolumes.push_back(0.0);
        m_dpdni.push_back(0.0);
    }
    return added;
}

vector<double> PengRobinsonAlphaGP::getCoeff(const std::string& iName)
{
    vector_fp spCoeff{ NAN, NAN, NAN };

    // Get number of species in the database
    // open xml file critProperties.xml
    XML_Node* doc = get_XML_File("critProperties.xml");
    size_t nDatabase = doc->nChildren();

    // Loop through all species in the database and attempt to match supplied
    // species to each. If present, calculate pureFluidParameters a_k and b_k
    // based on crit properties T_c and P_c:
    for (size_t isp = 0; isp < nDatabase; isp++) {
        XML_Node& acNodeDoc = doc->child(isp);
        std::string iNameLower = toLowerCopy(iName);
        std::string dbName = toLowerCopy(acNodeDoc.attrib("name"));

        // Attempt to match provided species iName to current database species
        //  dbName:
        if (iNameLower == dbName) {
            // Read from database and calculate a and b coefficients
            double vParams;
            double T_crit = 0.0, P_crit = 0.0, w_ac = 0.0;

            if (acNodeDoc.hasChild("Tc")) {
                vParams = 0.0;
                XML_Node& xmlChildCoeff = acNodeDoc.child("Tc");
                if (xmlChildCoeff.hasAttrib("value")) {
                    std::string critTemp = xmlChildCoeff.attrib("value");
                    vParams = strSItoDbl(critTemp);
                }
                if (vParams <= 0.0) { //Assuming that Pc and Tc are non zero.
                    throw CanteraError("PengRobinsonAlphaGP::getCoeff",
                        "Critical Temperature must be positive");
                }
                T_crit = vParams;
            }
            if (acNodeDoc.hasChild("Pc")) {
                vParams = 0.0;
                XML_Node& xmlChildCoeff = acNodeDoc.child("Pc");
                if (xmlChildCoeff.hasAttrib("value")) {
                    std::string critPressure = xmlChildCoeff.attrib("value");
                    vParams = strSItoDbl(critPressure);
                }
                if (vParams <= 0.0) { //Assuming that Pc and Tc are non zero.
                    throw CanteraError("PengRobinsonAlphaGP::getCoeff",
                        "Critical Pressure must be positive");
                }
                P_crit = vParams;
            }
            if (acNodeDoc.hasChild("omega")) {
                vParams = 0.0;
                XML_Node& xmlChildCoeff = acNodeDoc.child("omega");
                if (xmlChildCoeff.hasChild("value")) {
                    std::string acentric_factor = xmlChildCoeff.attrib("value");
                    vParams = strSItoDbl(acentric_factor);
                }
                w_ac = vParams;
            }

            spCoeff[0] = omega_a * (GasConstant * GasConstant) * (T_crit * T_crit) / P_crit; //coeff a
            spCoeff[1] = omega_b * GasConstant * T_crit / P_crit; // coeff b
            spCoeff[2] = w_ac; // acentric factor
            break;
        }
    }
    // If the species is not present in the database, throw an error
    if(isnan(spCoeff[0]))
    {
        throw CanteraError("PengRobinsonAlphaGP::getCoeff",
            "Species '{}' is not present in the database", iName);
    }
    return spCoeff;
}

void PengRobinsonAlphaGP::initThermo()
{
    for (auto& item : m_species) {
        // Read a and b coefficients and acentric factor w_ac from species input
        // information, specified in a YAML input file.
        if (item.second->input.hasKey("equation-of-state")) {
            auto eos = item.second->input["equation-of-state"].getMapWhere(
                "model", "Peng-Robinson");
            double a0 = 0;
            if (eos["a"].isScalar()) {
                a0 = eos.convert("a", "Pa*m^6/kmol^2");
            }
            double b = eos.convert("b", "m^3/kmol");
            // unitless acentric factor:
            double w = eos["acentric-factor"].asDouble();

            setSpeciesCoeffs(item.first, a0, b, w);
            if (eos.hasKey("binary-a")) {
                AnyMap& binary_a = eos["binary-a"].as<AnyMap>();
                const UnitSystem& units = binary_a.units();
                for (auto& item2 : binary_a) {
                    double a0 = 0;
                    if (item2.second.isScalar()) {
                        a0 = units.convert(item2.second, "Pa*m^6/kmol^2");
                    }
                    setBinaryCoeffs(item.first, item2.first, a0);
                }
            }
        } else {
            // Check if a and b are already populated for this species (only the
            // diagonal elements of a). If not, then search 'critProperties.xml'
            // to find critical temperature and pressure to calculate a and b.
            size_t k = speciesIndex(item.first);
            if (m_a_coeffs(k, k) == 0.0) {
                vector<double> coeffs = getCoeff(item.first);

                // Check if species was found in the database of critical
                // properties, and assign the results
                if (!isnan(coeffs[0])) {
                    setSpeciesCoeffs(item.first, coeffs[0], coeffs[1], coeffs[2]);
                }
            }
        }
    }

    // mix a
    for(size_t k = 0; k < m_kk; ++k){
        for (size_t j = 0; j < m_kk; j++) {
            if (k == j) {
                continue;
            }
            double a_kj = sqrt(m_a_coeffs(j,j) * m_a_coeffs(k,k));
            m_a_coeffs(j, k) = a_kj;
            m_a_coeffs(k, j) = a_kj;
        }
    }

    // mix alpha
    mixAlpha();
}

double PengRobinsonAlphaGP::sresid() const
{
    double smole = 0;
    double T = temperature();
    for(size_t k = 0; k < m_kk; ++k){
        smole = smole +  moleFractions_[k] * GPpredict(m_sresidGP[k], T, k);
    }
    std::cout<<"smole="<<hmole<<std::endl;
    
    return hmole- GasConstant * (mean_X(m_s0_R) - sum_xlogx()
        - std::log(pressure()/refPressure()));
    // double molarV = molarVolume();
    // double hh = m_b / molarV;
    // double zz = z();
    // double alpha_1 = daAlpha_dT();
    // double vpb = molarV + (1.0 + Sqrt2) * m_b;
    // double vmb = molarV + (1.0 - Sqrt2) * m_b;
    // double fac = alpha_1 / (2.0 * Sqrt2 * m_b);
    // double sresid_mol_R = log(zz*(1.0 - hh)) + fac * log(vpb / vmb) / GasConstant;
    // return GasConstant * sresid_mol_R;
}

double PengRobinsonAlphaGP::hresid() const
{
    double hmole = 0;
    double T = temperature();
    for(size_t k = 0; k < m_kk; ++k){
        hmole = hmole +  moleFractions_[k] * GPpredict(m_hresidGP[k], T, k);
    }
    std::cout<<"hmole="<<hmole<<std::endl;
    
    return hmole- RT() * mean_X(m_h0_RT);
    // double molarV = molarVolume();
    // double zz = z();
    // double aAlpha_1 = daAlpha_dT();
    // double T = temperature();
    // double vpb = molarV + (1 + Sqrt2) * m_b;
    // double vmb = molarV + (1 - Sqrt2) * m_b;
    // double fac = 1 / (2.0 * Sqrt2 * m_b);
    // return GasConstant * T * (zz - 1.0) + fac * log(vpb / vmb) * (T * aAlpha_1 - m_aAlpha_mix);
}

double PengRobinsonAlphaGP::liquidVolEst(double T, double& presGuess) const
{
    double v = m_b * 1.1;
    double atmp;
    double btmp;
    double aAlphatmp;
    calculateAB(atmp, btmp, aAlphatmp);
    double pres = std::max(psatEst(T), presGuess);
    double Vroot[3];
    bool foundLiq = false;
    int m = 0;
    while (m < 100 && !foundLiq) {
        int nsol = solveCubic(T, pres, atmp, btmp, aAlphatmp, Vroot);
        if (nsol == 1 || nsol == 2) {
            double pc = critPressure();
            if (pres > pc) {
                foundLiq = true;
            }
            pres *= 1.04;
        } else {
            foundLiq = true;
        }
    }

    if (foundLiq) {
        v = Vroot[0];
        presGuess = pres;
    } else {
        v = -1.0;
    }
    return v;
}

void PengRobinsonAlphaGP::setTemperature(const doublereal temp)
{
    Phase::setTemperature(temp);
    _updateReferenceStateThermo();
    // depends on mole fraction and temperature
    updateMixingExpressions();
    iState_ = phaseState(true);
}

void PengRobinsonAlphaGP::setPressure(doublereal p)
{
    // A pretty tricky algorithm is needed here, due to problems involving
    // standard states of real fluids. For those cases you need to combine the T
    // and P specification for the standard state, or else you may venture into
    // the forbidden zone, especially when nearing the triple point. Therefore,
    // we need to do the standard state thermo calc with the (t, pres) combo.
    
    double t = temperature();
    
    double rhoNow = density();
    if (forcedState_ == FLUID_UNDEFINED) {
        double rho = densityCalc(t);
        if (rho > 0.0) {
            setDensity(rho);
            iState_ = phaseState(true);
        } else {
            if (rho < -1.5) {
                rho = densityCalc(t);
                if (rho > 0.0) {
                    setDensity(rho);
                    iState_ = phaseState(true);
                } else {
                    throw CanteraError("PengRobinsonAlphaGP::setPressure",
                        "neg rho");
                }
            } else {
                throw CanteraError("PengRobinsonAlphaGP::setPressure",
                    "neg rho");
            }
        }
    } else if (forcedState_ == FLUID_GAS) {
        // Normal density calculation
        if (iState_ < FLUID_LIQUID_0) {
            double rho = densityCalc(t);
            if (rho > 0.0) {
                setDensity(rho);
                iState_ = phaseState(true);
                if (iState_ >= FLUID_LIQUID_0) {
                    throw CanteraError("PengRobinsonAlphaGP::setPressure",
                        "wrong state");
                }
            } else {
                throw CanteraError("PengRobinsonAlphaGP::setPressure",
                    "neg rho");
            }
        }
    } else if (forcedState_ > FLUID_LIQUID_0) {
        if (iState_ >= FLUID_LIQUID_0) {
            double rho = densityCalc(t);
            if (rho > 0.0) {
                setDensity(rho);
                iState_ = phaseState(true);
                if (iState_ == FLUID_GAS) {
                    throw CanteraError("PengRobinsonAlphaGP::setPressure",
                        "wrong state");
                }
            } else {
                throw CanteraError("PengRobinsonAlphaGP::setPressure",
                    "neg rho");
            }
        }
    }
}

double PengRobinsonAlphaGP::densityCalc(double T)//(double T, double presPa, int phaseRequested, double rhoGuess)
{
    double density = 0;
    for(size_t k = 0; k < m_kk; ++k){
        // std::cout<<m_density[k]<<std::endl;
        density =density +  moleFractions_[k] * GPpredict(m_denGP[k], T, k);
    }
    return density;
    // // It's necessary to set the temperature so that m_aAlpha_mix is set correctly.
    // updateAlpha(T, presPa);
    // setTemperature(T);
    // double tcrit = critTemperature();
    // double mmw = meanMolecularWeight();
    // if (rhoGuess == -1.0) {
    //     if (phaseRequested >= FLUID_LIQUID_0) {
    //         double lqvol = liquidVolEst(T, presPa);
    //         rhoGuess = mmw / lqvol;
    //     }
    // } else {
    //     // Assume the Gas phase initial guess, if nothing is specified to the routine
    //     rhoGuess = presPa * mmw / (GasConstant * T);
    // }

    // double volGuess = mmw / rhoGuess;
    // m_NSolns = solveCubic(T, presPa, m_a, m_b, m_aAlpha_mix, m_Vroot);

    // double molarVolLast = m_Vroot[0];
    // if (m_NSolns >= 2) {
    //     if (phaseRequested >= FLUID_LIQUID_0) {
    //         molarVolLast = m_Vroot[0];
    //     } else if (phaseRequested == FLUID_GAS || phaseRequested == FLUID_SUPERCRIT) {
    //         molarVolLast = m_Vroot[2];
    //     } else {
    //         if (volGuess > m_Vroot[1]) {
    //             molarVolLast = m_Vroot[2];
    //         } else {
    //             molarVolLast = m_Vroot[0];
    //         }
    //     }
    // } else if (m_NSolns == 1) {
    //     if (phaseRequested == FLUID_GAS || phaseRequested == FLUID_SUPERCRIT || phaseRequested == FLUID_UNDEFINED) {
    //         molarVolLast = m_Vroot[0];
    //     } else {
    //         return -2.0;
    //     }
    // } else if (m_NSolns == -1) {
    //     if (phaseRequested >= FLUID_LIQUID_0 || phaseRequested == FLUID_UNDEFINED || phaseRequested == FLUID_SUPERCRIT) {
    //         molarVolLast = m_Vroot[0];
    //     } else if (T > tcrit) {
    //         molarVolLast = m_Vroot[0];
    //     } else {
    //         return -2.0;
    //     }
    // } else {
    //     molarVolLast = m_Vroot[0];
    //     return -1.0;
    // }
    // return mmw / molarVolLast;
}

double PengRobinsonAlphaGP::densSpinodalLiquid() const
{
    double Vroot[3];
    double T = temperature();
    int nsol = solveCubic(T, pressure(), m_a, m_b, m_aAlpha_mix, Vroot);
    if (nsol != 3) {
        return critDensity();
    }

    auto resid = [this, T](double v) {
        double pp;
        return dpdVCalc(T, v, pp);
    };

    boost::uintmax_t maxiter = 100;
    std::pair<double, double> vv = bmt::toms748_solve(
        resid, Vroot[0], Vroot[1], bmt::eps_tolerance<double>(48), maxiter);

    double mmw = meanMolecularWeight();
    return mmw / (0.5 * (vv.first + vv.second));
}

double PengRobinsonAlphaGP::densSpinodalGas() const
{
    double Vroot[3];
    double T = temperature();
    int nsol = solveCubic(T, pressure(), m_a, m_b, m_aAlpha_mix, Vroot);
    if (nsol != 3) {
        return critDensity();
    }

    auto resid = [this, T](double v) {
        double pp;
        return dpdVCalc(T, v, pp);
    };

    boost::uintmax_t maxiter = 100;
    std::pair<double, double> vv = bmt::toms748_solve(
        resid, Vroot[1], Vroot[2], bmt::eps_tolerance<double>(48), maxiter);

    double mmw = meanMolecularWeight();
    return mmw / (0.5 * (vv.first + vv.second));
}

double PengRobinsonAlphaGP::dpdVCalc(double T, double molarVol, double& presCalc) const
{
    double denom = molarVol * molarVol + 2 * molarVol * m_b - m_b * m_b;
    double vpb = molarVol + m_b;
    double vmb = molarVol - m_b;
    double dpdv = -GasConstant * T / (vmb * vmb) + 2 * m_aAlpha_mix * vpb / (denom*denom);
    return dpdv;
}

void PengRobinsonAlphaGP::calculatePressureDerivatives() const
{
    double T = temperature();
    double mv = molarVolume();
    double pres = pressure();

    updateAlpha(T);
    mixAlpha();

    m_dpdV = dpdVCalc(T, mv, pres);
    double vmb = mv - m_b;
    double denom = mv * mv + 2 * mv * m_b - m_b * m_b;
    m_dpdT = (GasConstant / vmb - daAlpha_dT() / denom);
}

void PengRobinsonAlphaGP::updateMixingExpressions()
{
    // Update individual alpha
    //     for (size_t j = 0; j < m_kk; j++) {
    //         double critTemp_j = speciesCritTemperature(m_a_coeffs(j,j), m_b_coeffs[j]);
    //         double sqt_alpha = 1 + m_kappa[j] * (1 - sqrt(temp / critTemp_j));
    //         m_alpha[j] = sqt_alpha*sqt_alpha;
    //     }
    
    mixAlpha();
    //Update aAlpha_i, j
    //     for (size_t i = 0; i < m_kk; i++) {
    //         for (size_t j = 0; j < m_kk; j++) {
    //             m_aAlpha_binary(i, j) = sqrt(m_alpha[i] * m_alpha[j]) * m_a_coeffs(i,j);
    //         }
    //     }
    calculateAB(m_a,m_b,m_aAlpha_mix);
}

void PengRobinsonAlphaGP::calculateAB(double& aCalc, double& bCalc, double& aAlphaCalc) const
{
    bCalc = 0.0;
    aCalc = 0.0;
    aAlphaCalc = 0.0;
    for (size_t i = 0; i < m_kk; i++) {
        bCalc += moleFractions_[i] * m_b_coeffs[i];
        for (size_t j = 0; j < m_kk; j++) {
            aCalc += m_a_coeffs(i, j) * moleFractions_[i] * moleFractions_[j];
            aAlphaCalc += m_aAlpha_binary(i, j) * moleFractions_[i] * moleFractions_[j];
        }
    }
}

double PengRobinsonAlphaGP::daAlpha_dT() const
{
    int N;
    double kxi, mi, Tc, Trx, Tri, GammaT2;
    
    for (size_t k = 0; k < m_kk; ++k) {
        m_dalphadT[k] = 0;
        // N = m_AlphaKx[k].rows();
        // Tc = speciesCritTemperature(m_a_coeffs(k,k), m_b_coeffs[k]);
        // Trx = temperature() / Tc;
        // GammaT2 = 0; // m_KernelGamma[k][0] * m_KernelGamma[k][0];

        // m_dalphadT[k] = 0;
        // for (int i = 0; i < N; ++i) {
        //     kxi = m_AlphaKx[k](i,0);
        //     Tri = m_AlphaX[k](i,0);
        //     mi = m_Alpham[k](i,0);
        //     m_dalphadT[k] += -(Trx - Tri) / GammaT2 / Tc * mi * kxi;
        //     // // std::cout << m_Alpham[k] << "" << m_Alpham[k].rows() << " "<< m_Alpham[k].cols() << std::endl;
        //     // std::cout << kxi << " " << GammaT2 << " " << Tc << " "
        //     //           << mi << " " << (Trx - Tri) << std::endl;
        // }
    }

    // std::cout << "dalphadT:" << std::endl;
    // for (size_t k = 0; k < m_kk; ++k) {
    //     std::cout << m_dalphadT[k] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "alpha:" << std::endl;
    // for (size_t k = 0; k < m_kk; ++k) {
    //     std::cout << m_alpha[k] << " ";
    // }
    // std::cout << std::endl;

    // std::cout << "m_a_coeffs:" << std::endl;
    // for (size_t i = 0; i < m_kk; i++) {
    //     for (size_t j = 0; j < m_kk; j++) {
    //         std::cout << m_a_coeffs(i,j) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    // std::cout << "m_aAlpha_binary:" << std::endl;
    // for (size_t i = 0; i < m_kk; i++) {
    //     for (size_t j = 0; j < m_kk; j++) {
    //         std::cout << m_aAlpha_binary(i,j) << " ";
    //     }
    //     std::cout << std::endl;
    // }

    //Calculate mixture derivative
    double daAlphadT = 0.0;
    for (size_t i = 0; i < m_kk; i++) {
        for (size_t j = 0; j < m_kk; j++) {
            daAlphadT += moleFractions_[i] * moleFractions_[j] * 0.5 * m_aAlpha_binary(i, j)
                                             * (m_dalphadT[i] / m_alpha[i] + m_dalphadT[j] / m_alpha[j]);
        }
    }

    // std::cout << "daAlphadT:" << daAlphadT << std::endl;

    return daAlphadT;
}

double PengRobinsonAlphaGP::d2aAlpha_dT2() const
{
    int N;
    double kxi, mi, Tc, Trx, Tri, GammaT2, coeff1;
    
    for (size_t k = 0; k < m_kk; ++k) {
        m_dalphadT[k] = 0;
        m_d2alphadT2[k] = 0;
        // N = m_AlphaKx[k].rows();
        // Tc = speciesCritTemperature(m_a_coeffs(k,k), m_b_coeffs[k]);
        // Trx = temperature() / Tc;
        // GammaT2 = m_KernelGamma[k][0] * m_KernelGamma[k][0];

        // m_dalphadT[k] = 0;
        // m_d2alphadT2[k] = 0;
        // for (int i = 0; i < N; ++i) {
        //     kxi = m_AlphaKx[k](i,0);
        //     Tri = m_AlphaX[k](i,0);
        //     mi = m_Alpham[k](i,0);
        //     coeff1 = (Trx - Tri)*(Trx - Tri) / GammaT2 - 1;
        //     m_dalphadT[k] += -(Trx - Tri) / GammaT2 / Tc * mi * kxi;
        //     m_d2alphadT2[k] += coeff1 / GammaT2 / Tc / Tc * mi * kxi;
        // }
    }

    //Calculate mixture derivative
    double d2aAlphadT2 = 0.0;
    for (size_t i = 0; i < m_kk; i++) {
        double alphai = m_alpha[i];
        for (size_t j = 0; j < m_kk; j++) {
            double alphaj = m_alpha[j];
            double alphaij = alphai * alphaj;
            double term1 = m_d2alphadT2[i] / alphai + m_d2alphadT2[j] / alphaj;
            double term2 = 2 * m_dalphadT[i] * m_dalphadT[j] / alphaij;
            double term3 = m_dalphadT[i] / alphai + m_dalphadT[j] / alphaj;
            d2aAlphadT2 += 0.5 * moleFractions_[i] * moleFractions_[j] * m_aAlpha_binary(i, j)
                                       * (term1 + term2 - 0.5 * term3 * term3);
        }
    }
    return d2aAlphadT2;
}

void PengRobinsonAlphaGP::calcCriticalConditions(double& pc, double& tc, double& vc) const
{
    if (m_b <= 0.0) {
        tc = 1000000.;
        pc = 1.0E13;
        vc = omega_vc * GasConstant * tc / pc;
        return;
    }
    if (m_a <= 0.0) {
        tc = 0.0;
        pc = 0.0;
        vc = 2.0 * m_b;
        return;
    }
    tc = m_a * omega_b / (m_b * omega_a * GasConstant);
    pc = omega_b * GasConstant * tc / m_b;
    vc = omega_vc * GasConstant * tc / pc;
}

int PengRobinsonAlphaGP::solveCubic(double T, double pres, double a, double b, double aAlpha, double Vroot[3]) const
{
    // Derive the coefficients of the cubic polynomial (in terms of molar volume v) to solve.
    double bsqr = b * b;
    double RT_p = GasConstant * T / pres;
    double aAlpha_p = aAlpha / pres;
    double an = 1.0;
    double bn = (b - RT_p);
    double cn = -(2 * RT_p * b - aAlpha_p + 3 * bsqr);
    double dn = (bsqr * RT_p + bsqr * b - aAlpha_p * b);

    double tc = a * omega_b / (b * omega_a * GasConstant);
    double pc = omega_b * GasConstant * tc / b;
    double vc = omega_vc * GasConstant * tc / pc;

    int nSolnValues = MixtureFugacityTP::solveCubic(T, pres, a, b, aAlpha, Vroot, an, bn, cn, dn, tc, vc);

    return nSolnValues;
}

}